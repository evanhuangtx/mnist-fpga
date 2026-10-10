"""Train the binary-input MLP, select on validation, and evaluate the test set once."""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from data import PROJECT_ROOT, load_mnist
from model import MNISTMLP


def shift_images(inputs: torch.Tensor, max_shift: int) -> torch.Tensor:
    """Random zero-filled translations; preserve binary pixels and image size."""
    if max_shift == 0:
        return inputs
    images = nn.functional.pad(inputs.reshape(-1, 28, 28), (max_shift,) * 4)
    starts = torch.randint(0, 2 * max_shift + 1, (len(inputs), 2))
    rows = starts[:, 0, None, None] + torch.arange(28)[None, :, None]
    columns = starts[:, 1, None, None] + torch.arange(28)[None, None, :]
    return images[torch.arange(len(inputs))[:, None, None], rows, columns].reshape(-1, 784)


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                max_shift: int = 0) -> dict:
    model.train()
    total_loss, correct, count = 0.0, 0, 0
    for inputs, labels in loader:
        inputs = shift_images(inputs, max_shift)
        optimizer.zero_grad(set_to_none=True)
        scores = model(inputs)
        loss = nn.functional.cross_entropy(scores, labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (scores.argmax(dim=1) == labels).sum().item()
        count += len(labels)
    return {"loss": total_loss / count, "accuracy": correct / count, "correct": correct, "count": count}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> dict:
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    for inputs, labels in loader:
        scores = model(inputs)
        loss = nn.functional.cross_entropy(scores, labels, reduction="sum")
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite evaluation loss")
        total_loss += loss.item()
        correct += (scores.argmax(dim=1) == labels).sum().item()
        count += len(labels)
    return {"loss": total_loss / count, "accuracy": correct / count, "correct": correct, "count": count}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-shift", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "artifacts" / "mlp_float.pt")
    parser.add_argument("--results", type=Path, default=PROJECT_ROOT / "docs" / "training_results.json")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.threads < 1:
        parser.error("epochs, batch-size, lr, and threads must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    full_train, test = load_mnist(download=False)
    train, validation = random_split(full_train, [55000, 5000], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed + 1))
    validation_loader = DataLoader(validation, batch_size=1024)
    test_loader = DataLoader(test, batch_size=1024)
    model = MNISTMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr / 10)
    settings = {
        "architecture": [784, 64, 10], "hidden_activation": "ReLU", "output": "raw scores",
        "binarization": "raw uint8 pixel >= 128", "flatten_order": "row-major",
        "training_augmentation": "random zero-filled translations", "max_shift_pixels": args.max_shift,
        "epochs": args.epochs, "batch_size": args.batch_size, "seed": args.seed,
        "training_images": len(train), "validation_images": len(validation), "test_images": len(test),
        "split": "torch random_split, generator seed=seed; shuffle generator seed=seed+1",
        "optimizer": "Adam", "initial_lr": args.lr, "weight_decay": 0.0001,
        "scheduler": "CosineAnnealingLR", "minimum_lr": args.lr / 10,
        "loss": "CrossEntropyLoss", "device": "cpu", "threads": args.threads,
        "deterministic_algorithms": True, "python_version": sys.version.split()[0],
        "torch_version": str(torch.__version__), "numpy_version": np.__version__,
    }
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history, best_key = [], (-1.0, float('-inf'))
    for epoch in range(1, args.epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        train_metrics = train_epoch(model, train_loader, optimizer, args.max_shift)
        validation_metrics = evaluate(model, validation_loader)
        history.append({"epoch": epoch, "lr": lr, "train": train_metrics, "validation": validation_metrics})
        key = (validation_metrics["accuracy"], -validation_metrics["loss"])
        if key > best_key:
            best_key = key
            torch.save({"model_state_dict": model.state_dict(), "settings": settings,
                        "epoch": epoch, "validation": validation_metrics}, args.checkpoint)
        print(f"Epoch {epoch:02d}/{args.epochs}: train={train_metrics['accuracy']:.2%} "
              f"val={validation_metrics['accuracy']:.2%} loss={validation_metrics['loss']:.4f}", flush=True)
        scheduler.step()

    # Test accuracy never participates in checkpoint selection.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader)
    checkpoint["test"] = test_metrics
    torch.save(checkpoint, args.checkpoint)
    report = {"settings": settings, "selected_epoch": checkpoint["epoch"],
              "selected_validation": checkpoint["validation"], "test": test_metrics,
              "target_accuracy": 0.97, "target_met": test_metrics["accuracy"] >= 0.97,
              "elapsed_seconds": time.perf_counter() - started, "history": history}
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Selected epoch: {checkpoint['epoch']}", flush=True)
    print(f"Test accuracy: {test_metrics['accuracy']:.2%} ({test_metrics['correct']}/{test_metrics['count']})", flush=True)
    print(f"97% target: {'PASS' if report['target_met'] else 'NOT MET'}", flush=True)
    print(f"Checkpoint: {args.checkpoint}\nResults: {args.results}", flush=True)


if __name__ == "__main__":
    main()
