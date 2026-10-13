"""Measure hidden ReLU neurons that never activate on unaugmented MNIST."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import PROJECT_ROOT, load_mnist
from model import MNISTMLP


@torch.no_grad()
def measure_hidden(model: MNISTMLP, loader: DataLoader) -> dict:
    """A neuron is dataset-dead only if its output is exactly zero for every image."""
    model.eval()
    positive_counts = torch.zeros(model.fc1.out_features, dtype=torch.int64)
    maximum = torch.zeros(model.fc1.out_features)
    count = 0
    for inputs, _ in loader:
        hidden = model.relu(model.fc1(inputs))
        if not torch.isfinite(hidden).all():
            raise ValueError("Non-finite hidden activation")
        positive_counts += (hidden > 0).sum(dim=0)
        maximum = torch.maximum(maximum, hidden.amax(dim=0))
        count += len(inputs)
    if count == 0:
        raise ValueError("Cannot measure dead neurons on an empty dataset")
    dead = (positive_counts == 0).nonzero().flatten().tolist()
    return {
        "images": count, "hidden_neurons": len(positive_counts),
        "dead_count": len(dead), "dead_indices": dead,
        "active_count": len(positive_counts) - len(dead),
        "positive_counts": positive_counts.tolist(),
        "positive_fractions": (positive_counts.double() / count).tolist(),
        "maximum_activations": maximum.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "model training" / "artifacts" / "mlp_float.pt")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "docs" / "dead_neurons.json")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error("batch-size and threads must be positive")
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = MNISTMLP()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    train, test = load_mnist(download=False)
    results = {}
    for name, dataset in (("train", train), ("test", test)):
        metrics = measure_hidden(model, DataLoader(dataset, batch_size=args.batch_size))
        results[name] = metrics
        print(f"{name}: {metrics['images']:,} images; dead={metrics['dead_count']}/64; "
              f"active={metrics['active_count']}/64; dead indices={metrics['dead_indices']}")
    combined_dead = sorted(set(results["train"]["dead_indices"]) & set(results["test"]["dead_indices"]))
    report = {
        "definition": "ReLU output is exactly zero for every image in the named dataset; no epsilon threshold",
        "indexing": "zero-based hidden neuron indices, 0-63",
        "data": "Unaugmented binarized MNIST; train includes all 60,000 official training images (including validation holdout)",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "checkpoint_epoch": checkpoint["epoch"], "batch_size": args.batch_size,
        "torch_version": str(torch.__version__), **results,
        "combined": {"images": len(train) + len(test), "dead_count": len(combined_dead),
                     "dead_indices": combined_dead},
        "interpretation": "Dataset-level inactivity does not prove a neuron is inactive for every possible input.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Combined: {len(combined_dead)}/64 dead across all 70,000 images")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
