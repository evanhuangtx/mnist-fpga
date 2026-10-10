"""Load MNIST as row-major, binary float32 inputs and integer class labels."""

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import TensorDataset
from torchvision.datasets import MNIST

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"


def binarize_images(images: torch.Tensor) -> torch.Tensor:
    """Convert raw (N, 28, 28) uint8 pixels to (N, 784) float32 zeros/ones."""
    if images.dtype != torch.uint8 or images.ndim != 3 or images.shape[1:] != (28, 28):
        raise ValueError("Expected raw uint8 images with shape (N, 28, 28)")
    return (images >= 128).reshape(-1, 784).to(torch.float32)


def load_mnist(root: Path = DATA_ROOT, download: bool = True) -> tuple[TensorDataset, TensorDataset]:
    """Return the official training and test splits without normalizing pixels."""
    datasets = []
    for train in (True, False):
        raw = MNIST(root=str(root), train=train, download=download)
        datasets.append(TensorDataset(binarize_images(raw.data), raw.targets))
    return datasets[0], datasets[1]


def save_preview(root: Path, test: TensorDataset, output: Path) -> None:
    """Show the first 20 test digits: original on the left, binary on the right."""
    raw = MNIST(root=str(root), train=False, download=False)
    canvas = Image.new("RGB", (940, 480), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)
    for index in range(20):
        x, y = (index % 5) * 188 + 8, (index // 5) * 120 + 4
        label = int(test.tensors[1][index])
        draw.text((x, y), f"#{index}: label {label}", fill="black", font=font)
        original = Image.fromarray(raw.data[index].numpy())
        binary = Image.fromarray((test.tensors[0][index].reshape(28, 28) * 255).to(torch.uint8).numpy())
        for offset, image in ((0, original), (88, binary)):
            canvas.paste(image.resize((84, 84), Image.Resampling.NEAREST), (x + offset, y + 24))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DATA_ROOT, help="MNIST download directory")
    parser.add_argument("--preview", type=Path, default=PROJECT_ROOT / "docs" / "mnist_samples.png")
    args = parser.parse_args()

    # Exercise the threshold boundary and distinguish row-major from column-major order.
    probe = torch.zeros((1, 28, 28), dtype=torch.uint8)
    probe[0, 0, :4] = torch.tensor([0, 127, 128, 255], dtype=torch.uint8)
    probe[0, 1, 2] = 128
    expected = torch.zeros((1, 784))
    expected[0, [2, 3, 30]] = 1
    assert torch.equal(binarize_images(probe), expected), "Threshold or flatten order is incorrect"

    train, test = load_mnist(args.root)
    for name, dataset, count in (("train", train, 60000), ("test", test, 10000)):
        images, labels = dataset.tensors
        assert images.shape == (count, 784) and images.dtype == torch.float32
        assert labels.shape == (count,) and labels.dtype == torch.int64
        assert torch.all((images == 0) | (images == 1)), "Inputs must be binary"
        assert torch.all((labels >= 0) & (labels <= 9)), "Labels must be digits 0-9"
        print(f"{name}: {count:,} images, shape={tuple(images.shape)}, pixels=0/1, labels=0-9")
    save_preview(args.root, test, args.preview)
    print("PASS: threshold boundary, row-major order, split sizes, dtypes, and value ranges")
    print(f"Preview: {args.preview}")


if __name__ == "__main__":
    main()
