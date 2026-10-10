"""Check the training/evaluation loops without a full MNIST training run."""

import math
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import MNISTMLP
from train import evaluate, shift_images, train_epoch


class TrainingTestbench(unittest.TestCase):
    def test_evaluation_counts_and_uneven_batches(self) -> None:
        model = nn.Sequential(nn.Linear(784, 10), nn.Dropout(0.9))
        with torch.no_grad():
            model[0].weight.zero_()
            model[0].bias.zero_()
        loader = DataLoader(TensorDataset(torch.zeros(5, 784), torch.tensor([0, 1, 0, 2, 0])), batch_size=2)
        metrics = evaluate(model, loader)
        self.assertEqual(metrics['count'], 5)
        self.assertEqual(metrics['correct'], 3)
        self.assertEqual(metrics['accuracy'], 0.6)
        self.assertAlmostEqual(metrics['loss'], math.log(10), places=6)
        self.assertFalse(model.training)

    def test_training_loop_learns_simple_binary_examples(self) -> None:
        torch.manual_seed(42)
        model = MNISTMLP()
        inputs = torch.zeros(16, 784)
        inputs[8:, 0] = 1
        labels = torch.tensor([0] * 8 + [1] * 8)
        loader = DataLoader(TensorDataset(inputs, labels), batch_size=5)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
        before = evaluate(model, loader)['loss']
        for _ in range(20):
            metrics = train_epoch(model, loader, optimizer)
        after = evaluate(model, loader)
        self.assertEqual(metrics['count'], 16)
        self.assertEqual(after['accuracy'], 1.0)
        self.assertLess(after['loss'], before / 10)
        print(f"\nTraining loop: synthetic loss {before:.6f} -> {after['loss']:.6f}, accuracy={after['accuracy']:.0%}")

    def test_random_shifts_preserve_binary_pixels_and_use_zero_padding(self) -> None:
        inputs = torch.ones(32, 784)
        torch.manual_seed(42)
        shifted = shift_images(inputs, 2).reshape(-1, 28, 28)
        self.assertEqual(tuple(shifted.shape), (32, 28, 28))
        self.assertTrue(((shifted == 0) | (shifted == 1)).all().item())
        self.assertTrue((shifted[:, 2:26, 2:26] == 1).all().item())
        self.assertTrue((shifted == 0).any().item())
        torch.testing.assert_close(shift_images(inputs, 0), inputs, rtol=0, atol=0)
        # An isolated central pixel must survive and move by at most two pixels.
        impulses = torch.zeros(32, 784)
        impulses[:, 14 * 28 + 14] = 1
        moved = shift_images(impulses, 2).reshape(-1, 28, 28)
        self.assertTrue((moved.sum(dim=(1, 2)) == 1).all().item())
        locations = moved.nonzero()
        self.assertTrue(((locations[:, 1:] >= 12) & (locations[:, 1:] <= 16)).all().item())

    def test_nonfinite_scores_are_rejected(self) -> None:
        model = nn.Linear(784, 10)
        with torch.no_grad():
            model.bias.fill_(float('nan'))
        loader = DataLoader(TensorDataset(torch.zeros(1, 784), torch.tensor([0])))
        with self.assertRaisesRegex(RuntimeError, 'Non-finite evaluation loss'):
            evaluate(model, loader)
        with self.assertRaisesRegex(RuntimeError, 'Non-finite training loss'):
            train_epoch(model, loader, torch.optim.SGD(model.parameters(), lr=0.01))


if __name__ == '__main__':
    unittest.main(verbosity=2)
