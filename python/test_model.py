"""Self-checking Python testbenches for the floating-point MNIST MLP.

Run from the project root: .venv/Scripts/python.exe -m unittest discover -s python -p test_model.py -v
"""

import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data import load_mnist
from model import MNISTMLP


def known_model() -> MNISTMLP:
    """Use small integer parameters whose outputs can be calculated by hand."""
    model = MNISTMLP()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.fc1.weight[0, 0] = 2
        model.fc1.bias[0] = -1
        model.fc1.weight[1, 1] = -3
        model.fc1.bias[1] = 2
        model.fc1.weight[2, 783] = 4
        model.fc1.bias[2] = -2
        model.fc2.weight[0, :3] = torch.tensor([3, -2, 1])
        model.fc2.bias[0] = -5
        model.fc2.weight[1, :3] = torch.tensor([-1, 4, -3])
        model.fc2.bias[1] = 1
        model.fc2.bias[9] = -2
    return model


def known_inputs() -> torch.Tensor:
    inputs = torch.zeros((2, 784))
    inputs[1, [0, 1, 783]] = 1
    return inputs


class ModelTestbench(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_architecture_and_parameter_counts(self) -> None:
        model = MNISTMLP()
        self.assertEqual([type(block) for block in model.children()], [nn.Linear, nn.ReLU, nn.Linear])
        self.assertEqual(tuple(model.fc1.weight.shape), (64, 784))
        self.assertEqual(tuple(model.fc2.weight.shape), (10, 64))
        self.assertEqual(sum(p.numel() for name, p in model.named_parameters() if name.endswith('weight')), 50816)
        self.assertEqual(sum(p.numel() for name, p in model.named_parameters() if name.endswith('bias')), 74)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 50890)

    def test_hand_calculated_scores_and_predictions(self) -> None:
        # Zero image: h=[0,2,0], scores 0/1/9=[-9,9,-2].
        # Pixels 0,1,783 set: h=[1,0,2], scores 0/1/9=[0,-6,-2].
        expected = torch.zeros((2, 10))
        expected[0, [0, 1, 9]] = torch.tensor([-9, 9, -2], dtype=torch.float32)
        expected[1, [0, 1, 9]] = torch.tensor([0, -6, -2], dtype=torch.float32)
        actual = known_model()(known_inputs())
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(actual.argmax(dim=1).tolist(), [1, 0])
        # Exact negative scores also catch an unwanted output ReLU or softmax.
        print(f"\nHand calculation: expected={expected.tolist()} actual={actual.tolist()}")

    def test_relu_below_at_and_above_zero(self) -> None:
        model = known_model()
        with torch.no_grad():
            model.fc1.weight[0].zero_()
            model.fc1.weight[0, :2] = 1
            model.fc1.bias[0] = -1
        inputs = torch.zeros((3, 784))
        inputs[1, 0] = 1
        inputs[2, :2] = 1
        before = model.fc1(inputs)[:, 0]
        after = model.relu(model.fc1(inputs))[:, 0]
        torch.testing.assert_close(before, torch.tensor([-1., 0., 1.]), rtol=0, atol=0)
        torch.testing.assert_close(after, torch.tensor([0., 0., 1.]), rtol=0, atol=0)

    def test_matches_independent_numpy_reference(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.integers(0, 2, size=(17, 784), dtype=np.int64)
        w1 = rng.integers(-3, 4, size=(64, 784), dtype=np.int64)
        b1 = rng.integers(-4, 5, size=64, dtype=np.int64)
        w2 = rng.integers(-3, 4, size=(10, 64), dtype=np.int64)
        b2 = rng.integers(-4, 5, size=10, dtype=np.int64)
        expected = np.maximum(x @ w1.T + b1, 0) @ w2.T + b2
        model = MNISTMLP()
        with torch.no_grad():
            for parameter, values in ((model.fc1.weight, w1), (model.fc1.bias, b1),
                                      (model.fc2.weight, w2), (model.fc2.bias, b2)):
                parameter.copy_(torch.from_numpy(values))
            actual = model(torch.from_numpy(x).float()).numpy()
        np.testing.assert_array_equal(actual, expected)
        print("\nNumPy reference: all 170 output scores match exactly")

    def test_batch_sizes_and_input_preservation(self) -> None:
        model = MNISTMLP().eval()
        for count in (1, 7, 32):
            with self.subTest(batch_size=count), torch.no_grad():
                inputs = torch.randint(0, 2, (count, 784)).float()
                original = inputs.clone()
                scores = model(inputs)
                separate = torch.cat([model(row.unsqueeze(0)) for row in inputs])
                self.assertEqual(tuple(scores.shape), (count, 10))
                self.assertTrue(torch.isfinite(scores).all().item())
                torch.testing.assert_close(scores, separate, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(inputs, original, rtol=0, atol=0)

    def test_cross_entropy_gradients_and_optimizer_step(self) -> None:
        model = known_model()
        inputs = known_inputs()
        labels = torch.tensor([1, 0])
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss = criterion(model(inputs), labels)
        before_loss = loss.item()
        before_parameters = {name: p.detach().clone() for name, p in model.named_parameters()}
        loss.backward()
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all().item())
                self.assertGreater(parameter.grad.abs().sum().item(), 0)
        # Neuron 1 is negative for the second sample and positive for the first.
        # The first sample is all zeros, so its input-weight gradient is also zero.
        torch.testing.assert_close(model.fc1.weight.grad[1], torch.zeros(784), rtol=0, atol=0)
        self.assertGreater(model.fc1.weight.grad[0].abs().sum().item(), 0)
        optimizer.step()
        for name, parameter in model.named_parameters():
            self.assertFalse(torch.equal(parameter.detach(), before_parameters[name]), name)
        after_loss = criterion(model(inputs), labels).item()
        self.assertLess(after_loss, before_loss)
        print(f"\nOptimizer check: cross-entropy {before_loss:.6f} -> {after_loss:.6f}")


class MNISTIntegrationTestbench(unittest.TestCase):
    def test_real_mnist_batches(self) -> None:
        # Deliberately require the cached step-2 dataset; tests do not download data.
        train, test = load_mnist(download=False)
        torch.manual_seed(0)
        model = MNISTMLP()
        for name, dataset in (("train", train), ("test", test)):
            with self.subTest(split=name):
                inputs, labels = next(iter(DataLoader(dataset, batch_size=32, shuffle=False)))
                self.assertEqual(tuple(inputs.shape), (32, 784))
                self.assertTrue(((inputs == 0) | (inputs == 1)).all().item())
                scores = model(inputs)
                self.assertEqual(tuple(scores.shape), (32, 10))
                self.assertTrue(torch.isfinite(scores).all().item())
                loss = nn.CrossEntropyLoss()(scores, labels)
                self.assertTrue(torch.isfinite(loss).item())
                model.zero_grad()
                loss.backward()
                for parameter in model.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all().item())
                print(f"\nMNIST {name}: input=(32,784), scores=(32,10), finite loss={loss.item():.6f}, backward PASS")


if __name__ == "__main__":
    unittest.main(verbosity=2)
