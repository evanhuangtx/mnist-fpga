"""Self-checking tests for exact dataset-level dead-neuron measurement."""

import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from count_dead_neurons import measure_hidden
from model import MNISTMLP


class HiddenNeuronTestbench(unittest.TestCase):
    def setUp(self) -> None:
        self.model = MNISTMLP()
        with torch.no_grad():
            self.model.fc1.weight.zero_()
            self.model.fc1.bias.fill_(-1)
            self.model.fc1.weight[0, 0] = 1
            self.model.fc1.bias[0] = -0.5
            self.model.fc1.bias[1] = 0
            self.model.fc1.bias[3] = 1
            self.model.fc1.bias[4] = 1e-9
        inputs = torch.zeros(5, 784)
        inputs[3:, 0] = 1
        self.dataset = TensorDataset(inputs, torch.zeros(5, dtype=torch.int64))

    def test_known_dead_neurons_and_tiny_positive_activation(self) -> None:
        original = {name: value.clone() for name, value in self.model.state_dict().items()}
        metrics = measure_hidden(self.model, DataLoader(self.dataset, batch_size=2))
        self.assertEqual(metrics['images'], 5)
        self.assertEqual(metrics['dead_count'], 61)
        self.assertEqual(metrics['active_count'], 3)
        self.assertEqual(metrics['dead_indices'], [i for i in range(64) if i not in (0, 3, 4)])
        self.assertEqual(metrics['positive_counts'][:5], [2, 0, 0, 5, 5])
        self.assertEqual(metrics['positive_fractions'][:5], [0.4, 0.0, 0.0, 1.0, 1.0])
        self.assertEqual(metrics['maximum_activations'][:4], [0.5, 0.0, 0.0, 1.0])
        self.assertGreater(metrics['maximum_activations'][4], 0)
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)
        print("\nHidden-neuron check: expected 61 dead / 3 active; exact zero and tiny positive cases PASS")

    def test_counts_are_independent_of_batch_boundaries(self) -> None:
        reference = measure_hidden(self.model, DataLoader(self.dataset, batch_size=5))
        for size in (1, 2, 3):
            self.assertEqual(measure_hidden(self.model, DataLoader(self.dataset, batch_size=size)), reference)

    def test_empty_and_nonfinite_data_are_rejected(self) -> None:
        empty = TensorDataset(torch.zeros(0, 784), torch.zeros(0, dtype=torch.int64))
        with self.assertRaisesRegex(ValueError, 'empty dataset'):
            measure_hidden(self.model, DataLoader(empty))
        with torch.no_grad():
            self.model.fc1.bias[0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Non-finite hidden activation'):
            measure_hidden(self.model, DataLoader(self.dataset))


if __name__ == '__main__':
    unittest.main(verbosity=2)
