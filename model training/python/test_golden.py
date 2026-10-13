"""Test integer inference against hand values and independent sequential Python MACs."""

import copy
import unittest

import numpy as np
import torch
from torchvision.datasets import MNIST

from data import DATA_ROOT, PROJECT_ROOT
from golden import IntegerMLP, dense_integer, requantize


def scalar_dense(images, weights, biases, shift, relu):
    results = []
    for image in images:
        row = []
        for neuron_weights, bias in zip(weights, biases):
            acc = int(bias)
            for activation, weight in zip(image, neuron_weights):
                acc += int(activation) * int(weight)
                if not -2147483648 <= acc <= 2147483647:
                    raise OverflowError('Sequential reference overflow')
            value = acc // (2 ** shift)
            value = min(32767, max(-32768, value))
            row.append(max(0, value) if relu else value)
        results.append(row)
    return np.array(results, dtype=np.int16)


class GoldenTestbench(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifact = torch.load(PROJECT_ROOT / 'model training' / 'artifacts' / 'mlp_quantized.pt', weights_only=True)

    def test_shift_saturation_and_relu_boundaries(self):
        acc = np.array([[7, -7, 65536, -65538, 0]], dtype=np.int64)
        np.testing.assert_array_equal(requantize(acc, 1, False), [[3, -4, 32767, -32768, 0]])
        np.testing.assert_array_equal(requantize(acc, 1, True), [[3, 0, 32767, 0, 0]])
        limits = np.array([[-2147483648, 2147483647]], dtype=np.int64)
        np.testing.assert_array_equal(requantize(limits, 31, False), [[-1, 0]])
        np.testing.assert_array_equal(requantize(np.array([[-32768, 32767]], dtype=np.int64), 0, False), [[-32768, 32767]])

    def test_hand_calculated_dense_biases_and_signed_products(self):
        inputs = np.array([[1, 0], [0, 1]], dtype=np.int16)
        weights = np.array([[2, -3], [-4, 5]], dtype=np.int8)
        biases = np.array([1, -2], dtype=np.int32)
        actual = dense_integer(inputs, weights, biases, 1, False, 1)
        np.testing.assert_array_equal(actual, [[1, -3], [-1, 1]])
        self.assertEqual(actual.dtype, np.int16)
        # Saturation occurs before narrowing to 16-bit storage.
        np.testing.assert_array_equal(dense_integer(np.array([[0]], dtype=np.int16),
            np.array([[0], [0]], dtype=np.int8), np.array([65536, -65538], dtype=np.int32), 1, False, 1),
            [[32767, -32768]])

    def test_random_dense_matches_sequential_reference(self):
        rng = np.random.default_rng(42)
        inputs = rng.integers(-32768, 32768, size=(9, 7), dtype=np.int16)
        weights = rng.integers(-128, 128, size=(5, 7), dtype=np.int16).astype(np.int8)
        biases = rng.integers(-10000, 10000, size=5, dtype=np.int32)
        for shift in (0, 3, 12, 31):
            for relu in (False, True):
                np.testing.assert_array_equal(dense_integer(inputs, weights, biases, shift, relu),
                    scalar_dense(inputs.tolist(), weights.tolist(), biases.tolist(), shift, relu))

    def test_prefix_overflow_is_rejected_even_when_final_sum_fits(self):
        with self.assertRaisesRegex(OverflowError, 'MAC prefix'):
            dense_integer(np.array([[1, 1]], dtype=np.int16), np.array([[127, -127]], dtype=np.int8),
                          np.array([2147483640], dtype=np.int32), 0, False, 1)

    def test_invalid_inputs_and_scale_configuration_are_rejected(self):
        model = IntegerMLP(copy.deepcopy(self.artifact))
        for inputs in (np.zeros((1, 784), dtype=np.float32), np.full((1, 784), 2, dtype=np.int16),
                       np.zeros((1, 783), dtype=np.int16)):
            with self.assertRaises(ValueError):
                model.infer(inputs)
        for shift in (-1, 32, 1.5):
            with self.assertRaises(ValueError):
                requantize(np.array([[1]], dtype=np.int64), shift, False)
        broken = copy.deepcopy(self.artifact)
        broken['layers'][1]['shift'] += 1
        with self.assertRaisesRegex(ValueError, 'scale or shift'):
            IntegerMLP(broken)

    def test_signed_argmax_ties_choose_lowest_digit(self):
        artifact = copy.deepcopy(self.artifact)
        for tensor in artifact['integer_state_dict'].values():
            tensor.zero_()
        scores = [-8, -3, -3, -10, -7, -9, -6, -5, -4, -11]
        artifact['integer_state_dict']['fc2.bias'][:] = torch.tensor(scores, dtype=torch.int32) << artifact['layers'][1]['shift']
        outputs = IntegerMLP(artifact).infer(np.zeros((1, 784), dtype=np.int16))
        np.testing.assert_array_equal(outputs['scores'], [scores])
        self.assertEqual(outputs['digits'].tolist(), [1])

    def test_real_mnist_layers_match_sequential_reference(self):
        raw = MNIST(root=str(DATA_ROOT), train=False, download=False)
        inputs = (raw.data[:10].numpy() >= 128).astype(np.int16).reshape(10, 784)
        original = inputs.copy()
        model = IntegerMLP(self.artifact)
        outputs = model.infer(inputs)
        hidden = scalar_dense(inputs.tolist(), *[a.tolist() for a in model.parameters['fc1']],
                              model.layers[0]['shift'], True)
        scores = scalar_dense(hidden.tolist(), *[a.tolist() for a in model.parameters['fc2']],
                              model.layers[1]['shift'], False)
        np.testing.assert_array_equal(outputs['hidden'], hidden)
        np.testing.assert_array_equal(outputs['scores'], scores)
        np.testing.assert_array_equal(outputs['digits'], scores.argmax(axis=1))
        np.testing.assert_array_equal(inputs, original)
        self.assertEqual(outputs['hidden'].dtype, np.int16)
        self.assertEqual(outputs['scores'].dtype, np.int16)
        separate = [model.infer(image[None, :]) for image in inputs]
        np.testing.assert_array_equal(outputs['hidden'], np.concatenate([o['hidden'] for o in separate]))
        np.testing.assert_array_equal(outputs['scores'], np.concatenate([o['scores'] for o in separate]))
        print('\nSequential reference: all 640 hidden values, 100 output scores, and 10 predictions match exactly; every scalar MAC prefix fits int32')


if __name__ == '__main__':
    unittest.main(verbosity=2)
