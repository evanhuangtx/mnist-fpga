"""Self-checking tests for the quantization arithmetic contract."""

import unittest

import torch

from quantize import (accumulator_bounds, choose_output_fraction, choose_weight_fraction,
                      quantize_values, reachable_bounds)


class QuantizationTestbench(unittest.TestCase):
    def test_ties_to_even_and_signed_endpoints(self) -> None:
        values = torch.tensor([0.5, 1.5, -0.5, -1.5, -128, 127], dtype=torch.float64)
        result = quantize_values(values, 0, 8)
        self.assertEqual(result.dtype, torch.int8)
        self.assertEqual(result.tolist(), [0, 2, 0, -2, -128, 127])
        self.assertEqual(quantize_values(torch.tensor([0.25]), 5, 32).tolist(), [8])
        self.assertEqual(quantize_values(torch.tensor([-2147483648., 2147483647.], dtype=torch.float64), 0, 32).tolist(), [-2147483648, 2147483647])

    def test_invalid_parameters_are_rejected_without_clipping(self) -> None:
        for value in (128., -129., float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                quantize_values(torch.tensor([value]), 0, 8)
        with self.assertRaises(ValueError):
            quantize_values(torch.tensor([2147483648.], dtype=torch.float64), 0, 32)
        with self.assertRaises(ValueError):
            choose_weight_fraction(torch.tensor([[200.]]), torch.tensor([0.]), 0, 1)

    def test_weight_precision_and_wide_accumulator_bounds(self) -> None:
        fraction, weights, biases, bounds = choose_weight_fraction(torch.tensor([[-1., 0.5]]), torch.tensor([0.25]), 0, 1)
        self.assertEqual(fraction, 7)
        self.assertEqual(weights.tolist(), [[-128, 64]])
        self.assertEqual(biases.tolist(), [32])
        self.assertEqual(bounds.tolist(), [224])
        bounds = accumulator_bounds(torch.tensor([[-128]], dtype=torch.int8), torch.tensor([-2147483648], dtype=torch.int32), 32767)
        self.assertEqual(bounds.item(), 2147483648 + 128 * 32767)

    def test_reachable_intervals_and_smallest_safe_shift(self) -> None:
        weights = torch.tensor([[2, -3], [-4, 5]], dtype=torch.int8)
        biases = torch.tensor([7, -2], dtype=torch.int32)
        lower, upper = reachable_bounds(weights, biases, torch.tensor([10, 20]))
        self.assertEqual(lower.tolist(), [-53, -42])
        self.assertEqual(upper.tolist(), [27, 98])
        self.assertEqual(choose_output_fraction(torch.tensor([-65538]), torch.tensor([65536]), 8, False), (6, 2))
        self.assertEqual(choose_output_fraction(torch.tensor([-1000000]), torch.tensor([100]), 8, True), (8, 0))
        with self.assertRaises(ValueError):
            choose_output_fraction(torch.tensor([0]), torch.tensor([65536]), 0, False)
        print("\nQuantization checks: rounding, signed endpoints, no clipping, wide overflow bounds, and interval shifts PASS")


if __name__ == '__main__':
    unittest.main(verbosity=2)
