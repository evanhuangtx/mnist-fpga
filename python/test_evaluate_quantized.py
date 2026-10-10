"""Check accuracy accounting and the one-percentage-point acceptance threshold."""

import unittest

import numpy as np

from evaluate_quantized import summarize_predictions


class QuantizedEvaluationTestbench(unittest.TestCase):
    def test_gains_losses_disagreements_and_confusion(self):
        result = summarize_predictions(np.array([0, 1, 2, 3]), np.array([0, 1, 0, 0]), np.array([0, 0, 2, 1]))
        self.assertEqual(result['float_correct'], 2)
        self.assertEqual(result['integer_correct'], 2)
        self.assertEqual(result['prediction_disagreements'], 3)
        self.assertEqual(result['float_correct_integer_wrong'], 1)
        self.assertEqual(result['float_wrong_integer_correct'], 1)
        self.assertEqual(result['accuracy_drop_percentage_points'], 0)
        confusion = np.array(result['integer_confusion_matrix_rows_true_columns_predicted'])
        self.assertEqual(int(confusion.sum()), 4)
        self.assertEqual(int(confusion.trace()), 2)

    def test_acceptance_boundary_is_one_percentage_point(self):
        labels = np.zeros(10000, dtype=np.int64)
        float_digits = np.zeros_like(labels)
        integer_digits = np.zeros_like(labels)
        integer_digits[:100] = 1
        exact = summarize_predictions(labels, float_digits, integer_digits)
        self.assertEqual(exact['accuracy_drop_percentage_points'], 1.0)
        self.assertTrue(exact['within_one_percentage_point'])
        integer_digits[100] = 1
        self.assertFalse(summarize_predictions(labels, float_digits, integer_digits)['within_one_percentage_point'])

    def test_inconsistent_or_invalid_labels_are_rejected(self):
        with self.assertRaises(ValueError):
            summarize_predictions(np.array([0, 1]), np.array([0]), np.array([0, 1]))
        with self.assertRaises(ValueError):
            summarize_predictions(np.array([10]), np.array([0]), np.array([0]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
