import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from export_mem import decode_layer, encode_layer, export_parallel, read_words, write_words

class MemoryExportTestbench(unittest.TestCase):
    def parallel_fixture(self):
        return {'layers': [{'name': 'fc', 'inputs': 3, 'outputs': 10, 'shift': 0, 'relu': True}],
                'integer_state_dict': {'fc.weight': torch.arange(-8, 22, dtype=torch.int8).reshape(10, 3),
                                      'fc.bias': torch.tensor([-2147483648, -1, 0, 2147483647, 4, 5, 6, 7, -15, 9], dtype=torch.int32)}}

    def test_parallel_lane_order_signs_and_padding(self):
        artifact = self.parallel_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            result = export_parallel(path, artifact)
            weights = (path / 'weights_x8.mem').read_text().splitlines()
            biases = (path / 'biases_x8.mem').read_text().splitlines()
            self.assertEqual(len(weights), 6)
            self.assertEqual(weights[0], '0d0a070401fefbf8')
            self.assertEqual(weights[3], '0000000000001310')
            self.assertEqual(biases[0][-32:], '7fffffff00000000ffffffff80000000')
            self.assertEqual(biases[1], '0' * 48 + '00000009fffffff1')
            self.assertEqual(result['config']['layers'][0]['groups'], 2)
            for key, expected in artifact['integer_state_dict'].items():
                np.testing.assert_array_equal(result['artifact']['integer_state_dict'][key].numpy(), expected.numpy())

    def test_one_lane_export_preserves_scalar_order(self):
        artifact = self.parallel_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            export_parallel(path, artifact, 1)
            self.assertEqual(read_words(path / 'weights_x8.mem', 8, True, 30),
                             artifact['integer_state_dict']['fc.weight'].reshape(-1).tolist())
            self.assertEqual(read_words(path / 'biases_x8.mem', 32, True, 10),
                             artifact['integer_state_dict']['fc.bias'].tolist())

    def test_parallel_rejects_invalid_lane_count(self):
        with tempfile.TemporaryDirectory() as directory:
            for lanes in (0, 3, -1):
                with self.assertRaises(ValueError):
                    export_parallel(Path(directory), self.parallel_fixture(), lanes)

    def test_signed_endpoints_and_negative_hex_words(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.mem'
            for width, values, expected in (
                (8, [-128, -1, 0, 127], ['80', 'ff', '00', '7f']),
                (32, [-2147483648, -1, 0, 2147483647], ['80000000', 'ffffffff', '00000000', '7fffffff'])):
                write_words(path, values, width, True)
                self.assertEqual(path.read_text().splitlines(), expected)
                self.assertEqual(read_words(path, width, True, 4), values)
            write_words(path, [0, 7], 8, False)
            self.assertEqual(read_words(path, 8, False, 2), [0, 7])

    def test_malformed_files_and_overflow_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.mem'
            for text in ('0x01\n', '1\n', 'gg\n', '01\n02\n', '//comment\n'):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    read_words(path, 8, True, 1)
            for value in (-129, 128, 1.5):
                with self.assertRaises(ValueError):
                    write_words(path, [value], 8, True)
            with self.assertRaises(ValueError):
                write_words(path, [-1], 8, False)

    def test_layer_rom_matches_manual_field_encoding(self):
        cases = [
            ({'type': 'dense', 'inputs': 784, 'outputs': 64, 'weight_base': 0, 'bias_base': 0, 'shift': 0, 'relu': True},
             '00' + '0310' + '0040' + '00000000' + '00000000' + '00' + '01' + '00'),
            ({'type': 'dense', 'inputs': 64, 'outputs': 10, 'weight_base': 50176, 'bias_base': 64, 'shift': 7, 'relu': False},
             '00' + '0040' + '000a' + '0000c400' + '00000040' + '07' + '00' + '00')]
        for layer, expected in cases:
            encoded = encode_layer(layer)
            self.assertEqual(f'{encoded:032x}', expected)
            self.assertEqual(decode_layer(encoded), layer)
            with self.assertRaises(ValueError):
                decode_layer(encoded | 1)
        print('\nMemory checks: signed hex endpoints and both manually encoded 128-bit layer records PASS')

if __name__ == '__main__':
    unittest.main(verbosity=2)
