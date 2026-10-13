"""Test signed ROM encoding, strict file parsing, and packed layer field layout."""

import tempfile
import unittest
from pathlib import Path

from export_mem import decode_layer, encode_layer, read_words, write_words


class MemoryExportTestbench(unittest.TestCase):
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
