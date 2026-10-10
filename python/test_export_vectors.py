"""Test UART pixel bit order and signed test-vector file encoding."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from export_vectors import pack_images, read_vector, unpack_images, write_vector


class VectorExportTestbench(unittest.TestCase):
    def test_pixel_order_byte_boundaries_and_complete_round_trip(self):
        images = np.zeros((2, 784), dtype=np.int16)
        images[0, [0, 2, 7, 8, 783]] = 1
        images[1] = 1
        packed = pack_images(images)
        self.assertEqual(packed.shape, (2, 98))
        self.assertEqual(packed[0, :2].tolist(), [0x85, 0x01])
        self.assertEqual(int(packed[0, -1]), 0x80)
        self.assertTrue((packed[1] == 0xff).all())
        np.testing.assert_array_equal(unpack_images(packed), images)

    def test_signed_vector_words_and_image_major_order(self):
        values = np.array([[-32768, -1, 0], [1, 32767, -2]], dtype=np.int16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scores.mem'
            write_vector(path, values, 16, True)
            self.assertEqual(path.read_text().splitlines(), ['8000', 'ffff', '0000', '0001', '7fff', 'fffe'])
            np.testing.assert_array_equal(read_vector(path, 16, True, (2, 3)), values)
            self.assertNotIn(b'\r', path.read_bytes())

    def test_invalid_pixels_and_vector_files_are_rejected(self):
        with self.assertRaises(ValueError):
            pack_images(np.full((1, 784), 2, dtype=np.int16))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.mem'
            for text in ('ffff\n', 'ffff\n0000\n0001\n', '0x00\n0000\n'):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    read_vector(path, 16, True, (1, 2))
            with self.assertRaises(ValueError):
                write_vector(path, np.array([-1]), 8, False)
        print('\nVector checks: LSB-first packing, first/last pixels, signed scores, image order, and malformed-file rejection PASS')


if __name__ == '__main__':
    unittest.main(verbosity=2)
