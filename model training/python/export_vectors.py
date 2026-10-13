"""Export and reload RTL vectors for 10, 1,000, and 10,000 MNIST test images."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from torchvision.datasets import MNIST

from data import DATA_ROOT, PROJECT_ROOT
from export_mem import load_exports
from golden import IntegerMLP


def pack_images(images: np.ndarray) -> np.ndarray:
    images = np.asarray(images)
    if images.ndim != 2 or images.shape[1] != 784 or images.dtype.kind not in 'iu':
        raise ValueError('Expected integer images shaped (N,784)')
    if not np.all((images == 0) | (images == 1)):
        raise ValueError('Image pixels must be binary')
    return np.packbits(images.astype(np.uint8), axis=1, bitorder='little')


def unpack_images(packed: np.ndarray) -> np.ndarray:
    packed = np.asarray(packed)
    if packed.ndim != 2 or packed.shape[1] != 98 or packed.dtype != np.uint8:
        raise ValueError('Expected uint8 payloads shaped (N,98)')
    return np.unpackbits(packed, axis=1, bitorder='little').astype(np.int16)


def write_vector(path: Path, values: np.ndarray, width: int, signed: bool) -> None:
    """Stream bounded chunks instead of building millions of strings in memory."""
    values = np.asarray(values)
    if width not in (8, 16) or values.dtype.kind not in 'iu':
        raise ValueError('Expected integer vectors of width 8 or 16')
    lower = -(1 << (width - 1)) if signed else 0
    upper = (1 << (width - 1)) - 1 if signed else (1 << width) - 1
    if values.size == 0 or np.any(values < lower) or np.any(values > upper):
        raise ValueError('Vector values exceed the requested word format')
    flat = values.reshape(-1)
    mask = (1 << width) - 1
    with path.open('w', encoding='ascii', newline='\n') as stream:
        for start in range(0, len(flat), 4096):
            stream.write(''.join(f'{int(value) & mask:0{width // 4}x}\n' for value in flat[start:start + 4096]))


def read_vector(path: Path, width: int, signed: bool, shape: tuple) -> np.ndarray:
    if width not in (8, 16):
        raise ValueError('Expected vector word width 8 or 16')
    pattern = re.compile(f'[0-9a-fA-F]{{{width // 4}}}')
    count = int(np.prod(shape))
    dtype = np.dtype(f'{"int" if signed else "uint"}{width}')
    with path.open('r', encoding='ascii') as stream:
        def words():
            for line in stream:
                text = line.rstrip('\r\n')
                if pattern.fullmatch(text) is None:
                    raise ValueError(f'Malformed vector word in {path.name}')
                word = int(text, 16)
                if signed and word & (1 << (width - 1)):
                    word -= 1 << width
                yield word
        iterator = words()
        values = np.fromiter(iterator, dtype=dtype, count=count)
        if next(iterator, None) is not None:
            raise ValueError(f'Extra vector words in {path.name}')
    return values.reshape(shape)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--memory-dir', type=Path, default=PROJECT_ROOT / 'mem')
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'mem' / 'vectors')
    parser.add_argument('--export-report', type=Path, default=PROJECT_ROOT / 'docs' / 'export_verification.json')
    parser.add_argument('--report', type=Path, default=PROJECT_ROOT / 'docs' / 'test_vector_verification.json')
    args = parser.parse_args()
    export_report = json.loads(args.export_report.read_text(encoding='utf-8'))
    if not export_report['export_reload_exact'] or export_report['images_checked'] != 10000:
        raise ValueError('Step-10 exports must have passed full verification')
    for name, metadata in export_report['files'].items():
        if file_hash(args.memory_dir / name) != metadata['sha256']:
            raise ValueError(f'Parameter export changed after verification: {name}')
    artifact = load_exports(args.memory_dir)
    model = IntegerMLP(artifact)
    raw = MNIST(root=str(DATA_ROOT), train=False, download=False)
    images = (raw.data.numpy() >= 128).astype(np.int16).reshape(-1, 784)
    labels = raw.targets.numpy().astype(np.uint8)
    if len(images) != 10000:
        raise ValueError('Expected all 10,000 official MNIST test images')
    hidden = np.empty((10000, 64), dtype=np.int16)
    scores = np.empty((10000, 10), dtype=np.int16)
    digits = np.empty(10000, dtype=np.uint8)
    for start in range(0, 10000, 256):
        stop = min(start + 256, 10000)
        outputs = model.infer(images[start:stop])
        hidden[start:stop], scores[start:stop], digits[start:stop] = outputs['hidden'], outputs['scores'], outputs['digits']
    packed = pack_images(images)
    np.testing.assert_array_equal(unpack_images(packed), images)
    correct = int((digits == labels).sum())
    if correct != export_report['test_correct']:
        raise AssertionError('Vector predictions differ from verified exported-model accuracy')
    suites = []
    for count in (10, 1000, 10000):
        directory = args.output_dir / str(count)
        directory.mkdir(parents=True, exist_ok=True)
        formats = {
            'images.mem': (images[:count], 16, True, 784),
            'packed_images.mem': (packed[:count], 8, False, 98),
            'hidden.mem': (hidden[:count], 16, True, 64),
            'scores.mem': (scores[:count], 16, True, 10),
            'expected_digits.mem': (digits[:count], 8, False, 1),
            'labels.mem': (labels[:count], 8, False, 1),
        }
        metadata, restored = {}, {}
        for name, (values, width, signed, stride) in formats.items():
            path = directory / name
            write_vector(path, values, width, signed)
            restored[name] = read_vector(path, width, signed, values.shape)
            np.testing.assert_array_equal(restored[name], values)
            metadata[name] = {'word_bits': width, 'signed': signed, 'words_per_image': stride,
                              'total_words': int(values.size), 'bytes': path.stat().st_size, 'sha256': file_hash(path)}
        np.testing.assert_array_equal(unpack_images(restored['packed_images.mem']), restored['images.mem'])
        for start in range(0, count, 256):
            stop = min(start + 256, count)
            outputs = model.infer(restored['images.mem'][start:stop])
            np.testing.assert_array_equal(outputs['hidden'], restored['hidden.mem'][start:stop])
            np.testing.assert_array_equal(outputs['scores'], restored['scores.mem'][start:stop])
            np.testing.assert_array_equal(outputs['digits'], restored['expected_digits.mem'][start:stop])
        suite = {'schema_version': 1, 'images': count, 'split': 'first N official MNIST test images in original order',
                 'image_indices': [0, count - 1], 'flatten_order': 'row then column',
                 'packed_pixel_order': 'first pixel is bit 0; 98 payload bytes; no start byte included',
                 'source_integer_checkpoint_sha256': export_report['source_integer_checkpoint_sha256'],
                 'parameter_file_sha256': {name: item['sha256'] for name, item in export_report['files'].items()},
                 'layers': artifact['layers'], 'files': metadata, 'reload_exact': True,
                 'packed_payload_round_trip_exact': True, 'recomputed_layer_outputs_exact': True}
        (directory / 'manifest.json').write_text(json.dumps(suite, indent=2) + '\n', encoding='utf-8')
        suites.append({'images': count, 'path': str(directory.resolve()), 'files_verified': len(formats),
                       'hidden_values_verified': count * 64, 'scores_verified': count * 10,
                       'digits_verified': count, 'all_checks_passed': True})
        print(f'PASS: {count:,}-image suite; all six files reload exactly, packed pixels match, and layer outputs recompute exactly', flush=True)
    report = {'source_integer_checkpoint_sha256': export_report['source_integer_checkpoint_sha256'],
              'suites': suites, 'full_test_correct': correct, 'full_test_accuracy': correct / 10000,
              'all_checks_passed': True}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'All suites verified. Full golden accuracy: {correct}/10000 ({correct / 10000:.2%})')
    print(f'Vectors: {args.output_dir}\nReport: {args.report}')


if __name__ == '__main__':
    main()
