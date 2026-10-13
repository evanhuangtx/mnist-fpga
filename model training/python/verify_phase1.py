import json
import sys
import unittest
from pathlib import Path

import torch

from data import PROJECT_ROOT
from export_vectors import file_hash

def main() -> None:
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parent), pattern='test_*.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    def report(name):
        return json.loads((PROJECT_ROOT / 'docs' / name).read_text(encoding='utf-8'))

    float_hash = file_hash(PROJECT_ROOT / 'model training' / 'artifacts' / 'mlp_float.pt')
    integer_hash = file_hash(PROJECT_ROOT / 'model training' / 'artifacts' / 'mlp_quantized.pt')
    integer = torch.load(PROJECT_ROOT / 'model training' / 'artifacts' / 'mlp_quantized.pt', map_location='cpu', weights_only=True)
    accuracy = report('quantized_accuracy.json')
    exports = report('export_verification.json')
    vectors = report('test_vector_verification.json')
    dead = report('dead_neurons.json')
    if not (integer['settings']['source_checkpoint_sha256'] == float_hash == accuracy['float_checkpoint_sha256'] == dead['checkpoint_sha256']):
        raise ValueError('Float checkpoint provenance differs between reports')
    if not (integer_hash == accuracy['integer_checkpoint_sha256'] == exports['source_integer_checkpoint_sha256'] == vectors['source_integer_checkpoint_sha256']):
        raise ValueError('Integer checkpoint provenance differs between reports')
    if not (accuracy['step9_passed'] and accuracy['overflow_checks_passed'] and exports['export_reload_exact'] and vectors['all_checks_passed']):
        raise ValueError('A full evaluation or export verification did not pass')
    for name, metadata in exports['files'].items():
        if file_hash(PROJECT_ROOT / 'mem' / name) != metadata['sha256']:
            raise ValueError(f'Parameter file hash mismatch: {name}')
    files_checked = len(exports['files'])
    parallel_path = PROJECT_ROOT / 'docs' / 'parallel_export_verification.json'
    if parallel_path.exists():
        parallel = json.loads(parallel_path.read_text(encoding='utf-8'))
        if (parallel['source_integer_checkpoint_sha256'] != integer_hash or
                not all(parallel[key] for key in ('parameter_round_trip_exact', 'padding_zero', 'inference_exact'))):
            raise ValueError('Parallel export verification failed')
        for name, metadata in parallel['files'].items():
            path = PROJECT_ROOT / 'mem' / name
            if file_hash(path) != metadata['sha256'] or path.stat().st_size != metadata['bytes']:
                raise ValueError('Parallel export hash or size mismatch: ' + name)
            files_checked += 1
    for count in (10, 1000, 10000):
        directory = PROJECT_ROOT / 'mem' / 'vectors' / str(count)
        manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        if manifest['images'] != count or manifest['source_integer_checkpoint_sha256'] != integer_hash:
            raise ValueError('Vector manifest source or image count mismatch')
        if not all(manifest[key] for key in ('reload_exact', 'packed_payload_round_trip_exact', 'recomputed_layer_outputs_exact')):
            raise ValueError('Vector verification failed')
        if manifest['parameter_file_sha256'] != {name: value['sha256'] for name, value in exports['files'].items()}:
            raise ValueError('Vector parameter provenance differs')
        for name, metadata in manifest['files'].items():
            path = directory / name
            if file_hash(path) != metadata['sha256'] or path.stat().st_size != metadata['bytes']:
                raise ValueError(f'Vector file hash or size mismatch: {count}/{name}')
            if metadata['total_words'] != count * metadata['words_per_image']:
                raise ValueError('Vector stride or count mismatch')
            files_checked += 1
    verification = {
        'all_checks_passed': True, 'unit_tests_run': result.testsRun,
        'unit_test_failures': len(result.failures), 'unit_test_errors': len(result.errors),
        'artifact_files_hash_verified': files_checked,
        'float_checkpoint_sha256': float_hash, 'integer_checkpoint_sha256': integer_hash,
        'float_accuracy': accuracy['summary']['float_accuracy'],
        'integer_accuracy': accuracy['summary']['integer_accuracy'],
        'accuracy_loss_percentage_points': accuracy['summary']['accuracy_drop_percentage_points'],
        'scope': 'Fresh complete unit suite and artifact provenance/hash checks; full 10,000-image comparisons are recorded in hash-matched evaluation and export reports.',
    }
    (PROJECT_ROOT / 'docs' / 'phase1_verification.json').write_text(json.dumps(verification, indent=2) + '\n', encoding='utf-8')
    print(f'PASS: {result.testsRun} tests and {files_checked} parameter/vector files verified against full-evaluation reports')

if __name__ == '__main__':
    main()
