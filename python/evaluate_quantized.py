"""Compare float and integer MLP accuracy on the complete MNIST test split."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import MNIST

from data import DATA_ROOT, PROJECT_ROOT
from golden import IntegerMLP, integer_array
from model import MNISTMLP


def summarize_predictions(labels, float_digits, integer_digits) -> dict:
    labels = integer_array(labels, 'labels', 1, 0, 9)
    float_digits = integer_array(float_digits, 'float digits', 1, 0, 9)
    integer_digits = integer_array(integer_digits, 'integer digits', 1, 0, 9)
    if labels.shape != float_digits.shape or labels.shape != integer_digits.shape:
        raise ValueError('Prediction and label counts disagree')
    float_correct = float_digits == labels
    integer_correct = integer_digits == labels
    count = len(labels)
    confusion = np.zeros((10, 10), dtype=np.int64)
    np.add.at(confusion, (labels, integer_digits), 1)
    fc, qc = int(float_correct.sum()), int(integer_correct.sum())
    drop = 100 * (fc - qc) / count
    return {
        'images': count, 'float_correct': fc, 'integer_correct': qc,
        'float_accuracy': fc / count, 'integer_accuracy': qc / count,
        'accuracy_drop_percentage_points': drop,
        'maximum_allowed_drop_percentage_points': 1.0,
        'within_one_percentage_point': drop <= 1.0,
        'float_target_met': 100 * fc >= 97 * count,
        'integer_at_least_97_percent': 100 * qc >= 97 * count,
        'prediction_disagreements': int((float_digits != integer_digits).sum()),
        'float_correct_integer_wrong': int((float_correct & ~integer_correct).sum()),
        'float_wrong_integer_correct': int((~float_correct & integer_correct).sum()),
        'integer_confusion_matrix_rows_true_columns_predicted': confusion.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--float-checkpoint', type=Path, default=PROJECT_ROOT / 'artifacts' / 'mlp_float.pt')
    parser.add_argument('--integer-checkpoint', type=Path, default=PROJECT_ROOT / 'artifacts' / 'mlp_quantized.pt')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'docs' / 'quantized_accuracy.json')
    parser.add_argument('--predictions', type=Path, default=PROJECT_ROOT / 'docs' / 'test_predictions.csv')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('batch-size must be positive')
    torch.set_num_threads(4)
    float_hash = hashlib.sha256(args.float_checkpoint.read_bytes()).hexdigest()
    artifact = torch.load(args.integer_checkpoint, map_location='cpu', weights_only=True)
    if artifact['settings']['source_checkpoint_sha256'] != float_hash:
        raise ValueError('Integer parameters were not produced from this float checkpoint')
    checkpoint = torch.load(args.float_checkpoint, map_location='cpu', weights_only=True)
    float_model = MNISTMLP()
    float_model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    float_model.eval()
    integer_model = IntegerMLP(artifact)
    raw = MNIST(root=str(DATA_ROOT), train=False, download=False)
    images = (raw.data.numpy() >= 128).astype(np.int16).reshape(-1, 784)
    labels = raw.targets.numpy()
    if len(images) != 10000:
        raise ValueError('Expected the full 10,000-image official test split')
    float_digits, integer_digits = [], []
    layer_checks = []
    for index, layer in enumerate(integer_model.layers):
        weights, biases = integer_model.parameters[layer['name']]
        wide_weights, wide_biases = weights.astype(np.int64), biases.astype(np.int64)
        # Python integer sums independently recheck every prefix bound from step 7.
        maximum_input = 1 if index == 0 else 32767
        bounds = [abs(int(b)) + maximum_input * sum(abs(int(w)) for w in row)
                  for row, b in zip(wide_weights, wide_biases)]
        if max(bounds) > 2147483647 or bounds != layer['accumulator_absolute_bounds']:
            raise OverflowError('Accumulator bound failed or differs from quantization settings')
        layer_checks.append({'name': layer['name'], 'max_conservative_absolute_bound': max(bounds),
                             'every_mac_prefix_fits_int32': True,
                             'observed_accumulator_min': 2147483647, 'observed_accumulator_max': -2147483648,
                             'observed_output_min': 32767, 'observed_output_max': -32768,
                             'saturated_values': 0})
    for offset in range(0, len(images), args.batch_size):
        batch = images[offset:offset + args.batch_size]
        integer_outputs = integer_model.infer(batch)
        with torch.no_grad():
            float_scores = float_model(torch.from_numpy(batch).to(torch.float32))
            if not torch.isfinite(float_scores).all():
                raise ValueError('Non-finite float scores')
            float_digits.extend(float_scores.argmax(dim=1).tolist())
        integer_digits.extend(integer_outputs['digits'].tolist())
        incoming = batch
        for index, layer in enumerate(integer_model.layers):
            weights, biases = integer_model.parameters[layer['name']]
            acc = incoming.astype(np.int64) @ weights.astype(np.int64).T + biases.astype(np.int64)
            shifted = acc >> layer['shift']
            output = integer_outputs['hidden' if index == 0 else 'scores']
            stats = layer_checks[index]
            stats['observed_accumulator_min'] = min(stats['observed_accumulator_min'], int(acc.min()))
            stats['observed_accumulator_max'] = max(stats['observed_accumulator_max'], int(acc.max()))
            stats['observed_output_min'] = min(stats['observed_output_min'], int(output.min()))
            stats['observed_output_max'] = max(stats['observed_output_max'], int(output.max()))
            stats['saturated_values'] += int(((shifted < -32768) | (shifted > 32767)).sum())
            if np.any(acc < -2147483648) or np.any(acc > 2147483647):
                raise OverflowError('Observed accumulator exceeds int32')
            incoming = output
    float_digits = np.array(float_digits, dtype=np.int64)
    integer_digits = np.array(integer_digits, dtype=np.int64)
    summary = summarize_predictions(labels, float_digits, integer_digits)
    if summary['float_correct'] != checkpoint['test']['correct']:
        raise ValueError('Reloaded float accuracy differs from recorded checkpoint accuracy')
    report = {
        'split': 'all 10,000 official MNIST test images, unchanged original order',
        'binarization': 'raw pixel >= 128', 'batch_size': args.batch_size,
        'float_checkpoint_sha256': float_hash,
        'integer_checkpoint_sha256': hashlib.sha256(args.integer_checkpoint.read_bytes()).hexdigest(),
        'summary': summary, 'layers': layer_checks,
        'overflow_checks_passed': True,
        'step9_passed': summary['float_target_met'] and summary['within_one_percentage_point'],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['image_index', 'true_label', 'float_digit', 'integer_digit', 'float_correct', 'integer_correct'])
        for index, (label, f, q) in enumerate(zip(labels, float_digits, integer_digits)):
            writer.writerow([index, int(label), int(f), int(q), int(f == label), int(q == label)])
    print(f"Float: {summary['float_accuracy']:.2%} ({summary['float_correct']}/10000)")
    print(f"Integer: {summary['integer_accuracy']:.2%} ({summary['integer_correct']}/10000)")
    print(f"Accuracy drop: {summary['accuracy_drop_percentage_points']:.2f} percentage points; limit=1.00")
    print(f"Prediction disagreements: {summary['prediction_disagreements']}")
    for stats in layer_checks:
        print(f"{stats['name']}: bound={stats['max_conservative_absolute_bound']:,}, "
              f"observed accumulator=[{stats['observed_accumulator_min']}, {stats['observed_accumulator_max']}], "
              f"saturated values={stats['saturated_values']}")
    print(f"Step 9: {'PASS' if report['step9_passed'] else 'FAIL'}")
    print(f"Report: {args.output}\nPredictions: {args.predictions}")
    if not report['step9_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
