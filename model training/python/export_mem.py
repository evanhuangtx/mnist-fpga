import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import MNIST

from data import DATA_ROOT, PROJECT_ROOT
from golden import IntegerMLP

LAYER_FIELDS = [('type', 8), ('inputs', 16), ('outputs', 16), ('weight_base', 32),
                ('bias_base', 32), ('shift', 8), ('flags', 8), ('reserved', 8)]
LAYER_KEYS = ('name', 'type', 'inputs', 'outputs', 'weight_base', 'bias_base', 'relu',
              'input_fractional_bits', 'weight_fractional_bits',
              'accumulator_fractional_bits', 'output_fractional_bits', 'shift')

def write_words(path: Path, values, width: int, signed: bool) -> None:
    if width <= 0 or width % 4:
        raise ValueError('Word width must be a positive multiple of four')
    lower = -(1 << (width - 1)) if signed else 0
    upper = (1 << (width - 1)) - 1 if signed else (1 << width) - 1
    lines = []
    for value in values:
        if not isinstance(value, (int, np.integer)) or not lower <= int(value) <= upper:
            raise ValueError(f'Value does not fit {width}-bit word')
        lines.append(f'{int(value) & ((1 << width) - 1):0{width // 4}x}')
    path.write_text('\n'.join(lines) + '\n', encoding='ascii')

def read_words(path: Path, width: int, signed: bool, count: int) -> list[int]:
    if width <= 0 or width % 4:
        raise ValueError('Word width must be a positive multiple of four')
    lines = path.read_text(encoding='ascii').splitlines()
    if len(lines) != count or any(re.fullmatch(f'[0-9a-fA-F]{{{width // 4}}}', line) is None for line in lines):
        raise ValueError(f'Invalid word count or hexadecimal encoding in {path.name}')
    words = [int(line, 16) for line in lines]
    if signed:
        words = [word - (1 << width) if word & (1 << (width - 1)) else word for word in words]
    return words

def encode_layer(layer: dict) -> int:
    if layer['type'] != 'dense':
        raise ValueError('Only dense layers are supported in v1')
    values = {**layer, 'type': 0, 'flags': int(layer['relu']), 'reserved': 0}
    word = 0
    for name, width in LAYER_FIELDS:
        value = values[name]
        if type(value) is not int or not 0 <= value < (1 << width):
            raise ValueError(f'Layer field {name} does not fit {width} bits')
        word = (word << width) | value
    return word

def export_parallel(directory: Path, artifact: dict, lanes: int = 8) -> dict:
    if lanes < 1 or lanes & (lanes - 1):
        raise ValueError('Lane count must be a positive power of two')
    weight_words, bias_words, records = [], [], []
    restored = {}
    for layer in artifact['layers']:
        name = layer['name']
        weight = artifact['integer_state_dict'][name + '.weight'].numpy()
        bias = artifact['integer_state_dict'][name + '.bias'].numpy()
        outputs, inputs = weight.shape
        groups = (outputs + lanes - 1) // lanes
        padded = groups * lanes
        record = {'name': name, 'inputs': inputs, 'outputs': outputs,
                  'groups': groups, 'weight_base': len(weight_words),
                  'bias_base': len(bias_words), 'shift': layer['shift'],
                  'relu': layer['relu']}
        records.append(record)
        w = np.pad(weight, ((0, padded - outputs), (0, 0)))
        b = np.pad(bias, (0, padded - outputs))
        w = w.reshape(groups, lanes, inputs).transpose(0, 2, 1).reshape(-1, lanes)
        b = b.reshape(groups, lanes)
        for row in w:
            weight_words.append(sum((int(value) & 255) << (8 * k) for k, value in enumerate(row)))
        for row in b:
            bias_words.append(sum((int(value) & 0xffffffff) << (32 * k) for k, value in enumerate(row)))
    directory.mkdir(parents=True, exist_ok=True)
    write_words(directory / 'weights_x8.mem', weight_words, lanes * 8, False)
    write_words(directory / 'biases_x8.mem', bias_words, lanes * 32, False)
    weights = read_words(directory / 'weights_x8.mem', lanes * 8, False, len(weight_words))
    biases = read_words(directory / 'biases_x8.mem', lanes * 32, False, len(bias_words))
    def unpack(words, bits):
        mask, sign = (1 << bits) - 1, 1 << (bits - 1)
        return np.array([[((word >> (k * bits)) & mask) -
                         ((1 << bits) if ((word >> (k * bits)) & sign) else 0)
                         for k in range(lanes)] for word in words], dtype=np.int64)
    for record in records:
        name, inputs, outputs, groups = (record[k] for k in ('name', 'inputs', 'outputs', 'groups'))
        wb, bb = record['weight_base'], record['bias_base']
        w = unpack(weights[wb:wb + groups * inputs], 8)
        w = w.reshape(groups, inputs, lanes).transpose(0, 2, 1).reshape(groups * lanes, inputs)
        b = unpack(biases[bb:bb + groups], 32).reshape(-1)
        if np.any(w[outputs:]) or np.any(b[outputs:]):
            raise AssertionError('Padding lanes must be zero')
        restored[name + '.weight'] = torch.from_numpy(w[:outputs].astype(np.int8))
        restored[name + '.bias'] = torch.from_numpy(b[:outputs].astype(np.int32))
        for suffix in ('.weight', '.bias'):
            if not torch.equal(restored[name + suffix], artifact['integer_state_dict'][name + suffix]):
                raise AssertionError('Packed parameters differ: ' + name + suffix)
    config = {'lanes': lanes, 'weight_word_bits': lanes * 8, 'bias_word_bits': lanes * 32,
              'weight_words': len(weights), 'bias_words': len(biases),
              'lane_order': 'lane 0 in least significant bits', 'layers': records}
    (directory / 'layers_x8.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    params = [f'localparam integer LANES = {lanes};',
              f'localparam integer W8_DEPTH = {len(weights)};',
              f'localparam integer B8_DEPTH = {len(biases)};']
    for index, record in enumerate(records):
        params.extend([f"localparam integer W8_BASE_L{index} = {record['weight_base']};",
                       f"localparam integer B8_BASE_L{index} = {record['bias_base']};"])
    (directory / 'params.vh').write_text('\n'.join(params) + '\n', encoding='ascii')
    return {'config': config, 'artifact': {'integer_state_dict': restored, 'layers': artifact['layers']}}

def decode_layer(word: int) -> dict:
    if type(word) is not int or not 0 <= word < (1 << 128):
        raise ValueError('Layer word must fit 128 bits')
    fields = {}
    for name, width in reversed(LAYER_FIELDS):
        fields[name] = word & ((1 << width) - 1)
        word >>= width
    if fields['type'] != 0 or fields['reserved'] != 0 or fields['flags'] not in (0, 1):
        raise ValueError('Unsupported layer type, flags, or reserved bits')
    return {'type': 'dense', 'inputs': fields['inputs'], 'outputs': fields['outputs'],
            'weight_base': fields['weight_base'], 'bias_base': fields['bias_base'],
            'shift': fields['shift'], 'relu': bool(fields['flags'])}

def load_exports(directory: Path) -> dict:
    config = json.loads((directory / 'layers.json').read_text(encoding='utf-8'))
    if config['schema_version'] != 1 or config['layer_table_word_bits'] != 128:
        raise ValueError('Unsupported export schema')
    weights = read_words(directory / 'weights.mem', 8, True, 50816)
    biases = read_words(directory / 'biases.mem', 32, True, 74)
    shifts = read_words(directory / 'shifts.mem', 8, False, 2)
    table = read_words(directory / 'layer_table.mem', 128, False, 2)
    layers = config['layers']
    if len(layers) != 2:
        raise ValueError('Expected two layer records')
    state = {}
    for index, layer in enumerate(layers):
        decoded = decode_layer(table[index])
        if any(layer[key] != value for key, value in decoded.items()) or shifts[index] != decoded['shift']:
            raise ValueError('Layer table, shift ROM, and JSON settings disagree')
        count = layer['inputs'] * layer['outputs']
        start = decoded['weight_base']
        w = np.array(weights[start:start + count], dtype=np.int8).reshape(layer['outputs'], layer['inputs'])
        start = decoded['bias_base']
        b = np.array(biases[start:start + layer['outputs']], dtype=np.int32)
        state[f"{layer['name']}.weight"] = torch.from_numpy(w)
        state[f"{layer['name']}.bias"] = torch.from_numpy(b)
    artifact = {'integer_state_dict': state, 'layers': layers}
    IntegerMLP(artifact)  
    return artifact

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=PROJECT_ROOT / 'model training' / 'artifacts' / 'mlp_quantized.pt')
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'mem')
    parser.add_argument('--accuracy-report', type=Path, default=PROJECT_ROOT / 'docs' / 'quantized_accuracy.json')
    parser.add_argument('--report', type=Path, default=PROJECT_ROOT / 'docs' / 'export_verification.json')
    args = parser.parse_args()
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    accuracy = json.loads(args.accuracy_report.read_text(encoding='utf-8'))
    if not accuracy['step9_passed'] or accuracy['integer_checkpoint_sha256'] != checkpoint_hash:
        raise ValueError('The checkpoint must match the passing step-9 evaluation')
    artifact = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    original_model = IntegerMLP(artifact)
    parallel = export_parallel(args.output_dir, artifact)
    parallel_model = IntegerMLP(parallel['artifact'])
    layers = [{key: layer[key] for key in LAYER_KEYS} for layer in artifact['layers']]
    weights, biases = [], []
    for layer in layers:
        w, b = original_model.parameters[layer['name']]
        if layer['weight_base'] != len(weights) or layer['bias_base'] != len(biases):
            raise ValueError('Memory bases must concatenate layers without gaps')
        weights.extend(w.reshape(-1).tolist())
        biases.extend(b.tolist())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_words(args.output_dir / 'weights.mem', weights, 8, True)
    write_words(args.output_dir / 'biases.mem', biases, 32, True)
    write_words(args.output_dir / 'shifts.mem', [layer['shift'] for layer in layers], 8, False)
    write_words(args.output_dir / 'layer_table.mem', [encode_layer(layer) for layer in layers], 128, False)
    config = {'schema_version': 1, 'source_integer_checkpoint_sha256': checkpoint_hash,
              'weight_words': 50816, 'bias_words': 74, 'weight_word_bits': 8, 'bias_word_bits': 32,
              'layer_table_word_bits': 128, 'layer_table_fields_msb_first': LAYER_FIELDS,
              'type_codes': {'dense': 0}, 'flags': {'bit_0': 'ReLU'},
              'weight_order': 'output neuron then input index', 'pixel_order': 'row-major, packed bit 0 first',
              'layers': layers}
    (args.output_dir / 'layers.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    restored = load_exports(args.output_dir)
    for name, values in artifact['integer_state_dict'].items():
        if not torch.equal(values, restored['integer_state_dict'][name]):
            raise AssertionError(f'Reloaded parameters differ: {name}')
    restored_model = IntegerMLP(restored)
    raw = MNIST(root=str(DATA_ROOT), train=False, download=False)
    images = (raw.data.numpy() >= 128).astype(np.int16).reshape(-1, 784)
    if len(images) != 10000:
        raise ValueError('Expected all 10,000 MNIST test images')
    correct = 0
    for offset in range(0, len(images), 256):
        batch = images[offset:offset + 256]
        expected = original_model.infer(batch)
        actual = restored_model.infer(batch)
        parallel_actual = parallel_model.infer(batch)
        for key in ('hidden', 'scores', 'digits'):
            np.testing.assert_array_equal(actual[key], expected[key])
            np.testing.assert_array_equal(parallel_actual[key], expected[key])
        correct += int((actual['digits'] == raw.targets[offset:offset + len(batch)].numpy()).sum())
    if correct != accuracy['summary']['integer_correct']:
        raise AssertionError('Reloaded exports disagree with step-9 accuracy')
    filenames = ('weights.mem', 'biases.mem', 'shifts.mem', 'layer_table.mem', 'layers.json')
    report = {'source_integer_checkpoint_sha256': checkpoint_hash, 'weight_words': len(weights),
              'bias_words': len(biases), 'layers': len(layers), 'parameter_round_trip_exact': True,
              'images_checked': 10000, 'hidden_values_matched': 640000, 'scores_matched': 100000,
              'digits_matched': 10000, 'test_correct': correct, 'test_accuracy': correct / 10000,
              'export_reload_exact': True,
              'files': {name: {'sha256': hashlib.sha256((args.output_dir / name).read_bytes()).hexdigest(),
                               'bytes': (args.output_dir / name).stat().st_size} for name in filenames}}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    parallel_files = ('weights_x8.mem', 'biases_x8.mem', 'layers_x8.json', 'params.vh')
    parallel_report = {**parallel['config'], 'source_integer_checkpoint_sha256': checkpoint_hash,
                       'parameter_round_trip_exact': True, 'padding_zero': True,
                       'images_checked': 10000, 'inference_exact': True,
                       'files': {name: {'sha256': hashlib.sha256((args.output_dir / name).read_bytes()).hexdigest(),
                                        'bytes': (args.output_dir / name).stat().st_size} for name in parallel_files}}
    (args.report.parent / 'parallel_export_verification.json').write_text(json.dumps(parallel_report, indent=2) + '\n', encoding='utf-8')
    print(f'Exported {len(weights):,} weight words, {len(biases)} bias words, 2 shifts, and 2 layer records')
    print('PASS: reloaded parameters match exactly; all 640,000 hidden values, 100,000 scores, and 10,000 digits match')
    print(f'Reloaded test accuracy: {correct / 10000:.2%} ({correct}/10000)')
    print(f'Files: {args.output_dir}\nVerification: {args.report}')

if __name__ == '__main__':
    main()
