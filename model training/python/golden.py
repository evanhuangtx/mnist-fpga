"""Integer-only v1 inference: exact MACs, arithmetic shifts, saturation, and ReLU."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import MNIST

from data import DATA_ROOT, PROJECT_ROOT


def integer_array(values, name: str, ndim: int, lower: int, upper: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != ndim or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a {ndim}-dimensional integer array")
    if array.size == 0 or np.any(array < lower) or np.any(array > upper):
        raise ValueError(f"{name} must be nonempty and in [{lower}, {upper}]")
    return array.astype(np.int64, copy=False)


def requantize(accumulators, shift: int, relu: bool) -> np.ndarray:
    if not isinstance(shift, int) or isinstance(shift, bool) or not 0 <= shift <= 31:
        raise ValueError("shift must be an integer in 0..31")
    accumulators = integer_array(accumulators, "accumulators", 2, -2147483648, 2147483647)
    shifted = accumulators >> shift
    saturated = np.clip(shifted, -32768, 32767)
    if relu:
        saturated = np.maximum(saturated, 0)
    return saturated.astype(np.int16)


def dense_integer(inputs, weights, biases, shift: int, relu: bool,
                  max_input_magnitude: int = 32768) -> np.ndarray:
    """Exact int64 evaluation after proving all 32-bit MAC prefixes are safe."""
    if not isinstance(max_input_magnitude, int) or not 0 <= max_input_magnitude <= 32768:
        raise ValueError("max_input_magnitude must be an integer in 0..32768")
    inputs = integer_array(inputs, "inputs", 2, -max_input_magnitude, min(max_input_magnitude, 32767))
    weights = integer_array(weights, "weights", 2, -128, 127)
    biases = integer_array(biases, "biases", 1, -2147483648, 2147483647)
    if inputs.shape[1] != weights.shape[1] or len(biases) != weights.shape[0]:
        raise ValueError("Dense input, weight, and bias dimensions disagree")
    bounds = np.abs(biases) + max_input_magnitude * np.abs(weights).sum(axis=1)
    if np.any(bounds > 2147483647):
        raise OverflowError("Conservative bound exceeds int32; a MAC prefix could overflow")
    accumulators = inputs @ weights.T + biases
    return requantize(accumulators, shift, relu)


class IntegerMLP:
    """Return hidden activations, signed scores, and digits for (N,784) binary integers."""

    def __init__(self, artifact: dict) -> None:
        state = artifact["integer_state_dict"]
        self.layers = artifact["layers"]
        if len(self.layers) != 2:
            raise ValueError("Expected two dense layers")
        self.parameters = {}
        incoming_fraction = 0
        for layer, (name, inputs, outputs, relu, weight_base, bias_base) in zip(self.layers,
                (("fc1", 784, 64, True, 0, 0), ("fc2", 64, 10, False, 50176, 64))):
            if (layer['name'], layer['type'], layer['inputs'], layer['outputs'], layer['relu'],
                    layer['weight_base'], layer['bias_base']) != (name, 'dense', inputs, outputs, relu, weight_base, bias_base):
                raise ValueError("Layer configuration does not match the v1 MLP")
            fractions = [layer[key] for key in ('input_fractional_bits', 'weight_fractional_bits',
                                               'accumulator_fractional_bits', 'output_fractional_bits')]
            if any(type(value) is not int or value < 0 for value in fractions):
                raise ValueError("Fractional bit counts must be nonnegative integers")
            fi, fw, fa, fo = fractions
            shift = layer['shift']
            if type(shift) is not int or not 0 <= shift <= 31 or fi != incoming_fraction or fa != fi + fw or shift != fa - fo:
                raise ValueError("Layer scale or shift is inconsistent")
            weights, biases = state[f'{name}.weight'], state[f'{name}.bias']
            if weights.dtype != torch.int8 or biases.dtype != torch.int32:
                raise ValueError("Artifact weights/biases must be int8/int32")
            if tuple(weights.shape) != (outputs, inputs) or tuple(biases.shape) != (outputs,):
                raise ValueError("Artifact parameter shapes disagree with the layer")
            self.parameters[name] = (weights.cpu().numpy().copy(), biases.cpu().numpy().copy())
            incoming_fraction = fo

    @classmethod
    def load(cls, path: Path = PROJECT_ROOT / 'model training' / 'artifacts' / 'mlp_quantized.pt'):
        return cls(torch.load(path, map_location='cpu', weights_only=True))

    def infer(self, inputs) -> dict[str, np.ndarray]:
        inputs = integer_array(inputs, "binary images", 2, 0, 1)
        if inputs.shape[1] != 784:
            raise ValueError("Each image must contain 784 row-major pixels")
        outputs = []
        for index, layer in enumerate(self.layers):
            weights, biases = self.parameters[layer['name']]
            inputs = dense_integer(inputs, weights, biases, layer['shift'], layer['relu'],
                                   max_input_magnitude=1 if index == 0 else 32767)
            outputs.append(inputs)
        return {"hidden": outputs[0], "scores": outputs[1],
                "digits": outputs[1].argmax(axis=1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=PROJECT_ROOT / 'model training' / 'artifacts' / 'mlp_quantized.pt')
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'docs' / 'golden_samples.json')
    args = parser.parse_args()
    if not 1 <= args.count <= 10000:
        parser.error('count must be in 1..10000')
    raw = MNIST(root=str(DATA_ROOT), train=False, download=False)
    inputs = (raw.data[:args.count].numpy() >= 128).astype(np.int16)
    outputs = IntegerMLP.load(args.checkpoint).infer(inputs.reshape(args.count, 784))
    labels = raw.targets[:args.count].tolist()
    report = {
        'quantized_checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'split': 'first test images in original order', 'images': args.count,
        'hidden_dtype': str(outputs['hidden'].dtype), 'score_dtype': str(outputs['scores'].dtype),
        'samples': [{'index': i, 'true_label': labels[i], 'predicted_digit': int(outputs['digits'][i]),
                     'hidden': outputs['hidden'][i].tolist(), 'scores': outputs['scores'][i].tolist()}
                    for i in range(args.count)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    for i in range(args.count):
        print(f"Image {i}: true={labels[i]}, predicted={outputs['digits'][i]}, scores={outputs['scores'][i].tolist()}")
    print(f"Integer outputs: hidden={outputs['hidden'].shape}, scores={outputs['scores'].shape}")
    print(f"Samples: {args.output}")


if __name__ == '__main__':
    main()
