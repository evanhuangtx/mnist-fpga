"""Choose safe power-of-two scales and quantize the trained MLP parameters."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from data import PROJECT_ROOT
from model import MNISTMLP

INT32_MAX = 2147483647


def quantize_values(values: torch.Tensor, fractional_bits: int, width: int) -> torch.Tensor:
    """Round nearest with ties to even; reject overflow instead of clipping."""
    if width not in (8, 32) or not 0 <= fractional_bits <= 60:
        raise ValueError("Expected width 8/32 and fractional bits in 0..60")
    scaled = torch.round(values.to(torch.float64) * (2 ** fractional_bits))
    lower, upper = -(2 ** (width - 1)), 2 ** (width - 1) - 1
    if not torch.isfinite(scaled).all() or not ((scaled >= lower) & (scaled <= upper)).all():
        raise ValueError(f"Parameters do not fit signed {width}-bit storage at F={fractional_bits}")
    return scaled.to(torch.int8 if width == 8 else torch.int32)


def accumulator_bounds(weights: torch.Tensor, biases: torch.Tensor, max_input: int) -> torch.Tensor:
    """Absolute bounds cover every MAC prefix, using wide arithmetic."""
    return biases.to(torch.int64).abs() + max_input * weights.to(torch.int64).abs().sum(dim=1)


def choose_weight_fraction(weights: torch.Tensor, biases: torch.Tensor, input_fraction: int,
                           max_input: int) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
    for fraction in range(30, -1, -1):
        try:
            integer_weights = quantize_values(weights, fraction, 8)
            integer_biases = quantize_values(biases, input_fraction + fraction, 32)
        except ValueError:
            continue
        bounds = accumulator_bounds(integer_weights, integer_biases, max_input)
        if (bounds <= INT32_MAX).all():
            return fraction, integer_weights, integer_biases, bounds
    raise ValueError("No safe nonnegative weight fractional bits fit this layer")


def reachable_bounds(weights: torch.Tensor, biases: torch.Tensor,
                     input_maxima: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Interval bounds for nonnegative inputs, including all binary MNIST images."""
    weights = weights.to(torch.int64)
    biases = biases.to(torch.int64)
    maxima = input_maxima.to(torch.int64)
    lower = biases + (weights.clamp(max=0) * maxima).sum(dim=1)
    upper = biases + (weights.clamp(min=0) * maxima).sum(dim=1)
    return lower, upper


def choose_output_fraction(lower: torch.Tensor, upper: torch.Tensor,
                           accumulator_fraction: int, relu: bool) -> tuple[int, int]:
    """Keep maximum precision while preventing reachable 16-bit saturation."""
    for shift in range(min(31, accumulator_fraction) + 1):
        shifted_lower, shifted_upper = lower >> shift, upper >> shift
        # Hidden values below -32768 saturate and then become zero under ReLU.
        if (shifted_upper <= 32767).all() and (relu or (shifted_lower >= -32768).all()):
            return accumulator_fraction - shift, shift
    raise ValueError("No supported right shift can represent this layer's output")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "artifacts" / "mlp_float.pt")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts" / "mlp_quantized.pt")
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "docs" / "quantization_results.json")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint["model_state_dict"]
    MNISTMLP().load_state_dict(state, strict=True)
    input_fraction = 0
    input_maxima = torch.ones(784, dtype=torch.int64)
    integer_state, layers = {}, []
    for index, (name, inputs, outputs, relu, weight_base, bias_base) in enumerate(
        (("fc1", 784, 64, True, 0, 0), ("fc2", 64, 10, False, 50176, 64))
    ):
        weights, biases = state[f"{name}.weight"], state[f"{name}.bias"]
        max_input = 1 if index == 0 else 32767
        fraction, integer_weights, integer_biases, bounds = choose_weight_fraction(
            weights, biases, input_fraction, max_input)
        lower, upper = reachable_bounds(integer_weights, integer_biases, input_maxima)
        output_fraction, shift = choose_output_fraction(lower, upper, input_fraction + fraction, relu)
        assert shift == input_fraction + fraction - output_fraction and 0 <= shift <= 31
        integer_state[f"{name}.weight"] = integer_weights
        integer_state[f"{name}.bias"] = integer_biases
        weight_error = (integer_weights.double() / (2 ** fraction) - weights.double()).abs()
        bias_error = (integer_biases.double() / (2 ** (input_fraction + fraction)) - biases.double()).abs()
        layer = {
            "name": name, "type": "dense", "inputs": inputs, "outputs": outputs,
            "weight_base": weight_base, "bias_base": bias_base, "relu": relu,
            "input_fractional_bits": input_fraction, "weight_fractional_bits": fraction,
            "accumulator_fractional_bits": input_fraction + fraction,
            "output_fractional_bits": output_fraction, "shift": shift,
            "weight_integer_min": int(integer_weights.min()), "weight_integer_max": int(integer_weights.max()),
            "bias_integer_min": int(integer_biases.min()), "bias_integer_max": int(integer_biases.max()),
            "max_absolute_weight_error": weight_error.max().item(),
            "mean_absolute_weight_error": weight_error.mean().item(),
            "max_absolute_bias_error": bias_error.max().item(),
            "accumulator_absolute_bounds": bounds.tolist(), "max_accumulator_absolute_bound": int(bounds.max()),
            "reachable_accumulator_lower": lower.tolist(), "reachable_accumulator_upper": upper.tolist(),
            "reachable_saturation_avoided": True,
        }
        layers.append(layer)
        if relu:
            input_maxima = (upper >> shift).clamp(0, 32767)
        input_fraction = output_fraction
        print(f"{name}: weight F={fraction}, input F={layer['input_fractional_bits']}, "
              f"output F={output_fraction}, shift={shift}, accumulator bound={int(bounds.max()):,}")
    source_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    settings = {
        "source_checkpoint_sha256": source_hash, "source_epoch": checkpoint["epoch"],
        "architecture": [784, 64, 10], "weight_width": 8, "bias_width": 32,
        "activation_width": 16, "accumulator_width": 32,
        "parameter_rounding": "nearest, ties to even", "requantization": "arithmetic right shift then saturation then optional ReLU",
        "weight_order": "output neuron then input index", "argmax_ties": "lowest digit index",
        "scale_selection": "maximum safe power-of-two precision with interval bounds over all binary inputs",
        "accuracy_evaluated": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"integer_state_dict": integer_state, "layers": layers, "settings": settings}, args.output)
    # Read back the integer artifact to verify storage types and exact parameter preservation.
    restored = torch.load(args.output, map_location="cpu", weights_only=True)
    for name, values in integer_state.items():
        assert torch.equal(values, restored["integer_state_dict"][name])
        assert restored["integer_state_dict"][name].dtype == (torch.int8 if name.endswith('weight') else torch.int32)
    report = {"settings": settings, "layers": layers, "parameter_count": 50890,
              "quantized_artifact": str(args.output.resolve()),
              "checks": {"no_parameter_clipping": True, "signed_storage_ranges": True,
                         "all_accumulator_prefixes_fit_int32": True, "checkpoint_round_trip": True}}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: parameter ranges, accumulator bounds, scale consistency, and integer artifact reload")
    print(f"Parameters: {args.output}\nReport: {args.report}")


if __name__ == "__main__":
    main()
