# MNIST integer arithmetic and export rules

The integer golden model and SystemVerilog implementation must use the same
rules below for the 784 to 64 to 10 MLP. All 64 hidden neurons remain in the
network. Step 7 chooses numerical fractional bits and shifts from the trained
parameters; the conventions here define how those choices are interpreted.

## Number formats

| Quantity | Storage | Allowed integer values |
|---|---|---|
| Binarized input activation | Signed 16 bits | 0 or 1 |
| Weight | Signed 8 bits | -128 to 127 |
| Bias | Signed 32 bits | -2147483648 to 2147483647 |
| Product | Signed 24 bits before sign extension | Exact signed 16-bit activation times signed 8-bit weight |
| Accumulator | Signed 32 bits | -2147483648 to 2147483647 |
| Hidden activation after ReLU | Signed 16 bits | 0 to 32767 |
| Output score | Signed 16 bits | -32768 to 32767 |

Use two's-complement encoding for every signed value. Extend the product's
sign to 32 bits before accumulation. In SystemVerilog, declare operands and
intermediate signals signed and make expression widths explicit; a concatenation
or part-select can otherwise lose signed interpretation.

## Scales and bias conversion

An integer q with F fractional bits represents the real value q / 2**F.
Use one weight scale per layer and one activation scale per layer, shared by
all neurons. Fractional bit counts are nonnegative integers. Input pixels have
F_input = 0: their stored values are exactly 0 and 1.

For a layer with incoming activation fraction F_in, weight fraction F_w, and
outgoing activation fraction F_out:

```text
weight_integer = round_ties_to_even(weight_float * 2**F_w)
accumulator_fraction = F_in + F_w
bias_integer = round_ties_to_even(bias_float * 2**accumulator_fraction)
shift = accumulator_fraction - F_out
```

Offline conversion uses nearest rounding, with exact halfway values rounded
to the even integer. Examples: 0.5 becomes 0, 1.5 becomes 2, -0.5 becomes 0,
and -1.5 becomes -2. Reject non-finite values and parameters outside their
storage ranges; do not silently wrap or clip weights or biases. Choose a
supported scale instead.

The hidden output's F_out is the next layer's F_in. All ten final scores share
one F_out so their integers can be compared directly. The shift must be an
integer from 0 through 31. Negative shifts and runtime left shifts are excluded.

| Layer | Incoming fraction | Weight fraction | Outgoing fraction | Shift |
|---|---|---|---|---|
| Hidden dense | 0 | F_w1 | F_hidden | F_w1 - F_hidden |
| Output dense | F_hidden | F_w2 | F_scores | F_hidden + F_w2 - F_scores |

The numerical values F_w1, F_w2, F_hidden, F_scores, and both shifts are chosen
in step 7 and exported with the parameters. These are not training weights
or hardware defaults.

## Accumulation and requantization

For each output neuron, load its bias as the accumulator's starting value.
Visit input indices in ascending order and add each signed activation times
its signed weight. Apply requantization once, after the complete dot product.

```text
acc = bias_integer
for input_index from 0 through input_count - 1:
    acc = acc + activation[input_index] * weight[output_index][input_index]

shifted = arithmetic_right_shift(acc, shift)
saturated = min(32767, max(-32768, shifted))
output = max(0, saturated) if relu_enabled else saturated
```

Runtime requantization uses arithmetic right shift with no rounding increment.
This is floor division by 2**shift, including for negative accumulators; it is
not truncation toward zero. In Python use integer `acc >> shift`. In
SystemVerilog use `>>>` with a signed accumulator. Saturate the shifted value
while it is still wide, before assigning it to a 16-bit output.

| Accumulator | Shift | After shift | After saturation | With ReLU |
|---|---:|---:|---:|---:|
| 7 | 1 | 3 | 3 | 3 |
| -7 | 1 | -4 | -4 | 0 |
| 65536 | 1 | 32768 | 32767 | 32767 |
| -65538 | 1 | -32769 | -32768 | 0 |
| 32767 | 0 | 32767 | 32767 | 32767 |
| -32768 | 0 | -32768 | -32768 | 0 |

The hidden dense layer enables ReLU. The output dense layer disables ReLU
and returns raw signed scores. No softmax is used for integer inference.

For an illustrative first-layer scale, F_w = 5 and F_out = 3 give shift = 2.
Weights [0.5, -0.25] become [16, -8], and bias 0.25 becomes 8. Inputs [1, 0]
give accumulator 24, stored activation 6, and represented activation 6/8 = 0.75.
These numbers demonstrate the convention; step 7 selects the actual scales.

## Accumulator overflow checks

The supported computation never overflows its signed 32-bit accumulator.
Reject a parameter configuration that cannot meet this requirement; hardware
wraparound is not a valid inference result. Use Python integers or sufficiently
wide integer arithmetic for the golden model's checks, avoiding NumPy int32
arithmetic that can overflow before being checked.

A conservative bound for every intermediate sum of one neuron is:

```text
bound = abs(bias_integer) + max_input_magnitude * sum(abs(weight_integer))
require bound <= 2147483647
```

Use max_input_magnitude = 1 for the first dense layer and 32767 for the second
layer, covering the complete range of its ReLU inputs. Calculate absolute
values in a wider type, including for -128 weights and minimum signed biases.
The bound covers every prefix of the dot product, not only the final sum.

## Pixel and memory order

Flatten images by row then column: pixel index = row * 28 + column. For a
packed UART image, byte j contains pixels 8*j through 8*j+7, with the first
pixel in bit 0 and the last in bit 7. The payload has 98 bytes; the start byte
0xAA is separate. Example pixels [1,0,1,0,0,0,0,1] pack into byte 0x85.
Unpacked activation buffer words are 0000 or 0001.

Store dense weights with output neuron as the outer dimension and input index
as the inner dimension, matching PyTorch Linear weight shape [out, in]:

```text
weight_address = weight_base + output_index * input_count + input_index
bias_address = bias_base + output_index
```

| Layer | Inputs | Outputs | Weight base | Weight words | Bias base | Bias words | ReLU |
|---|---:|---:|---:|---:|---:|---:|---|
| Hidden dense | 784 | 64 | 0 | 50176 | 0 | 64 | Yes |
| Output dense | 64 | 10 | 50176 | 640 | 64 | 10 | No |

Weights and biases use separate memories. Bases are word addresses in those
memories, not byte offsets. The complete exports contain 50816 weight words
and 74 bias words. Hidden activation index j identifies hidden neuron j;
final score index k identifies digit k. Load inputs into buffer A, write hidden
outputs to B, then write final scores to A.

## Hexadecimal exports

Write one fixed-width hexadecimal word per line, with no 0x prefix or headers.
Use two hex digits for weights, eight for biases, and four for activations
and scores. Encode signed q of width N as `q & ((1 << N) - 1)` after checking
that q fits its signed range. Readers must interpret the high bit as the sign.

| Value and width | Hexadecimal |
|---|---|
| Weight -128, 8 bits | 80 |
| Weight -1, 8 bits | ff |
| Weight 127, 8 bits | 7f |
| Bias -1, 32 bits | ffffffff |
| Score -32768, 16 bits | 8000 |
| Score -1, 16 bits | ffff |
| Score 32767, 16 bits | 7fff |

Hexadecimal words in text files have no byte endianness. If optional raw scores
are transmitted over UART, send digits 0 through 9 in order, each as two's-
complement 16-bit bytes with the low byte first.

Export each layer's type, input/output dimensions, weight and bias bases,
shift, and ReLU flag with its parameters. Also record fractional bits for
software interpretation. Validate that the exported shift equals
F_in + F_w - F_out.

## Prediction and ties

Argmax compares scores as signed 16-bit values. Initialize the best score
from digit 0, then visit digits 1 through 9. Update the winner only when the
new score is strictly greater. Equal scores therefore select the lowest digit
index. Initializing the best score to zero would incorrectly handle an image
whose scores are all negative.

Integer saturation can create ties that did not exist in float inference.
The golden model and hardware must still follow the same tie rule; step 7
measures any resulting accuracy loss.
