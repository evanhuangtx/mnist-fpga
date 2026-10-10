# MNIST hardware memory exports

These files contain the verified v1 integer parameters. They are ASCII
hexadecimal files for `$readmemh`, with one fixed-width word per line.
`layers.json` records the scales and layout for software. The export verification
in `../docs/export_verification.json` identifies each file by SHA-256 hash.

| File | Words | Word width | Meaning |
|---|---:|---:|---|
| `weights.mem` | 50,816 | 8 bits | Signed two's-complement weights |
| `biases.mem` | 74 | 32 bits | Signed two's-complement biases |
| `shifts.mem` | 2 | 8 bits | Unsigned right shifts: 0, 7 |
| `layer_table.mem` | 2 | 128 bits | Packed layer controller records |

Weights are ordered by output neuron, then input index. Biases are ordered by
output neuron. Each layer follows the previous layer without gaps.

| Layer | Inputs | Outputs | Weight base | Bias base | Shift | ReLU |
|---|---:|---:|---:|---:|---:|---|
| Hidden dense | 784 | 64 | 0 | 0 | 0 | Yes |
| Output dense | 64 | 10 | 50,176 | 64 | 7 | No |

Bases are word addresses in separate memories. Hidden weights and activations
have 7 fractional bits. Output weights and scores have 6 fractional bits.
Biases use accumulator fractions of 7 and 13 respectively.

## SystemVerilog loading

Run the simulator from the repository root for these relative paths:

```systemverilog
logic signed [7:0] weight_rom [0:50815];
logic signed [31:0] bias_rom [0:73];
logic [7:0] shift_rom [0:1];
logic [127:0] layer_rom [0:1];

initial begin
    $readmemh("mem/weights.mem", weight_rom);
    $readmemh("mem/biases.mem", bias_rom);
    $readmemh("mem/shifts.mem", shift_rom);
    $readmemh("mem/layer_table.mem", layer_rom);
end
```

Use signed arithmetic for the weights and biases. The text word `ff` represents
weight -1, not 255. Part-selects and concatenations may need explicit signed
casts before arithmetic. Vivado project setup must make these files available
to synthesis; later synthesis checks confirm actual ROM inference.

## Layer table fields

Each record contains the following fields. The dense type code is 0. Flags
bit 0 enables ReLU; other flags and the reserved byte must be zero.

| Bits | Field |
|---|---|
| 127:120 | Type, 8 bits |
| 119:104 | Input count, 16 bits |
| 103:88 | Output count, 16 bits |
| 87:56 | Weight base, 32 bits |
| 55:24 | Bias base, 32 bits |
| 23:16 | Right shift, 8 bits |
| 15:8 | Flags, 8 bits; ReLU is record bit 8 |
| 7:0 | Reserved, 8 bits |

The separate shift ROM is convenient for independent requantization tests.
Its values match the shifts packed in the layer table.

## Regenerate and verify

From the repository root:

```powershell
.\.venv\Scripts\python.exe python\export_mem.py
.\.venv\Scripts\python.exe -m unittest discover -s python -p test_export_mem.py -v
```

Regeneration requires the quantized checkpoint and its passing step-9 accuracy
report. The exported parameters are reloaded from the files, compared exactly
with the checkpoint, then used to reproduce all layer outputs and predictions
on all 10,000 test images. The check matched 640,000 hidden activations,
100,000 output scores, and 10,000 predicted digits; reloaded accuracy was 97.42%.

Test vectors are available in `vectors/10/`, `vectors/1000/`, and `vectors/10000/`. See `vectors/README.md` for their layout.
