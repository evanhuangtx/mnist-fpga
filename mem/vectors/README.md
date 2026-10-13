# MNIST RTL test vector layout

Choose `10/` for initial integration, `1000/` for development, or `10000/`
for the full test. Each directory contains the first N images of the official
MNIST test split in original order. The smaller sets are prefixes of the full
set. Each manifest records file widths, word counts, hashes, parameter hashes,
and scale settings.

| File | Word width | Words per image | Purpose |
|---|---:|---:|---|
| `images.mem` | Signed 16 bits | 784 | Binary input activations, 0000 or 0001 |
| `packed_images.mem` | Unsigned 8 bits | 98 | UART image payload bytes |
| `hidden.mem` | Signed 16 bits | 64 | Expected hidden-layer outputs after ReLU |
| `scores.mem` | Signed 16 bits | 10 | Expected final raw scores |
| `expected_digits.mem` | Unsigned 8 bits | 1 | Expected golden-model digit |
| `labels.mem` | Unsigned 8 bits | 1 | Ground-truth MNIST label |

Each file contains one fixed-width hexadecimal word per line, with no headers
or 0x prefixes. Signed values use two's complement. Files use LF newlines.
For zero-based image i and element j, the address is:

```text
images:         i * 784 + j
packed_images:  i * 98  + j
hidden:         i * 64  + j
scores:         i * 10  + j
expected_digits and labels: i
```

Input pixels follow row-major order: j = row * 28 + column. Packed byte j
contains pixels 8*j through 8*j+7, with the earliest pixel in bit 0. The payload
files contain 98 bytes per image. Prepend the start byte 0xAA when driving the
UART protocol.

Use `expected_digits.mem` for exact RTL prediction checks. Use `labels.mem`
for recognition accuracy. The integer model classifies 9,742 of the 10,000
images correctly; the RTL must reproduce its expected outputs on all images,
including those it misclassifies.

Hidden activations have 7 fractional bits and scores have 6 fractional bits.
Compare their stored signed integers exactly, without converting to float.

## Example testbench memory loading

The following declarations load the ten-image set. For a larger set, change
both N and VECTOR_DIR.

```systemverilog
localparam int N = 10;
localparam string VECTOR_DIR = "mem/vectors/10";
logic signed [15:0] images [0:N*784-1];
logic [7:0] packed_images [0:N*98-1];
logic signed [15:0] expected_hidden [0:N*64-1];
logic signed [15:0] expected_scores [0:N*10-1];
logic [7:0] expected_digits [0:N-1];
logic [7:0] labels [0:N-1];

initial begin
    $readmemh({VECTOR_DIR, "/images.mem"}, images);
    $readmemh({VECTOR_DIR, "/packed_images.mem"}, packed_images);
    $readmemh({VECTOR_DIR, "/hidden.mem"}, expected_hidden);
    $readmemh({VECTOR_DIR, "/scores.mem"}, expected_scores);
    $readmemh({VECTOR_DIR, "/expected_digits.mem"}, expected_digits);
    $readmemh({VECTOR_DIR, "/labels.mem"}, labels);
end
```

Run the simulator from the repository root. The future core testbench should
load one image, wait for inference completion, and compare each completed
layer with its expected values before checking the predicted digit.

## Regenerate and verify

From the repository root:

```powershell
.\.venv\Scripts\python.exe "model training\python\export_vectors.py"
.\.venv\Scripts\python.exe -m unittest discover -s "model training\python" -p test_export_vectors.py -v
```

Generation loads the hardware parameter exports and checks their hashes against
the passing step-10 verification. All six files in every set are read back,
compared exactly with their source arrays, and used to recompute the layer
outputs. Packed bytes also undergo an exact unpacking check.

All three sets passed. The full set contains 640,000 expected hidden values,
100,000 expected scores, and 10,000 expected digits. Results are recorded in
`../../docs/test_vector_verification.json`; each set has its own `manifest.json`.
The three packing and file-format tests passed, including signed score
endpoints and first/last pixel positions. RTL simulation follows when the
hardware modules are implemented.
