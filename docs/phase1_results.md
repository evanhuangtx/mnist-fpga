# Phase 1 Python results

The 784 to 64 to 10 binary-input MLP and its hardware verification artifacts
meet the Phase 1 accuracy and arithmetic requirements. The trained float model
scores 97.43% and the integer golden model scores 97.42% on all 10,000 official
MNIST test images. Hardware implementation starts from the included parameters,
integer reference, and expected layer outputs.

## Training and dataset

Images are flattened row by row and binarized using raw pixel >=128. The official
60,000-image training split is divided into 55,000 optimization images and a
5,000-image validation holdout using seed 42. The 10,000-image test split is
separate. The final training run uses 40 epochs, batch size 128, Adam with weight
decay 0.0001, and a cosine schedule from learning rate 0.001 to 0.0001.

Training inputs receive zero-filled shifts of up to two pixels in each direction.
The best validation checkpoint is selected at epoch 40 with 97.36% validation
accuracy. Settings and all epoch metrics are in `training_results.json`.

The first completed run without augmentation reached 96.87% test accuracy.
The training/validation gap motivated adding small image shifts. Both completed
runs were evaluated on the test set; checkpoint selection within each run used
only validation metrics. The baseline weights and metrics are preserved for
comparison. The final reported test result is therefore a benchmark measurement
after two experiments, rather than an untouched final holdout study.

## Measured results

| Metric | Value | Evidence |
|---|---:|---|
| Float correct predictions | 9,743 / 10,000 | `quantized_accuracy.json` |
| Integer correct predictions | 9,742 / 10,000 | `quantized_accuracy.json` |
| Quantization accuracy loss | 0.01 percentage points | `quantized_accuracy.json` |
| Allowed accuracy loss | 1.00 percentage point | Arithmetic and evaluation convention |
| Float/integer prediction disagreements | 10 | `test_predictions.csv` |
| Float correct, integer wrong | 5 | `quantized_accuracy.json` |
| Float wrong, integer correct | 4 | `quantized_accuracy.json` |
| Inactive float hidden neurons | 1 / 64, zero-based index 13 | `dead_neurons.json` |
| Tests passed | 34 | `phase1_test_results.txt` |

Inactive means ReLU output exactly zero for every measured image. The activity
check covers all 60,000 official training images, including validation, and all
10,000 test images, without augmentation. This describes dataset-level activity.
All 64 hidden neurons remain in both models.

## Arithmetic safety

Weights use signed 8-bit integers, biases and accumulators signed 32-bit integers,
and activations signed 16-bit integers. Offline conversion uses nearest rounding
with ties to even; inference uses arithmetic right shift, saturation, and hidden
ReLU. No parameters were clipped.

| Layer | Weight fraction | Activation output fraction | Shift | Conservative absolute accumulator bound | Observed accumulator range |
|---|---:|---:|---:|---:|---|
| Hidden dense | 7 | 7 | 0 | 9,411 | -2,105 to 2,133 |
| Output dense | 6 | 6 | 7 | 48,495,796 | -288,961 to 152,170 |

Both conservative bounds fit signed 32-bit arithmetic and cover every MAC prefix.
No output values saturated during the full test. `integer_arithmetic.md` specifies
all signed formats, memory ordering, shift behavior, and argmax ties.

## Verification and exports

The scalar reference matched all 640 hidden activations, 100 scores, and ten
predictions on the first ten test images. Its MACs use Python integers and check
each prefix. Edge-case tests cover signed multiplication, biases, negative-value
shifts, saturation, ReLU, overflow rejection, and ties among negative scores.

Hardware export and reload preserved all 50,816 weights and 74 biases exactly.
Inference from the reloaded files matched 640,000 hidden activations, 100,000
scores, and 10,000 predicted digits. The separate 10-, 1,000-, and 10,000-image
vector sets passed complete file reload, pixel packing, and output recomputation
checks. File hashes identify the verified checkpoints, ROMs, and test vectors.

`phase1_verification.json` records the final passing unit suite and provenance
check of 23 parameter/vector files. Full comparison evidence is in
`export_verification.json` and `test_vector_verification.json`.

## Phase 2 handoff

Implement and test the MAC, requantization/ReLU, one-cycle-read activation buffer,
weight ROM, dense address generator, and signed argmax. Use the arithmetic contract
and the included small vector set for initial integration, then progress to
1,000 and 10,000 images. RTL equivalence, FPGA resource use, timing closure, and
UART/board operation are verified in later phases.
