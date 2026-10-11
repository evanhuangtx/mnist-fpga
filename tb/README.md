# Multiply–Accumulate (MAC)

We find a weighted sum for a single neuron by adding activation * weight and adding it to the accumulator each clock cycle. It loads a neuron’s bias before processing the neuron.

## Inputs and output

| Signal | Description |
|---|---|
| clk | Clock; the accumulator updates on rising edges |
| reset | sets accumulator to zero: Synchronous reset |
| load_bias | Replaces the accumulator with the bias |
| accumulation_enable | Enables adding one product to the accumulator |
| activation | Signed 16-bit input |
| weight | Signed 8-bit weight |
| bias | Signed 32-bit starting value |
| accumulator | Signed 32-bit running sum |

## Behavior

Control priority : reset, then load_bias, then accumulation_enable.
If none, it will hold its previous value.

Because the multiplication of 8 bit times 16 bits makes a 24 bit product, we must sign extend it to 32 bits. Also since we use wrap-around behavior to handle overflow all the sums must be within a 32 bit range(checked during the quantized verification phase to be compliant)


## Testbench

The current testbench prints results for:
- Reset
- Loading a bias
- Accumulating a negative product
- Holding while accumulation is disabled
- Loading a bias while accumulation is also enabled

The expected output sequence is 0, 5, -3, -3, -7.
These are printed checks; the testbench does not yet automatically
fail when a value is incorrect.

## Running the simulation

From the repository root, with Icarus Verilog installed:

mkdir -p tmp
iverilog -g2012 -Wall -s tb_mac -o tmp/tb_mac.vvp rtl/mac.sv tb/tb_mac.sv
vvp tmp/tb_mac.vvp
