`timescale 1ns/1ps
module tb_mac;
  logic clk=0;
  always #5 clk=~clk;
  logic reset=0, load_bias=0, accumulation_enable=0;
  logic signed [15:0] activation=0;
  logic signed [7:0] weight=0;
  logic signed [31:0] bias=0, accumulator;
  mac dut(.*);
  task automatic step(input bit rst_value, enable_value, load_value,
                      input int activation_value, weight_value, bias_value, expected);
    @(negedge clk);
    reset=rst_value; accumulation_enable=enable_value; load_bias=load_value;
    activation=16'(activation_value); weight=8'(weight_value); bias=32'(bias_value);
    @(posedge clk); #1;
    if(accumulator !== 32'(expected))
      $fatal(1,"MAC mismatch: reset=%0b en=%0b load=%0b expected=%0d got=%0d",
             reset,accumulation_enable,load_bias,expected,accumulator);
  endtask
  initial begin
    step(1,0,0,0,0,0,0);
    step(0,1,1,4,-2,5,-3);
    step(0,1,0,10,3,999,27);
    step(0,0,0,10,3,19,27);
    step(0,0,1,10,3,19,27);
    step(0,1,1,10,3,-7,23);
    step(0,1,1,-32768,-128,0,4194304);
    step(0,1,0,32767,-128,0,128);
    step(1,1,1,10,3,19,0);
    step(0,0,1,10,3,19,0);
    $display("PASS: MAC reset, first-product bias, accumulation, hold, and signed limits");
    $finish;
  end
endmodule
