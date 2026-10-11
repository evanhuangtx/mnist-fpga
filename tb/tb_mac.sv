// Code your testbench here
// or browse Examples
`timescale 1ns/1ps

module tb_mac;
    logic clk;
    logic reset;
    logic load_bias;
    logic accumulation_enable;

    logic signed [15:0] activation;
    logic signed [7:0] weight;
    logic signed [31:0] bias;
    logic signed [31:0] accumulator;
  
  mac dut(
    .clk(clk),
    .reset(reset),
    .load_bias(load_bias), .accumulation_enable(accumulation_enable),
        .activation(activation),
        .weight(weight),
        .bias(bias),
        .accumulator(accumulator)
    );
  
  
  initial clk = 0;
  always #5 clk = ~clk;
  
  initial begin
    load_bias = 0;
    accumulation_enable = 0;
    activation = 0;
    weight = 0;
    bias =0;
    reset =1;
    @(posedge clk);
    #1;
    $display("After reset: %0d", accumulator);
    
    @(negedge clk);
    reset= 0;
    load_bias = 1;
    bias = 5;
    accumulation_enable=0;
    @(posedge clk);
    #1;
    $display("After loading bias:%0d",accumulator);
    
    
    @(negedge clk);
    load_bias = 0;
    accumulation_enable =1;
    activation =4;
    weight = -2;
    @(posedge clk);
    #1;
    $display("after accumulator:%0d",accumulator);
    
    @(negedge clk);
    load_bias = 0;
    accumulation_enable =0;
    activation =10;
    weight = 3;
    @(posedge clk);
    #1;
    $display("after accumulator:%0d",accumulator);
    
    @(negedge clk);
    load_bias = 1;
    accumulation_enable =1;
    activation =10;
    weight = 3;
    bias = -7;
    @(posedge clk);
    #1;
    $display("after accumulator:%0d",accumulator);
    
    
    
    
    $finish;
    
    
    
    
    
    
  end
  
  
endmodule
