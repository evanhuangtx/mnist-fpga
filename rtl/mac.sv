`timescale 1ns/1ps
module mac (
    input  logic clk,
  input  logic reset,
  input  logic load_bias,
  input  logic accumulation_enable,
  input  logic signed [15:0]activation,
  input  logic signed [7:0]weight,
  input  logic signed [31:0]bias,
 
  output logic signed [31:0]accumulator
);
  logic signed [23:0]product;
  logic signed [31:0]extended_product;
  
  assign product = weight * activation;
  assign extended_product = {{8{product[23]}},product};
  
  always_ff @(posedge clk) begin
    
    if(reset) accumulator <= 32'sd0;
    else if(accumulation_enable) begin
      if(load_bias) accumulator <= bias + extended_product;
      else accumulator <= accumulator + extended_product;
    end
    
  end

  
endmodule 
