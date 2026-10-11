// Code your design here
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
  
  always@(posedge clk) begin
    
    if(reset) accumulator <= 32'sd0;
    else if(load_bias) accumulator <= bias;
    else if(accumulation_enable)accumulator <= accumulator + extended_product;
    
  end

  
endmodule 