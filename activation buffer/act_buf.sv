`timescale 1ns / 1ps

module act_buf #(
    parameter int DEPTH = 5408,
    parameter int WIDTH = 16,
    parameter int AW    = 13             
)(
    input  logic                    clk,

    
    input  logic                    we,       
    input  logic [AW-1:0]           waddr,
    input  logic signed [WIDTH-1:0] wdata,

    
    input  logic [AW-1:0]           raddr,
    output logic signed [WIDTH-1:0] rdata     
);
    logic signed [WIDTH-1:0] mem [0:DEPTH-1];

    always_ff @(posedge clk) begin
      if(we) mem[waddr] <= wdata;
      rdata <= mem[raddr];
    end
  
        

       
        

endmodule
