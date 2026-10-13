`timescale 1ns / 1ps
module rom #(
    parameter int    WIDTH    = 8,
    parameter int    DEPTH    = 50816,
    parameter int    AW       = $clog2(DEPTH),
    parameter string MEM_FILE = "mem/weights.mem"
)(
    input  logic                    clk,
    input  logic [AW-1:0]           addr,
    output logic signed [WIDTH-1:0] data
);
  logic signed [WIDTH-1:0] memory [0:DEPTH-1];

    initial begin
        $readmemh(MEM_FILE, memory);
    end

    always_ff @(posedge clk) begin
        data <= memory[addr];
    end
endmodule
