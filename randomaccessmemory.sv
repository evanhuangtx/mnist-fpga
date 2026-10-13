`timescale 1ns / 1ps
module rom #(
    parameter int    LANES    = 8,
    parameter int    LANE_W   = 8,
    parameter int    DEPTH    = 50816,
    parameter int    AW       = $clog2(DEPTH),
    parameter string MEM_FILE = "mem/weights.mem"
)(
    input  logic                         clk,
    input  logic [AW-1:0]                addr,
    output logic [LANES-1:0][LANE_W-1:0] data      // data[k] = lane k
);
    logic [LANES*LANE_W-1:0] memory [0:DEPTH-1];

    initial begin
        $readmemh(MEM_FILE, memory);
    end

    always_ff @(posedge clk) begin
        data <= memory[addr];
    end
endmodule
