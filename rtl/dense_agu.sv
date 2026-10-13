`timescale 1ns / 1ps

module dense_agu #(
    parameter int LANES = 8,
    parameter int LG = $clog2(LANES)
)(
    input logic clk,
    input logic rst,
    input logic start,
    input logic [12:0] in_size,
    input logic [12:0] out_size,
    input logic [15:0] w_base,
    input logic [6:0] b_base,
    output logic [12:0] act_raddr,
    output logic [15:0] w_addr,
    output logic [6:0] b_addr,
    output logic [12:0] out_idx,
    output logic valid,
    output logic first,
    output logic last,
    output logic done
);
    logic running;
    logic [12:0] i, g, in_last, grp_last;
    logic [6:0] b_base_r;

    initial begin
        if (LANES < 1 || (LANES & (LANES - 1)) != 0 || LG != $clog2(LANES))
            $fatal(1, "LANES must be a positive power of two with matching LG");
    end

    always_ff @(posedge clk) begin
        done <= 1'b0;
        if (rst) begin
            running <= 1'b0;
            i <= '0;
            g <= '0;
            w_addr <= '0;
            in_last <= '0;
            grp_last <= '0;
            b_base_r <= '0;
        end else if (start) begin
            i <= '0;
            g <= '0;
            w_addr <= w_base;
            b_base_r <= b_base;
            running <= (in_size != 0 && out_size != 0);
            done <= (in_size == 0 || out_size == 0);
            in_last <= in_size - 13'd1;
            grp_last <= 13'(((int'(out_size) + LANES - 1) >> LG) - 1);
        end else if (running) begin
            w_addr <= w_addr + 16'd1;
            if (i == in_last) begin
                i <= '0;
                if (g == grp_last) begin
                    running <= 1'b0;
                    done <= 1'b1;
                end else g <= g + 13'd1;
            end else i <= i + 13'd1;
        end
    end

    assign act_raddr = i;
    assign out_idx = g << LG;
    assign b_addr = b_base_r + g[6:0];
    assign valid = running;
    assign first = running && (i == 0);
    assign last = running && (i == in_last);
endmodule
