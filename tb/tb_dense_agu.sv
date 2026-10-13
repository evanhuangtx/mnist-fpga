`timescale 1ns / 1ps

module tb_dense_agu;
    logic complete1, complete8;
    agu_check #(.LANES(1)) single_lane(.complete(complete1));
    agu_check #(.LANES(8)) eight_lanes(.complete(complete8));
    initial begin
        wait (complete1 && complete8);
        $display("PASS: dense AGU with one and eight lanes");
        $finish;
    end
    initial begin
        #1000000;
        $fatal(1, "AGU test timed out");
    end
endmodule

module agu_check #(parameter int LANES = 8)(output logic complete = 0);
    logic clk = 0;
    always #5 clk = ~clk;
    logic rst = 1, start = 0;
    logic [12:0] in_size = 0, out_size = 0, act_raddr, out_idx;
    logic [15:0] w_base = 0, w_addr;
    logic [6:0] b_base = 0, b_addr;
    logic valid, first, last, done;
    dense_agu #(.LANES(LANES)) dut(.*);

    task automatic run_layer(int ins, int outs, int wb, int bb);
        int steps, groups;
        @(negedge clk);
        in_size = 13'(ins);
        out_size = 13'(outs);
        w_base = 16'(wb);
        b_base = 7'(bb);
        start = 1;
        @(negedge clk);
        start = 0;
        groups = (outs + LANES - 1) / LANES;
        steps = 0;
        for (int group_idx = 0; group_idx < groups; group_idx++) begin
            for (int input_idx = 0; input_idx < ins; input_idx++) begin
                if (valid !== 1 || int'(act_raddr) != input_idx || int'(out_idx) != group_idx * LANES ||
                    int'(w_addr) != wb + steps || int'(b_addr) != bb + group_idx ||
                    first !== (input_idx == 0) || last !== (input_idx == ins - 1) || done !== 0)
                    $fatal(1, "LANES=%0d group=%0d input=%0d address/control mismatch", LANES, group_idx, input_idx);
                steps++;
                @(negedge clk);
            end
        end
        if (done !== 1 || valid !== 0) $fatal(1, "Completion mismatch");
        @(negedge clk);
        if (done !== 0) $fatal(1, "Done did not clear");
    endtask

    initial begin
        repeat (2) @(negedge clk);
        rst = 0;
        if (valid !== 0 || first !== 0 || last !== 0 || w_addr !== 0 || b_addr !== 0)
            $fatal(1, "Reset outputs incorrect");
        run_layer(784, 64, 0, 0);
        run_layer(64, 10, LANES == 8 ? 6272 : 50176, LANES == 8 ? 8 : 64);
        run_layer(3, 10, 100, 5);
        run_layer(1, 1, 23, 2);
        run_layer(1, 128, 0, 0);
        run_layer(0, 10, 0, 0);
        run_layer(3, 0, 0, 0);
        complete = 1;
    end
endmodule
