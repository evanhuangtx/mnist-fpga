`timescale 1ns / 1ps

module tb_requant_relu;

    logic signed [31:0] acc;
    logic relu_en;
    logic [4:0] shift;
    logic signed [15:0] y;

    logic signed [31:0] exp_shifted;
    logic signed [15:0] expected;
    int errors;
    int i;

    requant_relu dut(
        .acc(acc),
        .relu_en(relu_en),
        .shift(shift),
        .y(y));

    initial begin
        errors = 0;

        // Directed tests: set inputs, wait, check y
        acc = 1000; shift = 2; relu_en = 0; #1;
        if(y !== 16'sd250) begin $display("FAIL: 1000>>>2 got %0d, expected 250", y); errors++; end

        acc = -5; shift = 1; relu_en = 0; #1;
        if(y !== -16'sd3) begin $display("FAIL: -5>>>1 got %0d, expected -3", y); errors++; end

        acc = 2000000; shift = 0; relu_en = 0; #1;
        if(y !== 16'sd32767) begin $display("FAIL: saturate high got %0d, expected 32767", y); errors++; end

        acc = -2000000; shift = 0; relu_en = 0; #1;
        if(y !== -16'sd32768) begin $display("FAIL: saturate low got %0d, expected -32768", y); errors++; end

        acc = -1000; shift = 0; relu_en = 1; #1;
        if(y !== 16'sd0) begin $display("FAIL: relu negative got %0d, expected 0", y); errors++; end

        acc = 1000; shift = 0; relu_en = 1; #1;
        if(y !== 16'sd1000) begin $display("FAIL: relu positive got %0d, expected 1000", y); errors++; end

        acc = 123; shift = 0; relu_en = 0; #1;
        if(y !== 16'sd123) begin $display("FAIL: shift 0 got %0d, expected 123", y); errors++; end

        // Random tests: compute the expected answer the same way, then compare
        for(i = 0; i < 10000; i++) begin
            acc = $random;
            shift = $urandom_range(0, 31);
            relu_en = $urandom_range(0, 1);
            #1;

            exp_shifted = acc >>> shift;
            if(exp_shifted > 32'sd32767) exp_shifted = 32'sd32767;
            else if(exp_shifted < -32'sd32768) exp_shifted = -32'sd32768;
            if(relu_en && exp_shifted < 32'sd0) exp_shifted = 0;
            expected = exp_shifted[15:0];

            if(y !== expected) begin
                $display("FAIL: acc=%0d shift=%0d relu=%0b got %0d, expected %0d", acc, shift, relu_en, y, expected);
                errors++;
            end
        end

        if(errors == 0) $display("PASS: all tests passed");
        else $display("FAIL: %0d errors", errors);
        $finish;
    end

endmodule