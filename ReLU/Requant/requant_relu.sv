`timescale 1ns / 1ps

module requant_relu(
    input logic signed [31:0]acc,
    input logic relu_en,
    input logic [4:0]shift,
    output logic signed [15:0] y);
    
    logic signed [31:0] shifted;
    always_comb begin
        shifted = acc >>> shift;
        if(shifted >32'sd32767) shifted = 32'sd32767;
        else if(shifted <-32'sd32768) shifted = -32'sd32768;
        else shifted = shifted;
   
        if(relu_en) begin
            if(shifted <32'sd0) shifted = 0;
            else shifted = shifted;
        end
        else shifted = shifted;
        
        
        
       y = shifted[15:0];
   
    end
    
endmodule
