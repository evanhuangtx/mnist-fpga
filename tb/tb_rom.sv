`timescale 1ns/1ps
module tb_rom;
  logic clk=0;always #5 clk=~clk;
  logic [12:0] addr=0;
  logic [3:0] baddr=0;
  logic [7:0][7:0] data;
  logic [7:0][31:0] bdata;
  logic signed [7:0] weights [0:50815];
  logic signed [31:0] biases [0:73];
  rom wr(.clk(clk),.addr(addr),.data(data));
  rom #(.LANE_W(32),.DEPTH(10),.MEM_FILE("mem/biases_x8.mem")) br(.clk(clk),.addr(baddr),.data(bdata));
  logic [63:0] held_weights;
  logic [255:0] held_biases;
  bit have_read=0;
  initial begin
    $readmemh("mem/weights.mem",weights);$readmemh("mem/biases.mem",biases);
    for(int layer=0;layer<2;layer++)begin
      for(int g=0;g<(layer==0?8:2);g++)begin
        for(int i=0;i<(layer==0?784:64);i++)begin
          @(negedge clk);
          held_weights=data; held_biases=bdata;
          addr=13'((layer==0?0:6272)+g*(layer==0?784:64)+i);
          baddr=4'((layer==0?0:8)+g);
          #1;
          if(have_read && (data!==held_weights || bdata!==held_biases))
            $fatal(1,"ROM output changed before clock");
          @(posedge clk);#1;have_read=1;
          for(int k=0;k<8;k++)begin
            if(layer==1 && g*8+k>=10)begin
              if(data[k]!==8'd0||bdata[k]!==32'd0)$fatal(1,"Padding lane not zero");
            end else begin
              if($signed(data[k])!==weights[(layer==0?0:50176)+(g*8+k)*(layer==0?784:64)+i])
                $fatal(1,"Weight mismatch at word %0d lane %0d",addr,k);
              if($signed(bdata[k])!==biases[(layer==0?0:64)+g*8+k])
                $fatal(1,"Bias mismatch at group %0d lane %0d",baddr,k);
            end
          end
        end
      end
    end
    $display("PASS: all 6400 packed weight words and 10 bias groups, including padding");$finish;
  end
endmodule
