// top_loopback_basys3.v — UART echo test for Basys3
// Receives any byte and immediately echoes it back.
// If this works, UART is fine and the problem is in the MLP logic.
// If this times out too, the problem is pin mapping or clock.
`default_nettype none

module top_loopback_basys3 (
    input  wire       clk,
    input  wire       btnC,
    input  wire       RsRx,
    output wire       RsTx,
    output wire [3:0] led
);
    wire rst = btnC;

    wire [7:0] rx_data;
    wire       rx_valid;

    uart_rx #(.CLK_HZ(100_000_000), .BAUD(115_200)) u_rx (
        .clk(clk), .rst(rst),
        .rx_pin(RsRx),
        .rx_data(rx_data), .rx_valid(rx_valid)
    );

    wire tx_busy;
    uart_tx #(.CLK_HZ(100_000_000), .BAUD(115_200)) u_tx (
        .clk(clk), .rst(rst),
        .tx_data(rx_data),
        .tx_start(rx_valid),
        .tx_busy(tx_busy),
        .tx_pin(RsTx)
    );

    reg [3:0] led_r;
    always @(posedge clk)
        if (rx_valid) led_r <= rx_data[3:0];
    assign led = led_r;

endmodule

`default_nettype wire
