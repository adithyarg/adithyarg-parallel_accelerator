# mlp_accel — MNIST MLP Accelerator on Basys3 FPGA

A fully open-source neural network inference accelerator implemented on the
Digilent Basys3 development board (Xilinx Artix-7 XC7A35T). Recognises handwritten
digits (0–9) from the MNIST dataset at **94% accuracy**, using Vivado 2024.2.

## Demo

Hold a handwritten digit in front of your webcam. The FPGA classifies it and
sends the result back over UART. The predicted digit appears on screen and on
the board's LEDs (LD0–LD3 in binary).

```
Host PC  ──USB──►  Basys3 Artix-7  ──LED[3:0]──►  predicted digit (binary)
         ◄──────   (115200 baud)
```

## Results

| Metric               | Value                        |
|----------------------|------------------------------|
| Test accuracy        | ~94% (MNIST test set)        |
| Inference latency    | ~2.1 ms (hardware only)      |
| Clock                | 100 MHz                      |
| UART baud rate       | 115,200                      |
| Target device        | XC7A35T-1CPG236C             |

## Network Architecture

```
Input (784)  →  Dense 64 (ReLU)  →  Dense 32 (ReLU)  →  Dense 10 (argmax)
```

**Fixed-point quantization:**
- Weights:      Q4.12 signed 16-bit  (scale = 4096)
- Activations:  Q8.8  unsigned 16-bit (scale = 256)
- Accumulator:  Q8.16 signed 32-bit
- Biases:       Q8.8  signed 16-bit

## Project Layout

```
mlp_accel/
├── README.md
├── report.md                         ← full technical report
├── rtl/
│   ├── mlp_accel.v                   ← accelerator: MAC + FSM + weight BRAMs
│   ├── top_basys3.v                  ← Basys3 top level: UART + pixel buffer + MLP
│   ├── top_loopback_basys3.v         ← UART loopback test (diagnostics)
│   ├── hex/                          ← quantized weights and biases (hex)
│   └── uart/
│       ├── uart_rx.v                 ← 8N1 UART receiver
│       └── uart_tx.v                 ← 8N1 UART transmitter
├── syn/
│   ├── basys3.xdc                    ← Vivado pin constraints (Artix-7 CPG236)
│   └── build_basys3.tcl              ← Vivado batch build script
├── training/
│   ├── train.py                      ← PyTorch training (MNIST, 784→64→32→10)
│   └── quantize.py                   ← weight quantization → hex file export
└── host/
    └── webcam_detect.py              ← live webcam inference via UART
```

## Prerequisites

**Hardware:**
- Digilent Basys3 (Artix-7 XC7A35T-1CPG236C)
- USB cable (micro-USB)

**Software:**
```bash
# FPGA toolchain
Vivado 2024.2 (or later)

# Python
pip install torch torchvision numpy pyserial opencv-python
```

## Quick Start

```bash
# 1. Train the model
python training/train.py

# 2. Export quantized weights as hex files
python training/quantize.py

# 3. Build bitstream in Vivado TCL console
#    (run from NN_Digits/ directory)
source syn/build_basys3.tcl

# 4. Program the board via Vivado Hardware Manager
#    open_hw_manager → connect → program top_basys3.bit

# 5. Run live webcam demo
python host/webcam_detect.py --port COM15
```

## Vivado Build (manual)

Add these sources to a new Vivado project targeting `xc7a35tcpg236-1`:

**Design Sources:**
- `rtl/top_basys3.v`
- `rtl/mlp_accel.v`
- `rtl/uart/uart_rx.v`
- `rtl/uart/uart_tx.v`

**Constraints:** `syn/basys3.xdc`

Set `top_basys3` as top module. Add `rtl/hex/` as an include directory so
`$readmemh` can locate the weight files during synthesis.

## Host Protocol

1. Host sends exactly **784 bytes** — raw uint8 pixel values, row-major (28×28)
2. FPGA buffers all 784 bytes, runs inference
3. FPGA sends **1 byte** — predicted digit 0–9

## Webcam Demo

```bash
python host/webcam_detect.py --port COM15
```

- Green box shows the crop region — keep your digit centred inside it
- Second window shows the 28×28 image sent to the FPGA
- Press **Space** to run inference, **Q** to quit
- Result overlaid on camera feed

## Pin Mapping (Basys3)

| Signal | Pin  | Description          |
|--------|------|----------------------|
| clk    | W5   | 100 MHz oscillator   |
| btnC   | U18  | Reset (centre button)|
| RsRx   | B18  | UART RX              |
| RsTx   | A18  | UART TX              |
| led[0] | U16  | Result bit 0 (LSB)   |
| led[1] | E19  | Result bit 1         |
| led[2] | U19  | Result bit 2         |
| led[3] | V19  | Result bit 3 (MSB)   |

## Toolchain Versions Tested

| Tool    | Version  |
|---------|----------|
| Vivado  | 2024.2   |
| Python  | 3.12     |
| PyTorch | 2.x      |

## License

MIT
