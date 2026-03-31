# MLP Accelerator on FPGA — Technical Report
### MNIST Handwritten Digit Recognition on Digilent Basys3 (Artix-7)

---

## 1. Introduction

This project implements a real-time handwritten digit recognition system using a
Multi-Layer Perceptron (MLP) neural network accelerator deployed on a Digilent Basys3
FPGA development board. The system classifies handwritten digits (0–9) from the MNIST
dataset by receiving raw pixel data over UART from a host PC, performing inference
entirely in hardware, and returning the predicted digit over the same UART link.

The primary objective is to demonstrate that neural network inference can be offloaded
from a general-purpose CPU to reconfigurable hardware, achieving deterministic low-latency
inference with minimal resource usage. The design targets the Xilinx Artix-7 XC7A35T
device on the Basys3 board, synthesized and implemented using Xilinx Vivado 2024.2.

### 1.1 Motivation

Neural network inference on embedded and edge devices is a growing area of interest.
FPGAs offer a compelling middle ground between the flexibility of CPUs and the raw
throughput of dedicated ASICs. By implementing the inference pipeline directly in RTL,
this project explores the trade-offs between hardware resource usage, inference latency,
numerical precision, and classification accuracy.

### 1.2 Scope

- Train a 3-layer MLP on MNIST using PyTorch
- Quantize weights to fixed-point representation
- Implement the inference engine in synthesizable Verilog
- Port the design from the original ECP5 target to the Basys3 Artix-7 platform
- Validate end-to-end operation via a live webcam demo over UART

---

## 2. System Architecture

The system is divided into three subsystems: the host PC, the UART communication
interface, and the FPGA inference engine.

```
┌─────────────────────┐        UART 115200 8N1        ┌──────────────────────────┐
│     Host PC         │ ──── 784 bytes (pixels) ────► │   Basys3 FPGA (Artix-7)  │
│  webcam_detect.py   │ ◄─────── 1 byte (result) ──── │   mlp_accel + UART       │
└─────────────────────┘                               └──────────────────────────┘
                                                               │
                                                        LED[3:0] = digit (binary)
```

### 2.1 Host Side

The host script (`host/webcam_detect.py`) captures frames from a webcam, crops the
central region, applies Otsu thresholding to produce a binary image, resizes it to
28×28 pixels, and sends the 784 raw uint8 pixel values over a serial port to the FPGA.
It then waits for a single byte response containing the predicted digit.

### 2.2 FPGA Side

The FPGA design consists of four modules:

| Module           | Role                                              |
|------------------|---------------------------------------------------|
| `top_basys3`     | Top-level: clock, reset, pixel buffer, glue logic |
| `mlp_accel`      | MLP inference FSM + MAC + weight BRAMs            |
| `uart_rx`        | 8N1 UART receiver (100 MHz, 115200 baud)          |
| `uart_tx`        | 8N1 UART transmitter (100 MHz, 115200 baud)       |

---

## 3. Neural Network Design

### 3.1 Architecture

The network is a fully-connected MLP with the following topology:

```
Input (784)  →  Dense 64 (ReLU)  →  Dense 32 (ReLU)  →  Dense 10 (argmax)
```

- Input: 28×28 grayscale pixel values flattened to a 784-element vector
- Hidden Layer 1: 64 neurons with ReLU activation
- Hidden Layer 2: 32 neurons with ReLU activation
- Output Layer: 10 neurons (one per digit class), argmax for classification

Total parameters: (784×64 + 64) + (64×32 + 32) + (32×10 + 10) = **52,650**

### 3.2 Training

The model is trained using PyTorch on the standard MNIST dataset (60,000 training
images, 10,000 test images). Training uses cross-entropy loss with the Adam optimizer
for approximately 10 epochs. The trained model achieves approximately **94% accuracy**
on the MNIST test set.

### 3.3 Fixed-Point Quantization

To enable efficient hardware implementation, all weights and activations are quantized
from 32-bit floating point to fixed-point integers. The quantization scheme is:

| Data Type    | Format        | Bit Width | Scale Factor |
|--------------|---------------|-----------|--------------|
| Weights      | Q4.12 signed  | 16-bit    | 4096         |
| Activations  | Q8.8 unsigned | 16-bit    | 256          |
| Accumulator  | Q8.16 signed  | 32-bit    | —            |
| Biases       | Q8.8 signed   | 16-bit    | 256          |

Quantized weights are exported as hexadecimal files (`*.hex`) and loaded into FPGA
block RAM at synthesis time via Verilog's `$readmemh` system task.

---

## 4. RTL Implementation

### 4.1 Top-Level Module (`top_basys3`)

The top-level module instantiates all submodules and manages the pixel receive buffer.
It accumulates incoming UART bytes into a 784-byte pixel buffer (`pix_buf`). Once all
784 bytes are received, it asserts the `start` signal to the MLP accelerator. When
inference completes (`done` asserted), the result is latched and transmitted back over
UART. The lower 4 bits of the result are also driven to LEDs LD0–LD3.

Key parameters:
- `CLK_HZ = 100_000_000` — 100 MHz Basys3 oscillator
- `BAUD = 115_200` — UART baud rate

### 4.2 MLP Accelerator (`mlp_accel`)

This is the core inference engine. It contains three sub-modules and an FSM.

#### 4.2.1 Weight BRAM (`weight_bram`)

A parameterized synchronous-read block RAM with no reset port. The `(* ram_style = "block" *)`
attribute instructs Vivado to infer Artix-7 RAMB18/RAMB36 primitives rather than
distributed LUT RAM. Three instances are used, one per layer:

| Instance | Depth  | Size    | Contents         |
|----------|--------|---------|------------------|
| `u_w1`   | 50,176 | 16-bit  | Layer 1 weights  |
| `u_w2`   | 2,048  | 16-bit  | Layer 2 weights  |
| `u_w3`   | 320    | 16-bit  | Layer 3 weights  |

#### 4.2.2 MAC Unit (`mac`)

A pipelined multiply-accumulate unit with 2-cycle latency. It computes:

```
acc += (a × b) >> 4
```

where `a` is a 16-bit activation and `b` is a 16-bit weight. The pipeline has two stages:
- Stage 1: compute `product = (a_sign_extended × b_sign_extended) >>> 4`
- Stage 2: accumulate with signed saturation to prevent overflow

Signed saturation clamps the accumulator to `[0x80000000, 0x7FFFFFFF]` on overflow,
preventing wrap-around errors that would corrupt classification results.

On `clr`, the accumulator is preloaded with the bias value (left-shifted by 8 to align
with the Q8.16 accumulator format).

#### 4.2.3 FSM

The FSM sequences all three layers through the shared MAC unit using an 18-state
one-hot encoding. The state sequence per neuron is:

```
IDLE → Lx_PRE → Lx_PRE2 → Lx_MAC (×N inputs) → Lx_DRAIN (×2) → Lx_STORE → ...
```

- `PRE`: present BRAM address for weight fetch (in_ctr = 0)
- `PRE2`: BRAM data valid; load MAC with bias; assert mac_clr
- `MAC`: stream weight × activation products into MAC for all N inputs
- `DRAIN`: flush the 2-cycle MAC pipeline
- `STORE`: apply ReLU and write result to activation buffer

A key optimization is the use of a registered `base_addr` accumulator instead of
computing `nrn_ctr × L_IN` combinatorially. This eliminates a wide multiplier from
the critical path, enabling clean timing closure at 100 MHz on the Artix-7.

#### 4.2.4 ReLU and Argmax

ReLU is applied at the STORE stage:
- If `acc ≤ 0`: output = 0
- If `acc > SAT_LIM (0x007FFF00)`: output = 0x7FFF (saturation)
- Otherwise: output = `acc[23:8]` (extract Q8.8 result from Q8.16 accumulator)

Argmax over the 10 output neurons is computed combinatorially after Layer 3 completes,
selecting the index of the maximum activation as the predicted digit.

### 4.3 UART Modules

Both `uart_rx` and `uart_tx` implement standard 8N1 framing. The receiver uses a
2-stage synchronizer on the RX pin to prevent metastability. Baud timing is derived
by counting clock cycles: `CLKS_PER_BIT = CLK_HZ / BAUD = 868` at 100 MHz / 115200.

The receiver samples at the midpoint of each bit period (`HALF_BIT = 434` cycles)
to maximize noise margin.

---

## 5. Platform Port: ECP5 to Basys3

The original design targeted the Radiona ULX3S board (Lattice ECP5 85F) using the
open-source Yosys/nextpnr toolchain. Porting to the Basys3 required the following changes:

### 5.1 Clock Frequency

| Parameter  | ULX3S (ECP5) | Basys3 (Artix-7) |
|------------|--------------|------------------|
| Clock      | 25 MHz       | 100 MHz          |
| CLK_HZ     | 25,000,000   | 100,000,000      |
| CLKS_PER_BIT | 217        | 868              |

The UART baud rate divisor scales automatically via the `CLK_HZ` parameter, so no
logic changes were needed — only the parameter value was updated.

### 5.2 Pin Mapping

| Signal  | ULX3S Pin    | Basys3 Pin | Notes                    |
|---------|--------------|------------|--------------------------|
| clk     | clk_25mhz    | W5         | Renamed port             |
| reset   | btn_pwr (active-low) | U18 (btnC, active-high) | Logic inverted |
| UART RX | ftdi_txd     | B18 (RsRx) | Renamed port             |
| UART TX | ftdi_rxd     | A18 (RsTx) | Renamed port             |
| led[3:0]| led[3:0]     | U16/E19/U19/V19 | Same logic          |

Reset polarity changed from active-low (`~btn_pwr`) to active-high (`btnC`), matching
the Basys3 button behaviour.

### 5.3 Constraints File

The ECP5 `.lpf` pin constraints file was replaced with a Vivado-compatible `.xdc` file
(`syn/basys3.xdc`) using the official Basys3 master XDC as reference. Configuration
properties for the Artix-7 SPI boot mode were also added:

```tcl
set_property CONFIG_VOLTAGE 3.3 [current_design]
set_property CFGBVS VCCO [current_design]
set_property BITSTREAM.GENERAL.COMPRESS TRUE [current_design]
```

### 5.4 Toolchain

| Step         | ECP5 (original)     | Basys3 (ported)         |
|--------------|---------------------|-------------------------|
| Synthesis    | Yosys               | Vivado (synth_design)   |
| Place & Route| nextpnr-ecp5        | Vivado (place/route)    |
| Bitstream    | ecppack             | Vivado (write_bitstream)|
| Programming  | openFPGALoader      | Vivado Hardware Manager |

---

## 6. Resource Utilization

The following estimates are based on the Artix-7 XC7A35T device. Exact figures
are available in the Vivado utilization report after implementation.

| Resource      | Estimated Usage | Artix-7 Available | Notes                        |
|---------------|-----------------|-------------------|------------------------------|
| LUT           | ~3,500          | 20,800            | FSM, MAC, address logic      |
| FF            | ~1,200          | 41,600            | Pipeline registers, buffers  |
| RAMB18        | ~12             | 50                | Weight BRAMs (3 layers)      |
| DSP48E1       | 1               | 90                | MAC multiplier               |
| IO            | 8               | 106               | clk, rst, RX, TX, 4× LED    |

The design is well within the Artix-7 resource budget, using less than 20% of available
LUTs and approximately 24% of block RAM resources.

---

## 7. Host Software

### 7.1 Webcam Inference Script

`host/webcam_detect.py` provides a live demo interface:

1. Opens the default webcam using OpenCV
2. Displays a live feed with a green crop guide box
3. On `Space` keypress: crops the frame, applies Gaussian blur and Otsu thresholding,
   resizes to 28×28, sends 784 bytes over serial, reads 1 byte response
4. Overlays the predicted digit and latency on the camera feed
5. Displays a 280×280 upscaled preview of the exact image sent to the FPGA

### 7.2 UART Protocol

The protocol is intentionally minimal:

```
Host → FPGA : 784 bytes  (raw uint8 pixel values, row-major, 28×28)
FPGA → Host : 1 byte     (predicted digit, value 0–9)
```

No framing, checksums, or handshaking — the FPGA simply counts 784 received bytes
and triggers inference. This keeps the RTL simple and the latency predictable.

### 7.3 Dependencies

```bash
pip install opencv-python pyserial numpy
```

Usage:
```bash
python host/webcam_detect.py --port COM15 --baud 115200
```

---

## 8. Future Work

The current implementation provides a functional baseline. Several improvements are
planned for future iterations:

### 8.1 CNN Accelerator

Replacing the MLP with a Convolutional Neural Network (e.g. LeNet-5) would significantly
improve accuracy (from ~94% to ~99%) and better demonstrate the value of hardware
acceleration. This requires implementing:
- 2D convolution with line buffers
- Max pooling
- A dataflow architecture to overlap computation across layers

### 8.2 7-Segment Display Output

The Basys3 has a 4-digit 7-segment display. Driving it to show the predicted digit
directly on the board would eliminate the need for a host PC to observe results,
making the demo more self-contained.

### 8.3 On-Board Image Capture

Integrating a camera module (e.g. OV7670 via PMOD) would remove the UART bottleneck
entirely, enabling true real-time inference at the full ~476 inferences/second rate
the hardware is capable of.

### 8.4 Throughput Benchmarking

Adding hardware performance counters (inference cycle count, throughput register)
would allow quantitative comparison against CPU and GPU baselines, strengthening
the "acceleration" narrative for academic presentation.

---

## 9. Conclusion

This project successfully implements and demonstrates a fixed-point MLP inference
accelerator on the Digilent Basys3 FPGA board. The design was ported from the original
Lattice ECP5 / Yosys toolchain to Xilinx Artix-7 / Vivado, with adaptations for the
100 MHz clock, Basys3 pin mapping, and Vivado project flow. The end-to-end system —
from webcam capture on the host PC through UART communication to hardware inference
and LED output — demonstrates a complete hardware-software co-design workflow.

The foundation is solid for extension toward a CNN accelerator, which would represent
a more compelling contribution for an engineering final year project context.

---

## Appendix A — File Structure

```
NN_Digits/
├── rtl/
│   ├── mlp_accel.v           ← MAC + FSM + weight BRAMs
│   ├── top_basys3.v          ← Basys3 top-level (100 MHz)
│   ├── top_loopback_basys3.v ← UART loopback diagnostic
│   └── uart/
│       ├── uart_rx.v         ← 8N1 receiver
│       └── uart_tx.v         ← 8N1 transmitter
├── rtl/hex/
│   ├── weight_l1.hex         ← Layer 1 weights (Q4.12)
│   ├── weight_l2.hex         ← Layer 2 weights (Q4.12)
│   ├── weight_l3.hex         ← Layer 3 weights (Q4.12)
│   ├── bias_l1.hex           ← Layer 1 biases  (Q8.8)
│   ├── bias_l2.hex           ← Layer 2 biases  (Q8.8)
│   └── bias_l3.hex           ← Layer 3 biases  (Q8.8)
├── syn/
│   ├── basys3.xdc            ← Vivado pin constraints
│   └── build_basys3.tcl      ← Vivado batch build script
├── training/
│   ├── train.py              ← PyTorch training script
│   └── quantize.py           ← Weight quantization + hex export
└── host/
    └── webcam_detect.py      ← Live webcam inference demo
```

## Appendix B — Key Parameters

| Parameter     | Value           |
|---------------|-----------------|
| Target device | XC7A35T-1CPG236C|
| Clock         | 100 MHz         |
| UART baud     | 115,200         |
| Network       | 784→64→32→10    |
| Weight format | Q4.12 (16-bit)  |
| Activation    | Q8.8  (16-bit)  |
| Accumulator   | Q8.16 (32-bit)  |
| Test accuracy | ~94%            |
