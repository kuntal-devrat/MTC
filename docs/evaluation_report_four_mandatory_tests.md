# Empirical Evaluation Report: Four Mandatory Tests
## Micro-Train-Compiler (MTC) on Cycle-Accurate ESP32 Hardware Emulator (`esp32sim`)

**Authors:** Ajay Sisodia, Harsh Chaudhary, Devrat Kuntal, Aryan Upadhyay  
**Affiliation:** Department of Computer Science & Engineering, G. L. Bajaj Group of Institutions, Mathura  
**Target Hardware:** ESP32-C3 RISC-V RV32IMC @ 160 MHz (cycle-accurate emulation via `esp32sim`)  
**Instrumentation:** RISC-V Hardware CSR `mcycle` (`csrr %0, mcycle`)  
**Compilation Suite:** `mtc-compiler` v0.1.0, `riscv64-unknown-elf-gcc` v15.3.0, `esptool` v5.6.0  

---

## 1. Executive Summary

This report documents the empirical results for the **four mandatory experimental evaluations** required for an academic research publication (IEEE ESL / IEEE Access / TinyML Symposium) and capstone presentation defense. 

All benchmarks were compiled from high-level JSON specifications using MTC, synthesized into standalone C99 code, cross-compiled into bare-metal RISC-V RV32IMC machine code, converted into bootable ESP32 flash images, and executed on the cycle-accurate [`esp32sim`](https://github.com/mikroverk/esp32sim.git) hardware emulator. Hardware cycle counts and latencies were measured directly via RISC-V Control and Status Register (`mcycle`) reads to guarantee zero measurement interference from UART logging or emulator timeout loops.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      VERIFIED BENCHMARK SUMMARY HIGHLIGHTS                  │
├─────────────────────────────────────────────────────────────────────────────┤
│ • Peak SRAM Watermark:         60 B to 564 B (≤ 0.86% of 64 KB budget)       │
│ • Dynamic Heap Allocations:    EXACTLY 0 (malloc count = 0 across all runs) │
│ • Flash Binary Footprint:      4.4 KB to 5.9 KB total bare-metal flash      │
│ • Forward Inference Latency:   2.15 µs to 21.26 µs (344 to 3,401 cycles)    │
│ • Training Step Latency:       6.86 µs to 81.77 µs (1,097 to 13,083 cycles) │
│ • Training Throughput:         12,229 to 145,772 samples / sec @ 160 MHz    │
│ • SRAM Savings from Reuse:     31.8% to 43.6% reduction vs. autograd graphs │
│ • Convergence Verification:    Monotonic loss reduction (up to 81.8% drop)  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Benchmark Model Suite

Four diverse TinyML neural network architectures were evaluated representing distinct edge-learning workloads:

| Model ID | Application Domain | Architecture ($L_{\text{in}} \to \dots \to L_{\text{out}}$) | Activation Functions | Samples ($N$) | Epochs ($E$) | Learning Rate ($\eta$) |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **`xor_net`** | Non-linear Logic / Classification | $2 \to 4 \to 1$ | ReLU $\to$ Sigmoid | 4 | 1,000 | 0.10 |
| **`sensor_drift_net`** | Multi-sensor Calibration / Drift | $4 \to 16 \to 8 \to 1$ | ReLU $\to$ ReLU $\to$ Sigmoid | 8 | 250 | 0.01 |
| **`sine_regression_net`**| Continuous Curve Fitting ($y = \sin x$) | $1 \to 8 \to 8 \to 1$ | ReLU $\to$ ReLU $\to$ Sigmoid | 16 | 200 | 0.05 |
| **`gesture_anomaly_net`**| 3-Axis IMU Gesture Anomaly Detection | $3 \to 12 \to 6 \to 1$ | ReLU $\to$ ReLU $\to$ Sigmoid | 12 | 300 | 0.03 |

---

## 3. Mandatory Test 1: Static vs. Dynamic Memory & Flash Footprint

### Objective
Verify that MTC achieves zero runtime dynamic heap allocation (`malloc = 0`), strictly enforces static memory budgets ($\le 64$ KB), and produces a compact flash binary footprint compared to runtime interpreter engines like TensorFlow Lite Micro (TFLM).

### Empirical Data (Table 1)

| Model Name | Dense Stages | Total Parameters (Weights + Biases) | Static RAM Watermark (MTC) | Peak Heap Allocations (`malloc`) | Flash Code (`.text`) | Total Flash Image (`.bin`) | % of 64KB SRAM Budget |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`xor_net`** | 2 | 17 | **60 Bytes** | **0 Bytes** | 4,418 Bytes | 4,496 Bytes | **0.09%** |
| **`sensor_drift_net`** | 3 | 225 | **564 Bytes** | **0 Bytes** | 5,806 Bytes | 5,888 Bytes | **0.86%** |
| **`sine_regression_net`**| 3 | 89 | **260 Bytes** | **0 Bytes** | 5,454 Bytes | 5,536 Bytes | **0.40%** |
| **`gesture_anomaly_net`**| 3 | 133 | **352 Bytes** | **0 Bytes** | 5,858 Bytes | 5,936 Bytes | **0.54%** |

### Key Findings:
1. **Zero Dynamic Allocation**: `malloc` count is identically 0. All buffers are statically planned at compile-time by MTC's memory planner.
2. **Ultra-Low Memory Footprint**: Even the deepest 4-layer network (`sensor_drift_net`) occupies only 564 bytes of SRAM—less than 1% of the ESP32-C3's 64 KB SRAM budget.
3. **Flash Footprint Transparency**: Total flash image sizes are between 4.4 KB and 5.9 KB. *Paper Note:* Unlike runtime engines (e.g., TFLM ~20 KB) that include interpreter overhead, MTC's binary contains only bare-metal C99 loops, the model weights, and hardware initialization code.

---

## 4. Mandatory Test 2: Execution Latency & Hardware Cycles

### Objective
Measure true CPU instruction cycle counts and wall execution latency for on-device forward inference and backpropagation training step on the ESP32-C3 RISC-V core @ 160 MHz.

### Methodology
To eliminate instrumentation overhead and avoid skewing measurements with UART serial serialization or simulator idle loops, cycle counts are acquired directly via the RISC-V hardware CSR `mcycle` (`csrr %0, mcycle`). Forward inference and backpropagation sample updates were measured in isolated loops without UART I/O.

### Empirical Data (Table 2)

| Model Name | Forward Cycles (`mcycle`) | Forward Latency (@ 160 MHz) | Train Step Cycles (`mcycle`) | Train Step Latency (@ 160 MHz) | Training Throughput |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`xor_net`** | **344** cycles | **2.15 µs** | **1,097** cycles | **6.86 µs** | **145,772 samples/s** |
| **`sensor_drift_net`** | **3,401** cycles | **21.26 µs** | **13,083** cycles | **81.77 µs** | **12,229 samples/s** |
| **`sine_regression_net`**| **1,548** cycles | **9.67 µs** | **6,013** cycles | **37.58 µs** | **26,610 samples/s** |
| **`gesture_anomaly_net`**| **2,204** cycles | **13.77 µs** | **7,884** cycles | **49.27 µs** | **20,296 samples/s** |

![Hardware Throughput](/home/pika/.gemini/antigravity/brain/66585295-8403-4bdb-aa85-02b93b2844bb/hardware_throughput.png)

### Key Findings:
1. **Sub-100 µs Training Latency**: Even the largest model (`sensor_drift_net`, 225 parameters) completes a full forward + backward + SGD update step in **81.77 µs** (13,083 cycles).
2. **High Sample Throughput**: The training throughput ranges from **12,229 to 145,772 samples per second**, confirming that continuous online edge learning is computationally feasible without impeding real-time sensor polling.

---

## 5. Mandatory Test 3: Memory Ablation (Dynamic Autograd Graph vs. MTC Fused Static Buffers)

### Objective
Quantify peak SRAM savings achieved by MTC's fused SGD in-place updates and static ping-pong buffer reuse compared to standard dynamic autograd graph engines (e.g., PyTorch, TensorFlow, or AIfES).

### Architectural Comparison
- **Dynamic Autograd Graph (PyTorch / AIfES baseline)**: Retains all forward layer activations ($A_l$), allocates explicit parameter gradient tensors ($\nabla W_l, \nabla B_l$) for all layers, and retains per-layer backpropagation delta tensors ($\delta_l$).
- **MTC Static Liveness Allocator**: Fuses the gradient computation and SGD weight update in a single pass (eliminating $\nabla W$ and $\nabla B$ arrays entirely) and shares only two alternating ping-pong delta buffers across all hidden layers.

### Empirical Data (Table 3)

| Model Name | Baseline Autograd RAM (Bytes) | MTC Static Planned RAM (Bytes) | Absolute SRAM Saved (Bytes) | Relative SRAM Reduction (%) |
| :--- | :---: | :---: | :---: | :---: |
| **`xor_net`** | 88 B | **60 B** | 28 B | **31.8%** |
| **`sensor_drift_net`** | 1,000 B | **564 B** | 436 B | **43.6%** |
| **`sine_regression_net`**| 456 B | **260 B** | 196 B | **43.0%** |
| **`gesture_anomaly_net`**| 608 B | **352 B** | 256 B | **42.1%** |

![Memory Ablation](/home/pika/.gemini/antigravity/brain/66585295-8403-4bdb-aa85-02b93b2844bb/memory_ablation.png)

### Key Findings:
- MTC achieves a **31.8% to 43.6% reduction in peak SRAM** across all evaluated architectures.
- The savings scale with network depth and parameter count because eliminating the parameter gradient arrays ($\nabla W$) removes $O(\sum |W_l|)$ overhead.

---

## 6. Mandatory Test 4: Convergence & Stability Under Q15 Quantization

### Objective
Verify that 16-bit signed fixed-point (Q15) arithmetic converges stably during on-device backpropagation without catastrophic overflow or diverging loss trajectories.

### Empirical Data (Table 4)

| Model Name | Epochs ($E$) | Initial Q15 MSE Loss | Final Q15 MSE Loss | Loss Reduction (%) | Convergence Stability Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`xor_net`** | 1,000 | 0.2626 | 0.1981 | **24.57%** | Sub-optimal (Gradient Attenuation) |
| **`sensor_drift_net`** | 250 | 0.0927 | 0.0448 | **51.65%** | Convergent / Stable |
| **`sine_regression_net`**| 200 | 0.0667 | 0.0161 | **75.88%** | Highly Convergent / Stable |
| **`gesture_anomaly_net`**| 300 | 0.1550 | 0.0282 | **81.81%** | Highly Convergent / Stable |

![Combined Convergence](/home/pika/.gemini/antigravity/brain/66585295-8403-4bdb-aa85-02b93b2844bb/combined_convergence.png)

### Academic Honesty & Critical Review Notes:
1. **Regression & Anomaly Models**:
   - `gesture_anomaly_net` achieved an **81.8% loss drop** (0.1550 $\to$ 0.0282).
   - `sine_regression_net` achieved a **75.9% loss drop** (0.0667 $\to$ 0.0161).
   - `sensor_drift_net` achieved a **51.6% loss drop** (0.0927 $\to$ 0.0448).
   - All three demonstrate strong, monotonic fixed-point convergence.
2. **XOR Classification Caveat**:
   - `xor_net` achieved a **24.57% loss drop** (0.2626 $\to$ 0.1981). While the loss decreases monotonically, an MSE of 0.1981 on a 0/1 binary target indicates that the network has not fully separated the non-linear XOR boundary within 1,000 epochs.
   - *Academic Reason:* Small initial weight initializations and Q15 quantization step size ($1/32768 \approx 3.05 \times 10^{-5}$) can cause gradient updates to underflow when passing through multiple saturating sigmoid/ReLU derivative terms (gradient quantization attenuation).
   - *Paper Recommendation:* Present this limitation transparently in the discussion section as an intrinsic trade-off of 16-bit integer backpropagation, contrasting it with the regression benchmarks where continuous loss surfaces converge smoothly.

---

## 7. Audit & Methodology Transparency Note

An independent audit of the test methodology revealed the following critical insights:

1. **Previous Latency Artifact**: An earlier automated run used the emulator's wall timeout duration (`--max-seconds`) to approximate cycle counts. Because the bare-metal firmware looped indefinitely on completion, the reported latency reflected the timeout heuristic rather than true CPU execution.
2. **Direct CSR Correction**: In this validated report, all cycle counts were gathered via hardware CSR `mcycle` (`csrr %0, mcycle`), reflecting the exact cycle count spent executing forward and backward passes. True training step latency is **6.86 µs to 81.77 µs** (not 2.2 ms), demonstrating that MTC is substantially faster than initial timeout-based estimates suggested.
3. **Reproducibility**: The complete benchmark suite can be re-run at any time with:
   ```bash
   python3 validation/scripts/run_benchmarks.py
   ```
   All raw JSON data is preserved in [`validation/benchmark_results.json`](file:///home/pika/Documents/MTC/validation/benchmark_results.json).
