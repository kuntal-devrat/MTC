# Micro-Train-Compiler (MTC): Final Presentation Script & Defense Battlecard
**Academic Session:** 2026–2027  
**Department:** Computer Science & Engineering, G. L. Bajaj Group of Institutions, Mathura  
**Team Members:**  
1. **Ajay Sisodia** (`2405111530012`) — Compiler Frontend, AST Validation & Toolchain  
2. **Harsh Chaudhary** (`2405111530067`) — Fixed-Point Mathematics, Q15 Arithmetic & Interval Safety (IBP)  
3. **Devrat Kuntal** (`2405111530055`) — Static Memory Planning, Theorem 1, Fused SGD & C99 Codegen  
4. **Aryan Upadhyay** (`2405111530037`) — Bare-Metal RISC-V Firmware, Emulation, Benchmarks & CSR Profiling  

---

## Part 1: Presentation Delivery Rules & Stage Dynamics

1. **Total Target Duration:** 12 to 14 minutes (leaving 5–6 minutes for Q&A).
2. **Individual Timing:** Exactly 3 to 3.5 minutes per speaker.
3. **Stage Positioning:**
   - All 4 stand in a confident line facing the panel.
   - The active speaker steps forward 1 pace to speak.
   - When handing off, verbally cue the next speaker by name; the incoming speaker steps forward and makes eye contact with the jury.
4. **Tone:** Crisp, assertive systems-engineering tone. Zero casual hesitations ("um", "like", "basically"). Ground every claim in exact empirical data (e.g., *"60 to 564 Bytes"*, *"12,229 samples per second"*, *"CSR mcycle"*).

---

## Part 2: Slide-by-Slide Word-for-Word Presentation Script

```
   [Ajay Sisodia]          [Harsh Chaudhary]          [Devrat Kuntal]          [Aryan Upadhyay]
  Slides 1 - 3 (Intro)    Slides 4 - 6 (Q15 Math)    Slides 7 - 9 (Liveness)  Slides 10 - 13 (Results)
      (00:00 - 03:15)         (03:15 - 06:30)            (06:30 - 09:45)          (09:45 - 13:00)
```

---

### SPEAKER 1: Ajay Sisodia (Roll No: 2405111530012)
*Focus: Problem Formulation, The Embedded Memory Wall, Project Goals & AST Frontend*  
*Duration: 00:00 – 03:15*

#### Slide 1: Title Slide & Team Introduction
> **Visual:** Title, College Crest, Student Names, Supervisor: Ms. Gaurisha Sisodia.  
> **Script:**  
> "Respected judges, external evaluators, and faculty members—a very good morning.  
>
> Today, our team is presenting our B.Tech Minor Project titled: **Micro-Train-Compiler (MTC): An Ahead-of-Time Static Compiler for Zero-Heap On-Device Neural Network Training on 32-Bit Microcontrollers**.  
>
> I am Ajay Sisodia, and presenting with me today are my colleagues Harsh Chaudhary, Devrat Kuntal, and Aryan Upadhyay, under the mentorship of Ms. Gaurisha Sisodia.  
>
> To begin, let us examine the fundamental dilemma that dominates edge artificial intelligence today."

#### Slide 2: The Problem — Frozen Edge Models & The Embedded SRAM Wall
> **Visual:** Cloud Training $\to$ Quantize $\to$ Frozen Flash $\to$ Real-World Failures (Sensor Drift, User Variance).  
> **Script:**  
> "Over the last five years, TinyML frameworks such as TensorFlow Lite for Microcontrollers and CMSIS-NN have successfully enabled neural network **inference** on low-power silicon. However, the entire industry operates under a rigid **'train-in-the-cloud, deploy-frozen'** paradigm. Once an edge firmware binary is flashed into ROM, its weights are permanently static.  
>
> In real physical deployments, this leads to three catastrophic failure modes:  
> 1. **Sensor Transducer Aging:** Low-cost metal-oxide gas sensors and optical photodiodes degrade and drift over operational cycles. A frozen model gradually loses accuracy and misclassifies valid signals.  
> 2. **Inter-User Variance:** Wearable biomedical devices encounter distinct biomechanical kinematics across different human bodies; a population-average model fails on individual patients.  
> 3. **Ambient Resonant Shifts:** Industrial vibration monitors experience different baseline frequencies depending on mounting stiffness.  
>
> The obvious solution is **on-device training**. But why can't we simply run PyTorch or backpropagation directly on an ESP32 or an ARM Cortex-M?  
>
> Because of **the embedded memory wall**:  
> Traditional autograd engines dynamically build computational Directed Acyclic Graphs (DAGs) on the heap at runtime using interleaved `malloc()` and `free()`. Low-cost 32-bit microcontrollers possess only 16 to 64 Kilobytes of SRAM and **lack a Memory Management Unit (MMU)**. Dynamic memory allocations quickly fragment flat physical memory, resulting in fatal heap exhaustion, pointer corruption, and hardware watchdog resets. Furthermore, conventional backpropagation retains dedicated gradient arrays ($\nabla W, \nabla b$) across all layers, instantly doubling parameter RAM."

#### Slide 3: Project Vision & Stage 1 Compiler Frontend
> **Visual:** 4-Stage Compiler Pipeline (`system_architecture.png`) highlighting Stage 1.  
> **Script:**  
> "Our solution is **Micro-Train-Compiler (MTC)**. MTC abandons runtime graph interpretation entirely. Instead, it operates as an **ahead-of-time (AOT) static compiler** that parses a high-level network topology and emits freestanding C99 code where all tensor addresses, scratchpads, and execution schedules are statically fixed at compile time—guaranteeing **`malloc = 0`**.  
>
> As shown on the slide, the compilation pipeline executes in four stages:  
> In **Stage 1**, our frontend parser ingests a declarative JSON model specification containing layer widths, activations, and embedded training samples. It normalizes these into our compiler Intermediate Representation (IR) and enforces strict topological contracts: verifying input-output dimension matching between consecutive layers, validating hyperparameter bounds ($\eta > 0, \text{epochs} \ge 1$), and preparing static memory tables.  
>
> Once the AST is validated, the computational challenge shifts to mathematical execution: how can an integer microcontroller execute gradients without floating-point hardware?  
>
> To explain our fixed-point mathematics and interval safety analysis, I invite my teammate **Harsh Chaudhary**."

*(Handoff: Ajay steps back; Harsh steps forward.)*

---

### SPEAKER 2: Harsh Chaudhary (Roll No: 2405111530067)
*Focus: Q15 Fixed-Point Arithmetic, Rounding Bias, Interval Bound Propagation (IBP)*  
*Duration: 03:15 – 06:30*

#### Slide 4: Why Integer Math? The 16-Bit Q15 Format
> **Visual:** Mathematical definition of Q15, multiplication with rounding bias $2^{14}$, saturation logic.  
> **Script:**  
> "Thank you, Ajay.  
>
> Low-cost microcontrollers, including our target ESP32-C3 RISC-V core, do not feature a hardware Floating-Point Unit (FPU). Emulating 32-bit floating-point math in software requires 20 to 80 clock cycles per operation, destroying real-time control throughput.  
>
> To achieve microsecond-scale execution, MTC formulates the entire forward pass, loss calculation, and backpropagation pass in **signed 16-bit fixed-point arithmetic**, specifically the **Q15 format**.  
>
> A real decimal $x \in [-1.0, 1.0 - 2^{-15}]$ is stored as a 16-bit signed two's-complement integer:  
> $$X_{\text{raw}} = \lfloor x \cdot 32768 \rfloor \in [-32768, 32767]$$  
>
> When multiplying two Q15 numbers, their product produces a 32-bit intermediate integer. To scale it back to 16 bits without introducing systematic truncation drift, MTC applies an analytical rounding bias of $2^{14} = 16384$, shifts right by 15 bits, and applies saturating clamping:  
> $$Z_{\text{raw}} = \text{SAT16}\left( \frac{(X_{\text{raw}} \cdot Y_{\text{raw}}) + 16384}{32768} \right)$$  
>
> This enables single-cycle integer hardware multipliers to execute neural network training."

#### Slide 5: Stage 2 — Conservative Interval Bound Propagation (IBP)
> **Visual:** Mathematical equations of Interval Bound Propagation across Dense and Activation nodes.  
> **Script:**  
> "However, integer arithmetic introduces a dangerous vulnerability: **arithmetic overflow**. Accumulating matrix-vector dot products across multi-layer networks can easily exceed the 16-bit boundary of $[-32768, 32767]$, turning positive numbers into negative numbers and destabilizing gradient descent.  
>
> To guarantee mathematical safety **before a single line of C code is synthesized**, MTC implements **Conservative Interval Bound Propagation (IBP)** in Stage 2.  
>
> Given an input interval $[x_{\min}, x_{\max}]$, the compiler tracks interval bounds across every layer. For a dense linear transformation:  
> $$y_i^{\min} = \sum_j \min(w_{ij} x_j^{\min}, w_{ij} x_j^{\max}) + b_i$$  
> $$y_i^{\max} = \sum_j \max(w_{ij} x_j^{\min}, w_{ij} x_j^{\max}) + b_i$$  
>
> For non-linear activations, ReLU bounds the interval to $[0, \max(0, y^{\max})]$, while Sigmoid bounds it strictly to $[0, 32767]$.  
>
> If any intermediate layer interval exceeds $[-32768, 32767]$, the compiler halts and flags an arithmetic saturation hazard, alerting the developer to rescale initial weights or learning rates. This provides formal compile-time numerical safety."

#### Slide 6: Activation Derivatives in Q15 Fixed Point
> **Visual:** ReLU derivative and Sigmoid derivative integer formulations.  
> **Script:**  
> "During the backward pass, MTC evaluates activation derivatives directly on the integer lattice:  
> - For **ReLU**, the derivative is a simple conditional branch: 1 (stored as $32767$) if pre-activation $z > 0$, and $0$ otherwise.  
> - For **Sigmoid**, the derivative $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ is evaluated using the retained forward activation $a$:  
> $$\sigma'(z)_{\text{Q15}} = \frac{a \cdot (32767 - a) + 16384}{32768}$$  
>
> With numerical safety certified and arithmetic routines verified, the primary compiler challenge is memory planning: how do we eliminate dynamic heap allocation and minimize static SRAM?  
>
> To present our static liveness planner, Theorem 1, and in-place SGD fusion, I hand over to **Devrat Kuntal**."

*(Handoff: Harsh steps back; Devrat steps forward.)*

---

### SPEAKER 3: Devrat Kuntal (Roll No: 2405111530055)
*Focus: Static Liveness Analysis, Theorem 1 Proof, Ping-Pong Buffers, Fused SGD Updates*  
*Duration: 06:30 – 09:45*

#### Slide 7: The Memory Dilemma — Autograd Graph vs. Static Liveness
> **Visual:** Memory comparison diagram (`ping_pong_fused_liveness.png`).  
> **Script:**  
> "Thank you, Harsh.  
>
> Please look closely at the diagram on this slide. On the left, you see how conventional autodiff libraries like PyTorch or AIfES manage memory: they maintain a dynamic tape on the heap, store every forward activation, retain dedicated gradient buffers ($\nabla W, \nabla b$) for all layers, and allocate separate error vectors ($\delta^{(l)}$). For an embedded device, this is unsustainable.  
>
> On the right, you see MTC's memory model: **pure compile-time static placement**.  
>
> MTC achieves this through two fundamental algorithmic innovations:  
> 1. Fused In-Place SGD Parameter Updates.  
> 2. Alternating Ping-Pong Delta Buffering based on a formal liveness theorem."

#### Slide 8: Fused In-Place Parameter Updates ($\text{Mem}(\nabla W) = 0$)
> **Visual:** Mathematical derivation and C99 code snippet of the fused update inner loop.  
> **Script:**  
> "First, why do standard libraries maintain explicit weight gradient buffers ($\nabla W$)? Because they separate the backward pass from the optimizer step: first compute $\nabla W = \delta \cdot a^T$, then update $W \leftarrow W - \eta \nabla W$. Storing $\nabla W$ requires duplicating every weight matrix in SRAM.  
>
> In MTC, we prove that this decoupling is redundant. Substituting the gradient definition into the SGD rule gives:  
> $$W_{ij}^{(l)} \leftarrow W_{ij}^{(l)} - \eta \left( \delta_i^{(l)} a_j^{(l-1)} \right)$$  
>
> Because each parameter element $W_{ij}$ depends only on the current layer's error $\delta_i$ and the predecessor activation $a_j$, MTC executes the update **in place** immediately inside the backward loop!  
>
> This completely eliminates parameter gradient arrays from SRAM:  
> $$\text{Memory}(\nabla W) = 0 \quad \text{and} \quad \text{Memory}(\nabla b) = 0$$  
>
> Crucially, our compiler enforces the strict mathematical sequence: error is propagated backward to layer $l-1$ using the **unmodified original weights** *before* the in-place SGD step overwrites $W^{(l)}$. Thus, mathematical gradient equivalence is 100% preserved while eliminating hundreds of bytes of RAM."

#### Slide 9: Theorem 1 — Disjoint Liveness & Ping-Pong Buffers
> **Visual:** Formal statement of Theorem 1, pointer swap animation ($\text{Buf}_A \leftrightarrow \text{Buf}_B$).  
> **Script:**  
> "Second, how do we store activation error gradients ($\delta^{(l)}$)? A naive static allocator provisions a separate array for every layer: $\sum_{l=1}^L d_l$.  
>
> In MTC, we formulate **Theorem 1: Disjoint Liveness of Error Gradients**.  
>
> **Theorem 1 states:** In a sequential feedforward network, during backward error propagation from layer $l+1$ to layer $l$, the computation of $\delta^{(l)}$ depends strictly on $\delta^{(l+1)}$ and $W^{(l+1)}$. Once $\delta^{(l)}$ is evaluated and parameters of layer $l+1$ are updated, $\delta^{(l+1)}$ is never referenced again in the execution lifecycle.  
>
> **The proof is direct:** From the chain rule recurrence relation, $\delta^{(l)}$ depends exclusively on $(W^{(l+1)})^T \delta^{(l+1)}$. All downstream derivatives $\delta^{(k)}$ for $k > l+1$ have already completed their parameter updates. Because no backward skip-connections exist, the active working set of error gradients at any instant during the backward pass is **at most two**: the current gradient $\delta^{(l)}$ and the incoming gradient $\delta^{(l+1)}$.  
>
> Exploiting Theorem 1, MTC statically provisions **exactly two alternating scratchpad buffers**: $\text{Buf}_A$ and $\text{Buf}_B$, sized to the maximum layer width.  
> Layer $L$ writes to $\text{Buf}_A$; Layer $L-1$ reads from $\text{Buf}_A$ and writes to $\text{Buf}_B$; Layer $L-2$ reads from $\text{Buf}_B$ and writes to $\text{Buf}_A$, alternating via simple pointer swaps.  
>
> Backward scratchpad memory is bounded by $2 \times \max(d_l)$, completely decoupling memory consumption from network depth!  
>
> Finally, Stage 4 synthesizes standalone C99 files—`mtc_model.h`, `mtc_model.c`, and `mtc_train.c`—requiring only `<stdint.h>` and completely omitting `<stdlib.h>`.  
>
> To present our bare-metal RISC-V hardware cross-compilation, cycle-accurate benchmarks, and empirical results, I invite **Aryan Upadhyay**."

*(Handoff: Devrat steps back; Aryan steps forward.)*

---

### SPEAKER 4: Aryan Upadhyay (Roll No: 2405111530037)
*Focus: Bare-Metal RISC-V Harness, CSR `mcycle` Profiling, 4 Mandatory Tests, Limitations & Conclusion*  
*Duration: 09:45 – 13:00*

#### Slide 10: Bare-Metal Firmware Harness & Hardware Instrumentation
> **Visual:** Linker layout (`iram`, `dram`), `entry.S`, UART0 @ `0x60000000`, CSR `mcycle` assembly.  
> **Script:**  
> "Thank you, Devrat.  
>
> To evaluate MTC on real embedded silicon, we targeted the **Espressif ESP32-C3**, featuring a 32-bit RISC-V core implementing the `RV32IMC` instruction set architecture: base integer, hardware multiply/divide, and 16-bit compressed instructions.  
>
> Rather than relying on heavy operating system layers like ESP-IDF or FreeRTOS, we engineered a freestanding bare-metal firmware harness:  
> 1. A custom linker script (`linker.ld`) mapping code to Instruction RAM (`0x40380000`) and data to Data RAM (`0x3FC8B800`).  
> 2. Startup assembly (`entry.S`) that initializes the RISC-V stack pointer register (`sp`) to `0x3FCC0000` and enters `esp_main`.  
> 3. A memory-mapped hardware UART driver addressing the UART0 FIFO register directly at physical address `0x60000000`, eliminating `libc`'s bloated `printf` library.  
>
> Furthermore, to eliminate serial logging distortion, we did not use wall-clock software timers. Instead, we instrumented execution using inline assembly reading the **RISC-V hardware Control and Status Register `mcycle`** (`csrr %0, mcycle`). Querying active CPU cycles immediately before and after execution guarantees laboratory-grade, cycle-accurate measurement."

#### Slide 11: Empirical Results — The 4 Mandatory Tests
> **Visual:** Combined Results Slide: Tables 5.2, 5.3, 5.4, 5.5 and Figures 5.1, 5.2, 5.3.  
> **Script:**  
> "We evaluated MTC across four representative TinyML edge workloads: `xor_net`, `sensor_drift_net`, `sine_regression_net`, and `gesture_anomaly_net`. All binaries were cross-compiled with `riscv64-unknown-elf-gcc` at `-O2` and evaluated on the cycle-accurate `esp32sim` hardware emulator.  
>
> Here are our empirical findings across the four mandatory evaluations:  
>
> 1. **Test 1: Static Memory & Zero Heap:**  
> Inspecting the binary section headers via `riscv64-unknown-elf-size` confirmed strictly **0 Bytes of dynamic heap allocation (`malloc = 0`)**. Peak SRAM consumption across the entire network ranges from **60 Bytes to 564 Bytes**—less than 0.86% of the 64~KB user SRAM budget. The entire bootable flash binary occupies under 6.0~KB!  
>
> 2. **Test 2: Execution Latency & Throughput:**  
> Forward inference completes in 2.15~$\mu$s to 21.26~$\mu$s. A complete forward-backward training step executes in **6.86~$\mu$s to 81.77~$\mu$s**, delivering training throughput between **12,229 and 145,772 samples per second** at 160~MHz. This proves that weight adaptation can run seamlessly inside real-time sensor sampling loops.  
>
> 3. **Test 3: Memory Ablation:**  
> Comparing MTC against the dynamic autograd baseline demonstrates an average **40.1% peak SRAM reduction** across all topologies. In the 4-layer `sensor_drift_net`, MTC saves 436 Bytes of SRAM by fusing parameter updates and sharing ping-pong delta buffers.  
>
> 4. **Test 4: Numerical Convergence:**  
> Continuous regression and anomaly detection workloads converged with high fidelity under pure Q15 arithmetic: `gesture_anomaly_net` achieved an **81.81% loss reduction**, `sine_regression_net` achieved **75.88%**, and `sensor_drift_net` achieved **51.65%** with monotonic loss decay."

#### Slide 12: Transparent Academic Critique — Quantization Attenuation on XOR
> **Visual:** Mathematical equation for Sigmoid derivative, truncation to zero at $\epsilon_{\text{Q15}} = 2^{-15}$, and engineering mitigations.  
> **Script:**  
> "Academic integrity requires reporting limitations transparently. While regression workloads converged cleanly, the non-linear classification benchmark (`xor_net`) achieved a modest **24.57% loss reduction** over 1,000 epochs.  
>
> Why did this occur?  
> In Q15 fixed-point arithmetic, the machine epsilon is $\epsilon = 2^{-15} \approx 3.05 \times 10^{-5}$.  
> In backpropagation through Sigmoid activations, as outputs approach saturated states, the derivative $\sigma'(z)$ falls below $0.10$. When multiplied by learning rate $\eta = 0.10$ and fixed-point activations, the mathematical gradient evaluates to a value smaller than $2^{-15}$.  
> In integer arithmetic with truncation, this fractional update rounds down to **zero** (`int32 >> 15 == 0`), causing parameter updates to stall. We define this phenomenon as **gradient quantization attenuation**.  
>
> For Phase 2, we have formulated two concrete engineering mitigations:  
> 1. **Accumulator Register Widening:** Maintaining weights in 32-bit fixed-point (Q16.16) while keeping activations in Q15, allowing sub-epsilon gradients to accumulate across steps.  
> 2. **Stochastic Rounding:** Adding pseudo-random dither prior to right-shifting to preserve unbiased gradient expectations."

#### Slide 13: Conclusion, Future Roadmap & Q&A Handoff
> **Visual:** Summary bullet points, Phase 2 roadmap, "Thank You" with team details.  
> **Script:**  
> "To conclude:  
> Micro-Train-Compiler successfully proves that on-device deep learning backpropagation can transition from fragile runtime interpreters to an **ahead-of-time static compiler**. By eliminating dynamic memory allocation (`malloc = 0`), fusing in-place parameter updates, and proving disjoint delta liveness, MTC achieves sub-1~KB SRAM footprints and sub-100~$\mu$s training steps on bare-metal RISC-V hardware.  
>
> For Phase 2, we will expand MTC's Intermediate Representation to support 1D Convolutional layers and investigate RISC-V Packed SIMD vectorization.  
>
> Our complete source code, bare-metal harnesses, and research paper are openly documented in our repository.  
>
> Thank you for your time. The team is now ready to take your questions."

---

## Part 3: Exhaustive Q&A Defense Battlecard

### Q&A Responder Strategy
- **Compiler Frontend / IR / AST / Toolchain:** Ajay Sisodia
- **Fixed-Point Mathematics / Q15 / IBP Bounds:** Harsh Chaudhary
- **Static Liveness / Theorem 1 / Fused SGD:** Devrat Kuntal
- **Bare-Metal RISC-V / Linker / Emulation / Benchmarks:** Aryan Upadhyay
- **XOR Quantization Attenuation:** Aryan Upadhyay & Harsh Chaudhary

---

### Category A: Compiler & Systems Architecture

#### Q1: "Why did you build an ahead-of-time (AOT) compiler instead of just using an interpreter like TensorFlow Lite Micro or PyTorch Mobile?"
- **Responder:** Ajay Sisodia
- **Defense Answer:**  
  "TFLM and PyTorch Mobile were architected with an interpreter dispatch model. While TFLM uses a pre-allocated `tensor_arena` for *inference*, its runtime interpreter has zero support for backward gradient DAGs or parameter updates. In contrast, PyTorch Mobile and autograd libraries dynamically allocate graph nodes on the heap during execution.  
  On resource-constrained microcontrollers with flat physical memory and no MMU, dynamic allocation causes severe heap fragmentation that crashes the firmware over long training runs.  
  By adopting an Ahead-of-Time (AOT) compilation strategy, MTC resolves all tensor addresses, array lifecycles, and update schedules at compile time on the host PC. The microcontroller receives pure freestanding C99 machine code with zero interpreter dispatch overhead, zero dynamic memory allocations (`malloc = 0`), and predictable execution timing."

#### Q2: "Why not use TVM or MicroTVM? Doesn't Apache TVM already compile models to C?"
- **Responder:** Ajay Sisodia (Supported by Devrat)
- **Defense Answer:**  
  "Apache TVM and MicroTVM are exceptional optimizing compilers, but they are strictly **inference-only**. Their intermediate representations (Relay and TIR) and memory planners optimize feedforward computational graphs. They do not generate reverse-mode automatic differentiation kernels, do not compute backward activation derivatives, and cannot execute parameter updates on-device.  
  MTC specifically solves the *on-device training* problem by compiling reverse-mode autodiff, liveness-planned delta scratchpads, and fused parameter updates into a static C99 engine."

#### Q3: "How do you guarantee that `malloc` is never called? What if a linked standard library function calls it secretly?"
- **Responder:** Aryan Upadhyay
- **Defense Answer:**  
  "We enforce zero dynamic allocation at both the source level and the linker level:  
  1. **Source Level:** Our generated C99 code (`mtc_model.c`, `mtc_model.h`, `mtc_train.c`) includes only `<stdint.h>`. It explicitly omits `<stdlib.h>`, meaning `malloc()` and `free()` are not even declared.  
  2. **Linker Level:** We cross-compile with the flag `-nostdlib` using `riscv64-unknown-elf-gcc`. We supply our own assembly entry point (`entry.S`) and our own linker script (`linker.ld`). If any function attempted to call `malloc()`, the GNU linker would halt compilation immediately with an `undefined reference to malloc` error.  
  3. **Binary Inspection:** We verified all ELF binaries using `riscv64-unknown-elf-nm -u` and `size -A -d`. There are zero undefined heap symbols; all memory resides in static `.bss` and `.data` sections."

---

### Category B: Mathematics & Algorithmic Invariants

#### Q4: "In standard backpropagation, you compute all gradients first, and then update weights in an optimizer step. In MTC, you update weights in-place immediately. Won't that corrupt the weight values needed to backpropagate errors to earlier layers?"
- **Responder:** Devrat Kuntal *(Crucial question — must answer with absolute precision!)*
- **Defense Answer:**  
  "That is a fundamental question, and the exact ordering of operations in MTC is designed to prevent that corruption.  
  In backpropagation, layer $l$ requires weight matrix $W^{(l)}$ for two distinct purposes:  
  1. To backpropagate the error delta to predecessor layer $l-1$: $\delta^{(l-1)} = ((W^{(l)})^T \delta^{(l)}) \odot \sigma'(z^{(l-1)})$.  
  2. To update its own weights: $W^{(l)} \leftarrow W^{(l)} - \eta (\delta^{(l)} (a^{(l-1)})^T)$.  
  
  MTC enforces a strict **analytical execution sequence**:  
  - **Step 1:** Layer $l$ receives $\delta^{(l)}$ from the incoming buffer.  
  - **Step 2:** Layer $l$ computes $\delta^{(l-1)}$ and writes it into the alternate ping-pong buffer using the **unmodified original weights** $W^{(l)}$.  
  - **Step 3:** Only *after* $\delta^{(l-1)}$ is safely written does MTC execute the in-place SGD update loop on $W^{(l)}$.  
  
  Because Step 2 strictly precedes Step 3, the earlier layers receive mathematically exact gradients identical to standard autograd engines, while allowing $W^{(l)}$ to be updated in place without dedicating a separate gradient buffer $\nabla W$."

#### Q5: "Can you explain Theorem 1 in simple terms? Why are two buffers enough for any number of layers?"
- **Responder:** Devrat Kuntal
- **Defense Answer:**  
  "Consider how information flows backward. Layer 4 computes error $\delta^{(4)}$. To compute $\delta^{(3)}$, you only need $\delta^{(4)}$ and $W^{(4)}$.  
  Once $\delta^{(3)}$ is computed and $W^{(4)}$ is updated, you will never need $\delta^{(4)}$ again in the entire training loop—its lifecycle is over.  
  Next, to compute $\delta^{(2)}$, you only need $\delta^{(3)}$. So you can overwrite the memory where $\delta^{(4)}$ was stored with $\delta^{(2)}$.  
  This is identical to a two-player ping-pong game: one buffer holds the *current* error, the other buffer receives the *previous* error. They alternate pointers at each layer.  
  Regardless of whether the neural network has 3 layers, 10 layers, or 50 layers, at any single microsecond during backpropagation, only two error vectors need to be alive simultaneously. Hence, backward scratchpad memory is $2 \times \max(d_l)$ instead of $\sum d_l$."

---

### Category C: Fixed-Point Arithmetic & Numerical Safety

#### Q6: "Why did you choose Q15 instead of standard 8-bit integers (INT8) used in TensorFlow Lite?"
- **Responder:** Harsh Chaudhary
- **Defense Answer:**  
  "INT8 is well-suited for **inference** because forward activations have pre-determined dynamic ranges that can be scaled using pre-computed calibration tables.  
  However, for **training and backpropagation**, INT8 has only 256 quantization bins. Gradient updates involve multiplying small error deltas ($\delta$) by learning rates ($\eta = 0.01$) and activations. In an 8-bit space, almost every gradient update truncates to zero.  
  Q15 provides 65,536 quantization bins with a resolution of $3.05 \times 10^{-5}$, which provides sufficient numerical precision for accumulating weight updates while fitting neatly into single-cycle 16-bit register operations on a 32-bit integer CPU."

#### Q7: "What is Interval Bound Propagation (IBP), and why is it needed if you already have saturating arithmetic?"
- **Responder:** Harsh Chaudhary
- **Defense Answer:**  
  "Saturating arithmetic is a **reactive** runtime safety net—if an overflow occurs, it clamps the number to $+32767$ or $-32768$. However, if a neural network frequently saturates its activations during forward propagation, its gradients become zero, killing model learning.  
  Interval Bound Propagation is a **proactive** compile-time formal verification technique. In Stage 2, MTC takes the input domain $[-1.0, +1.0]$ and mathematically propagates worst-case intervals through all linear weight transforms and activation functions.  
  If the theoretical interval exceeds 16-bit bounds, MTC alerts the engineer *before code generation*, allowing them to normalize inputs or adjust initial weight distributions rather than discovering silent saturation during physical deployment."

#### Q8: "Why did `xor_net` only achieve a 24.57% loss reduction while the other three models achieved 50% to 81%? Did your compiler fail on XOR?"
- **Responder:** Aryan Upadhyay (Supported by Harsh)
- **Defense Answer:**  
  "That is an insightful observation and one we highlighted transparently in Chapter 5 and Section VII of our research paper.  
  The compiler did not fail—it executed the exact mathematical operations specified. The phenomenon observed is **gradient quantization attenuation**.  
  In `xor_net`, the model has a non-linear Sigmoid output layer solving a sharp binary classification boundary. When the output units move toward saturation (e.g., $\sigma \approx 0.85$), the Sigmoid derivative $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ drops to approximately $0.12$.  
  When multiplied by a learning rate $\eta = 0.10$ and fixed-point activations, the mathematical update evaluates to approximately $0.00002$. In Q15 fixed-point, the smallest representable number is $\epsilon = 1/32768 \approx 0.0000305$.  
  Because the update is smaller than machine epsilon, integer right-shifting truncates the fractional update to **zero**. The weights experience zero update, stalling loss convergence.  
  In contrast, continuous regression workloads (`sine_regression_net`, `sensor_drift_net`, and `gesture_anomaly_net`) operate on smooth loss surfaces without saturated Sigmoid boundaries, converging rapidly (up to 81.81% loss reduction).  
  In our report, we have formulated the exact engineering mitigations for this: widening weight accumulators to 32-bit (Q16.16) and applying stochastic rounding."

---

### Category D: Hardware, Emulation & Bare-Metal Setup

#### Q9: "Why did you use `esp32sim` instead of running code on a physical ESP32 board or on Wokwi?"
- **Responder:** Aryan Upadhyay
- **Defense Answer:**  
  "We utilized both physical ESP32 targets and cycle-accurate emulation, but `esp32sim` was chosen for academic benchmarking for three specific scientific reasons:  
  1. **Cycle-Accurate Instruction Timing:** `esp32sim` is a Rust-based hardware emulator that boots the actual ESP32-C3 mask ROM ELFs, emulates the exact RV32IMC pipeline, and models memory-mapped peripheral timing cycle-for-cycle.  
  2. **Zero Physical Probe Jitter:** Physical breadboard wires and USB-UART bridges introduce baud-rate latency, interrupt jitter, and power rail fluctuations. `esp32sim` provides deterministic, laboratory-reproducible cycle counts.  
  3. **Direct CSR `mcycle` Verification:** We verified our assembly routine `csrr %0, mcycle` inside the emulator, ensuring that cycle metrics reflect pure CPU execution with UART transmission completely disabled during timing blocks.  
  Furthermore, our generated code is completely standard C99; as demonstrated in our repository, the exact same C code compiles directly in the Arduino IDE and Wokwi for physical ESP32 deployment."

#### Q10: "Explain the bare-metal harness. What do `entry.S` and `linker.ld` do?"
- **Responder:** Aryan Upadhyay
- **Defense Answer:**  
  "In standard embedded development, runtime frameworks like ESP-IDF initialize a massive C runtime library (`libc`), FreeRTOS threads, memory heap allocators, and interrupt dispatchers. That runtime alone consumes over 25~KB of SRAM before application code even starts.  
  To measure pure neural network performance:  
  - Our **Linker Script (`linker.ld`)** directly partitions the ESP32-C3's physical memory map: placing `.text` instructions in Instruction RAM at origin `0x40380000`, and `.data` and `.bss` variables in Data RAM at origin `0x3FC8B800`.  
  - Our **Startup Assembly (`entry.S`)** provides the `_start` symbol. It sets the hardware stack pointer (`sp`) to `0x3FCC0000` (the top of available SRAM) and immediately branches to our C entry point `esp_main`.  
  - Our **UART Driver** writes raw characters directly into the hardware FIFO register at memory-mapped address `0x60000000`.  
  This freestanding harness guarantees that every measured clock cycle and every byte of SRAM belongs 100% to MTC."

---

### Category E: Practical Value & "Hostile / Skeptical Judge" Inquiries

#### Q11: "A 4-layer perceptron with 225 parameters is tiny. Can MTC train a large model like a ResNet or a CNN on an ESP32?"
- **Responder:** Ajay Sisodia (Supported by Devrat)
- **Defense Answer:**  
  "MTC is designed specifically for **TinyML edge adaptation**, not training massive vision backbones from scratch.  
  A ResNet-50 has 25 million parameters and requires over 100 Megabytes of memory—it cannot fit on *any* sub-dollar microcontroller regardless of compiler optimizations.  
  However, in real-world edge IoT systems, 90% of field failures are caused by local drift: temperature sensors drifting, vibration baselines shifting, or biometric sensors adapting to a user's gait. These signals are low-dimensional temporal streams (typically 1 to 16 sensor channels).  
  A 4-layer network with 200 parameters is mathematically capable of continuous recalibration, anomaly detection, and curve fitting. By enabling these networks to train in **under 564 Bytes of SRAM at 12,000 samples per second**, MTC solves real industrial adaptation problems on 50-cent microcontrollers."

#### Q12: "Why not just buy a more expensive chip like an ARM Cortex-M7 with 2MB PSRAM?"
- **Responder:** Ajay Sisodia
- **Defense Answer:**  
  "In industrial manufacturing and mass consumer IoT, unit economics dictate hardware choices. An ARM Cortex-M7 with external PSRAM costs between \$7 and \$15, draws significantly higher battery power, and requires complex multi-layer PCB routing.  
  In contrast, an ESP32-C3 or ARM Cortex-M0+ costs under \$1.00 and can run for months on a coin-cell battery.  
  If an embedded engineer can perform local model adaptation in **564 Bytes of internal SRAM**, they do not need external PSRAM or expensive silicon. MTC delivers an order-of-magnitude reduction in bill-of-materials (BOM) cost and power consumption."

#### Q13: "What are each member's specific contributions to this project?"
- **Defense Answer:**  
  - **Ajay Sisodia:** "I developed the compiler frontend, the declarative JSON parsing schema, the static AST dimensional validator, and the automated build integration toolchain."  
  - **Harsh Chaudhary:** "I formulated the 16-bit signed Q15 arithmetic specification, implemented the Interval Bound Propagation engine to certify overflow safety, and authored the saturating integer math macros."  
  - **Devrat Kuntal:** "I designed the static liveness memory planner, proved Theorem 1 for alternating ping-pong delta buffers, formulated the fused in-place SGD update pass, and wrote the C99 backend code generator."  
  - **Aryan Upadhyay:** "I authored the bare-metal RISC-V execution harness, configured the custom linker script and startup assembly, integrated the `esp32sim` hardware emulator, instrumented cycle counts via CSR `mcycle`, and evaluated the empirical benchmark suite."

---

## Part 4: Stage Delivery Checklist

- [ ] **Ajay:** Projector remote / slide clicker in hand. Speak with commanding opening energy.
- [ ] **Harsh:** Crisp blackboard-style explanation of Q15 fixed-point arithmetic; show why rounding bias $16384$ matters.
- [ ] **Devrat:** Use the pointer to trace the ping-pong pointer swaps on Figure 2; emphasize that original weights are preserved before SGD in-place updates.
- [ ] **Aryan:** Present benchmark numbers with exact metrics (60–564 Bytes, 12k–145k samples/s, CSR `mcycle`). Own the XOR gradient attenuation critique confidently.
- [ ] **All:** Rehearsed seamless verbal handoffs without awkward pauses.
