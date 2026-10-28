# Research Publication Roadmap: Micro-Train-Compiler (MTC)
## From Capstone Project to Peer-Reviewed IEEE / ACM Publication

**Project:** Micro-Train-Compiler (MTC)  
**Authors:** Ajay Sisodia, Harsh Chaudhary, Devrat Kuntal, Aryan Upadhyay  
**Affiliation:** Department of Computer Science & Engineering, G. L. Bajaj Group of Institutions, Mathura  
**Target Submission Window:** Early 2027 / Final Year  

---

## 1. The Core Secret of Academic Publishing

Academic reviewers at IEEE and ACM do **not** accept papers just because "the code works" or "we built a software system."  
A research paper is accepted based on **three core elements**:

1. **A Clear Novel Problem / Gap:** What exact limitation exists in the current state-of-the-art (e.g., TFLM, MCUNetV3, AIfES)?
2. **A Distinct Methodological Novelty:** What clever algorithmic or compiler technique did you design to solve it?
3. **Empirical Quantitative Evidence:** Proof on real physical hardware (ESP32) comparing your tool against existing baselines with tables, ablation studies, and convergence curves.

---

## 2. MTC's Unique Research Angle (The "Novelty Claim")

To publish, your paper needs a sharp, defensible title and claim. Here is the winning research angle for MTC:

### Proposed Paper Title:
> **"MTC: Compile-Time Interval-Bounded Automatic Differentiation for Zero-Allocation Neural Network Training on Microcontrollers"**  
> *(Alternative: "Safe and Lean: A Static Compiler for On-Device Training on 32-bit Microcontrollers with Guaranteed Q15 Overflow Bounds")*

### The 3 Novel Contributions to Claim in the Abstract:
1. **Source-to-Source Analytical Autodiff:** Instead of building dynamic computational tapes at runtime (like PyTorch) or relying on generic runtime interpreters (like AIfES/TFLM), MTC analytically lowers forward-backward passes into pure, unrolled C99 loops ahead-of-time, achieving **zero dynamic heap allocation (`malloc = 0`)**.
2. **Compile-Time Numerical Safety Certification (Novelty!):** Fixed-point Q15 arithmetic on low-power MCUs suffers from catastrophic two's-complement integer wrap-around. MTC is the **first edge compiler to integrate Interval Bound Propagation (IBP)** directly into the lowering pass, certifying that all activation and gradient ranges stay strictly within $[-1.0, +1.0)$ *before flashing to silicon*.
3. **Deterministic Memory Planning:** A liveness-based interference graph allocator that maps disjoint forward/backward tensors into an overlapping unified static buffer, enforcing a strict peak SRAM watermark of $\le 64$ KB.

---

## 3. Target Publication Venues

Choose the venue matching your timeline and publication ambitions:

| Tier | Venue Name | Type | Acceptance / Timeline | Why It Fits MTC |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1 (Premier TinyML)** | **TinyML Research Symposium** (held with MLSys / DAC) | International Conference | High Prestige, 3–4 months review | The dedicated worldwide venue for TinyML systems research. |
| **Tier 1 (Embedded Letters)** | **IEEE Embedded Systems Letters (ESL)** | Rapid SCI/Scopus Journal (4 pages) | Fast turnaround (2–3 months), 4-page strict limit | Perfect for concise, high-impact compiler/hardware prototypes like MTC. |
| **Tier 2 (High Scopus Journal)**| **IEEE Access** / **ACM Transactions on Embedded Computing (TECS)** | Open Access SCI/Scopus Journal | 2–4 months review, high visibility | Excellent for full 8–10 page technical papers with complete hardware benchmarks. |
| **Tier 2 (Reputable Conference)**| **IEEE International Conference on VLSI Design & Embedded Systems (VLSID)** | International Conference | Annual deadline (August–October) | Prominent Indian/international IEEE conference with high recognition for B.Tech students. |

---

## 4. The Experimental Evaluation Plan (What You Must Measure)

Reviewers judge a systems paper by its **Evaluation Section**. To write a convincing paper, we must collect 4 specific experiments on the ESP32:

### Experiment 1: SRAM Memory Footprint vs Baselines
- **Compare MTC against:**
  1. AIfES (Fraunhofer static C library)
  2. TensorFlow Lite for Microcontrollers (Inference baseline)
  3. Naive Float32 C implementation
- **Metrics to Report:** Peak SRAM watermark (bytes), Flash ROM footprint (KB), dynamic memory allocations (`malloc` count = 0).

### Experiment 2: Training Latency & Throughput (Clock Cycles)
- Measure the execution time for one full training step (Forward + Analytical Backprop + SGD Update) on ESP32 running at 240 MHz.
- **Metric:** Milliseconds per epoch and microjoules ($\mu\text{J}$) per sample.

### Experiment 3: Numerical Convergence & Stability Verification
- Train a model (e.g. Non-linear XOR, Sine wave regression, or 3-axis Accelerometer Gesture Classifier) on the ESP32.
- Log the loss curve over 100 epochs via UART.
- **Comparison:** Show that MTC's Q15 fixed-point training matches Float32 training curve without exploding or overflowing.

### Experiment 4: Ablation Study of Static Buffer Reuse
- Show a bar chart comparing:
  - Total Memory without buffer reuse (sum of all tensor sizes)
  - Memory with MTC's lifetime reuse allocator (proves the $40\% - 60\%$ SRAM reduction).

---

## 5. Standard IEEE / ACM Paper Structure

A research paper follows a strict 6-section structure:

```text
├── 1. Introduction
│    ├── Motivation: Sensor drift, privacy, cloud communication bandwidth
│    ├── The Memory Wall: Why existing autograd tools fail on MCUs (< 512KB SRAM)
│    └── Summary of 3 Contributions (Bulleted list)
├── 2. Related Work
│    ├── On-Device Training Frameworks (MCUNetV3, TinyTL, AIfES)
│    └── Embedded ML Compilers & Memory Scheduling (TVM, Glow, CMSIS-NN)
├── 3. Compiler Architecture & System Design
│    ├── Front-End Ingestion & AST Validation
│    ├── Analytical Reverse-Mode Automatic Differentiation
│    └── Static Interval Bound Propagation (IBP for Q15 Stability)
├── 4. Deterministic Memory Allocation
│    ├── Tensor Liveness Analysis & Interference Graph
│    └── Buffer Overlay and Zero-Allocation C99 Code Generation
├── 5. Experimental Evaluation
│    ├── Hardware Setup: ESP32 (Xtensa Dual-Core 240MHz, 520KB SRAM)
│    ├── Benchmark Datasets (XOR, Sensor Anomaly Detection)
│    ├── Memory & Latency Comparisons vs AIfES
│    └── Convergence Accuracy & Fixed-Point Stability Curves
└── 6. Conclusion & Future Work
     └── Summary of results, roadmap toward convolutional layers and INT8 quant.
```

---

## 6. Step-by-Step Action Plan

1. **Step 1: Finish Compiler Engine (Phase 1 & 2):**
   - Lower AST $\to$ `IrGraph` computation DAG (`mtc_compiler/ir.py`).
   - Implement the reverse-mode autodiff pass to emit gradient nodes analytically.
2. **Step 2: Collect Hardware Telemetry (Phase 4):**
   - Flash the emitted `mtc_model.c` to an actual ESP32 board.
   - Stream epoch loss over serial UART to our Python telemetry script.
   - Save the raw numbers into CSV files for plotting.
3. **Step 3: Draft the Overleaf / LaTeX Paper:**
   - Use the official IEEE 2-column conference template (`IEEEtran.cls`).
   - Generate high-resolution vector plots (`matplotlib` saved as PDF/EPS).
4. **Step 4: Guide Review & Submission:**
   - Share the draft with your project supervisor (Asst. Prof. Gaurisha Sisodia) for faculty review and institutional endorsement.
   - Submit to target venue.
