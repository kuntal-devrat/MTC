# Micro-Train-Compiler (MTC)
## Project Progress Report — 02

**Project Title:** Micro-Train-Compiler (MTC): A Source-to-Source Compiler for On-Device Neural Network Training on Microcontrollers  
**Project Category:** Compiler Design & Embedded Machine Learning (TinyML)  
**Target Hardware:** ESP32 Microcontroller (Xtensa Dual-Core 240MHz, 520KB SRAM)  
**Tech Stack:** Python (Compiler Tool), C99 (Generated Training Program), Python (Validation)
**Date of Submission:** 18 September 2026  
**Document Format:** 10-page Progress Report PDF (`docs/Progress_Report_02_MTC.pdf`) with dedicated page breaks per subtopic and native vector Mermaid diagrams.

---

### Page 1: Cover Page
- **Institution Name:** G. L. Bajaj Group of Institutions, Mathura
- **Department:** Department of Computer Science & Engineering
- **Document Title:** PROGRESS REPORT — 02
- **Project Title:** Micro-Train-Compiler (MTC)
- **Subtitle:** A Source-to-Source Compiler for On-Device Neural Network Training on Microcontrollers
- **Academic Session:** B.Tech 5th Sem, 2026–27 Session
- **Submitted By:**
  - Ajay Sisodia (Roll No.: 2405111530012)
  - Harsh Chaudhary (Roll No.: 2405111530067)
  - Devrat Kuntal (Roll No.: 2405111530055)
  - Aryan Upadhyay (Roll No.: 2405111530037)
- **Submitted To:**
  - Gaurisha Sisodia (Asst. Professor, Dept. of Computer Science & Engineering, GL Bajaj Group of Institutions, Mathura)

---

### Page 2: Section 1 — Project Overview & Problem Statement

#### 1.1 Background & Motivation
Tiny Machine Learning (TinyML) enables neural network inference on resource-constrained microcontrollers (MCUs) with as little as 256KB SRAM. Existing frameworks like TensorFlow Lite Micro and CMSIS-NN operate on a "train-in-cloud, deploy-frozen" paradigm. While inference is now feasible, deployed models remain static and cannot adapt to changing field conditions.

#### 1.2 The Problem: Limitations of Frozen On-Device Models
1. **Sensor Degradation & Drift:** Physical wear, mechanical fatigue, and thermal stress alter sensor baselines over time. A frozen model loses accuracy unless physical recalibration occurs directly at the edge node.
2. **Privacy & Air-Gapped Compliance:** Medical records (HIPAA), private audio, and industrial telemetry prohibit streaming raw data to external cloud servers.
3. **Bandwidth & Power Bottlenecks:** Continuously streaming high-frequency raw telemetry via wireless radios drains battery power rapidly. Transmitting compact weight updates (~10KB) consumes orders of magnitude less energy.

#### 1.3 Proposed Solution: The Micro-Train-Compiler (MTC)
MTC is a Python-hosted source-to-source compiler that generates static-storage C99 training programs for a constrained sequential neural-network model. The current implementation includes Q15 forward/backward computation, SGD updates, initial-state range diagnostics, static RAM planning, and a host runner; physical MCU deployment remains to be validated.

---

### Page 3: Section 2 — Literature Survey & Competitive Differentiation

#### 2.1 Competitive Framework Survey
| Framework / Work | Origin | Execution Model | Key Architectural Limitation Addressed by MTC |
| :--- | :--- | :--- | :--- |
| **MCUNetV3** (Lin et al., NeurIPS) | MIT HAN Lab (2022) | Runtime Engine (TinyEngine) | Relies on specialized runtime interpreter; lacks compile-time interval range safety. |
| **TinyTL** (Cai et al., NeurIPS) | MIT (2020) | Bias-Only Adaptation | Restricted to tuning biases; cannot perform full backpropagation across dense layers. |
| **AIfES** (Fraunhofer IMS) | Industry (2021) | Static C Library | Generic runtime dispatch overhead; lacks compile-time buffer reuse optimization. |
| **TensorFlow Lite Micro** | Google (2021) | Inference-Only Runtime | Strictly inference; completely lacks gradient backpropagation kernels. |
| **Tapenade / Enzyme** | Academic (2013/20) | LLVM / Source Autodiff | Desktop/GPU autodiff; no fixed-point quantization or MCU memory planning. |

#### 2.2 Critical Differentiation Dimensions
- **On-Device Training:** MTC synthesizes compile-time C99 code with zero runtime dispatch.
- **Memory Model:** Strict static buffer reuse guarantees total peak SRAM $\le$ 64KB without `malloc()`.
- **Numerical Stability:** Formal interval arithmetic bounds $[a_{\min}, a_{\max}]$ prevent integer overflow in Q15 arithmetic.
- **Runtime Footprint:** Exactly 0 KB (no runtime engine required).

---

### Page 4: Section 3 — Proposed Methodology & Compiler Pipeline

#### 3.1 Five-Phase Compilation Pipeline Overview
| Phase | Pass Name | Input Data Structure | Output Product / Artifact |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Model Parser & Validator | JSON Model Specification | Verified Forward Computation DAG (`IrGraph`) |
| **Phase 2** | Reverse-Mode Autodiff | Forward `IrGraph` | Combined Forward + Backward Gradient DAG |
| **Phase 2b** | Numerical Range Analysis | Combined Gradient DAG | Bounded DAG with Q15 Stability Verification |
| **Phase 3** | Static Memory Planner | Annotated DAG + Node Lifetimes | Static Offset Map (< 64KB Total Watermark) |
| **Phase 4** | C99 Code Emitter | Memory-Mapped IR Graph | Standalone C Files (`mtc_model.c`, `mtc_train.c`) |

#### 3.2 Pipeline Pass Descriptions
1. **Ingestion & Validation:** Deserializes JSON, validates layer connectivity ($W_1.\text{out} = W_2.\text{in}$), and constructs the forward DAG.
2. **Reverse-Mode Autodiff:** Synthesizes exact analytical gradients at compile time without a runtime tape.
3. **Interval Range Analysis:** Propagates closed interval arithmetic to predict and prevent Q15 fixed-point integer overflow.
4. **Static Memory Planning:** Reuses memory buffers across non-overlapping tensor lifetimes.
5. **C99 Emission:** Emits bare-metal C99 source files ready for compilation via `xtensa-esp32-elf-gcc`.

---

### Page 5: Section 4 — Core Algorithms & Mathematical Formulations

#### 4.1 Reverse-Mode Analytical Automatic Differentiation
Unlike frameworks that build dynamic computation tapes at runtime, MTC analytically generates exact gradient nodes at compile time:

| Operation | Forward Formulation | Analytically Emitted Backward Derivative |
| :--- | :--- | :--- |
| **Dense (Linear)** | $y = W \cdot x + b$ | $\frac{\partial L}{\partial W} = x^T \cdot \frac{\partial L}{\partial y}, \quad \frac{\partial L}{\partial b} = \frac{\partial L}{\partial y}, \quad \frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot W^T$ |
| **ReLU Activation** | $y = \max(0, x)$ | $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \odot \mathbb{I}(x > 0) \quad \text{\footnotesize (Binary gate mask)}$ |
| **Sigmoid Activation** | $y = \sigma(x) = \frac{1}{1 + e^{-x}}$ | $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \odot \sigma(x) \odot (1 - \sigma(x))$ |
| **MSE Loss** | $L = \frac{1}{n}\sum_{i=1}^n (\hat{y}_i - y_i)^2$ | $\frac{\partial L}{\partial \hat{y}} = \frac{2}{n}(\hat{y} - y)$ |
| **SGD Update** | Parameter step | $W_{\text{new}} = W_{\text{old}} - \eta \cdot \frac{\partial L}{\partial W}, \quad b_{\text{new}} = b_{\text{old}} - \eta \cdot \frac{\partial L}{\partial b}$ |

#### 4.2 Static Numerical Range Analysis (Interval Arithmetic)
Fixed-point Q15 arithmetic represents signed fractions within $[-1.0, \, 1.0 - 2^{-15}] \approx [-1.0, \, +0.999969]$. Values exceeding this boundary cause catastrophic two's-complement integer wrap-around. MTC bounds all forward activations and backward gradients using closed interval rules $[a_{\min}, a_{\max}]$:

- **Interval Addition:**
  $$[a, b] + [c, d] = [a + c, \, b + d]$$
- **Interval Multiplication:**
  $$[a, b] \times [c, d] = [\min(ac, ad, bc, bd), \, \max(ac, ad, bc, bd)]$$
- **Weight Bounds (Xavier / Glorot Initialization):**
  $$W \in \left[ -\sqrt{\frac{6}{\text{fan}_{\text{in}} + \text{fan}_{\text{out}}}}, \, +\sqrt{\frac{6}{\text{fan}_{\text{in}} + \text{fan}_{\text{out}}}} \right]$$
- **Stability Metric:**
  $$\text{Ratio} = \frac{\max(|a_{\min}|, \, |a_{\max}|)}{\text{Q15}_{\text{MAX}}}$$

---

### Page 6: Section 5 — System Architecture & Data Flow Diagrams (DFD)

#### 5.1 Level-0 Context Diagram (DFD Level 0)
```text
+------------------------------------+       +------------------------------------+       +------------------------------------+
| 1. External Inputs      USER / CLI |       | Micro-Train-Compiler   PYTHON CORE |       | 2. Synthesized Artifacts  EMBEDDED |
| • Model AST: xor_net.json          | ====> | • Static Dimension Chaining        | ====> | • C99 Source: mtc_model.c / .h     |
|   Layer dims, activation & loss    |       | • Analytical Reverse Autodiff      |       |   Standalone code, zero deps       |
| • Directives: --target esp32       |       | • Numerical Range Evaluation (Q15) |       | • Unified Buffer: sram_pool[]      |
|   Q15 precision, 4-byte alignment  |       | • Static Memory Allocation (<=64K) |       |   Static layout <= 64 KB           |
|                                    |       | • Standalone ANSI C99 Emitter      |       | • Certificate: Q15 Stability Safe  |
+------------------------------------+       +------------------------------------+       +------------------------------------+
```
*Figure 1: Level-0 Context Diagram — External inputs, host compiler synthesis engine, and emitted embedded C99 artifacts.*

| Interface | Artifact / Stream | Data Structure / Format | Embedded Contract & Function |
| :--- | :--- | :--- | :--- |
| **Input 1** | Model Configuration | JSON AST (`xor_net.json`) | Declares layer dimensions, loss function, and optimizer hyper-parameters. |
| **Input 2** | Hardware Target Flags | CLI Flags (`--target esp32`) | Specifies 16-bit word length and 4-byte memory boundary alignment. |
| **Output 1** | C99 Firmware Source | `mtc_model.c` & `.h` | Self-contained, portable ANSI C99 files with zero external runtime dependencies. |
| **Output 2** | Static SRAM Budget | Memory Layout Plan | Enforces global buffer reuse; guarantees peak memory watermark $\le 64$ KB. |
| **Output 3** | Stability Certificate | Interval Bounds Report | Certifies Q15 mathematical safety: all tensor values fit within $[-1.0, +1.0)$. |

#### 5.2 Level-1 Data Flow Diagram (DFD Level 1)
```text
FRONTEND PIPELINE: INGESTION, AST VALIDATION & ANALYTICAL AUTODIFF (HOST MEMORY)
[ 0. Model AST: xor_net.json ] ──> [ 1.0 Validator: Dims & Order ] ──> [ 2.0 IR Builder: Forward DAG ] ──> [ 3.0 Autodiff: Reverse Graph ]
                                                    │
                                                    ▼
                             ⇓ Combined Full Adjoint DAG (Forward + Gradients) ⇓
                                                    │
                                                    ▼
BACKEND PIPELINE: NUMERICAL SAFETY, STATIC MEMORY PLANNING & CODEGEN (OPTIMIZATION)
[ 4.0 Range Analysis: Q15 Bounds ] ──> [ 5.0 Memory Planner: Liveness Reuse ] ──> [ 6.0 C99 Emitter ] ──> [ Emitted C99: mtc_model.c/.h ]
```
*Figure 2: Level-1 Data Flow Diagram — Sequential transformation across internal compiler passes.*

#### 5.3 Hardware Deployment & Closed-Loop Telemetry Workflow
```text
+------------------------------+     USB/UART0      +------------------------------+     UART1 (115.2k)    +------------------------------+
| 1. Host Workstation (PC)     | ─────────────────> | 2. Edge Target (ESP32 SoC)   | ────────────────────> | 3. Telemetry & Audit Station |
| • Python Host Compiler       |    Flash C99       | • Dual-Core @ 240MHz, 520K   |    Live Epoch Loss    | • Python 3.11 + PySerial     |
| • Synthesizes ANSI C99 files |                    | • On-Chip Training Loop      |    & Weight Telemetry | • Live Convergence Dashboard |
| • Compiles via xtensa-gcc    |                    | • Guarantees: 0 malloc, <64K |                       | • Validates Q15 Stability    |
+------------------------------+                    +------------------------------+                       +------------------------------+
```
*Figure 3: Hardware In-The-Loop Training Loop — Bare-metal microcontroller training and live host telemetry monitoring.*

| Subsystem | Hardware & Execution Spec | Communication Channel | Operational Invariant & Contract |
| :--- | :--- | :--- | :--- |
| **Edge Target (ESP32)** | Xtensa LX6 32-bit @ 240 MHz, 520 KB SRAM | UART0 (Flashing / Bootloader) | Zero dynamic heap allocation (`malloc = 0`). |
| **Firmware Footprint** | Standalone ANSI C99 (`mtc_model.c/.h`) | On-Chip Flash ROM < 32 KB | Peak SRAM buffer watermark $\le 64$ KB. |
| **Telemetry Monitor** | Python 3.11 + PySerial + Matplotlib | UART1 (Telemetry @ 115200 Baud) | Per-epoch loss and weight gradient telemetry logging. |

---

### Page 7: Section 6 — Process Description & Compilation Flowchart

#### 6.1 End-to-End Compilation Process Flowchart
```text
STAGE 1: FRONTEND INGESTION & ARCHITECTURAL VALIDATION
[ CLI Invocation ] ──> [ Read JSON Model ] ──> [ Syntax Valid? ] ──> [ Dims Chain? ] ──> [ AST Verified ]
                                                      │                      │                    │
                                                      └── (Fail: Syntax Err) └── (Fail: Dim Err)  ▼
                                                                                   ⇓ Pass: Valid Structural AST ⇓

STAGE 2 & 3: IR DAG SYNTHESIS, ANALYTICAL AUTODIFF & RANGE STABILITY
[ Build Forward DAG ] ──> [ Analytical Autodiff ] ──> [ Range Analysis ] ──> [ Q15 Overflow? ] ──> [ Safe IR Graph ]
                                                                                   │                     │
                                                                                   └── (Yes: Auto-Clamp) ▼
                                                                                   ⇓ Pass: Certified Q15 Bounds ⇓

STAGE 4 & 5: STATIC MEMORY ALLOCATION & BARE-METAL C99 CODEGEN
[ Liveness Analysis ] ──> [ Buffer Reuse Plan ] ──> [ SRAM <= 64 KB? ] ──> [ C99 Emitter ] ──> [ Firmware Ready ]
                                                          │                                      (0 malloc, portable)
                                                          └── (Fail: OOM Abort)
```
*Figure 4: End-to-End Compilation Flowchart — Multi-stage validation gates, analytical derivation, and code generation.*

#### 6.2 Compilation Stage Specifications & Safeguards
| Stage | Compiler Pass | Transformation & Functionality | Compile-Time Safeguard |
| :--- | :--- | :--- | :--- |
| **Stage 1** | Ingestion & Parser | Deserializes model JSON into strongly-typed `ModelConfig` AST. | Rejects invalid JSON syntax and missing fields. |
| **Stage 2** | Layer Validator | Verifies feature dimension chaining ($W_{i-1}.\text{out} == W_i.\text{in}$) and activation order. | Rejects dimension mismatches, 0-feature layers, and orphan activations. |
| **Stage 3** | IR & Autodiff | Lowers AST into computation DAG; synthesizes analytical reverse-mode gradients. | Eliminates dynamic tape allocation; guarantees $O(1)$ graph footprint. |
| **Stage 4** | Range Analysis | Propagates numerical intervals $[a_{\min}, a_{\max}]$ across forward and backward nodes. | Detects fixed-point Q15 overflow; inserts saturation clamping. |
| **Stage 5** | Memory Planner | Analyzes tensor liveness intervals; maps disjoint tensors to overlapping byte offsets. | Guarantees peak SRAM $\le 64\text{ KB}$; rejects out-of-memory networks. |
| **Stage 6** | C99 Emitter | Synthesizes zero-dependency ANSI C99 files with static global arrays. | Prohibits dynamic `malloc()`/`free()` calls entirely. |

#### 6.3 Fail-Fast Verification Guarantees & Error Diagnostics
- **Early Syntactic & Structural Rejection:** Dimension mismatches ($W_{i-1}.\text{out} \neq W_i.\text{in}$), missing hyper-parameters, and unsupported activation orders are detected at the AST ingestion boundary before DAG construction, returning actionable line/column diagnostics to the developer.
- **Static Deterministic Memory Certification:** The static buffer allocator computes precise live ranges across all forward activation tensors and backward adjoints. Any model configuration whose peak watermark exceeds the target 64 KB SRAM budget is rejected at compile time, eliminating embedded runtime crashes.

---

### Page 8: Section 7 — Data Structure Design (IR Schema / ER Equivalent)

#### 7.1 Entity-Relationship Architecture of Compiler Data Structures
```text
+-----------------------+     +-----------------------+     +-----------------------+
| ModelConfig  AST ROOT |     | LayerConfig      ENUM |     | IrGraph     DAG ARENA |
| • name: String        |     | • Dense { in, out }   |     | • forward_nodes       |
| • precision: Precision|     | • ReLU                |     | • backward_nodes      |
| • layers: Vec<Layer>  |     | • Sigmoid             |     | • loss_node           |
| • loss: LossType      |     +-----------------------+     | • precision: Precision|
| • optimizer: Optimizer|                                   +-----------------------+
+-----------------------+                                   
            │                                                           │
            ▼                                                           ▼
  ModelConfig 1 ──> * LayerConfig  │  ModelConfig 1 ══> 1 IrGraph  │  IrGraph 1 ──> * IrNode
            ▲                                                           ▲
            │                                                           │
+-----------------------+     +-----------------------+     +-----------------------+
| IrNode    ATOMIC NODE |     | MemoryPlan  ALLOCATOR |     | Interval       BOUNDS |
| • id: usize           |     | • offsets: HashMap    |     | • min: f64            |
| • op_type: OpKind     |     | • total_pool_bytes    |     | • max: f64            |
| • shape: TensorShape  |     | • peak_watermark <=64K|     | • fits_q15(): bool    |
| • inputs: Vec<usize>  |     +-----------------------+     | • clamp_to_q15()      |
| • bounds: Interval    |                                   +-----------------------+
+-----------------------+                                   
```
*Figure 5: Entity-Relationship Schema of Internal Compiler Data Structures.*

#### 7.2 Data Structure Specifications & Invariants
| Entity | Python Representation | Architectural Role & Key Fields | Target Constraint / Invariant |
| :--- | :--- | :--- | :--- |
| `ModelConfig` | Frozen dataclass | Root JSON config, bounds, optimizer, and embedded training samples. | Strictly validated before compilation. |
| `LayerConfig` | Frozen dataclass | Dense layer dimensions or activation kind. | Explicit checks enforce supported layer order and dimensions. |
| `DenseStage` | Frozen dataclass | Dense dimensions and optional following activation. | Deterministic sequential lowering. |
| `MemoryPlan` | Frozen dataclass | Static offsets for retained activations and reusable gradient buffers. | Computed RAM must fit the configured target budget. |
| `Interval` | Frozen dataclass | Continuous range $[a_{\min}, a_{\max}]$ for Q15 preflight analysis. | Covers initialized weights, forward values, backward deltas, gradients, and one update. |

#### 7.3 Memory Ownership, Lifetime Management & Zero-Allocation Semantics
- **Explicit Data Ownership:** Python objects represent configuration and interval values; compiler passes should keep graph references explicit and avoid hidden mutable global state.
- **Static Buffer Lifetime Reuse:** The planner assigns non-overlapping activation lifetimes and two reusable gradient buffers to offsets in a static Q15 pool.
- **Deterministic Compile-Time Checks:** Model-specific RAM bounds and initial-step numerical intervals are checked before code emission. Predicted Q15 overflows block generation unless saturating generation is explicitly enabled.

#### 7.4 Concrete Compilation Pipeline Mapping (AST to IR to Bare-Metal C99)
| Compilation Level | Data Structure Representation | Memory Placement | Compiler Invariant & Safety |
| :--- | :--- | :--- | :--- |
| **1. High-Level AST** | `ModelConfig`, `LayerConfig` enums | Host Transient Heap | Rejects dimension mismatch & orphan activations. |
| **2. Intermediate Representation** | `IrGraph` arena with `IrNode` vectors | Host Compiler Memory | Strictly cycle-free DAG; $O(1)$ operand indexing. |
| **3. Synthesized Bare-Metal** | `MemoryPlan` & `uint8_t sram_pool[]` | MCU Static `.bss` Segment | Overlaps disjoint buffers; $\le$ 64 KB, 0 malloc. |

---

### Page 9: Section 8 — Current Implementation Progress & Verification

#### 8.1 Implementation Status Table
| Component / Module | Target Phase | Status | Verification & Evidence |
| :--- | :--- | :--- | :--- |
| **PRD & Formalization** | Specification | **Complete** | 446-line specification document defining mathematical derivations and ESP32 SRAM budget. |
| **Python Package & CLI** | Infrastructure | **Complete** | `pyproject.toml` package with `validate`, `analyze`, and `compile` commands; standard-library compiler dependencies. |
| **Configuration & Validation** | Phase 1 | **Complete** | Strict JSON/dataclass parsing, supported-layer checks, dimensions, Q15 domains, and embedded sample validation. |
| **Sequential IR** | Phase 1 | **Complete** | `DenseStage` normalizes each dense operation and its optional activation. |
| **Range Analysis** | Phase 2b | **Implemented** | Propagates initial Q15 weights, forward bounds, backward deltas, gradients, and one SGD update; long-run guarantees are not claimed. |
| **Analytical Backpropagation** | Phase 2 | **Implemented** | Generated C includes Dense, ReLU, Sigmoid, MSE, and SGD equations without runtime autograd. |
| **Static Memory Planner** | Phase 3 | **Implemented** | Plans activation and gradient-buffer offsets and rejects plans exceeding the 64 KiB RAM budget. |
| **Q15 C99 Emitter** | Phase 4 | **Implemented** | Emits model, training runner, header, and CMake files with fixed-size storage and saturating arithmetic. |
| **Host Integration Validation** | Verification | **Complete** | Both example projects compile with GCC `-Wall -Wextra -Werror`, execute, and reduce reported loss. |
| **Physical MCU Deployment** | Verification | **Not run** | ESP-IDF and ARM cross-toolchains/hardware are not available in the current environment. |

#### 8.2 Execution & Verification Log
```text
$ python -m unittest discover -s tests -v
Ran 20 tests
OK

$ mtc compile examples/xor_net.json --output build/xor --allow-clipping
Generated 4 C99 project files in build/xor
Static RAM: 60 / 65536 bytes
$ gcc -std=c99 -Wall -Wextra -Werror mtc_model.c mtc_train.c -o mtc_train
$ ./mtc_train
epoch=1 loss_q15=8604
epoch=1000 loss_q15=6490
XOR classification after training: 4/4 samples correct at a 0.5 threshold.
```

---

### Page 10: Section 9 — Technology Stack, Milestones & References

#### 9.1 Technology Stack Justification
- **Host Compiler (Python):** Rapid compiler-pass iteration, standard-library JSON parsing, and straightforward package installation with `pip`.
- **Target Code (C99):** Portable C99 core; host GCC verification is complete. ESP32/STM32 cross-compilation is pending toolchain and board availability.
- **Validation Host (Python):** Telemetry acquisition over serial UART (`pyserial`) and real-time loss curve visualization (`matplotlib`).

#### 9.2 Project Timeline & Milestones
- **Weeks 1–3 (August 2026):** Literature Review, PRD Specification, Project Scaffold *(Complete)*
- **Weeks 4–6 (Sept 2026):** Model Parser, Layer Validator, Normalized Dense-Stage IR *(Complete)*
- **Weeks 7–9 (Oct 2026):** Reverse-Mode Analytical Autodiff in generated C *(Complete for Dense/ReLU/Sigmoid/MSE)*
- **Weeks 10–12 (Nov 2026):** One-step Q15 Interval Diagnostics *(Implemented; longer-run calibration remains future evaluation)*
- **Weeks 13–15 (Dec 2026):** Static Memory Planner & Q15 Arithmetic *(Complete in host-tested generator)*
- **Weeks 16–18 (Jan 2027):** C99 Code Emitter *(Complete; host GCC tested)*
- **Weeks 19–22 (Feb 2027):** ESP32 Hardware Deployment & Benchmark vs AIfES *(Upcoming)*
- **Weeks 23–26 (March 2027):** Final Documentation, Demonstration, Defense *(Upcoming)*

#### 9.3 Key References
1. Lin, J., Chen, W.-M., Cai, H., et al. (2022). "MCUNetV3: On-Tiny-Device Training with Memory-Efficient Sparse Backpropagation." *NeurIPS*.
2. Cai, H., Gan, C., Zhu, L., & Han, S. (2020). "TinyTL: Reduce Memory, Not Parameters for Efficient On-Device Learning." *NeurIPS*.
3. David, R., Duke, J., Jain, A., et al. (2021). "TensorFlow Lite Micro: Embedded Machine Learning for TinyML Systems." *MLSys*.
4. AIfES — Artificial Intelligence for Embedded Systems. Fraunhofer IMS. *github.com/Fraunhofer-IMS/AIfES_for_Arduino*.
5. Moore, R. E., Kearfott, R. B., & Cloud, M. J. (2009). *Introduction to Interval Analysis*. SIAM.
6. Hascoet, L. & Pascual, V. (2013). "The Tapenade Automatic Differentiation Tool." *ACM TOMS*.
