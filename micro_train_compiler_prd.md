# Product Requirement Document (PRD)

## Micro-Train-Compiler (MTC)

**Version:** 2.0  
**Last Updated:** September 2026  
**Project Type:** Final Year Project — Compiler Design & Embedded Machine Learning

---

## 1. Abstract

Micro-Train-Compiler (MTC) is a Python-hosted source-to-source compiler that translates a supported sequential neural-network definition and embedded training samples into standalone C99 code. The generated program implements Q15 forward inference, analytical backpropagation, MSE loss, and SGD updates using fixed-size static storage and no runtime heap allocation.

MTC performs model validation, initial-state interval analysis, static memory planning, and C generation on the host. The emitted program contains the training loop and model-specific operations; it does not embed a Python runtime or rely on a neural-network library on the MCU.

**Research contribution under evaluation:** MTC propagates interval bounds through the initialized forward graph and one analytical backward/SGD step to identify Q15 ranges that may saturate before generation. This is narrower than a guarantee over an entire training run: weights change during training, and input/target bounds are assumptions supplied by the model author. The potential contribution is a practical preflight analysis for generated fixed-point training, measured against an integer reference and generated-code behavior.

---

## 2. Problem Statement

Modern edge AI deployment follows a **train-in-cloud, deploy-frozen** paradigm. Frameworks like TensorFlow Lite Micro and CMSIS-NN enable efficient inference on microcontrollers, but the deployed models are static — they cannot adapt to post-deployment changes in sensor characteristics, environmental conditions, or hardware drift.

This rigidity creates real problems:

- **Sensor drift:** Physical sensors degrade over time. A vibration model trained on factory-fresh hardware will lose accuracy as bearings wear, temperatures shift, or components age. Without on-device adaptation, the device must be recalled or manually re-calibrated.
- **Data privacy constraints:** In applications governed by HIPAA, GDPR, or operating in air-gapped environments (e.g., remote infrastructure, medical wearables), raw sensor data cannot leave the device. On-device training keeps data local — only learned parameters (weight updates) need to be transmitted.
- **Bandwidth and power costs:** Continuously streaming raw sensor telemetry over cellular, LoRa, or satellite links is expensive in both power and bandwidth. A model that learns locally and transmits only compact weight deltas dramatically reduces communication overhead.

**MTC addresses this gap** by generating C code that enables small neural networks to train directly on microcontrollers with limited RAM (≤256KB) and Flash (≤1MB), making post-deployment model adaptation feasible without cloud connectivity.

---

## 3. Related Work

On-device training, quantized training, compile-time training-system optimization, and fixed-point network analysis all have prior work. Relevant comparisons include:

| Project | Approach | Differentiation from MTC |
|---|---|---|
| **On-Device Training Under 256KB Memory** (Lin et al., NeurIPS 2022) | Quantization-Aware Scaling, sparse updates, and the Tiny Training Engine; the backward graph is pruned at compile time. | Establishes both MCU training and compile-time training-system transformations. MTC focuses instead on fixed-point range diagnostics for generated Dense-network training. |
| **Fully Quantized DNN Training on Cortex-M** (Deutel et al., 2024) | Quantized forward/backward computation, dynamic scaling, and hardware evaluation on Cortex-M devices. | Demonstrates quantized MCU training. MTC's question is whether interval preflight can predict saturation in a generated training step. |
| **Fixed-Point Code Synthesis for Neural Networks** (Benmaghnia et al., 2022) | Uses bounded input ranges and linear constraints to select fixed-point formats for network inference. | Establishes interval/range-driven fixed-point synthesis for inference; MTC applies range propagation to training gradients and weight updates. |
| **TinyTL** (Cai et al., MIT, 2020) | Memory-efficient transfer learning by training only bias parameters, reducing activation storage. | Targets fine-tuning of pre-trained models. MTC targets full from-scratch training of small networks. |
| **AIfES** (Fraunhofer IMS) | Open-source C library for inference and training on MCUs, using ARM CMSIS acceleration. | Ships as a linkable library. MTC generates self-contained code with no library dependencies. |
| **TensorFlow Lite Micro** (Google) | Inference-only runtime for microcontrollers. No training support. | Inference only — no backpropagation capability. |

**MTC's intended niche:** model-specific C generation combined with interval diagnostics for Q15 backward-pass and SGD intermediates. The software also handles validation, Q15 lowering, static RAM planning, and training-loop generation; those capabilities are engineering scope, not individually claimed as novel.

**Novelty caveat:** The preliminary search identified fixed-point range analysis for inference and quantized backpropagation on MCUs, but did not establish that no prior work analyzes fixed-point training gradients statically. A publication must make that claim only after a systematic search across compiler, embedded-ML, and formal-analysis literature.

---

## 4. System Architecture

MTC operates as a command-line compiler tool that runs on the host machine (laptop/desktop). It reads a model definition, processes it through five compiler passes, and outputs C source files ready for cross-compilation.

```
┌─────────────────┐   ┌────────────────┐   ┌────────────────┐   ┌─────────────────────┐
│ Model Definition│   │ Phase 1:       │   │ Phase 2:       │   │ Phase 2b: ★ NOVEL   │
│ (JSON/TOML)     │──▶│ Parser & IR    │──▶│ Reverse-Mode   │──▶│ Numerical Range     │
│                 │   │ Graph Builder  │   │ Autodiff       │   │ Analysis (Interval  │
└─────────────────┘   └────────────────┘   └────────────────┘   │ Arithmetic)         │
                                                                 └──────────┬──────────┘
                                                                            │
                                              ┌─ Overflow warnings/errors ◀─┤
                                              │  Auto gradient clipping     │
                                              │                             ▼
┌─────────────────┐   ┌────────────────┐   ┌────────────────────────────────────┐
│ Output: .c + .h │   │ Phase 4:       │   │ Phase 3: Memory Planner &          │
│ files for MCU   │◀──│ C Code Emitter │◀──│ Fixed-Point Quantizer              │
└─────────────────┘   └────────────────┘   └────────────────────────────────────┘
                              │
                              ▼
                      ┌────────────────┐
                      │ Cross-compile  │
                      │ → Flash to MCU │
                      └────────────────┘
```

### 4.1 Target Hardware

| Specification | Constraint |
|---|---|
| **Primary target MCU** | Espressif ESP32 (Xtensa Dual-Core, 520KB SRAM, 4MB Flash) |
| **Secondary target (stretch)** | STM32F401 (Cortex-M4, 96KB SRAM, 512KB Flash) |
| **Max runtime RAM usage** | ≤64KB during peak backward pass |
| **Max binary size** | ≤128KB compiled Flash footprint |
| **Generated code dependencies** | None. Zero external libraries. Bare-metal C99. |
| **Dynamic allocation** | Forbidden. All memory is statically allocated at compile time. |

### 4.2 Tech Stack

| Component | Technology | Rationale |
|---|---|---|
| **Compiler core** | Python 3.10+ | Rapid iteration on parser, analysis, and code-generation passes; standard-library JSON support and a simple CLI entry point. |
| **Emitted target code** | C99 | Universal MCU toolchain support. Every embedded target has a C compiler (`arm-none-eabi-gcc`, `xtensa-esp32-elf-gcc`). |
| **Model config parsing** | Python `json` module | Standard-library parsing of model definitions without runtime dependencies. |
| **Validation scripts** | Python (`pyserial`, `matplotlib`) | Reading UART output from the MCU and plotting training convergence curves. |
| **Build system** | `pip` / `pyproject.toml`; CMake or ESP-IDF for generated C | Python packaging for the host compiler and standard tooling for embedded targets. |

### 4.3 Why Python for the Compiler

1. **Fast iteration.** Python keeps compiler-pass development concise, which is useful for an academic prototype with evolving IR and numerical-analysis requirements.
2. **Accessible data modeling.** Dataclasses and explicit validation provide readable structures for model configurations and intermediate representations.
3. **Low setup overhead.** The current host tool uses the standard library and can be installed with `pip install -e .` without external runtime dependencies.
4. **Ecosystem fit.** Python is already used for the project's validation scripts and integrates naturally with numerical experiments and embedded telemetry tooling.

> **Note:** Python is used to *write the host compiler*. Its generated output is C99 code for the target MCU.

---

## 5. Technical Design

### Phase 1: Model Parser & Intermediate Representation (IR)

**Objective:** Parse a structured model definition file and build a typed, directed acyclic graph (DAG) representing the network's forward pass.

**Input format (JSON example):**
```json
{
  "name": "sensor_drift_net",
  "precision": "int16",
  "input_range": [0.0, 1.0],
  "target_range": [0.0, 1.0],
  "epochs": 250,
  "seed": 11,
  "layers": [
    { "type": "dense", "in_features": 4, "out_features": 16 },
    { "type": "relu" },
    { "type": "dense", "in_features": 16, "out_features": 8 },
    { "type": "relu" },
    { "type": "dense", "in_features": 8, "out_features": 1 },
    { "type": "sigmoid" }
  ],
  "loss": "mse",
  "optimizer": { "type": "sgd", "learning_rate": 0.01 },
  "training_data": [
    {"input": [0.0, 0.0, 0.0, 0.0], "target": [0.0]},
    {"input": [1.0, 0.0, 0.0, 0.0], "target": [0.25]}
  ]
}
```

`training_data` is embedded as constant Q15 input/target arrays in the generated runner. `input_range` and `target_range` bound values used by the range pass; both default to `[-1.0, 1.0]`. `epochs` defaults to 100 and `seed` to 1.

**Requirements:**
- Parse model JSON into typed Python dataclasses and reject unsupported fields and values with actionable errors.
- Validate layer connectivity: ensure `out_features` of layer N matches `in_features` of layer N+1.
- Normalize the sequential model into ordered dense stages with dimensions and associated activations for analysis and code generation.
- Reject invalid configurations with clear error messages (e.g., "Layer 3: Dense expects in_features=16 but previous layer outputs 8").

**Supported layer types (v1.0):**
| Layer | Forward Operation |
|---|---|
| Dense (Linear) | `y = Wx + b` |
| ReLU | `y = max(0, x)` |
| Sigmoid | `y = 1 / (1 + exp(-x))` |

**Supported loss functions (v1.0):**
| Loss | Formula |
|---|---|
| Mean Squared Error (MSE) | `L = (1/n) Σ(y_pred - y_true)²` |

**Supported optimizers (v1.0):**
| Optimizer | Update Rule |
|---|---|
| SGD | `w = w - lr * ∂L/∂w` |

---

### Phase 2: Reverse-Mode Automatic Differentiation

**Objective:** Traverse the forward IR graph in reverse topological order and generate corresponding backward-pass IR nodes that compute gradients using the chain rule.

**Requirements:**
- For each forward node type, implement the analytical derivative:
  - **Dense:** `∂L/∂W = xᵀ · ∂L/∂y`, `∂L/∂b = ∂L/∂y`, `∂L/∂x = ∂L/∂y · Wᵀ`
  - **ReLU:** `∂L/∂x = ∂L/∂y ⊙ (x > 0)` (element-wise mask)
  - **Sigmoid:** `∂L/∂x = ∂L/∂y ⊙ σ(x) ⊙ (1 - σ(x))`
  - **MSE:** `∂L/∂y_pred = (2/n)(y_pred - y_true)`
- Produce a unified forward+backward IR graph where each backward node references the forward activations it depends on.
- The differentiation is fully **symbolic/analytical** — no numerical approximation, no runtime autograd tape. All gradient computation paths are resolved at compile time.

---

### Phase 2b: Static Numerical Range Analysis ★ (Novel Research Contribution)

**Objective:** Before emitting C code, analyze Q15 intervals through the initialized forward graph, analytical backward pass, and one SGD update. This is a conservative preflight for the configured input/target domains, not a proof of safety or convergence for an entire training run.

**Research positioning:** On-device and quantized training, compile-time training-system optimization, and fixed-point range analysis for inference all have prior work. The research question is whether this implementation's preflight bounds add useful warning power for Q15 training intermediates beyond those existing methods. Novelty must be stated only after a systematic literature review.

**How it works — Interval Arithmetic Propagation:**

Each intermediate in the supported dense network is assigned a conservative **value range** `[min, max]`. Initial weights use clipped Xavier intervals; input and target domains come from the model configuration. The current analysis evaluates one initialized forward/backward/SGD step:

| Operation | Range Propagation Rule |
|---|---|
| **Dense (matmul)** | `[min, max]_out = sum of products of input ranges × weight ranges` |
| **Bias add** | `[min + b, max + b]` |
| **ReLU** | `[max(0, min), max(0, max)]` |
| **Sigmoid** | `[σ(min), σ(max)]` (monotonic, so range is preserved) |
| **MSE gradient** | `[2/n * (min_pred - max_true), 2/n * (max_pred - min_true)]` |
| **Dense backward (∂L/∂W)** | Range of `xᵀ · ∂L/∂y` using input and gradient ranges |
| **Q15 multiply** | `[a_min * b_min, a_max * b_max]` with overflow check at ±1.0 |

**Compiler actions based on analysis:**

1. **Safe** — range fits within Q15 `[-1.0, +0.9999]`: emit code as-is.
2. **Warning** — range approaches Q15 boundary (>80% of representable range): emit code with a compiler warning suggesting learning rate reduction.
3. **Overflow detected** — by default, compilation fails with the first offending operation and its predicted interval. `--allow-clipping` permits generation with saturating Q15 arithmetic. This option does not insert selective per-layer clipping and does not guarantee convergence; saturation behavior must be evaluated against an integer reference and training runs.

**Requirements:**
- Represent ranges as `Interval { min: float, max: float }` for supported intermediate values.
- Propagate ranges through the forward pass, backward pass, and one SGD weight update.
- For weight initialization ranges, assume Xavier/Glorot initialization bounds: `±sqrt(6 / (fan_in + fan_out))`.
- For input data ranges, accept user-specified bounds in the model config (e.g., `"input_range": [-1.0, 1.0]`) or default to `[-1.0, 1.0]`.
- The analysis must over-approximate the configured one-step arithmetic. Its guarantees do not extend beyond the configured input/target intervals, Xavier initialization assumption, or one optimizer update.
- Report a **stability score** for the overall network: ratio of predicted gradient range to Q15 representable range, per layer.

**Paper contribution framing:** Evaluate whether interval analysis of generated Q15 training steps can predict actual saturation events across a defined set of models, input domains, and learning rates. Report false-positive/negative rates against an integer reference implementation; do not infer long-run training divergence from a one-step bound alone.

---

### Phase 3: Memory Planner & Fixed-Point Quantizer

**Objective:** Map the IR graph's tensor requirements to a concrete, statically-allocated memory layout that fits within the target MCU's RAM budget, and convert all floating-point operations to fixed-point integer arithmetic.

#### 3a. Static Memory Planning

- Perform **liveness analysis** on the IR graph: determine the lifetime (first-use to last-use) of every intermediate tensor (activations, gradient buffers).
- Apply **buffer reuse**: when two tensors have non-overlapping lifetimes, assign them to the same static memory region. This is analogous to register allocation in traditional compilers.
- Compute the **peak memory watermark** (maximum simultaneously-live memory). If this exceeds the target constraint (64KB), emit an error with a breakdown showing which layers/tensors are the bottleneck.
- All allocations are resolved to offsets within a single `static uint8_t memory_pool[N];` array. No `malloc`, no `new`, no heap.

*Reference: This technique is used by TinyEngine (Lin et al., 2020) and TVM's memory planning pass.*

#### 3b. Fixed-Point Quantization

- Convert all weights, biases, activations, and gradients from floating-point to **Q15 format** (signed 16-bit fixed-point, 1 sign bit + 15 fractional bits).
- Generate appropriate scale factors and handle overflow/saturation during multiply-accumulate operations.
- Emit fixed-point arithmetic helpers (fixed-point multiply with rounding, saturating add) as inline C functions.

**Scope limitation:** v1.0 supports only uniform int16 (Q15) quantization. Mixed-precision quantization (per-layer int8/int4) is deferred to future work.

---

### Phase 4: C Code Emitter

**Objective:** Lower the final annotated IR graph into compilable, standalone C99 source files.

**Output structure:**
```
output/
├── mtc_model.h        // Type definitions, buffer sizes, function declarations
├── mtc_model.c        // Forward pass, backward pass, weight update functions
├── mtc_train.c        // Embedded training samples and epoch loss output
└── CMakeLists.txt     // Build configuration for the target toolchain
```

**Requirements:**
- Emit bounded nested loops with layer dimensions fixed in generated source. Loop-unrolled kernels and target-specific SIMD optimization are future work.
- All arrays are declared as `static const` (weights) or `static` (mutable buffers) with sizes computed by the memory planner.
- No dynamic memory allocation. No `malloc`, `calloc`, `realloc`, `new`, or `free` anywhere in the generated code.
- Generated code must compile cleanly with `-std=c99 -Wall -Wextra -Werror` on a host C compiler. Cross-compiler and device builds are an additional validation gate and have not yet been run in this environment.
- Emit epoch number and Q15 loss to standard output in the generated host runner. Embedded projects may route the same output through their platform's UART layer.
- Keep target-specific startup and hardware serial setup outside the portable generated core; the current output is validated as a host C99 program.
- Emit a `Generated by MTC 0.1.0. Do not edit.` header in all generated C files.

**Code style of generated output:**
```c
// mtc_model.c (excerpt — generated code)
#include "mtc_model.h"

static int16_t memory_pool[MEMORY_POOL_SIZE];

void mtc_forward(const int16_t* input) {
    // Layer 0: Dense(4 -> 16)
    int16_t* act_0 = (int16_t*)&memory_pool[OFFSET_ACT_0];
    for (int i = 0; i < 16; i++) {
        int32_t acc = (int32_t)bias_0[i] << 15;
        for (int j = 0; j < 4; j++) {
            acc += (int32_t)weight_0[i * 4 + j] * (int32_t)input[j];
        }
        act_0[i] = saturate_q15(acc >> 15);
    }
    // Layer 1: ReLU
    for (int i = 0; i < 16; i++) {
        act_0[i] = act_0[i] > 0 ? act_0[i] : 0;
    }
    // ... remaining layers
}
```

---

## 6. Scope & Deliverables

### In Scope (v1.0 — Final Year Submission)

| Deliverable | Description |
|---|---|
| **Python CLI compiler** | `mtc validate`, `mtc analyze`, and `mtc compile model.json --target esp32 --output ./build/` |
| **Supported layers** | Dense (Linear), ReLU, Sigmoid |
| **Supported loss** | MSE |
| **Supported optimizer** | SGD |
| **Quantization** | Uniform int16 (Q15) fixed-point |
| **Memory planning** | Static liveness analysis with buffer reuse |
| ★ **Numerical range analysis** | Compile-time interval arithmetic overflow prediction for fixed-point backprop |
| ★ **Overflow handling** | Reject predicted Q15 overflow by default; optional generation with saturating arithmetic |
| **Target MCU** | ESP32 (primary), STM32F401 (stretch) |
| **Native validation demo** | Train the supplied XOR and sensor-drift examples from generated C on the host; both strict-GCC integration tests require loss to decrease |
| **On-device validation** | Flash generated code to the target MCU and verify UART output (not yet performed) |
| ★ **Benchmark comparison** | MTC (compiled) vs. AIfES on the same hardware and network (not yet performed) |

### Out of Scope (Future Work)

| Feature | Reason |
|---|---|
| Convolutional layers (Conv2D) | Requires im2col transforms and significantly more complex backpropagation |
| Recurrent layers (LSTM, GRU) | Requires backpropagation through time (BPTT) and hidden state management |
| Mixed-precision quantization (int8/int4) | Research-grade complexity, separate project in itself |
| Sparse gradient updates (top-k selection) | Valuable optimization but not essential for proof-of-concept |
| Cross-entropy loss / Softmax | Classification support deferred — MSE regression is sufficient for demo |
| Live dashboard UI (Streamlit/Web) | Scope creep — a matplotlib plot from serial data is sufficient validation |
| PyTorch model import (`.pt` files) | Would require full ONNX/TorchScript parsing — far beyond scope |
| RTOS support | Generated code targets bare-metal only |

---

## 7. Validation Plan

### 7.1 Correctness Validation

1. **Numerical equivalence test:** Implement the same network in PyTorch (float32) and train it with generated C (int16 Q15) on the same synthetic dataset. Compare loss convergence curves. (Planned; PyTorch is not a runtime compiler dependency.)
   - **Pass criterion:** The MTC-generated model's final loss is within 2× of the PyTorch baseline (expected due to quantization noise).

2. **Gradient verification:** For each layer type, compare MTC's analytically derived gradients against PyTorch's `autograd` output on the same input tensors. Use a small tolerance (ε = 1e-3 for float, 2 LSB for Q15).

### 7.2 Resource Validation

3. **Memory budget compliance:** The compiler must report peak memory usage. Verify that the generated code for the demo network uses ≤64KB RAM by inspecting the linker map file (`.map`) after cross-compilation.

4. **Binary size compliance:** The compiled binary for the demo network must be ≤128KB Flash. Verify via the linker map.

### 7.3 On-Device Validation

5. **ESP32 training demo:** Flash the generated code to an ESP32 dev board. Run 100+ training epochs on synthetic data stored in Flash. Capture UART loss output via `pyserial`. Plot the convergence curve to demonstrate that the model is learning.

### 7.4 ★ Range Analysis Validation (Research Contribution)

6. **Overflow prediction accuracy:** Create a test suite of 10+ network configurations with varying depths, widths, learning rates, and input ranges. For each:
   - Run the MTC range analysis to get predicted overflow/safe status per layer.
   - Run actual Q15 training (on desktop in emulated Q15 or on ESP32) and record whether overflow actually occurs.
   - Compute **true positive rate** (correctly predicted overflows), **false positive rate** (flagged as overflow but actually safe), and **false negative rate** (missed real overflows — must be 0%).
   - **Pass criterion:** Zero false negatives. False positive rate ≤30%.

7. **Clipping effectiveness:** For networks flagged as overflow-prone, compare:
   - Training **without** clipping (expected: divergence)
   - Training **with** compiler-inserted clipping (expected: convergence, possibly slower)
   - Training with **manually reduced learning rate** (expected: convergence)

### 7.5 ★ Compiled vs. Runtime Benchmark

8. **MTC vs. AIfES comparison:** Train the same 3-layer network on the same ESP32 using both MTC-generated code and AIfES library. Measure:
   - Binary size (Flash footprint)
   - Peak RAM usage
   - CPU cycles per training epoch
   - Time to convergence (epochs to reach target loss)
   - Final training accuracy

---

## 8. Constraints & Non-Functional Requirements

### 8.1 Generated Code Safety
- All loop indices must use bounded iteration counts derived from compile-time tensor dimensions. No unbounded loops.
- Integer arithmetic must use saturating operations to prevent overflow-induced undefined behavior.
- No pointer arithmetic beyond array indexing. No casts between unrelated types.

### 8.2 Compiler Robustness
- The Python compiler must reject invalid model configurations with actionable error messages rather than crashing.
- All compiler passes must be deterministic — the same input must always produce the same output C code.

### 8.3 Maintainability
- Each compiler phase (parser, autodiff, memory planner, code emitter) is implemented as a separate Python module with a clean interface boundary.
- Adding a new layer type should require implementing a `Layer` trait (forward IR generation, backward IR generation, memory size estimation) without modifying the core pipeline.

---

## 9. Project Timeline (Estimated)

| Phase | Duration | Milestone |
|---|---|---|
| Literature review & architecture design | Weeks 1–3 | Architecture document, related work survey |
| Phase 1: Parser & IR | Weeks 4–6 | JSON → IR graph with validation |
| Phase 2: Autodiff engine | Weeks 7–9 | Forward + backward IR for all v1.0 layers |
| ★ Phase 2b: Range analysis | Weeks 10–12 | Interval arithmetic propagation, overflow detection, clipping insertion |
| Phase 3: Memory planner + quantizer | Weeks 13–15 | Static allocation, Q15 conversion |
| Phase 4: C code emitter | Weeks 16–18 | Compilable C output for ESP32 |
| Integration & on-device testing | Weeks 19–21 | End-to-end demo on real hardware |
| ★ Benchmark: MTC vs. AIfES | Weeks 21–22 | Comparative evaluation on ESP32 |
| Report writing & presentation prep | Weeks 23–26 | Final report, paper draft, demo video, defense prep |

---

## 10. Limitations & Honest Assessment

This section exists to set accurate expectations for evaluators.

1. **MTC is a proof-of-concept, not a production framework.** It demonstrates the feasibility of compile-time training code generation for a restricted class of neural networks (small, sequential, fully-connected).

2. **The supported network architecture is deliberately minimal.** Dense + ReLU + Sigmoid + MSE is sufficient to prove the core idea (compile-time autodiff → C code → on-device training). Broader layer support (Conv2D, Attention, etc.) would require significantly more engineering effort.

3. **Fixed-point quantization introduces accuracy loss.** The Q15 format provides ~4.5 decimal digits of precision. For the small networks targeted by MTC, this is acceptable, but deeper or wider networks would suffer from accumulated quantization error.

4. **MTC does not compete with MCUNetV3 or TinyEngine on performance.** Those systems use hand-optimized CMSIS-NN kernels and hardware-specific SIMD instructions. MTC generates portable C code that prioritizes correctness and readability over peak throughput.

5. **The "training" demonstrated is on small synthetic datasets stored in Flash.** Real-world on-device training from live sensor streams would require additional engineering (DMA buffering, real-time scheduling) that is outside the scope of this project.

---

## 11. References

1. Lin, J., Chen, W.-M., Cai, H., Gan, C., & Han, S. (2022). *MCUNetV3: On-Tiny-Device Training with Memory-Efficient Sparse Backpropagation.* NeurIPS 2022.
2. Cai, H., Gan, C., Zhu, L., & Han, S. (2020). *TinyTL: Reduce Memory, Not Parameters for Efficient On-Device Learning.* NeurIPS 2020.
3. Lin, J., Chen, W.-M., Lin, Y., Cohn, J., Gan, C., & Han, S. (2020). *MCUNet: Tiny Deep Learning on IoT Devices.* NeurIPS 2020.
4. Chen, T., Xu, B., Zhang, C., & Guestrin, C. (2016). *Training Deep Nets with Sublinear Memory Cost.* arXiv:1604.06174.
5. AIfES — Artificial Intelligence for Embedded Systems. Fraunhofer IMS. https://github.com/Fraunhofer-IMS/AIfES_for_Arduino
6. David, R., Duke, J., Jain, A., et al. (2021). *TensorFlow Lite Micro: Embedded Machine Learning for TinyML Systems.* MLSys 2021.
7. Yates, R. & Lyons, R. (2004). *Fixed-Point Arithmetic: An Introduction.* IEEE Signal Processing Magazine.
8. Moore, R. E., Kearfott, R. B., & Cloud, M. J. (2009). *Introduction to Interval Analysis.* SIAM. *(Foundation for the range analysis pass)*
9. Hascoet, L. & Pascual, V. (2013). *The Tapenade Automatic Differentiation Tool: Principles, Model, and Specification.* ACM TOMS. *(Source-to-source AD — does not perform numerical stability analysis)*
10. Moses, W. S. & Churavy, V. (2020). *Instead of Rewriting Foreign Code for Machine Learning, Automatically Synthesize Fast Gradients.* NeurIPS 2020. *(Enzyme — LLVM-level AD, no overflow prediction)*
11. Narodytska, N. et al. (2018). *Verifying Properties of Binarized Deep Neural Networks.* AAAI 2018. *(Formal verification of inference, not training)*

---

*This document describes a final-year academic project. All claims are scoped to what is achievable within the project timeline and team capabilities.*
