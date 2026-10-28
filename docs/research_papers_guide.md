# Micro-Train-Compiler (MTC)
## Curated Research Paper Reading & Review Guide (20 Papers)

**Academic Session:** B.Tech 5th Sem, 2026–27 Session  
**Institution:** G. L. Bajaj Group of Institutions, Mathura  
**Department:** Department of Computer Science & Engineering  
**Project Title:** Micro-Train-Compiler (MTC): A Source-to-Source Compiler for On-Device Neural Network Training on Microcontrollers  

---

## 1. Student Paper Allocation Matrix

Each student is assigned **5 dedicated research papers** tailored to their difficulty level. All papers are open-access with direct PDF/arXiv links so they can be read immediately without paywalls.

| Student Name | Roll Number | Track Difficulty | Core Focus Area | Assigned Papers |
| :--- | :--- | :--- | :--- | :--- |
| **Ajay Sisodia** | `2405111530012` | **Foundational / Accessible** | TinyML Fundamentals, Embedded Inference & MCU Hardware | Papers 1 – 5 |
| **Harsh Chaudhary** | `2405111530067` | **Foundational / Accessible** | Quantization, INT8/Q15 Arithmetic & Applied Edge Training | Papers 6 – 10 |
| **Devrat Kuntal** | `2405111530055` | **Advanced / Algorithmic** | On-Device Backpropagation, Activation Memory & Pruning | Papers 11 – 15 |
| **Aryan Upadhyay** | `2405111530037` | **Advanced / Compilers** | Deep Learning Compilers, Reverse Autodiff & Interval Safety | Papers 16 – 20 |

---

## 2. Standard Student Review Template (4 Mandatory Sections)

For each paper, each student should write their review following this exact 4-part structure:

```markdown
### Paper Review: [Paper Title]
**Authors:** [Author List] | **Venue & Year:** [e.g., MLSys 2021] | **Link:** [arXiv URL]

#### 1. Summary (Executive Overview)
- What real-world problem does the paper address?
- Why is existing software/hardware insufficient?
- What is the primary contribution or proposed system?

#### 2. Workflow & System Architecture
- Step-by-step pipeline from input data/model to execution.
- Key modules/components in the architecture.
- Draw or describe the end-to-end dataflow.

#### 3. Methodology & Core Algorithms
- What mathematical formulations, loss functions, or algorithms are used?
- How is memory, compute, or precision handled?
- Any specific hardware optimizations or constraints?

#### 4. Conclusion & Key Takeaways for MTC
- What were the primary quantitative results (SRAM saved, speedup, accuracy)?
- What are the main limitations acknowledged by the authors?
- How does this paper directly inform or inspire the Micro-Train-Compiler project?
```

---

## 3. Detailed Paper Breakdown by Student

---

### Track A: Foundational & Applied TinyML (Student 1 & Student 2)

---

### Student 1: Ajay Sisodia (Roll No: 2405111530012)
*Focus: TinyML Systems Architecture, Microcontroller Inference Engines, Benchmarking & Edge Deployment*

#### Paper 1: TensorFlow Lite Micro: Embedded Machine Learning on TinyML Systems
- **Authors:** Robert David, Jared Duke, Advait Jain, Vijay Janapa Reddi, Pete Warden, et al.
- **Venue:** MLSys 2021
- **Direct Link:** [https://arxiv.org/abs/2010.08678](https://arxiv.org/abs/2010.08678)
- **Difficulty:** Accessible / Systems Overview
- **Why Read:** It is the industry-standard runtime for running ML on microcontrollers (Arm Cortex-M, ESP32).
- **What to Look For:**
  - *Summary:* Why mobile TF Lite cannot run on microcontrollers without eliminating dynamic memory (`malloc`) and OS dependencies.
  - *Workflow:* Model flatbuffer parsing $\rightarrow$ static tensor arena allocation $\rightarrow$ interpreter kernel execution.
  - *Methodology:* Single contiguous memory pool allocator (`AllOpsResolver`, `MicroInterpreter`).
  - *Conclusion / MTC Connection:* TFLM only does *inference*; MTC bridges the missing half: *on-device training*.

#### Paper 2: Tiny Machine Learning: Progress and Futures
- **Authors:** Ji Lin, Ligeng Zhu, Wei-Ming Chen, Wei-Chung Wang, Song Han
- **Venue:** IEEE Circuits and Systems Magazine / arXiv, 2024
- **Direct Link:** [https://arxiv.org/abs/2403.19076](https://arxiv.org/abs/2403.19076)
- **Direct PDF:** [https://arxiv.org/pdf/2403.19076.pdf](https://arxiv.org/pdf/2403.19076.pdf)
- **Companion Reference:** Partha Pratim Ray, *A Review on TinyML: State-of-the-Art and Prospects*, Journal of King Saud University, 2022 ([ScienceDirect Open Access](https://doi.org/10.1016/j.jksuci.2021.11.019))
- **Difficulty:** Accessible / Literature Survey
- **Why Read:** Provides the broadest, clearest high-level picture of the entire TinyML landscape, hardware classes, and toolchains.
- **What to Look For:**
  - *Summary:* Definition of TinyML, memory boundaries (< 1MB Flash, < 512KB SRAM), energy profiles.
  - *Workflow:* Sensor data acquisition $\rightarrow$ local edge intelligence $\rightarrow$ actuation without cloud roundtrip.
  - *Methodology:* Categorization of hardware platforms (MCU vs DSP vs NPU) and software frameworks.
  - *Conclusion / MTC Connection:* Highlights that edge training is the "holy grail" of TinyML future directions.

#### Paper 3: CMSIS-NN: Efficient Neural Network Kernels for Arm Cortex-M CPUs
- **Authors:** Liangzhen Lai, Naveen Suda, Vikas Chandra (Arm Research)
- **Venue:** arXiv preprint, 2018
- **Direct Link:** [https://arxiv.org/abs/1801.06601](https://arxiv.org/abs/1801.06601)
- **Difficulty:** Accessible / Embedded Optimization
- **Why Read:** Shows how bare-metal C functions perform matrix multiplication and convolutions on resource-constrained microcontrollers.
- **What to Look For:**
  - *Summary:* Achieving maximum MACs (multiply-accumulate) per cycle on integer-only MCUs without FPU.
  - *Workflow:* Fixed-point Q7 and Q15 matrix multiplication using SIMD instructions.
  - *Methodology:* Fixed-point arithmetic: $y = \text{SAT}((\sum W \cdot x) \gg \text{shift})$.
  - *Conclusion / MTC Connection:* MTC synthesizes standalone C99 code inspired by CMSIS-NN fixed-point Q15 kernels.

#### Paper 4: Benchmarking TinyML Systems: Challenges and Directions
- **Authors:** Colby Banbury, Vijay Janapa Reddi, Max Lam, William Fu, et al.
- **Venue:** arXiv preprint / MLPerf Tiny, 2020
- **Direct Link:** [https://arxiv.org/abs/2003.04821](https://arxiv.org/abs/2003.04821)
- **Difficulty:** Accessible / Empirical Evaluation
- **Why Read:** Explains how TinyML systems are benchmarked for latency, memory, and energy consumption.
- **What to Look For:**
  - *Summary:* Why standard ML benchmarks (MLPerf) fail for microcontrollers; the need for MLPerf Tiny.
  - *Workflow:* 4 core TinyML tasks: Keyword Spotting (KWS), Visual Wake Words (VWW), Anomaly Detection (AD), Image Classification (IC).
  - *Methodology:* Metrics: Peak SRAM watermark, Flash binary size, latency in milliseconds, and energy (mJ) per inference.
  - *Conclusion / MTC Connection:* Sets the evaluation standard for our ESP32 benchmarks against AIfES.

#### Paper 5: Edge Intelligence: Paving the Last Mile of Artificial Intelligence with Edge Computing
- **Authors:** Zhi Zhou, Xu Chen, En Li, Liekang Zeng, Ke Luo, Junshan Zhang
- **Venue:** Proceedings of the IEEE, 2019
- **Direct Link:** [https://arxiv.org/abs/1905.10083](https://arxiv.org/abs/1905.10083)
- **Difficulty:** Accessible / Edge Computing Survey
- **Why Read:** Broad conceptual foundation on edge vs cloud AI trade-offs, privacy, and communication bottlenecks.
- **What to Look For:**
  - *Summary:* Taxonomy of Edge Intelligence: AI *on* Edge vs AI *for* Edge.
  - *Workflow:* Edge-cloud collaboration, bandwidth constraints, and privacy-preserving computing.
  - *Methodology:* Latency-energy-accuracy trade-offs; transmission cost vs computation cost on embedded nodes.
  - *Conclusion / MTC Connection:* Explains why local on-device learning avoids catastrophic privacy breaches and battery drain.

---

### Student 2: Harsh Chaudhary (Roll No: 2405111530067)
*Focus: Quantization-Aware Training, Fixed-Point Arithmetic, Model Compression & Edge Adaptation*

#### Paper 6: Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference
- **Authors:** Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, et al. (Google)
- **Venue:** IEEE CVPR 2018
- **Direct Link:** [https://arxiv.org/abs/1712.05877](https://arxiv.org/abs/1712.05877)
- **Difficulty:** Accessible / Foundational Quantization
- **Why Read:** The foundational paper establishing affine 8-bit integer quantization ($r = S(q - Z)$) for neural networks.
- **What to Look For:**
  - *Summary:* Converting floating-point math to pure integer arithmetic without accuracy loss.
  - *Workflow:* Floating point weights $\rightarrow$ Scale ($S$) and Zero-Point ($Z$) calibration $\rightarrow$ Integer-only matrix multiplication.
  - *Methodology:* Quantization equation: $r = S \cdot (q - Z)$; integer multiplication with bit-shifting: $M = 2^{-n} M_0$.
  - *Conclusion / MTC Connection:* Directly inspires MTC's Q15 fixed-point representation and scaling formulas.

#### Paper 7: A Survey of Model Compression and Acceleration for Deep Neural Networks
- **Authors:** Tejalal Choudhary, Vipul Kumar Mishra, Anurag Goswami, Jagannathan Sarangapani
- **Venue:** IEEE Access / arXiv, 2020
- **Direct Link:** [https://arxiv.org/abs/1710.09282](https://arxiv.org/abs/1710.09282)
- **Difficulty:** Accessible / Compression Overview
- **Why Read:** Covers the four primary pillars of model compression: parameter pruning, low-rank factorization, weight sharing, and quantization.
- **What to Look For:**
  - *Summary:* Why modern neural networks are heavily overparameterized and can be shrunk by $10\times$ without losing accuracy.
  - *Workflow:* Training $\rightarrow$ Weight sensitivity analysis $\rightarrow$ Pruning/Quantization $\rightarrow$ Retraining.
  - *Methodology:* Structured vs unstructured pruning; linear vs non-linear quantization.
  - *Conclusion / MTC Connection:* Explains how keeping the network small allows full forward-backward graphs to fit in ESP32's 520KB SRAM.

#### Paper 8: On-Device Machine Learning: An Algorithms and Learning Theory Perspective
- **Authors:** Sauptik Dhar, Junyao Guo, Jiayi Liu, Samarth Tripathi, Unmesh Kurup, Mohak Shah
- **Venue:** ACM / arXiv, 2019/2020
- **Direct Link:** [https://arxiv.org/abs/1911.00623](https://arxiv.org/abs/1911.00623)
- **Direct PDF:** [https://arxiv.org/pdf/1911.00623.pdf](https://arxiv.org/pdf/1911.00623.pdf)
- **Difficulty:** Accessible / State-of-the-Art Overview
- **Why Read:** Comprehensive taxonomy comparing cloud-centric training with on-device adaptation methods.
- **What to Look For:**
  - *Summary:* The necessity of continual edge learning for personalized AI and sensor drift compensation.
  - *Workflow:* Distributed federated updates vs fully autonomous local on-device adaptation.
  - *Methodology:* Memory bottlenecks of backpropagation (storing activations) and strategies to bypass full backprop.
  - *Conclusion / MTC Connection:* Formulates the core problem statement solved by MTC: training on the device, not in the cloud.

#### Paper 9: MicroNets: Neural Network Architectures for Deploying TinyML Applications
- **Authors:** Colby Banbury, Chuteng Zhou, Igor Fedorov, Rong-Ming Huang, et al.
- **Venue:** MLSys 2021
- **Direct Link:** [https://arxiv.org/abs/2010.11267](https://arxiv.org/abs/2010.11267)
- **Difficulty:** Accessible / Neural Architecture Design
- **Why Read:** Shows how to design extremely compact neural networks specifically tailored to fit microcontrollers.
- **What to Look For:**
  - *Summary:* Designing architectures that strictly honor MCU hardware limits (e.g., peak memory < 256KB).
  - *Workflow:* Neural architecture search (NAS) with hardware memory constraints in the loss function.
  - *Methodology:* Depthwise-separable convolutions, channel scaling, and operator-level memory profiling.
  - *Conclusion / MTC Connection:* Provides realistic model topologies to test in our MTC compiler (`examples/xor_net.json` and edge sensor classifiers).

#### Paper 10: Training Deep Neural Networks with Low Precision Multiplications
- **Authors:** Matthieu Courbariaux, Yoshua Bengio, Jean-Pierre David
- **Venue:** ICLR 2015
- **Direct Link:** [https://arxiv.org/abs/1412.7024](https://arxiv.org/abs/1412.7024)
- **Difficulty:** Accessible / Numerical Stability
- **Why Read:** Demonstrates that gradients and weights do *not* need 32-bit floating point to converge during backpropagation.
- **What to Look For:**
  - *Summary:* Training networks where forward and backward multiplications use low-precision fixed-point numbers.
  - *Workflow:* Forward pass with low precision $\rightarrow$ backward gradient calculation with low precision $\rightarrow$ accumulated weight update.
  - *Methodology:* Stochastic rounding vs deterministic rounding; fixed-point dynamic range scaling.
  - *Conclusion / MTC Connection:* Validates that MTC's Q15 fixed-point training can achieve convergence without requiring a hardware FPU.

---

### Track B: Advanced Systems, Compilers & Algorithms (Student 3 & Student 4)

---

### Student 3: Devrat Kuntal (Roll No: 2405111530055)
*Focus: On-Device Backpropagation, Memory-Efficient Training, Activation Management & Sublinear Systems*

#### Paper 11: MCUNetV3: On-Device Training Under 256KB Memory
- **Authors:** Ji Lin, Wei-Ming Chen, Han Cai, Chuang Gan, Song Han (MIT HAN Lab)
- **Venue:** NeurIPS 2022
- **Direct Link:** [https://arxiv.org/abs/2206.15472](https://arxiv.org/abs/2206.15472)
- **Difficulty:** Advanced / Edge Training Breakthrough
- **Why Read:** The benchmark paper that first proved full on-device training is possible on a 256KB SRAM microcontroller.
- **What to Look For:**
  - *Summary:* The "activation memory wall" of backpropagation; why PyTorch/TensorFlow cannot fit on microcontrollers.
  - *Workflow:* Quantization-Aware Scaling (QAS) $\rightarrow$ Sparse layer/tensor update $\rightarrow$ TinyEngine compile-time code generation.
  - *Methodology:* Updating only selective layers (sparse updates); compile-time graph pruning eliminating autodiff runtime overhead.
  - *Conclusion / MTC Connection:* Direct competitor and motivation for MTC; MTC extends this by adding formal compile-time interval range safety.

#### Paper 12: TinyTL: Reduce Memory, Not Parameters for Efficient On-Device Learning
- **Authors:** Han Cai, Chuang Gan, Ligeng Zhu, Song Han (MIT HAN Lab)
- **Venue:** NeurIPS 2020
- **Direct Link:** [https://arxiv.org/abs/2007.11622](https://arxiv.org/abs/2007.11622)
- **Difficulty:** Advanced / Algorithmic Efficiency
- **Why Read:** Proves mathematically that memory footprint during training is dominated by activations, not parameter storage.
- **What to Look For:**
  - *Summary:* Parameter-efficient transfer learning (updating only biases) drastically cuts memory by avoiding storing activation maps for weight gradients.
  - *Workflow:* Pre-trained frozen backbone $\rightarrow$ LiteResidual modules $\rightarrow$ Bias-only gradient updates.
  - *Methodology:* Memory breakdown: $M_{\text{train}} = M_{\text{param}} + M_{\text{activation}} + M_{\text{workspace}}$. LiteResidual reduces activation memory by $6.5\times$.
  - *Conclusion / MTC Connection:* Provides mathematical backing for MTC's static memory planner and activation lifetime buffer reuse.

#### Paper 13: MCUNet: Tiny Deep Learning on IoT Devices
- **Authors:** Ji Lin, Wei-Ming Chen, Yujun Lin, Chuang Gan, Song Han (MIT HAN Lab)
- **Venue:** NeurIPS 2020
- **Direct Link:** [https://arxiv.org/abs/2007.10319](https://arxiv.org/abs/2007.10319)
- **Difficulty:** Advanced / System-Algorithm Co-Design
- **Why Read:** Explains the design of TinyEngine, an embedded execution engine that eliminates dynamic memory interpreters.
- **What to Look For:**
  - *Summary:* Why interpreter-based runtimes (like TFLM) incur huge RAM overhead; the superiority of compile-time code generation.
  - *Workflow:* Two-stage co-design: TinyNAS (neural architecture search) + TinyEngine (compile-time code generator).
  - *Methodology:* Ahead-of-Time (AOT) static compilation: memory scheduling, in-place computation, loop unrolling.
  - *Conclusion / MTC Connection:* MTC adopts the exact same AOT compiler philosophy: emitting pure, standalone C99 code without any runtime engine.

#### Paper 14: POET: Training Neural Networks on Tiny Devices with Integrated Rematerialization and Paging
- **Authors:** Shishir G. Patil, Paras Jain, Prabal Dutta, Ion Stoica, Joseph E. Gonzalez (UC Berkeley)
- **Venue:** ICML 2022
- **Direct Link:** [https://arxiv.org/abs/2207.07697](https://arxiv.org/abs/2207.07697)
- **Direct PDF:** [https://arxiv.org/pdf/2207.07697.pdf](https://arxiv.org/pdf/2207.07697.pdf)
- **Difficulty:** Advanced / Memory Optimization
- **Why Read:** Explores how to mathematically schedule gradient computation when SRAM is smaller than the model's intermediate layers.
- **What to Look For:**
  - *Summary:* Formulating edge training memory management as a mathematical integer linear programming (ILP) optimization problem.
  - *Workflow:* Computation DAG $\rightarrow$ Energy/Memory profiling $\rightarrow$ Optimal schedule (rematerialization vs paging).
  - *Methodology:* Rematerialization (recomputing activations during backward pass rather than caching them in RAM) and paging to Flash.
  - *Conclusion / MTC Connection:* Informs MTC's static memory reuse algorithm: trade cheap forward recomputation to save precious SRAM.

#### Paper 15: Training Deep Nets with Sublinear Memory Cost
- **Authors:** Tianqi Chen, Bing Xu, Chiyuan Zhang, Carlos Guestrin
- **Venue:** arXiv preprint, 2016
- **Direct Link:** [https://arxiv.org/abs/1604.06174](https://arxiv.org/abs/1604.06174)
- **Difficulty:** Advanced / Foundational Theory
- **Why Read:** The seminal paper that introduced "gradient checkpointing", reducing training memory from $O(n)$ to $O(\sqrt{n})$.
- **What to Look For:**
  - *Summary:* Resolving the memory bottleneck of training by discarding intermediate activations and recomputing them on demand.
  - *Workflow:* Full forward graph $\rightarrow$ Selected checkpoint nodes kept in RAM $\rightarrow$ Discarded nodes re-evaluated during reverse pass.
  - *Methodology:* Graph partitioning algorithm; $O(\sqrt{n})$ memory complexity proof with a 30% compute overhead.
  - *Conclusion / MTC Connection:* Directly applicable to MTC when scaling from shallow MLPs (XOR net) to multi-layer deep networks on ESP32.

---

### Student 4: Aryan Upadhyay (Roll No: 2405111530037)
*Focus: Deep Learning Compilers, Intermediate Representations, Automatic Differentiation & Interval Verification*

#### Paper 16: TVM: An Automated End-to-End Optimizing Compiler for Deep Learning
- **Authors:** Tianqi Chen, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Yan, Haichen Shen, et al.
- **Venue:** USENIX OSDI 2018
- **Direct Link:** [https://arxiv.org/abs/1802.04799](https://arxiv.org/abs/1802.04799)
- **Difficulty:** Advanced / Compiler Architecture
- **Why Read:** The definitive modern deep learning compiler paper; shows how high-level computation graphs lower to bare-metal code.
- **What to Look For:**
  - *Summary:* Overcoming the gap between high-level frameworks (PyTorch, TensorFlow) and diverse hardware backends through multi-level IRs.
  - *Workflow:* High-level computational graph (Relay IR) $\rightarrow$ Operator-level tensor IR (TIR) $\rightarrow$ Hardware target code generation.
  - *Methodology:* Loop transformations, memory hierarchy scheduling, and automated code generation via cost models.
  - *Conclusion / MTC Connection:* MTC implements a specialized, lightweight equivalent: lowering JSON AST $\rightarrow$ `IrGraph` DAG $\rightarrow$ standalone ANSI C99.

#### Paper 17: Glow: An Optimizing Compiler for High-Performance Neural Network Hardware
- **Authors:** Nadav Rotem, Jordan Fix, Saleem Abdulrasool, Summer Deng, Roman Dzhabarov, et al. (Meta)
- **Venue:** arXiv preprint, 2018
- **Direct Link:** [https://arxiv.org/abs/1805.00907](https://arxiv.org/abs/1805.00907)
- **Difficulty:** Advanced / Compiler Optimization
- **Why Read:** Demonstrates two-level Intermediate Representations (High-Level DAG + Low-Level Linear IR) and memory reuse optimization.
- **What to Look For:**
  - *Summary:* A domain-specific neural network compiler providing static graph optimizations and hardware-independent lowering.
  - *Workflow:* Graph optimization (constant folding, dead code elimination) $\rightarrow$ Memory allocation $\rightarrow$ Machine code emission.
  - *Methodology:* Liveness analysis and static buffer allocation to minimize the lifetime and footprint of intermediate tensors.
  - *Conclusion / MTC Connection:* MTC's Phase 3 static memory planner directly mirrors Glow's liveness analysis and buffer overlay algorithms.

#### Paper 18: Automatic Differentiation in Machine Learning: A Survey
- **Authors:** Atilim Gunes Baydin, Barak A. Pearlmutter, Alexey Andreyevich Radul, Jeffrey Mark Siskind
- **Venue:** Journal of Machine Learning Research (JMLR), 2018
- **Direct Link:** [https://arxiv.org/abs/1502.05767](https://arxiv.org/abs/1502.05767)
- **Difficulty:** Advanced / Mathematical Foundations
- **Why Read:** The definitive tutorial on automatic differentiation (AD), rigorously distinguishing AD from symbolic and numerical differentiation.
- **What to Look For:**
  - *Summary:* Mathematical taxonomy of differentiation methods: numerical (finite differences), symbolic, and automatic (forward vs reverse mode).
  - *Workflow:* Primal computational evaluation trace (Wengert list) $\rightarrow$ Adjoint reverse propagation using the multivariate chain rule.
  - *Methodology:* Reverse-mode autodiff complexity: computing gradients of $f: \mathbb{R}^n \to \mathbb{R}$ with computational cost bounded by $O(1) \times \text{Cost}(f)$.
  - *Conclusion / MTC Connection:* MTC replaces dynamic runtime Wengert tapes with *compile-time analytical reverse-mode graph derivation*.

#### Paper 19: The Tapenade Automatic Differentiation Tool: Principles, Model, and Specification
- **Authors:** Laurent Hascoët, Valérie Pascual (Inria)
- **Venue:** ACM Transactions on Mathematical Software (TOMS), 2013
- **Direct Link (HAL Open Access):** [https://hal.science/hal-00813735/](https://hal.science/hal-00813735/)
- **Direct PDF (HAL):** [https://hal.science/hal-00813735/document](https://hal.science/hal-00813735/document)
- **Official DOI:** [https://doi.org/10.1145/2450153.2450158](https://doi.org/10.1145/2450153.2450158)
- **Companion Modern Compiler AD:** Enzyme (*Fast Gradients in LLVM*, NeurIPS 2020: [https://arxiv.org/abs/2010.01709](https://arxiv.org/abs/2010.01709))
- **Difficulty:** Advanced / Source-to-Source Autodiff
- **Why Read:** Explains how source-to-source compilers take existing code and synthesize derivative source code ahead of time.
- **What to Look For:**
  - *Summary:* Source-to-source transformation model for reverse automatic differentiation in Fortran and C.
  - *Workflow:* Abstract Syntax Tree (AST) parsing $\rightarrow$ Call graph & data-dependence analysis $\rightarrow$ Synthesizing adjoint derivative functions.
  - *Methodology:* Activity analysis (identifying which variables depend on parameters), checkpointing strategies, and dead-code elimination.
  - *Conclusion / MTC Connection:* MTC uses the source-to-source paradigm to emit exact C99 forward and backward loops (`mtc_forward()`, `mtc_backward()`).

#### Paper 20: Efficient Formal Safety Verification of Neural Networks via Interval Bound Propagation
- **Authors:** Sven Gowal, Krishnamurthy Dvijotham, Robert Stanforth, Rudy Bunel, Chongli Qin, Pushmeet Kohli, et al. (Google DeepMind)
- **Venue:** arXiv preprint, 2018
- **Direct Link:** [https://arxiv.org/abs/1810.12715](https://arxiv.org/abs/1810.12715)
- **Difficulty:** Advanced / Mathematical Verification
- **Why Read:** The foundational paper applying closed interval arithmetic to bound neural network activations and prove mathematical bounds.
- **What to Look For:**
  - *Summary:* Using Interval Bound Propagation (IBP) to establish rigorous upper and lower bounds on all hidden layer activations.
  - *Workflow:* Input interval $[x_{\min}, x_{\max}] \to$ Layer-by-layer interval propagation $\to$ Output safety certification.
  - *Methodology:* Interval linear algebra: $\underline{z} = W_+ \underline{x} + W_- \overline{x} + b$, $\overline{z} = W_+ \overline{x} + W_- \underline{x} + b$; monotonic activation rules.
  - *Conclusion / MTC Connection:* MTC adapts DeepMind's IBP to verify fixed-point Q15 stability: certifying that all tensor ranges stay strictly within $[-1.0, +1.0)$.

---

## 4. How to Use These Reviews in Progress Report & Capstone Defense

When your team completes these 20 summaries:
1. **Literature Survey Chapter:** Condense the 20 summaries into Section 2 of your Final Capstone Report to demonstrate a comprehensive, publication-grade bibliography.
2. **Design Justification:** When faculty evaluators ask *"Why did you use compile-time autodiff instead of a runtime engine?"*, Student 3 can cite **MCUNetV3 (Paper 11)** and Student 4 can cite **Tapenade (Paper 19)**.
3. **Hardware Constraints:** When asked *"How can you guarantee the ESP32 won't crash from an Out-of-Memory error?"*, Student 1 can cite **TFLM (Paper 1)** and Student 3 can cite **TinyTL (Paper 12)**.
4. **Numerical Stability:** When asked *"How do you prevent fixed-point overflow in Q15 arithmetic?"*, Student 2 can cite **Jacob et al. (Paper 6)** and Student 4 can cite **Gowal et al. (Paper 20)**.
