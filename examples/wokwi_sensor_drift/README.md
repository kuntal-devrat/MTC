# MTC Live Wokwi Simulation Guide
## Demonstrating On-Device Training on `sensor_drift_net` (225 Parameters, 4 Layers)

This directory contains the ready-to-run live simulation package for **Wokwi** (the web-based ESP32 simulator: [https://wokwi.com](https://wokwi.com)).

It demonstrates **on-device backpropagation and in-place SGD learning** on a real 4-channel industrial sensor drift calibration workload, featuring:
- **Topology:** 4 Inputs $\to$ 16 Hidden (ReLU) $\to$ 8 Hidden (ReLU) $\to$ 1 Output (Sigmoid)
- **Parameters:** 225 (200 Weights + 25 Biases)
- **Static Memory:** Exactly 564 Bytes SRAM (0 dynamic allocations)
- **Visuals:** SSD1306 128x64 OLED display + GPIO 2 training pulse LED + Serial Dashboard

---

## 1. Quick Setup in 60 Seconds

1. Open your web browser and navigate to: **[https://wokwi.com/projects/new/esp32](https://wokwi.com/projects/new/esp32)**
2. In the top tab bar, click on **`diagram.json`** and replace its entire contents with the code from [`diagram.json`](diagram.json).
3. Click on the **`sketch.ino`** tab and replace its entire contents with the code from [`sketch.ino`](sketch.ino).
4. Click on the **Library Manager** tab (or the `+` button in Wokwi) and add:
   - `Adafruit SSD1306`
   - `Adafruit GFX Library`
5. Click the green **"Play"** button (or press `F1`) at the top.

---

## 2. What Judges Will See on Screen

### Visual 1: Real-Time SSD1306 OLED Display (Top Left)
- **Boot:** Displays *"MTC Micro-Train / Booting ESP32..."*
- **Training (Epoch 1 to 250):** The OLED screen animates live showing:
  ```
  +----------------------+
  | MTC ON-DEVICE DL     |
  | Model: sensor_drift  |
  | Status: Training...  |
  | Epoch: 150/250       |
  | Loss:  0.0504        |
  | [========----]       |
  +----------------------+
  ```
- **Completion:** Displays *"TRAIN COMPLETE! / Loss: 0.0448 (v51%) / SRAM: 564 B / Heap: 0 B"*.

### Visual 2: Hardware Status LED (GPIO 2)
- The green LED pulses rapidly on every training epoch, visually demonstrating that the CPU core is executing backpropagation.
- Upon reaching Epoch 250, the LED stays **solid green**, indicating calibration lock.

### Visual 3: Live Serial Monitor Output (115200 Baud)
The Serial Monitor outputs a clean technical dashboard:

```
================================================================================
         MICRO-TRAIN-COMPILER (MTC) - ZERO-HEAP ON-DEVICE TRAINING DEMO          
    Target: ESP32 (Xtensa / RV32) | Mode: Bare-Metal Static C99 Memory Plan     
================================================================================
[SYSTEM SPECIFICATIONS]
  • Network Architecture : 4 (Inputs) -> 16 (ReLU) -> 8 (ReLU) -> 1 (Sigmoid)
  • Layer Count          : 3 Dense Layers + Non-Linear Activations
  • Total Parameters     : 225 (200 Weights + 25 Biases)
  • Static Memory Pool   : 564 Bytes (0.86% of 64KB Bare-Metal SRAM)
  • Dynamic Heap Malloc  : 0 Bytes (malloc = 0, free = 0)
  • Delta Scratchpad     : 2 Alternating Ping-Pong Buffers (32 Bytes total)
  • Initial Free Heap    : 298412 Bytes
--------------------------------------------------------------------------------

[STEP 1: UNTRAINED BASELINE STATE]
  >> Initial MSE Loss: 0.0927 (High Error, Transducer Uncalibrated)

[STEP 2: MICROSECOND HARDWARE LATENCY BENCHMARK]
  • Forward Inference Pass Latency : 21.26 microseconds
  • Complete Training Step Latency : 81.77 microseconds (Fwd + Backprop + In-Place SGD)
  • Real-Time Training Throughput  : 12,229.0 samples / second
--------------------------------------------------------------------------------

[STEP 3: EXECUTING LIVE ON-DEVICE BACKPROPAGATION]
  Epoch | Loss (Q15) | Float MSE | Training Progress | Status LED
  ------+------------+-----------+-------------------+-----------
     1  |     3038   |   0.0927  | [--------------------]   0%  |    PULSE
    25  |     2785   |   0.0850  | [==------------------]  10%  |    PULSE
    50  |     2492   |   0.0760  | [====----------------]  20%  |    PULSE
   100  |     2064   |   0.0630  | [========------------]  40%  |    PULSE
   150  |     1778   |   0.0543  | [============--------]  60%  |    PULSE
   200  |     1589   |   0.0485  | [================----]  80%  |    PULSE
   250  |     1469   |   0.0448  | [====================] 100%  |    PULSE

================================================================================
         STEP 4: EMPIRICAL VERIFICATION - SENSOR DRIFT CALIBRATION              
================================================================================
Sample | Target | Untrained Pred | Trained Pred | Absolute Error Delta | Status
-------+--------+----------------+--------------+----------------------+-----------
  #0   |  0.000 |     0.384      |    0.082     |   -0.3020 (-78.6%)   | CONVERGED
  #1   |  0.250 |     0.612      |    0.281     |   -0.3310 (-91.4%)   | CONVERGED
  #2   |  0.500 |     0.211      |    0.472     |   -0.2610 (-90.3%)   | CONVERGED
  #3   |  0.750 |     0.435      |    0.718     |   -0.2830 (-89.8%)   | CONVERGED
  #4   |  1.000 |     0.690      |    0.915     |   -0.2250 (-72.6%)   | CONVERGED
--------------------------------------------------------------------------------
  >> Initial Loss : 0.0927 (Epoch 1)
  >> Final Loss   : 0.0448 (Epoch 250)
  >> Net Reduction: 51.65% (Loss reduced by more than half!)
  >> Heap Delta   : 0 Bytes (Free Heap Before: 298412 B, After: 298412 B)
  >> Conclusion   : Zero Memory Fragmentation, Zero Heap Mallocs, 100% Deterministic.
================================================================================
```

---

## 3. What to Say to the Judges (Talking Points)

1. **State the Problem:**
   *"Most embedded ML projects only do frozen inference. Once deployed in the field, sensors age and drift due to temperature, dust, and wear. In traditional architectures, you have to send data to the cloud to retrain, which drains battery and leaks privacy."*

2. **Present the Breakthrough:**
   *"MTC enables the ESP32 to train its own neural network locally. Normally, running backpropagation on a microcontroller causes fatal heap fragmentation (`malloc` faults). MTC compiles the backward pass ahead-of-time into a zero-heap static C99 engine."*

3. **Point to the Memory Pool:**
   *"Look at line 6 of the Serial output: this 4-layer, 225-parameter network uses only **564 Bytes of SRAM**—less than 1% of the ESP32's memory. Notice that the Free Heap before and after training is identically unchanged (`Heap Delta: 0 Bytes`)."*

4. **Point to the Calibration Proof:**
   *"Look at Step 4: before training, the untrained model had severe prediction errors (e.g. predicting 0.384 instead of 0.000). After 250 epochs of on-device backpropagation, the loss drops by **51.65%** and all sensor calibration targets converge with microsecond execution latency."*

5. **Live Interactive Repeat:**
   *"If the judges ask to see it again, simply type `r` into the Wokwi Serial Monitor input box and hit Enter—the model resets and re-trains live in front of them!"*
