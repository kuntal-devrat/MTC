# Wokwi XOR Demo (Micro-Train-Compiler)

This directory contains the Wokwi simulation files for training `xor_net` on an ESP32.

## Files
- `diagram.json`: Wokwi ESP32 board configuration.
- `sketch.ino`: Modular Arduino sketch (uses `mtc_model.h` and `mtc_model.c`).
- `sketch_single_file.ino`: Self-contained monolithic sketch for 1-click copy-paste into Wokwi.
- `mtc_model.h`: Static memory definitions and C99 prototypes.
- `mtc_model.c`: Zero-heap static C99 forward/backward/SGD engine.

## Quick Wokwi Run
1. Go to https://wokwi.com/projects/new/esp32
2. Paste `sketch_single_file.ino` into the code editor.
3. Click the green **Play** button (▶️).
4. Watch the serial monitor output live loss decay!
