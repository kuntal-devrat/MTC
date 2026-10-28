# Wokwi ESP32 XOR Demo

This project is generated from `examples/xor_net.json` with the MTC `wokwi` target. It uses the same Q15 model C that the native runner uses; the Arduino sketch supplies the embedded samples, prints epoch loss over Serial, then prints each final prediction. With the checked-in seed and 1,000 epochs, the native C integration test verifies all four XOR classifications at a 0.5 threshold.

## Run in Wokwi

1. Open [Wokwi ESP32](https://wokwi.com/projects/new/esp32).
2. Replace `sketch.ino` and `diagram.json` with the files in this directory.
3. Add `mtc_model.c` and `mtc_model.h` as project files beside `sketch.ino`.
4. Start the simulation and open the Serial Monitor at 115200 baud.

Expected output begins with the model name and Q15 settings, followed by `epoch=... loss=...` lines, one `sample=... target=... output=...` line per example, and `training complete`.

## Regenerate

From `mtc-compiler/`:

```bash
mtc compile examples/xor_net.json --target wokwi --output examples/wokwi_esp32 --allow-clipping
```

`--allow-clipping` is required for this example because the static Q15 analysis predicts saturating operations. It enables saturating arithmetic; it does not guarantee training convergence or accuracy.