# Micro-Train-Compiler (MTC)

MTC is a Python-hosted compiler that turns a sequential neural-network definition and embedded training samples into a standalone C99 Q15 training program. The supported v1 model contains Dense layers with optional ReLU or Sigmoid activations, MSE loss, and SGD.

## Setup

```bash
cd mtc-compiler
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Python 3.10 or newer is required. The compiler itself uses only the standard library. On Windows, activate with `.venv\\Scripts\\activate`.

## Use

Validate a model and its embedded samples:

```bash
mtc validate examples/xor_net.json
```

Inspect conservative Q15 range diagnostics and static RAM use:

```bash
mtc analyze examples/xor_net.json --target esp32
```

Generate C sources and a CMake project:

```bash
mtc compile examples/xor_net.json --output build/xor --allow-clipping
cd build/xor
gcc -std=c99 -Wall -Wextra -Werror mtc_model.c mtc_train.c -o mtc_train
./mtc_train
```

Compilation rejects predicted Q15 overflow by default. `--allow-clipping` permits generation using saturating fixed-point operations and reports the predicted overflow ranges. It does not assert that saturation preserves model accuracy.

The JSON model includes `input_range`, `target_range`, `epochs`, `seed`, and `training_data`. Samples are compiled as constant data in the generated trainer. The seed makes Xavier initialization reproducible.

## Generated Project

`mtc compile` writes `mtc_model.h`, `mtc_model.c`, `mtc_train.c`, and `CMakeLists.txt`. The C code has fixed-size static storage, no heap allocation, generated forward/backward passes, SGD updates, and epoch loss output in Q15 units.

The compiler reports its computed static RAM use against a 64 KiB budget. The generated C is portable C99; ESP-IDF and ARM cross-compilers are not bundled. This workspace has been verified with host GCC, not flashed to physical hardware.

For an online ESP32 simulator, compile with `--target wokwi` and follow [the Wokwi example instructions](mtc-compiler/examples/wokwi_esp32/README.md).

## Tests

```bash
python -m unittest discover -s tests -v
```

The suite checks parsing, validation, interval analysis, memory planning, and compiles/runs both example projects with GCC when it is available.

## Project Structure

```text
mtc-compiler/
├── pyproject.toml
├── mtc_compiler/
│   ├── config.py          # JSON model and embedded sample parsing
│   ├── validate.py        # Dimensions and sample contract checks
│   ├── ir.py              # Normalized dense stages and tensor shapes
│   ├── analysis.py        # Q15 forward/backward interval propagation
│   ├── memory_planner.py  # Static activation and gradient-buffer offsets
│   ├── codegen.py         # Standalone C99 project generation
│   └── cli.py             # validate, analyze, and compile commands
├── examples/
└── tests/
```

## License

MIT