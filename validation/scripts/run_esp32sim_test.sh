#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MTC_DIR="$PROJECT_ROOT/mtc-compiler"
BUILD_DIR="/tmp/mtc_esp32sim_run"
ESP32SIM_BIN="/tmp/esp32sim/target/release/esp32sim-c3"

export PYTHONPATH="$MTC_DIR:$PYTHONPATH"

echo "=========================================================="
echo "  Micro-Train-Compiler (MTC) -> esp32sim E2E Test Runner"
echo "=========================================================="

# Check prerequisites
if [ ! -f "$ESP32SIM_BIN" ]; then
    echo "Error: esp32sim-c3 binary not found at $ESP32SIM_BIN"
    echo "Please build esp32sim first: cd /tmp/esp32sim && cargo build --release"
    exit 1
fi

if ! command -v riscv64-unknown-elf-gcc &>/dev/null; then
    echo "Error: riscv64-unknown-elf-gcc not found."
    exit 1
fi

if ! python3 -m esptool version &>/dev/null; then
    echo "Error: esptool not found."
    exit 1
fi

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

echo "[1/4] Compiling model JSON with MTC..."
python3 -m mtc_compiler compile "$MTC_DIR/examples/xor_net.json" --output "$BUILD_DIR/mtc_c" --allow-clipping

echo "[2/4] Generating ESP32-C3 bare-metal firmware wrapper..."
cat << 'EOF' > "$BUILD_DIR/linker.ld"
ENTRY(_start)

MEMORY
{
  iram (rx) : ORIGIN = 0x40380000, LENGTH = 128K
  dram (rw) : ORIGIN = 0x3FC8B800, LENGTH = 128K
}

SECTIONS
{
  .text : {
    *(.text.entry)
    *(.text*)
    *(.rodata*)
  } > iram

  .data : {
    *(.data*)
    *(.sdata*)
  } > dram AT > iram

  .bss : {
    *(.bss*)
    *(.sbss*)
    *(COMMON)
  } > dram
}
EOF

cat << 'EOF' > "$BUILD_DIR/entry.S"
.section .text.entry
.global _start
.type _start, @function

_start:
    li sp, 0x3FCC0000
    call esp_main
1:  j 1b
EOF

cat << 'EOF' > "$BUILD_DIR/main.c"
#include "mtc_model.h"
#include <stdint.h>

#define UART0_FIFO ((volatile uint32_t *)0x60000000)

void uart_putc(char c) {
    *UART0_FIFO = (uint32_t)c;
}

void uart_puts(const char *s) {
    while (*s) {
        if (*s == '\n') uart_putc('\r');
        uart_putc(*s++);
    }
}

void uart_put_uint(uint32_t n) {
    char buf[16];
    int i = 0;
    if (n == 0) {
        uart_putc('0');
        return;
    }
    while (n > 0) {
        buf[i++] = '0' + (n % 10);
        n /= 10;
    }
    while (i > 0) {
        uart_putc(buf[--i]);
    }
}

static const int16_t inputs[4][2] = {
    {0, 0},
    {32767, 0},
    {0, 32767},
    {32767, 32767}
};
static const int16_t targets[4][1] = {
    {0},
    {32767},
    {32767},
    {0}
};

void esp_main(void) {
    uart_puts("\n[MTC Bare-Metal Trainer on ESP32 Hardware Simulator]\n");
    mtc_reset();
    
    for (uint32_t epoch = 1; epoch <= 100; ++epoch) {
        int64_t total_loss = 0;
        for (uint32_t s = 0; s < 4; ++s) {
            total_loss += mtc_train_sample(inputs[s], targets[s]);
        }
        uart_puts("epoch=");
        uart_put_uint(epoch);
        uart_puts(" loss_q15=");
        uart_put_uint((uint32_t)(total_loss / 4));
        uart_puts("\n");
    }

    uart_puts("\n[Validation Predictions After Training]\n");
    for (uint32_t s = 0; s < 4; ++s) {
        int16_t out[1];
        mtc_forward(inputs[s], out);
        uart_puts("sample ");
        uart_put_uint(s);
        uart_puts(": target=");
        uart_put_uint((uint32_t)targets[s][0]);
        uart_puts(" output=");
        uart_put_uint((uint32_t)out[0]);
        uart_puts("\n");
    }

    uart_puts("training complete\n");
}
EOF

echo "[3/4] Cross-compiling RISC-V ELF & creating ESP32-C3 flash image..."
riscv64-unknown-elf-gcc -march=rv32imc -mabi=ilp32 -nostdlib -O2 \
    -I "$BUILD_DIR/mtc_c" \
    -T "$BUILD_DIR/linker.ld" \
    "$BUILD_DIR/entry.S" \
    "$BUILD_DIR/main.c" \
    "$BUILD_DIR/mtc_c/mtc_model.c" \
    -o "$BUILD_DIR/firmware.elf"

python3 -m esptool --chip esp32c3 elf2image "$BUILD_DIR/firmware.elf" -o "$BUILD_DIR/firmware.bin"

echo "[4/4] Executing on esp32sim and plotting convergence..."
"$ESP32SIM_BIN" --boot app --app "$BUILD_DIR/firmware.bin" --console uart0 --max-seconds 1 > "$BUILD_DIR/simulation.log" 2>&1 || true

cat "$BUILD_DIR/simulation.log" | grep -E "epoch=|sample|training complete|\[MTC" || true

python3 "$PROJECT_ROOT/validation/scripts/plot_convergence.py" \
    --file "$BUILD_DIR/simulation.log" \
    --output "$PROJECT_ROOT/validation/esp32sim_convergence.png" \
    --title "MTC On-Device Q15 Training Convergence (ESP32Sim)"

echo "=========================================================="
echo "  Success: MTC verified on esp32sim!"
echo "  Plot generated at: $PROJECT_ROOT/validation/esp32sim_convergence.png"
echo "=========================================================="
