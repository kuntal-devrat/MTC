#!/usr/bin/env python3
"""
Validation script: Read training loss from ESP32 UART serial output (or log file / stdin)
and plot the convergence curve.

Usage:
    # Live from serial:
    python plot_convergence.py --port /dev/ttyUSB0 --baud 115200 --output loss_curve.png

    # From file or stdin:
    python plot_convergence.py --file run.log --output loss_curve.png
    ./mtc_train | python plot_convergence.py --stdin --output loss_curve.png
"""

import argparse
import re
import sys
from pathlib import Path
from typing import TextIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPOCH_LOSS_REGEX = re.compile(
    r"epoch=(\d+)\s+(?:loss=([0-9.eE+-]+)|loss_q15=(-?\d+))"
)


def parse_stream(stream: TextIO) -> tuple[list[int], list[float]]:
    epochs: list[int] = []
    losses: list[float] = []

    for line in stream:
        line_str = line.strip()
        match = EPOCH_LOSS_REGEX.search(line_str)
        if match:
            epoch = int(match.group(1))
            if match.group(2) is not None:
                loss = float(match.group(2))
            else:
                loss = float(int(match.group(3))) / 32768.0
            epochs.append(epoch)
            losses.append(loss)
        if "training complete" in line_str.lower():
            break

    return epochs, losses


def read_serial(port: str, baud: int, timeout: float = 10.0) -> tuple[list[int], list[float]]:
    try:
        import serial
    except ImportError:
        print("Error: pyserial is required for --port. Install with: pip install pyserial", file=sys.stderr)
        sys.exit(1)

    print(f"Opening serial port {port} at {baud} baud...")
    epochs: list[int] = []
    losses: list[float] = []

    with serial.Serial(port, baudrate=baud, timeout=timeout) as ser:
        while True:
            raw_line = ser.readline()
            if not raw_line:
                break
            line_str = raw_line.decode("utf-8", errors="replace").strip()
            print(line_str)
            match = EPOCH_LOSS_REGEX.search(line_str)
            if match:
                epoch = int(match.group(1))
                if match.group(2) is not None:
                    loss = float(match.group(2))
                else:
                    loss = float(int(match.group(3))) / 32768.0
                epochs.append(epoch)
                losses.append(loss)
            if "training complete" in line_str.lower():
                break

    return epochs, losses


def plot_loss(epochs: list[int], losses: list[float], output_path: Path, title: str = "MTC Training Convergence") -> None:
    if not epochs:
        print("Warning: No epoch loss measurements parsed to plot.", file=sys.stderr)
        return

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, losses, color="#1f77b4", linewidth=1.8, label="MSE Loss (Q15 Normalized)")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (MSE)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)

    initial_loss = losses[0]
    final_loss = losses[-1]
    reduction = ((initial_loss - final_loss) / initial_loss) * 100.0 if initial_loss != 0 else 0.0

    plt.annotate(
        f"Initial: {initial_loss:.4f}\nFinal: {final_loss:.4f}\nReduction: {reduction:.1f}%",
        xy=(epochs[-1], final_loss),
        xytext=(0.70, 0.80),
        textcoords="axes fraction",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.5),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"),
    )
    plt.legend(loc="upper right")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Convergence plot saved to: {output_path.resolve()}")
    print(f"Summary: {len(epochs)} epochs | Initial Loss: {initial_loss:.6f} | Final Loss: {final_loss:.6f} ({reduction:.2f}% reduction)")


def main() -> None:
    parser = argparse.ArgumentParser(description="MTC Training Loss Plotter")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--port", help="Serial port (e.g. /dev/ttyUSB0, COM3)")
    input_group.add_argument("--file", type=Path, help="Log file containing serial or training output")
    input_group.add_argument("--stdin", action="store_true", help="Read training output from standard input")

    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    parser.add_argument("--output", type=Path, default=Path("loss_curve.png"), help="Output image file (default: loss_curve.png)")
    parser.add_argument("--title", type=str, default="MTC Training Convergence", help="Plot title")

    args = parser.parse_args()

    if args.port:
        epochs, losses = read_serial(args.port, args.baud)
    elif args.file:
        if not args.file.exists():
            print(f"Error: file not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8") as f:
            epochs, losses = parse_stream(f)
    elif args.stdin:
        epochs, losses = parse_stream(sys.stdin)
    else:
        parser.print_help()
        sys.exit(1)

    plot_loss(epochs, losses, args.output, title=args.title)


if __name__ == "__main__":
    main()
