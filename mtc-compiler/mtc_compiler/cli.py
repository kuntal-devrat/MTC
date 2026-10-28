"""Command-line interface for the MTC compiler."""

import argparse
from pathlib import Path
from typing import Sequence

from .analysis import analyze_ranges
from .codegen import TARGET_RAM_BUDGETS, emit_project
from .config import ConfigError, load_model
from .ir import build_stages
from .memory_planner import plan_memory
from .validate import validate_model


def _add_model_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model", help="path to a JSON model definition")


def _print_analysis(model_path: str, target: str) -> None:
    model = load_model(model_path)
    validate_model(model)
    analysis = analyze_ranges(model, build_stages(model))
    budget = TARGET_RAM_BUDGETS[target]
    memory = plan_memory(model, build_stages(model), budget)
    print(f"Model: {model.name}")
    print(f"Target: {target} (static RAM budget {budget} bytes)")
    print(f"Planned static RAM: {memory.total_static_ram_bytes} bytes")
    print("Stage  Operation                  Range                         Status")
    for record in analysis.records:
        if record.overflows_q15:
            status = "OVERFLOW"
        elif record.near_overflow:
            status = "NEAR LIMIT"
        elif record.may_underflow:
            status = "UNDERFLOW RISK"
        else:
            status = "safe"
        print(
            f"{record.stage:5}  {record.operation:26} "
            f"[{record.interval.min:+.6f}, {record.interval.max:+.6f}]  {status}"
        )
    print(f"Maximum interval/Q15 ratio: {analysis.stability_score:.4f}")
    print(f"Predicted overflow ranges: {len(analysis.overflows)}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mtc", description="Micro-Train-Compiler")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate", help="validate a JSON model")
    _add_model_argument(validate_parser)

    analyze_parser = subparsers.add_parser(
        "analyze", help="report Q15 ranges and static RAM requirements"
    )
    _add_model_argument(analyze_parser)
    analyze_parser.add_argument(
        "--target", choices=TARGET_RAM_BUDGETS, default="esp32"
    )

    compile_parser = subparsers.add_parser(
        "compile", help="generate a standalone C99 training project"
    )
    _add_model_argument(compile_parser)
    compile_parser.add_argument("--target", choices=TARGET_RAM_BUDGETS, default="esp32")
    compile_parser.add_argument("--output", type=Path, default=Path("build/mtc"))
    compile_parser.add_argument(
        "--allow-clipping",
        action="store_true",
        help="allow predicted Q15 overflow; generated operations saturate at Q15 limits",
    )
    args = parser.parse_args(argv)

    try:
        model = load_model(args.model)
        validate_model(model)
        if args.command == "analyze":
            _print_analysis(args.model, args.target)
            return 0
        if args.command == "compile":
            compilation = emit_project(
                model,
                args.output,
                target=args.target,
                allow_clipping=args.allow_clipping,
            )
            print(f"Generated {len(compilation.files)} C99 project files in {args.output}")
            print(
                f"Static RAM: {compilation.memory.total_static_ram_bytes} / "
                f"{compilation.memory.budget_bytes} bytes"
            )
            print(f"Predicted Q15 overflow ranges: {len(compilation.analysis.overflows)}")
            if compilation.analysis.overflows:
                print("Overflowing operations use saturating Q15 arithmetic.")
            return 0
    except ConfigError as error:
        parser.exit(2, f"mtc: error: {error}\n")

    print(f"Model '{model.name}' is valid ({len(model.layers)} layers).")
    if model.training_data:
        print(f"Embedded training samples: {len(model.training_data)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
