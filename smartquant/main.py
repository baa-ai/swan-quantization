"""
SmartQuant CLI entry point.

Usage:
    python -m smartquant analyze \
        --input-dir ~/smartquant/models/maverick-bf16 \
        --output ~/smartquant/analysis/manifest.json \
        --device cpu
"""

import argparse
import sys
from pathlib import Path

from .config import QuantConfig
from .shard_processor import discover_and_print_patterns, process_all_shards
from .utils import setup_logging


def cmd_analyze(args: argparse.Namespace) -> None:
    """Run sensitivity analysis on model weights."""
    log_file = Path.home() / "smartquant" / "logs" / "analysis.log"
    logger = setup_logging(log_file=log_file)

    config = QuantConfig(
        sensitivity_threshold_8bit=args.threshold_8bit,
        sensitivity_threshold_16bit=args.threshold_16bit,
        svd_rank=args.svd_rank,
        device=args.device,
    )
    config.validate()

    input_dir = Path(args.input_dir).expanduser()
    output_path = Path(args.output).expanduser()

    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    logger.info(f"SmartQuant Analysis")
    logger.info(f"  Input: {input_dir}")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Device: {args.device}")
    logger.info(f"  8-bit threshold: {args.threshold_8bit}")
    logger.info(f"  16-bit threshold: {args.threshold_16bit}")
    logger.info(f"  SVD rank: {args.svd_rank}")

    if args.discover_only:
        discover_and_print_patterns(input_dir)
        return

    manifest = process_all_shards(input_dir, config, output_path)

    logger.info("Analysis complete.")


def cmd_discover(args: argparse.Namespace) -> None:
    """Discover tensor name patterns without running full analysis."""
    logger = setup_logging()
    input_dir = Path(args.input_dir).expanduser()

    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    discover_and_print_patterns(input_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="smartquant",
        description="SmartQuant: Intelligent mixed-precision quantization analysis",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Run sensitivity analysis on model weights",
    )
    analyze_parser.add_argument(
        "--input-dir",
        required=True,
        help="Path to the BF16 model directory",
    )
    analyze_parser.add_argument(
        "--output",
        required=True,
        help="Path to write the analysis manifest JSON",
    )
    analyze_parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "mps"],
        help="Compute device for analysis (default: cpu)",
    )
    analyze_parser.add_argument(
        "--threshold-8bit",
        type=float,
        default=0.45,
        help="Composite score threshold for 8-bit quantization (default: 0.45)",
    )
    analyze_parser.add_argument(
        "--threshold-16bit",
        type=float,
        default=0.85,
        help="Composite score threshold for 16-bit preservation (default: 0.85)",
    )
    analyze_parser.add_argument(
        "--svd-rank",
        type=int,
        default=256,
        help="Rank for randomized SVD (default: 256)",
    )
    analyze_parser.add_argument(
        "--discover-only",
        action="store_true",
        help="Only discover tensor patterns, don't run full analysis",
    )
    analyze_parser.set_defaults(func=cmd_analyze)

    # discover command (shorthand)
    discover_parser = subparsers.add_parser(
        "discover",
        help="Discover tensor name patterns in a model",
    )
    discover_parser.add_argument(
        "--input-dir",
        required=True,
        help="Path to the model directory",
    )
    discover_parser.set_defaults(func=cmd_discover)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
