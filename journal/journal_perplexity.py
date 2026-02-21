#!/usr/bin/env python3
"""
SmartQuant Perplexity Measurement — Experiment 3.

Standardized wrapper around mlx_lm.perplexity for consistent measurements
across all model variants. Ensures identical parameters for fair comparison.

Usage:
    python journal/journal_perplexity.py \
        --model ~/smartquant/models/qwen3-8b-bf16 \
        --variant bf16 \
        --output-dir journal/results \
        [--model-name qwen3_8b] \
        [--seq-len 2048] [--num-samples 256] [--seed 42]

    # Batch mode: measure multiple models
    python journal/journal_perplexity.py \
        --batch batch_config.json \
        --output-dir journal/results
"""

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Standard parameters for all perplexity measurements
DEFAULT_SEQ_LEN = 2048
DEFAULT_NUM_SAMPLES = 256
DEFAULT_SEED = 42


def measure_perplexity(model_path: str, seq_len: int = DEFAULT_SEQ_LEN,
                       num_samples: int = DEFAULT_NUM_SAMPLES,
                       seed: int = DEFAULT_SEED) -> dict:
    """Run mlx_lm.perplexity with standardized parameters.

    Returns:
        Dict with perplexity value, timing, and metadata.
    """
    model_path = str(Path(model_path).expanduser().resolve())

    cmd = [
        sys.executable, "-m", "mlx_lm.perplexity",
        "--model", model_path,
        "--sequence-length", str(seq_len),
        "--num-samples", str(num_samples),
        "--seed", str(seed),
    ]

    logger.info(f"Running: {' '.join(cmd)}")
    start = time.time()

    result = subprocess.run(
        cmd, capture_output=True, text=True,
        timeout=14400,  # 4 hour timeout for large models
    )
    elapsed = time.time() - start

    output = result.stdout + "\n" + result.stderr

    if result.returncode != 0:
        logger.error(f"Perplexity measurement failed (exit code {result.returncode})")
        logger.error(f"stderr: {result.stderr[-2000:]}")
        return {
            "perplexity": None,
            "error": result.stderr[-1000:],
            "elapsed_seconds": elapsed,
            "raw_output": output[-3000:],
        }

    ppl = _parse_perplexity(output)

    if ppl is None:
        logger.warning("Could not parse perplexity from output")
        logger.info(f"Raw output:\n{output[-2000:]}")

    return {
        "perplexity": ppl,
        "elapsed_seconds": elapsed,
        "raw_output": output[-3000:],
    }


def _parse_perplexity(output: str) -> float:
    """Parse perplexity value from mlx_lm.perplexity output.

    Handles various output formats:
      - "ppl = 6.1234, ..."
      - "Perplexity: 6.1234"
      - Lines with running ppl updates (take the last one)
    """
    # Try specific patterns first
    for pattern in [
        r"ppl\s*[=:]\s*([\d]+\.[\d]+)",
        r"[Pp]erplexity\s*[=:]\s*([\d]+\.[\d]+)",
        r"final\s+ppl\s*[=:]\s*([\d]+\.[\d]+)",
    ]:
        matches = re.findall(pattern, output)
        if matches:
            return float(matches[-1])  # Take last match (final value)

    # Fallback: find last line with "ppl" and extract float
    for line in reversed(output.strip().split("\n")):
        line_lower = line.lower()
        if "ppl" in line_lower or "perplexity" in line_lower:
            floats = re.findall(r"([\d]+\.[\d]+)", line)
            if floats:
                return float(floats[-1])

    return None


def get_model_info(model_path: str) -> dict:
    """Get basic model info: size, config details."""
    model_dir = Path(model_path).expanduser().resolve()
    info = {"path": str(model_dir)}

    # Total safetensors size
    safetensor_files = list(model_dir.glob("*.safetensors"))
    total_bytes = sum(f.stat().st_size for f in safetensor_files)
    info["size_gb"] = total_bytes / (1024 ** 3)
    info["num_shards"] = len(safetensor_files)

    # Read config.json for model details
    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        info["model_type"] = config.get("model_type", "unknown")
        info["num_hidden_layers"] = config.get("num_hidden_layers")
        info["hidden_size"] = config.get("hidden_size")
        info["vocab_size"] = config.get("vocab_size")

        # Check quantization config
        quant_config = config.get("quantization", config.get("quantization_config", {}))
        if quant_config:
            info["quantization"] = quant_config

    return info


def run_single(args):
    """Run perplexity for a single model."""
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = args.model_name or Path(args.model).expanduser().name
    variant = args.variant or "unknown"

    logger.info(f"Measuring perplexity: {model_name} ({variant})")
    logger.info(f"Parameters: seq_len={args.seq_len}, num_samples={args.num_samples}, seed={args.seed}")

    model_info = get_model_info(args.model)
    ppl_result = measure_perplexity(
        args.model, seq_len=args.seq_len,
        num_samples=args.num_samples, seed=args.seed,
    )

    results = {
        "model_name": model_name,
        "variant": variant,
        "model_info": model_info,
        "parameters": {
            "sequence_length": args.seq_len,
            "num_samples": args.num_samples,
            "seed": args.seed,
        },
        "perplexity": ppl_result["perplexity"],
        "elapsed_seconds": ppl_result["elapsed_seconds"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    if ppl_result.get("error"):
        results["error"] = ppl_result["error"]

    output_path = output_dir / f"perplexity_{model_name}_{variant}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_path}")
    if ppl_result["perplexity"] is not None:
        logger.info(f"Perplexity: {ppl_result['perplexity']:.4f}")
        logger.info(f"Model size: {model_info.get('size_gb', 0):.1f} GB")
        logger.info(f"Time: {ppl_result['elapsed_seconds']:.0f}s")
    else:
        logger.error("Perplexity measurement failed")

    return results


def run_batch(args):
    """Run perplexity for multiple models from a batch config file.

    Batch config JSON format:
    {
        "models": [
            {"path": "~/path/to/model", "name": "qwen3_8b", "variant": "bf16"},
            {"path": "~/path/to/model", "name": "qwen3_8b", "variant": "smartquant"},
            ...
        ]
    }
    """
    with open(args.batch) as f:
        batch_config = json.load(f)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for entry in batch_config["models"]:
        model_path = entry["path"]
        model_name = entry.get("name", Path(model_path).expanduser().name)
        variant = entry.get("variant", "unknown")

        logger.info(f"\n{'='*60}")
        logger.info(f"Measuring: {model_name} ({variant})")
        logger.info(f"{'='*60}")

        model_info = get_model_info(model_path)
        ppl_result = measure_perplexity(
            model_path, seq_len=args.seq_len,
            num_samples=args.num_samples, seed=args.seed,
        )

        result = {
            "model_name": model_name,
            "variant": variant,
            "model_info": model_info,
            "parameters": {
                "sequence_length": args.seq_len,
                "num_samples": args.num_samples,
                "seed": args.seed,
            },
            "perplexity": ppl_result["perplexity"],
            "elapsed_seconds": ppl_result["elapsed_seconds"],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        all_results.append(result)

        # Save individual result
        individual_path = output_dir / f"perplexity_{model_name}_{variant}.json"
        with open(individual_path, "w") as f:
            json.dump(result, f, indent=2)

    # Save combined results
    combined = {
        "parameters": {
            "sequence_length": args.seq_len,
            "num_samples": args.num_samples,
            "seed": args.seed,
        },
        "results": all_results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    combined_path = output_dir / "perplexity_comparison.json"
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)

    # Print comparison table
    logger.info(f"\n{'='*70}")
    logger.info("PERPLEXITY COMPARISON")
    logger.info(f"{'='*70}")
    logger.info(f"{'Model':<25} {'Variant':<15} {'PPL':>10} {'Size (GB)':>10} {'Time (s)':>10}")
    logger.info(f"{'-'*25} {'-'*15} {'-'*10} {'-'*10} {'-'*10}")
    for r in all_results:
        ppl = r["perplexity"]
        ppl_str = f"{ppl:.4f}" if ppl else "FAILED"
        size = r["model_info"].get("size_gb", 0)
        logger.info(f"{r['model_name']:<25} {r['variant']:<15} {ppl_str:>10} "
                     f"{size:>10.1f} {r['elapsed_seconds']:>10.0f}")

    return combined


def main():
    parser = argparse.ArgumentParser(description="SmartQuant Perplexity Measurement")
    parser.add_argument("--model",
                        help="Path to model directory (single mode)")
    parser.add_argument("--variant",
                        help="Model variant name (e.g., bf16, smartquant, uniform_4bit)")
    parser.add_argument("--model-name",
                        help="Model name for output files")
    parser.add_argument("--batch",
                        help="Path to batch config JSON (batch mode)")
    parser.add_argument("--output-dir", default="journal/results",
                        help="Output directory for results")
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN,
                        help=f"Sequence length (default: {DEFAULT_SEQ_LEN})")
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES,
                        help=f"Number of samples (default: {DEFAULT_NUM_SAMPLES})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed (default: {DEFAULT_SEED})")
    args = parser.parse_args()

    if args.batch:
        run_batch(args)
    elif args.model:
        run_single(args)
    else:
        parser.error("Either --model or --batch must be specified")


if __name__ == "__main__":
    main()
