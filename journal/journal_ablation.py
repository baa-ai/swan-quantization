#!/usr/bin/env python3
"""
SmartQuant Ablation Study — Experiment 1.

Tests SmartQuant with 10 different metric weight configurations to prove
that the composite scoring design is sound. Reuses per-metric scores from
a base manifest, recomputing only the composite + bit decisions for each config.

Workflow per config:
  1. Load base manifest (with per-tensor scores)
  2. Recompute composite scores with new metric weights
  3. Re-derive bit decisions from new composites
  4. Write ablation manifest
  5. Convert model with mlx_lm.convert using ablation manifest
  6. Measure perplexity with mlx_lm.perplexity
  7. Clean up converted model to save disk

Also runs baselines: BF16, uniform-4bit, uniform-8bit.

Usage:
    python journal/journal_ablation.py \
        --bf16-model ~/smartquant/models/qwen3-8b-bf16 \
        --output-dir journal/results \
        [--base-manifest journal/analysis/qwen3-8b/manifest_base.json] \
        [--seq-len 2048] [--num-samples 256] [--seed 42]
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Ablation configurations ──────────────────────────────────────────────────
# Each config: (name, {metric_weights})
# Weights must sum to 1.0.
ABLATION_CONFIGS = [
    # Solo metrics (4)
    ("svd_only",         {"svd": 1.00, "kurtosis": 0.00, "output_sensitivity": 0.00, "cross_layer": 0.00}),
    ("kurtosis_only",    {"svd": 0.00, "kurtosis": 1.00, "output_sensitivity": 0.00, "cross_layer": 0.00}),
    ("output_only",      {"svd": 0.00, "kurtosis": 0.00, "output_sensitivity": 1.00, "cross_layer": 0.00}),
    ("position_only",    {"svd": 0.00, "kurtosis": 0.00, "output_sensitivity": 0.00, "cross_layer": 1.00}),
    # Select pairs (4)
    ("svd_kurtosis",     {"svd": 0.50, "kurtosis": 0.50, "output_sensitivity": 0.00, "cross_layer": 0.00}),
    ("svd_output",       {"svd": 0.50, "kurtosis": 0.00, "output_sensitivity": 0.50, "cross_layer": 0.00}),
    ("output_position",  {"svd": 0.00, "kurtosis": 0.00, "output_sensitivity": 0.50, "cross_layer": 0.50}),
    ("kurtosis_position",{"svd": 0.00, "kurtosis": 0.50, "output_sensitivity": 0.00, "cross_layer": 0.50}),
    # Composite variants (2)
    ("composite_default",{"svd": 0.30, "kurtosis": 0.20, "output_sensitivity": 0.30, "cross_layer": 0.20}),
    ("equal_weights",    {"svd": 0.25, "kurtosis": 0.25, "output_sensitivity": 0.25, "cross_layer": 0.25}),
]


def generate_base_manifest(bf16_model: Path, output_path: Path) -> dict:
    """Run SmartQuant analysis to generate the base manifest with per-metric scores."""
    logger.info(f"Generating base manifest for {bf16_model}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Import and run directly for better control
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from smartquant.config import QuantConfig
    from smartquant.shard_processor import process_all_shards

    config = QuantConfig()
    config.validate()

    manifest = process_all_shards(bf16_model, config, output_path)
    logger.info(f"Base manifest written to {output_path}")
    return manifest


def recompute_manifest(base_manifest: dict, weights: dict,
                       threshold_2bit: float = 0.10,
                       threshold_8bit: float = 0.65,
                       threshold_16bit: float = 0.90) -> dict:
    """Recompute composite scores and bit decisions from saved per-metric scores.

    This is the key efficiency trick: we don't re-run the expensive SVD/kurtosis
    analysis. We just reweight the already-computed scores.
    """
    manifest = deepcopy(base_manifest)
    manifest["config"]["metric_weights"] = weights

    bits_counts = {2: 0, 4: 0, 8: 0, 16: 0}
    total_params = 0

    for shard_data in manifest["shards"].values():
        for tensor_name, info in shard_data["tensors"].items():
            num_params = info["num_params"]
            total_params += num_params
            scores = info["scores"]

            # Recompute composite — handle both v1 (cross_layer) and v2 (error_proxy) keys
            composite = sum(weights.get(k, 0) * scores.get(k, 0) for k in weights)
            info["composite_score"] = composite

            # Re-derive bit decision
            classification = info.get("classification", "default")
            if classification == "protected":
                bits, reason = 16, "protected_pattern"
            elif info["decision"].get("reason") == "1d_tensor":
                bits, reason = 16, "1d_tensor"
            elif composite >= threshold_16bit:
                bits, reason = 16, f"composite_score={composite:.3f} >= 16bit_threshold"
            elif classification == "sensitive":
                bits, reason = 8, "sensitive_pattern"
            elif composite >= threshold_8bit:
                bits, reason = 8, f"composite_score={composite:.3f} >= 8bit_threshold"
            elif composite <= threshold_2bit:
                bits, reason = 2, f"composite_score={composite:.3f} <= 2bit_threshold"
            else:
                bits, reason = 4, "default"

            info["decision"] = {"bits": bits, "reason": reason}
            bits_counts[bits] = bits_counts.get(bits, 0) + num_params

    # Update summary
    from smartquant.utils import estimate_quantized_size
    manifest["summary"]["bits_distribution"] = {
        str(b): {"params": c, "percentage": c / total_params * 100 if total_params else 0}
        for b, c in sorted(bits_counts.items())
    }
    manifest["summary"]["average_bits"] = (
        sum(b * c for b, c in bits_counts.items()) / total_params if total_params else 0
    )
    manifest["summary"]["estimated_size_gb"] = estimate_quantized_size(bits_counts)

    return manifest


def convert_model(bf16_path: Path, mlx_path: Path, manifest_path: Path,
                  dtype: str = "float16") -> bool:
    """Convert model using SmartQuant manifest. Returns True on success."""
    if mlx_path.exists():
        shutil.rmtree(mlx_path)

    cmd = [
        sys.executable, str(Path(__file__).resolve().parent.parent / "convert_model.py"),
        "--hf-path", str(bf16_path),
        "--mlx-path", str(mlx_path),
        "--manifest", str(manifest_path),
        "--dtype", dtype,
    ]
    logger.info(f"Converting: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        logger.error(f"Conversion failed: {result.stderr[-2000:]}")
        return False
    return True


def convert_uniform(bf16_path: Path, mlx_path: Path, bits: int,
                    dtype: str = "float16") -> bool:
    """Convert model with uniform quantization (all tensors same bits)."""
    if mlx_path.exists():
        shutil.rmtree(mlx_path)

    cmd = [
        sys.executable, "-m", "mlx_lm.convert",
        "--hf-path", str(bf16_path),
        "--mlx-path", str(mlx_path),
        "--quantize",
        "--q-bits", str(bits),
        "--dtype", dtype,
    ]
    logger.info(f"Uniform-{bits}bit convert: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        logger.error(f"Uniform conversion failed: {result.stderr[-2000:]}")
        return False
    return True


def measure_perplexity(model_path: Path, seq_len: int = 2048,
                       num_samples: int = 256, seed: int = 42) -> dict:
    """Run mlx_lm.perplexity and parse output. Returns dict with ppl and metadata."""
    cmd = [
        sys.executable, "-m", "mlx_lm.perplexity",
        "--model", str(model_path),
        "--sequence-length", str(seq_len),
        "--num-samples", str(num_samples),
        "--seed", str(seed),
    ]
    logger.info(f"Perplexity: {' '.join(cmd)}")
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    elapsed = time.time() - start

    if result.returncode != 0:
        logger.error(f"Perplexity failed: {result.stderr[-2000:]}")
        return {"perplexity": None, "error": result.stderr[-500:], "elapsed_seconds": elapsed}

    # Parse perplexity from output. mlx_lm.perplexity prints lines like:
    # "Perplexity: 6.1234, ..."  or  final line with the number
    output = result.stdout + result.stderr
    ppl = _parse_perplexity(output)

    return {
        "perplexity": ppl,
        "elapsed_seconds": elapsed,
        "raw_output": output[-2000:],
    }


def _parse_perplexity(output: str) -> float:
    """Parse perplexity value from mlx_lm.perplexity output."""
    import re
    # Look for patterns like "ppl = 6.123" or "Perplexity: 6.123"
    for pattern in [
        r"ppl\s*[=:]\s*([\d.]+)",
        r"[Pp]erplexity\s*[=:]\s*([\d.]+)",
        r"final ppl\s*[=:]\s*([\d.]+)",
    ]:
        match = re.search(pattern, output)
        if match:
            return float(match.group(1))

    # Fallback: look for last float on a line containing "ppl" or "perplexity"
    for line in reversed(output.strip().split("\n")):
        if "ppl" in line.lower() or "perplexity" in line.lower():
            floats = re.findall(r"([\d]+\.[\d]+)", line)
            if floats:
                return float(floats[-1])

    logger.warning("Could not parse perplexity from output")
    return None


def get_model_size_gb(model_path: Path) -> float:
    """Get total size of safetensors files in GB."""
    total = sum(f.stat().st_size for f in model_path.glob("*.safetensors"))
    return total / (1024 ** 3)


def run_ablation(args):
    """Main ablation study execution."""
    bf16_model = Path(args.bf16_model).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model name for output files
    model_name = args.model_name or bf16_model.name

    # Work directory for temporary converted models
    work_dir = Path(args.work_dir).expanduser().resolve() if args.work_dir else output_dir.parent / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    analysis_dir = output_dir.parent / "analysis" / model_name
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Generate or load base manifest ────────────────────────────
    base_manifest_path = Path(args.base_manifest).expanduser().resolve() if args.base_manifest else (
        analysis_dir / "manifest_base.json"
    )

    if base_manifest_path.exists() and not args.force_reanalyze:
        logger.info(f"Loading existing base manifest: {base_manifest_path}")
        with open(base_manifest_path) as f:
            base_manifest = json.load(f)
    else:
        base_manifest = generate_base_manifest(bf16_model, base_manifest_path)

    total_tensors = base_manifest.get("total_tensors", 0)
    logger.info(f"Base manifest: {total_tensors} tensors, "
                f"avg {base_manifest['summary']['average_bits']:.2f} bits")

    # ── Step 2: Run ablation configs ──────────────────────────────────────
    results = {
        "model": str(bf16_model),
        "model_name": model_name,
        "base_manifest": str(base_manifest_path),
        "perplexity_params": {
            "sequence_length": args.seq_len,
            "num_samples": args.num_samples,
            "seed": args.seed,
        },
        "configs": {},
        "baselines": {},
    }

    for config_name, weights in ABLATION_CONFIGS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Ablation config: {config_name}")
        logger.info(f"Weights: {weights}")
        logger.info(f"{'='*60}")

        # Recompute manifest
        ablation_manifest = recompute_manifest(base_manifest, weights)
        manifest_path = analysis_dir / f"manifest_{config_name}.json"
        with open(manifest_path, "w") as f:
            json.dump(ablation_manifest, f, indent=2)

        avg_bits = ablation_manifest["summary"]["average_bits"]
        est_gb = ablation_manifest["summary"]["estimated_size_gb"]
        logger.info(f"Config {config_name}: avg {avg_bits:.2f} bits, est {est_gb:.1f} GB")

        # Bit distribution summary
        dist = ablation_manifest["summary"]["bits_distribution"]
        for b in sorted(dist.keys(), key=int):
            logger.info(f"  {b}-bit: {dist[b]['percentage']:.1f}%")

        # Convert
        mlx_path = work_dir / f"{model_name}-ablation-{config_name}"
        success = convert_model(bf16_model, mlx_path, manifest_path)
        if not success:
            results["configs"][config_name] = {
                "weights": weights,
                "average_bits": avg_bits,
                "estimated_size_gb": est_gb,
                "perplexity": None,
                "error": "conversion_failed",
            }
            continue

        actual_size = get_model_size_gb(mlx_path)

        # Measure perplexity
        ppl_result = measure_perplexity(
            mlx_path, seq_len=args.seq_len,
            num_samples=args.num_samples, seed=args.seed,
        )

        results["configs"][config_name] = {
            "weights": weights,
            "average_bits": avg_bits,
            "estimated_size_gb": est_gb,
            "actual_size_gb": actual_size,
            "bits_distribution": dist,
            "perplexity": ppl_result["perplexity"],
            "ppl_elapsed_seconds": ppl_result["elapsed_seconds"],
        }

        logger.info(f"Config {config_name}: ppl={ppl_result['perplexity']}, "
                     f"avg_bits={avg_bits:.2f}, size={actual_size:.1f} GB")

        # Cleanup converted model to save disk
        if not args.keep_models:
            logger.info(f"Cleaning up {mlx_path}")
            shutil.rmtree(mlx_path, ignore_errors=True)

        # Save intermediate results
        _save_results(results, output_dir / f"ablation_{model_name}.json")

    # ── Step 3: Run baselines ─────────────────────────────────────────────
    if not args.skip_baselines:
        # BF16 baseline (perplexity on original model)
        logger.info(f"\n{'='*60}")
        logger.info("Baseline: BF16 (original model)")
        logger.info(f"{'='*60}")
        ppl_bf16 = measure_perplexity(
            bf16_model, seq_len=args.seq_len,
            num_samples=args.num_samples, seed=args.seed,
        )
        bf16_size = get_model_size_gb(bf16_model)
        results["baselines"]["bf16"] = {
            "perplexity": ppl_bf16["perplexity"],
            "size_gb": bf16_size,
            "bits": 16,
            "elapsed_seconds": ppl_bf16["elapsed_seconds"],
        }
        _save_results(results, output_dir / f"ablation_{model_name}.json")

        # Uniform 4-bit baseline
        logger.info(f"\n{'='*60}")
        logger.info("Baseline: Uniform 4-bit")
        logger.info(f"{'='*60}")
        mlx_path_u4 = work_dir / f"{model_name}-uniform-4bit"
        if convert_uniform(bf16_model, mlx_path_u4, bits=4):
            ppl_u4 = measure_perplexity(
                mlx_path_u4, seq_len=args.seq_len,
                num_samples=args.num_samples, seed=args.seed,
            )
            results["baselines"]["uniform_4bit"] = {
                "perplexity": ppl_u4["perplexity"],
                "size_gb": get_model_size_gb(mlx_path_u4),
                "bits": 4,
                "elapsed_seconds": ppl_u4["elapsed_seconds"],
            }
            if not args.keep_models:
                shutil.rmtree(mlx_path_u4, ignore_errors=True)
        _save_results(results, output_dir / f"ablation_{model_name}.json")

        # Uniform 8-bit baseline
        logger.info(f"\n{'='*60}")
        logger.info("Baseline: Uniform 8-bit")
        logger.info(f"{'='*60}")
        mlx_path_u8 = work_dir / f"{model_name}-uniform-8bit"
        if convert_uniform(bf16_model, mlx_path_u8, bits=8):
            ppl_u8 = measure_perplexity(
                mlx_path_u8, seq_len=args.seq_len,
                num_samples=args.num_samples, seed=args.seed,
            )
            results["baselines"]["uniform_8bit"] = {
                "perplexity": ppl_u8["perplexity"],
                "size_gb": get_model_size_gb(mlx_path_u8),
                "bits": 8,
                "elapsed_seconds": ppl_u8["elapsed_seconds"],
            }
            if not args.keep_models:
                shutil.rmtree(mlx_path_u8, ignore_errors=True)
        _save_results(results, output_dir / f"ablation_{model_name}.json")

    # ── Final output ──────────────────────────────────────────────────────
    _save_results(results, output_dir / f"ablation_{model_name}.json")
    _print_summary(results)
    return results


def _save_results(results: dict, path: Path):
    """Save results JSON with timestamp."""
    results["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {path}")


def _print_summary(results: dict):
    """Print a summary table of all ablation results."""
    logger.info(f"\n{'='*80}")
    logger.info("ABLATION STUDY SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"{'Config':<20} {'Avg Bits':>10} {'Size (GB)':>10} {'Perplexity':>12}")
    logger.info(f"{'-'*20} {'-'*10} {'-'*10} {'-'*12}")

    # Baselines
    for name, data in sorted(results.get("baselines", {}).items()):
        ppl = data.get("perplexity")
        ppl_str = f"{ppl:.4f}" if ppl else "FAILED"
        logger.info(f"[BASE] {name:<14} {data.get('bits', '?'):>10} "
                     f"{data.get('size_gb', 0):>10.1f} {ppl_str:>12}")

    # Ablation configs
    for name, data in results.get("configs", {}).items():
        ppl = data.get("perplexity")
        ppl_str = f"{ppl:.4f}" if ppl else "FAILED"
        avg_bits = data.get("average_bits", 0)
        size = data.get("actual_size_gb", data.get("estimated_size_gb", 0))
        logger.info(f"{name:<20} {avg_bits:>10.2f} {size:>10.1f} {ppl_str:>12}")


def main():
    parser = argparse.ArgumentParser(description="SmartQuant Ablation Study")
    parser.add_argument("--bf16-model", required=True,
                        help="Path to BF16 model directory")
    parser.add_argument("--output-dir", default="journal/results",
                        help="Output directory for results JSON")
    parser.add_argument("--base-manifest",
                        help="Path to existing base manifest (skip re-analysis)")
    parser.add_argument("--model-name",
                        help="Model name for output files (default: directory name)")
    parser.add_argument("--work-dir",
                        help="Working directory for temporary converted models")
    parser.add_argument("--seq-len", type=int, default=2048,
                        help="Sequence length for perplexity (default: 2048)")
    parser.add_argument("--num-samples", type=int, default=256,
                        help="Number of samples for perplexity (default: 256)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for perplexity (default: 42)")
    parser.add_argument("--force-reanalyze", action="store_true",
                        help="Force re-running sensitivity analysis")
    parser.add_argument("--keep-models", action="store_true",
                        help="Don't delete converted models after perplexity")
    parser.add_argument("--skip-baselines", action="store_true",
                        help="Skip BF16/uniform baseline perplexity runs")
    parser.add_argument("--configs", nargs="*",
                        help="Run only these config names (default: all)")
    args = parser.parse_args()
    run_ablation(args)


if __name__ == "__main__":
    main()
