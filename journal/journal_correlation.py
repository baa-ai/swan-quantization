#!/usr/bin/env python3
"""
SmartQuant Correlation Study — Experiment 2.

Validates that SmartQuant's sensitivity metrics actually predict quantization
degradation. For every 2D+ tensor in a BF16 model:

  1. Compute SmartQuant's 4 per-metric scores + composite
  2. Simulate 4-bit quantization (group-wise RTN) and measure reconstruction error
  3. Compute Spearman & Pearson correlations (each metric vs error, composite vs error)
  4. Compute 4x4 inter-metric correlation matrix (prove metrics are non-redundant)

Memory-efficient: loads one safetensor shard at a time.

Usage:
    python journal/journal_correlation.py \
        --model ~/smartquant/models/qwen3-8b-bf16 \
        --output-dir journal/results \
        [--model-name qwen3_8b]
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def simulate_quant_error(tensor: torch.Tensor, bits: int = 4,
                         group_size: int = 128) -> float:
    """Simulate group-wise round-to-nearest (RTN) quantization and return NRMSE.

    This is the ground truth: how much does quantize-then-dequantize change the tensor?
    Uses the same approach as MLX's quantization kernel.

    Returns:
        Normalized root mean square error (RMSE / RMS of original tensor).
    """
    t = tensor.float()
    original_shape = t.shape

    # Flatten to 2D: [rows, cols]
    if t.dim() == 1:
        t = t.unsqueeze(0)
    if t.dim() == 3:
        # Packed experts: reshape to 2D
        t = t.reshape(-1, t.shape[-1])
    elif t.dim() > 3:
        t = t.reshape(t.shape[0], -1)

    rows, cols = t.shape
    # Pad cols to multiple of group_size
    if cols % group_size != 0:
        pad = group_size - (cols % group_size)
        t = torch.nn.functional.pad(t, (0, pad))
        cols = t.shape[1]

    # Reshape into groups: [rows, num_groups, group_size]
    num_groups = cols // group_size
    t_grouped = t.reshape(rows, num_groups, group_size)

    # Per-group min/max
    g_min = t_grouped.min(dim=-1, keepdim=True).values
    g_max = t_grouped.max(dim=-1, keepdim=True).values

    # Quantization levels
    n_levels = (1 << bits) - 1  # e.g., 15 for 4-bit
    scale = (g_max - g_min) / n_levels
    scale = scale.clamp(min=1e-12)

    # Quantize (round-to-nearest)
    quantized = torch.round((t_grouped - g_min) / scale).clamp(0, n_levels)

    # Dequantize
    dequantized = quantized * scale + g_min

    # Trim padding and compute error
    dequantized = dequantized.reshape(rows, cols)
    t_flat = t.reshape(rows, cols)

    # Only compare non-padded region
    orig_cols = tensor.reshape(-1, tensor.shape[-1]).shape[-1] if tensor.dim() >= 2 else tensor.shape[-1]
    diff = dequantized[:, :orig_cols] - t_flat[:, :orig_cols]

    rmse = diff.pow(2).mean().sqrt().item()
    rms_orig = t_flat[:, :orig_cols].pow(2).mean().sqrt().item()

    if rms_orig < 1e-12:
        return 0.0

    return rmse / rms_orig  # Normalized RMSE


def compute_raw_scores(tensor: torch.Tensor, layer_idx=None, total_layers=None) -> dict:
    """Compute raw (unclamped) metric values for correlation analysis.

    The main SensitivityAnalyzer clamps scores to [0,1] which can cause saturation
    (e.g., output_sensitivity=1.0 for all tensors in small models). For correlation
    analysis we need the full dynamic range.
    """
    t = tensor.float()
    if t.dim() == 3:
        # Sample experts
        n = t.shape[0]
        if n <= 4:
            t = t.reshape(-1, t.shape[-1])
        else:
            indices = torch.linspace(0, n - 1, 4).long()
            t = t[indices].reshape(-1, t.shape[-1])
    elif t.dim() > 3:
        t = t.reshape(t.shape[0], -1)

    raw = {}

    # SVD: raw concentration ratio (before normalization)
    try:
        rows, cols = t.shape
        k = min(256, min(rows, cols))
        if min(rows, cols) <= 512:
            _, s, _ = torch.linalg.svd(t, full_matrices=False)
        else:
            _, s, _ = torch.svd_lowrank(t, q=k)
        s = s.abs()
        total_energy = (s ** 2).sum().item()
        if total_energy > 1e-12:
            top_k = max(1, len(s) // 10)
            top_energy = (s[:top_k] ** 2).sum().item()
            raw["svd_concentration"] = top_energy / total_energy
        else:
            raw["svd_concentration"] = 0.0
    except Exception:
        raw["svd_concentration"] = 0.0

    # Kurtosis: raw excess kurtosis
    flat = t.flatten().float()
    mean = flat.mean()
    centered = flat - mean
    var = (centered ** 2).mean()
    if var > 1e-12:
        std = var.sqrt()
        raw["kurtosis_raw"] = ((centered / std) ** 4).mean().item() - 3.0
    else:
        raw["kurtosis_raw"] = 0.0

    # Output sensitivity: raw relative delta (before /0.1 normalization)
    if t.dim() >= 2:
        rows, cols = t.shape[0], t.shape[1]
        num_probes = min(32, cols)
        x = torch.randn(cols, num_probes, device=t.device, dtype=t.dtype)
        x = x / (x.norm(dim=0, keepdim=True) + 1e-8)
        y_clean = t @ x
        w_range = t.max() - t.min()
        noise_scale = w_range / 16.0
        noise = torch.rand_like(t) * noise_scale - noise_scale / 2
        y_noisy = (t + noise) @ x
        delta = (y_noisy - y_clean).norm() / (y_clean.norm() + 1e-8)
        raw["output_delta"] = delta.item()
    else:
        raw["output_delta"] = 0.0

    # Cross-layer: raw U-score
    if layer_idx is not None and total_layers is not None and total_layers > 1:
        pos = layer_idx / (total_layers - 1)
        raw["position_u_score"] = 4.0 * (pos - 0.5) ** 2
        raw["layer_position"] = pos
    else:
        raw["position_u_score"] = 0.5
        raw["layer_position"] = 0.5

    return raw


def run_correlation(args):
    """Main correlation study execution."""
    model_dir = Path(args.model).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = args.model_name or model_dir.name

    # Load model index
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        # Single-shard model
        single_shard = model_dir / "model.safetensors"
        if single_shard.exists():
            weight_map = None  # Handle below
        else:
            logger.error(f"No model index or single shard found at {model_dir}")
            return
    else:
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})

    # Detect total layers
    if weight_map:
        all_names = list(weight_map.keys())
    else:
        # Single shard: enumerate tensor names
        with safe_open(str(model_dir / "model.safetensors"), framework="pt") as f:
            all_names = list(f.keys())

    layer_indices = set()
    for name in all_names:
        match = re.search(r"layers\.(\d+)", name)
        if match:
            layer_indices.add(int(match.group(1)))
    total_layers = max(layer_indices) + 1 if layer_indices else 0

    # Set up SmartQuant analyzer
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from smartquant.analyzer import SensitivityAnalyzer
    from smartquant.config import QuantConfig
    from smartquant.utils import classify_tensor, is_1d_tensor

    config = QuantConfig()
    analyzer = SensitivityAnalyzer(config)

    # Group tensors by shard
    if weight_map:
        shard_tensors = {}
        for tensor_name, shard_name in weight_map.items():
            shard_tensors.setdefault(shard_name, []).append(tensor_name)
    else:
        shard_tensors = {"model.safetensors": all_names}

    # ── Process each tensor ───────────────────────────────────────────────
    tensor_records = []
    start_time = time.time()
    processed = 0
    skipped = 0

    for shard_name in sorted(shard_tensors.keys()):
        tensor_names = shard_tensors[shard_name]
        shard_path = model_dir / shard_name

        logger.info(f"Processing shard {shard_name} ({len(tensor_names)} tensors)")

        with safe_open(str(shard_path), framework="pt") as f:
            for tensor_name in tensor_names:
                tensor = f.get_tensor(tensor_name)
                shape = tuple(tensor.shape)

                # Skip 1D tensors (norms, biases) and tiny tensors
                if is_1d_tensor(shape) or tensor.numel() < config.min_tensor_params:
                    skipped += 1
                    del tensor
                    continue

                # Skip protected patterns (embeddings, lm_head, etc.)
                classification = classify_tensor(
                    tensor_name, config.protected_patterns, config.sensitive_patterns
                )
                if classification == "protected":
                    skipped += 1
                    del tensor
                    continue

                # Extract layer index
                layer_match = re.search(r"layers\.(\d+)", tensor_name)
                layer_idx = int(layer_match.group(1)) if layer_match else None

                # Compute SmartQuant scores
                analysis = analyzer.analyze_tensor(
                    tensor, name=tensor_name,
                    layer_idx=layer_idx, total_layers=total_layers,
                )

                if analysis.get("skipped"):
                    skipped += 1
                    del tensor
                    continue

                # Compute reconstruction error
                quant_error = simulate_quant_error(tensor, bits=4, group_size=128)

                # Compute raw (unclamped) scores for better correlation analysis
                raw_scores = compute_raw_scores(tensor, layer_idx, total_layers)

                record = {
                    "name": tensor_name,
                    "shape": list(shape),
                    "num_params": tensor.numel(),
                    "classification": classification,
                    "layer_idx": layer_idx,
                    "scores": analysis["scores"],
                    "raw_scores": raw_scores,
                    "composite_score": analysis["composite_score"],
                    "quant_error_4bit": quant_error,
                }
                tensor_records.append(record)
                processed += 1

                if processed % 50 == 0:
                    logger.info(f"  Processed {processed} tensors, skipped {skipped}")

                del tensor

    elapsed = time.time() - start_time
    logger.info(f"Processed {processed} tensors in {elapsed:.1f}s (skipped {skipped})")

    # ── Compute correlations ──────────────────────────────────────────────
    if len(tensor_records) < 5:
        logger.error("Too few tensors for meaningful correlation analysis")
        return

    from scipy import stats

    # Extract arrays
    composites = np.array([r["composite_score"] for r in tensor_records])
    errors = np.array([r["quant_error_4bit"] for r in tensor_records])
    svd_scores = np.array([r["scores"]["svd"] for r in tensor_records])
    kurt_scores = np.array([r["scores"]["kurtosis"] for r in tensor_records])
    output_scores = np.array([r["scores"]["output_sensitivity"] for r in tensor_records])
    position_scores = np.array([r["scores"]["cross_layer"] for r in tensor_records])

    metric_arrays = {
        "svd": svd_scores,
        "kurtosis": kurt_scores,
        "output_sensitivity": output_scores,
        "cross_layer": position_scores,
        "composite": composites,
    }

    # Metric vs error correlations
    metric_vs_error = {}
    for metric_name, scores in metric_arrays.items():
        # Filter out constant arrays (would cause NaN correlation)
        if np.std(scores) < 1e-10 or np.std(errors) < 1e-10:
            metric_vs_error[metric_name] = {
                "spearman_r": 0.0, "spearman_p": 1.0,
                "pearson_r": 0.0, "pearson_p": 1.0,
                "note": "constant_array",
            }
            continue

        sp_r, sp_p = stats.spearmanr(scores, errors)
        pe_r, pe_p = stats.pearsonr(scores, errors)
        metric_vs_error[metric_name] = {
            "spearman_r": float(sp_r),
            "spearman_p": float(sp_p),
            "pearson_r": float(pe_r),
            "pearson_p": float(pe_p),
        }
        logger.info(f"  {metric_name:>20s} vs error: "
                     f"Spearman r={sp_r:.4f} (p={sp_p:.2e}), "
                     f"Pearson r={pe_r:.4f} (p={pe_p:.2e})")

    # Inter-metric correlation matrix (4x4)
    metric_names_4 = ["svd", "kurtosis", "output_sensitivity", "cross_layer"]
    inter_metric = {"metrics": metric_names_4, "spearman": [], "pearson": []}

    for i, mi in enumerate(metric_names_4):
        sp_row, pe_row = [], []
        for j, mj in enumerate(metric_names_4):
            ai, aj = metric_arrays[mi], metric_arrays[mj]
            if np.std(ai) < 1e-10 or np.std(aj) < 1e-10:
                sp_row.append(0.0)
                pe_row.append(0.0)
            else:
                sp_r, _ = stats.spearmanr(ai, aj)
                pe_r, _ = stats.pearsonr(ai, aj)
                sp_row.append(float(sp_r))
                pe_row.append(float(pe_r))
        inter_metric["spearman"].append(sp_row)
        inter_metric["pearson"].append(pe_row)

    logger.info("\nInter-metric Spearman correlation matrix:")
    header = f"{'':>20s} " + " ".join(f"{m:>12s}" for m in metric_names_4)
    logger.info(header)
    for i, mi in enumerate(metric_names_4):
        row = f"{mi:>20s} " + " ".join(f"{v:>12.4f}" for v in inter_metric["spearman"][i])
        logger.info(row)

    # Raw score correlations (unclamped, better for analysis)
    raw_metric_vs_error = {}
    raw_names = {
        "svd_concentration": np.array([r["raw_scores"]["svd_concentration"] for r in tensor_records]),
        "kurtosis_raw": np.array([r["raw_scores"]["kurtosis_raw"] for r in tensor_records]),
        "output_delta": np.array([r["raw_scores"]["output_delta"] for r in tensor_records]),
        "position_u_score": np.array([r["raw_scores"]["position_u_score"] for r in tensor_records]),
    }
    logger.info("\nRaw (unclamped) metric vs error correlations:")
    for metric_name, scores in raw_names.items():
        if np.std(scores) < 1e-10:
            raw_metric_vs_error[metric_name] = {
                "spearman_r": 0.0, "spearman_p": 1.0,
                "pearson_r": 0.0, "pearson_p": 1.0,
                "note": "constant_array",
            }
            logger.info(f"  {metric_name:>20s} vs error: CONSTANT (no variance)")
            continue
        sp_r, sp_p = stats.spearmanr(scores, errors)
        pe_r, pe_p = stats.pearsonr(scores, errors)
        raw_metric_vs_error[metric_name] = {
            "spearman_r": float(sp_r), "spearman_p": float(sp_p),
            "pearson_r": float(pe_r), "pearson_p": float(pe_p),
            "mean": float(np.mean(scores)), "std": float(np.std(scores)),
            "min": float(np.min(scores)), "max": float(np.max(scores)),
        }
        logger.info(f"  {metric_name:>20s} vs error: "
                     f"Spearman r={sp_r:.4f} (p={sp_p:.2e}), "
                     f"Pearson r={pe_r:.4f} (p={pe_p:.2e}), "
                     f"range=[{np.min(scores):.4f}, {np.max(scores):.4f}]")

    # Score distribution statistics
    score_stats = {}
    for metric_name, scores in metric_arrays.items():
        score_stats[metric_name] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "median": float(np.median(scores)),
            "p25": float(np.percentile(scores, 25)),
            "p75": float(np.percentile(scores, 75)),
        }

    error_stats = {
        "mean": float(np.mean(errors)),
        "std": float(np.std(errors)),
        "min": float(np.min(errors)),
        "max": float(np.max(errors)),
        "median": float(np.median(errors)),
        "p25": float(np.percentile(errors, 25)),
        "p75": float(np.percentile(errors, 75)),
    }

    # ── Save results ──────────────────────────────────────────────────────
    results = {
        "model": str(model_dir),
        "model_name": model_name,
        "total_layers": total_layers,
        "tensors_analyzed": processed,
        "tensors_skipped": skipped,
        "analysis_time_seconds": elapsed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "metric_vs_error": metric_vs_error,
        "raw_metric_vs_error": raw_metric_vs_error,
        "inter_metric_correlation": inter_metric,
        "score_statistics": score_stats,
        "error_statistics": error_stats,
        "tensor_records": tensor_records,
    }

    output_path = output_dir / f"correlation_{model_name}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    # ── Print summary ─────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("CORRELATION STUDY SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Tensors analyzed: {processed}")
    logger.info(f"\nMetric vs Reconstruction Error:")
    logger.info(f"{'Metric':>20s} {'Spearman r':>12s} {'p-value':>12s} {'Pearson r':>12s} {'p-value':>12s}")
    for metric_name, corr in metric_vs_error.items():
        logger.info(f"{metric_name:>20s} {corr['spearman_r']:>12.4f} {corr['spearman_p']:>12.2e} "
                     f"{corr['pearson_r']:>12.4f} {corr['pearson_p']:>12.2e}")

    # Verification checks
    comp_sp = metric_vs_error.get("composite", {}).get("spearman_p", 1.0)
    logger.info(f"\nVerification: Composite Spearman p-value = {comp_sp:.2e} "
                f"({'PASS' if comp_sp < 0.001 else 'FAIL'}: p < 0.001)")

    max_off_diag = 0
    for i in range(4):
        for j in range(4):
            if i != j:
                max_off_diag = max(max_off_diag, abs(inter_metric["spearman"][i][j]))
    logger.info(f"Verification: Max inter-metric |r| = {max_off_diag:.4f} "
                f"({'PASS' if max_off_diag < 0.7 else 'FAIL'}: |r| < 0.7)")

    return results


def main():
    parser = argparse.ArgumentParser(description="SmartQuant Correlation Study")
    parser.add_argument("--model", required=True,
                        help="Path to BF16 model directory")
    parser.add_argument("--output-dir", default="journal/results",
                        help="Output directory for results JSON")
    parser.add_argument("--model-name",
                        help="Model name for output files (default: directory name)")
    args = parser.parse_args()
    run_correlation(args)


if __name__ == "__main__":
    main()
