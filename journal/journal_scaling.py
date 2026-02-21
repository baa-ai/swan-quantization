#!/usr/bin/env /Users/tk/smartquant-env/bin/python3
"""
Scaling Study: Compare SmartQuant sensitivity profiles across 3 models.

Compares:
  - Qwen3-8B (dense, 8B params, 36 layers)
  - Llama4-Maverick (MoE 128E, 400B params, 48 layers)
  - Qwen3.5-397B (MoE 512E, 400B params, 60 layers)

Outputs comprehensive JSON to journal/results/scaling_study.json.
"""

import sys
sys.path.insert(0, '/Users/tk/smartquant')

import json
import re
import os
import statistics
from collections import defaultdict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MANIFESTS = {
    "qwen3_8b": {
        "path": "/Users/tk/smartquant/journal/analysis/qwen3_8b/manifest_base.json",
        "label": "Qwen3-8B (Dense)",
        "param_scale": "8B",
        "architecture": "dense",
    },
    "maverick": {
        "path": "/Users/tk/smartquant/analysis/manifest.json",
        "label": "Llama4-Maverick-17B-128E (MoE)",
        "param_scale": "400B",
        "architecture": "moe_128e",
    },
    "qwen35_397b": {
        "path": "/Users/tk/smartquant/analysis/qwen35-397b/smartquant_manifest.json",
        "label": "Qwen3.5-397B-A17B-512E (MoE)",
        "param_scale": "400B",
        "architecture": "moe_512e",
    },
}

OUTPUT_PATH = "/Users/tk/smartquant/journal/results/scaling_study.json"

# Tensor type classification
ATTENTION_KEYS = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP_KEYS = ["gate_proj", "up_proj", "down_proj", "gate_up_proj"]

LAYER_RE = re.compile(r'layers\.(\d+)')

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def classify_tensor_type(name):
    """Classify tensor into attention / mlp / embedding / norm / other."""
    low = name.lower()
    # Embeddings and head
    if "embed_tokens" in low or "lm_head" in low or "patch_embed" in low:
        return "embedding"
    # Norms
    if "layernorm" in low or "norm" in low:
        return "norm"
    # Router
    if "router" in low or (".gate.weight" in low and "gate_proj" not in low
                           and "gate_up" not in low and "shared_expert_gate" not in low):
        return "router"
    # Vision
    if "vision" in low or "visual" in low or "multi_modal" in low:
        return "vision"
    # MTP (multi-token prediction) head
    if low.startswith("mtp."):
        return "mtp"
    # Attention
    for k in ATTENTION_KEYS:
        if k in low and "feed_forward" not in low and "mlp" not in low:
            return "attention"
    # Linear attention (Qwen3.5 GatedDeltaNet)
    if "linear_attn" in low:
        return "linear_attention"
    # MLP / experts
    for k in MLP_KEYS:
        if k in low:
            if "shared_expert" in low:
                return "mlp_shared_expert"
            if "experts" in low:
                return "mlp_expert"
            return "mlp"
    return "other"


def extract_layer_idx(name):
    """Extract layer index from tensor name, or None."""
    m = LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def score_stats(values):
    """Compute summary statistics for a list of floats."""
    if not values:
        return {"count": 0, "mean": 0, "std": 0, "median": 0, "p25": 0, "p75": 0,
                "min": 0, "max": 0}
    s = sorted(values)
    n = len(s)
    mean = statistics.mean(s)
    std = statistics.pstdev(s)
    median = statistics.median(s)
    p25 = s[max(0, int(n * 0.25))]
    p75 = s[max(0, int(n * 0.75))]
    return {
        "count": n,
        "mean": round(mean, 6),
        "std": round(std, 6),
        "median": round(median, 6),
        "p25": round(p25, 6),
        "p75": round(p75, 6),
        "min": round(s[0], 6),
        "max": round(s[-1], 6),
    }


def analyze_manifest(path):
    """Full analysis of a single manifest."""
    with open(path) as f:
        manifest = json.load(f)

    summary = manifest.get("summary", {})
    total_layers = manifest.get("total_layers", 0)
    total_tensors = manifest.get("total_tensors", 0)

    # Collect all tensors
    all_tensors = {}
    for shard_name, shard_data in manifest.get("shards", {}).items():
        for tname, tinfo in shard_data.get("tensors", {}).items():
            all_tensors[tname] = tinfo

    # Bit distribution from summary
    bits_dist = summary.get("bits_distribution", {})

    # Score distributions per metric
    metric_values = defaultdict(list)
    composite_values = []

    # Tensor type breakdown by bit width: type -> bits -> {count, params}
    type_bit_dist = defaultdict(lambda: defaultdict(lambda: {"count": 0, "params": 0}))
    type_scores = defaultdict(list)

    # Layer-level analysis
    layer_data = defaultdict(lambda: {"scores": [], "bits": [], "tensor_count": 0})

    for tname, tinfo in all_tensors.items():
        bits = tinfo["decision"]["bits"]
        params = tinfo.get("num_params", 0)
        scores = tinfo.get("scores", {})
        composite = tinfo.get("composite_score", 0)
        ttype = classify_tensor_type(tname)
        layer_idx = extract_layer_idx(tname)

        # Metric distributions
        for metric, val in scores.items():
            if val is not None:
                metric_values[metric].append(val)
        if composite is not None:
            composite_values.append(composite)

        # Type-bit distribution
        type_bit_dist[ttype][bits]["count"] += 1
        type_bit_dist[ttype][bits]["params"] += params
        if composite is not None:
            type_scores[ttype].append(composite)

        # Layer data
        if layer_idx is not None:
            layer_data[layer_idx]["scores"].append(composite or 0)
            layer_data[layer_idx]["bits"].append(bits)
            layer_data[layer_idx]["tensor_count"] += 1

    # Compute metric stats
    metric_stats = {}
    for metric, vals in sorted(metric_values.items()):
        metric_stats[metric] = score_stats(vals)
    metric_stats["composite_score"] = score_stats(composite_values)

    # Compute type-bit as serializable dict
    type_bit_result = {}
    for ttype in sorted(type_bit_dist.keys()):
        bd = type_bit_dist[ttype]
        total_type_params = sum(bd[b]["params"] for b in bd)
        total_type_count = sum(bd[b]["count"] for b in bd)
        per_bit = {}
        for b in sorted(bd.keys()):
            pct_params = (bd[b]["params"] / total_type_params * 100) if total_type_params > 0 else 0
            pct_count = (bd[b]["count"] / total_type_count * 100) if total_type_count > 0 else 0
            per_bit[str(b)] = {
                "count": bd[b]["count"],
                "params": bd[b]["params"],
                "pct_params": round(pct_params, 2),
                "pct_count": round(pct_count, 2),
            }
        avg_bits_type = 0
        if total_type_params > 0:
            avg_bits_type = sum(b * bd[b]["params"] for b in bd) / total_type_params

        type_bit_result[ttype] = {
            "total_count": total_type_count,
            "total_params": total_type_params,
            "average_bits": round(avg_bits_type, 4),
            "score_stats": score_stats(type_scores.get(ttype, [])),
            "bit_distribution": per_bit,
        }

    # Layer-level summary
    layer_summary = {}
    layers_8plus = []
    for lidx in sorted(layer_data.keys()):
        ld = layer_data[lidx]
        avg_score = statistics.mean(ld["scores"]) if ld["scores"] else 0
        avg_bits = statistics.mean(ld["bits"]) if ld["bits"] else 0
        bits_counter = defaultdict(int)
        for b in ld["bits"]:
            bits_counter[b] += 1
        has_8plus = any(b >= 8 for b in ld["bits"])
        if has_8plus:
            layers_8plus.append(lidx)
        layer_summary[str(lidx)] = {
            "tensor_count": ld["tensor_count"],
            "avg_composite_score": round(avg_score, 6),
            "avg_bits": round(avg_bits, 4),
            "bit_counts": dict(bits_counter),
            "has_8plus_bit_tensors": has_8plus,
        }

    return {
        "total_params": summary.get("total_params", 0),
        "total_tensors": total_tensors or len(all_tensors),
        "total_layers": total_layers,
        "estimated_size_gb": round(summary.get("estimated_size_gb", 0), 2),
        "average_bits": round(summary.get("average_bits", 0), 4),
        "bits_distribution": bits_dist,
        "score_distributions": metric_stats,
        "tensor_type_breakdown": type_bit_result,
        "layer_summary": layer_summary,
        "layers_with_8plus_bits": layers_8plus,
        "num_layers_with_8plus": len(layers_8plus),
        "pct_layers_with_8plus": round(len(layers_8plus) / total_layers * 100, 1) if total_layers > 0 else 0,
    }


def compute_cross_model_patterns(results):
    """Identify patterns that hold across all models."""
    patterns = {}

    # Pattern 1: Attention vs MLP average bits
    attn_vs_mlp = {}
    for model_key, data in results.items():
        tb = data["tensor_type_breakdown"]
        attn_bits = tb.get("attention", {}).get("average_bits", 0)
        mlp_types = ["mlp", "mlp_expert", "mlp_shared_expert"]
        mlp_params = sum(tb.get(t, {}).get("total_params", 0) for t in mlp_types)
        mlp_weighted_bits = sum(
            tb.get(t, {}).get("average_bits", 0) * tb.get(t, {}).get("total_params", 0)
            for t in mlp_types
        )
        mlp_bits = mlp_weighted_bits / mlp_params if mlp_params > 0 else 0
        attn_vs_mlp[model_key] = {
            "attention_avg_bits": round(attn_bits, 4),
            "mlp_avg_bits": round(mlp_bits, 4),
            "attention_higher": attn_bits > mlp_bits,
            "delta": round(attn_bits - mlp_bits, 4),
        }
    patterns["attention_vs_mlp_bits"] = attn_vs_mlp
    patterns["attention_higher_across_all"] = all(
        v["attention_higher"] for v in attn_vs_mlp.values()
    )

    # Pattern 2: Attention vs MLP composite scores
    attn_vs_mlp_scores = {}
    for model_key, data in results.items():
        tb = data["tensor_type_breakdown"]
        attn_score = tb.get("attention", {}).get("score_stats", {}).get("mean", 0)
        mlp_types = ["mlp", "mlp_expert", "mlp_shared_expert"]
        mlp_count = sum(tb.get(t, {}).get("score_stats", {}).get("count", 0) for t in mlp_types)
        mlp_weighted = sum(
            tb.get(t, {}).get("score_stats", {}).get("mean", 0) *
            tb.get(t, {}).get("score_stats", {}).get("count", 0)
            for t in mlp_types
        )
        mlp_score = mlp_weighted / mlp_count if mlp_count > 0 else 0
        attn_vs_mlp_scores[model_key] = {
            "attention_mean_score": round(attn_score, 6),
            "mlp_mean_score": round(mlp_score, 6),
            "attention_higher": attn_score > mlp_score,
        }
    patterns["attention_vs_mlp_scores"] = attn_vs_mlp_scores

    # Pattern 3: Early/late layer sensitivity
    layer_position_patterns = {}
    for model_key, data in results.items():
        ls = data["layer_summary"]
        if not ls:
            continue
        indices = sorted(int(k) for k in ls.keys())
        n = len(indices)
        if n < 4:
            continue
        first_quarter = indices[:n // 4]
        last_quarter = indices[3 * n // 4:]
        middle_half = indices[n // 4: 3 * n // 4]

        first_scores = [ls[str(i)]["avg_composite_score"] for i in first_quarter]
        last_scores = [ls[str(i)]["avg_composite_score"] for i in last_quarter]
        middle_scores = [ls[str(i)]["avg_composite_score"] for i in middle_half]

        first_bits = [ls[str(i)]["avg_bits"] for i in first_quarter]
        last_bits = [ls[str(i)]["avg_bits"] for i in last_quarter]
        middle_bits = [ls[str(i)]["avg_bits"] for i in middle_half]

        layer_position_patterns[model_key] = {
            "first_quarter_avg_score": round(statistics.mean(first_scores), 6),
            "middle_half_avg_score": round(statistics.mean(middle_scores), 6),
            "last_quarter_avg_score": round(statistics.mean(last_scores), 6),
            "first_quarter_avg_bits": round(statistics.mean(first_bits), 4),
            "middle_half_avg_bits": round(statistics.mean(middle_bits), 4),
            "last_quarter_avg_bits": round(statistics.mean(last_bits), 4),
            "early_more_sensitive": statistics.mean(first_scores) > statistics.mean(middle_scores),
            "late_more_sensitive": statistics.mean(last_scores) > statistics.mean(middle_scores),
        }
    patterns["layer_position_sensitivity"] = layer_position_patterns

    # Pattern 4: MoE expert tensors vs dense MLP
    expert_patterns = {}
    for model_key, data in results.items():
        tb = data["tensor_type_breakdown"]
        if "mlp_expert" in tb:
            expert_patterns[model_key] = {
                "expert_avg_bits": tb["mlp_expert"]["average_bits"],
                "expert_count": tb["mlp_expert"]["total_count"],
                "expert_params": tb["mlp_expert"]["total_params"],
                "expert_mean_score": tb["mlp_expert"]["score_stats"]["mean"],
            }
        elif "mlp" in tb:
            expert_patterns[model_key] = {
                "mlp_avg_bits": tb["mlp"]["average_bits"],
                "mlp_count": tb["mlp"]["total_count"],
                "mlp_params": tb["mlp"]["total_params"],
                "mlp_mean_score": tb["mlp"]["score_stats"]["mean"],
            }
    patterns["expert_vs_dense_mlp"] = expert_patterns

    # Pattern 5: 8-bit fraction of attention across models
    attn_8bit_fraction = {}
    for model_key, data in results.items():
        tb = data["tensor_type_breakdown"]
        attn = tb.get("attention", {})
        bd = attn.get("bit_distribution", {})
        total_count = attn.get("total_count", 0)
        count_8 = bd.get("8", {}).get("count", 0)
        count_16 = bd.get("16", {}).get("count", 0)
        pct_8plus = ((count_8 + count_16) / total_count * 100) if total_count > 0 else 0
        attn_8bit_fraction[model_key] = {
            "attention_tensor_count": total_count,
            "at_8bit": count_8,
            "at_16bit": count_16,
            "pct_8plus_bit": round(pct_8plus, 1),
        }
    patterns["attention_8plus_bit_fraction"] = attn_8bit_fraction

    return patterns


def print_summary(results, patterns):
    """Print human-readable summary to stdout."""
    print("=" * 80)
    print("SMARTQUANT SCALING STUDY - CROSS-MODEL COMPARISON")
    print("=" * 80)

    # Overview table
    print("\n--- Model Overview ---")
    print(f"{'Model':<35} {'Params':>12} {'Tensors':>8} {'Layers':>6} {'Avg Bits':>9} {'Size (GB)':>10}")
    print("-" * 85)
    for key, cfg in MANIFESTS.items():
        d = results[key]
        params_str = f"{d['total_params'] / 1e9:.1f}B"
        print(f"{cfg['label']:<35} {params_str:>12} {d['total_tensors']:>8} "
              f"{d['total_layers']:>6} {d['average_bits']:>9.4f} {d['estimated_size_gb']:>10.2f}")

    # Bit distributions
    print("\n--- Bit Width Distribution (% by params) ---")
    print(f"{'Model':<35} {'4-bit':>10} {'8-bit':>10} {'16-bit':>10}")
    print("-" * 70)
    for key, cfg in MANIFESTS.items():
        bd = results[key]["bits_distribution"]
        p4 = bd.get("4", {}).get("percentage", 0)
        p8 = bd.get("8", {}).get("percentage", 0)
        p16 = bd.get("16", {}).get("percentage", 0)
        print(f"{cfg['label']:<35} {p4:>9.1f}% {p8:>9.1f}% {p16:>9.1f}%")

    # Attention vs MLP
    print("\n--- Pattern: Attention vs MLP Average Bits ---")
    print(f"{'Model':<35} {'Attn Bits':>10} {'MLP Bits':>10} {'Delta':>8} {'Attn Higher?':>14}")
    print("-" * 82)
    for key in MANIFESTS:
        avm = patterns["attention_vs_mlp_bits"][key]
        print(f"{MANIFESTS[key]['label']:<35} {avm['attention_avg_bits']:>10.4f} "
              f"{avm['mlp_avg_bits']:>10.4f} {avm['delta']:>+8.4f} "
              f"{'YES' if avm['attention_higher'] else 'NO':>14}")
    verdict = "YES" if patterns["attention_higher_across_all"] else "NO"
    print(f"\n  --> Attention gets higher bits across ALL models: {verdict}")

    # Attention 8+ bit fraction
    print("\n--- Pattern: Attention Tensors at 8+ Bits ---")
    print(f"{'Model':<35} {'Total Attn':>10} {'At 8-bit':>10} {'At 16-bit':>10} {'% 8+':>8}")
    print("-" * 78)
    for key in MANIFESTS:
        af = patterns["attention_8plus_bit_fraction"][key]
        print(f"{MANIFESTS[key]['label']:<35} {af['attention_tensor_count']:>10} "
              f"{af['at_8bit']:>10} {af['at_16bit']:>10} {af['pct_8plus_bit']:>7.1f}%")

    # Layer position patterns
    print("\n--- Pattern: Layer Position Sensitivity (Composite Score) ---")
    print(f"{'Model':<35} {'First 25%':>10} {'Middle 50%':>11} {'Last 25%':>10} {'Early>Mid':>10} {'Late>Mid':>10}")
    print("-" * 91)
    for key in MANIFESTS:
        lp = patterns["layer_position_sensitivity"].get(key)
        if lp:
            print(f"{MANIFESTS[key]['label']:<35} {lp['first_quarter_avg_score']:>10.4f} "
                  f"{lp['middle_half_avg_score']:>11.4f} {lp['last_quarter_avg_score']:>10.4f} "
                  f"{'YES' if lp['early_more_sensitive'] else 'NO':>10} "
                  f"{'YES' if lp['late_more_sensitive'] else 'NO':>10}")

    # Layer position bits
    print("\n--- Pattern: Layer Position Average Bits ---")
    print(f"{'Model':<35} {'First 25%':>10} {'Middle 50%':>11} {'Last 25%':>10}")
    print("-" * 70)
    for key in MANIFESTS:
        lp = patterns["layer_position_sensitivity"].get(key)
        if lp:
            print(f"{MANIFESTS[key]['label']:<35} {lp['first_quarter_avg_bits']:>10.4f} "
                  f"{lp['middle_half_avg_bits']:>11.4f} {lp['last_quarter_avg_bits']:>10.4f}")

    # Tensor type breakdown
    print("\n--- Tensor Type Breakdown (Average Bits) ---")
    all_types = set()
    for key in MANIFESTS:
        all_types.update(results[key]["tensor_type_breakdown"].keys())
    all_types = sorted(all_types)

    header = f"{'Type':<25}"
    for key in MANIFESTS:
        header += f" {key:>15}"
    print(header)
    print("-" * (25 + 16 * len(MANIFESTS)))
    for ttype in all_types:
        row = f"{ttype:<25}"
        for key in MANIFESTS:
            tb = results[key]["tensor_type_breakdown"].get(ttype)
            if tb:
                row += f" {tb['average_bits']:>14.4f}"
            else:
                row += f" {'---':>14}"
        print(row)

    # Layers with 8+ bits
    print("\n--- Layers with 8+ Bit Tensors ---")
    for key, cfg in MANIFESTS.items():
        d = results[key]
        print(f"  {cfg['label']}: {d['num_layers_with_8plus']}/{d['total_layers']} "
              f"({d['pct_layers_with_8plus']:.1f}%) layers have at least one 8+ bit tensor")

    # Score distribution summaries
    print("\n--- Composite Score Distribution ---")
    print(f"{'Model':<35} {'Mean':>8} {'Std':>8} {'Median':>8} {'P25':>8} {'P75':>8} {'Max':>8}")
    print("-" * 90)
    for key, cfg in MANIFESTS.items():
        cs = results[key]["score_distributions"]["composite_score"]
        print(f"{cfg['label']:<35} {cs['mean']:>8.4f} {cs['std']:>8.4f} "
              f"{cs['median']:>8.4f} {cs['p25']:>8.4f} {cs['p75']:>8.4f} {cs['max']:>8.4f}")

    print("\n" + "=" * 80)
    print(f"Results saved to: {OUTPUT_PATH}")
    print("=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    results = {}
    for key, cfg in MANIFESTS.items():
        print(f"Analyzing {cfg['label']}...")
        results[key] = analyze_manifest(cfg["path"])

    patterns = compute_cross_model_patterns(results)

    # Build output
    output = {
        "study": "SmartQuant Scaling Study",
        "description": "Cross-model comparison of SmartQuant sensitivity-driven quantization",
        "models": {},
        "cross_model_patterns": patterns,
    }
    for key, cfg in MANIFESTS.items():
        output["models"][key] = {
            "label": cfg["label"],
            "param_scale": cfg["param_scale"],
            "architecture": cfg["architecture"],
            "manifest_path": cfg["path"],
            **results[key],
        }

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print_summary(results, patterns)


if __name__ == "__main__":
    main()
