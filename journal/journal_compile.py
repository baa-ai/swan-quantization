#!/usr/bin/env python3
"""
SmartQuant Results Compiler — generates publication figures and tables.

Reads all JSON result files from experiments and produces:
  - LaTeX tables (5): ablation, correlation, inter-metric, perplexity comparison, scaling
  - Matplotlib figures (5): ablation bar chart, correlation scatter, inter-metric heatmap,
    scaling bit allocation, perplexity-vs-bits curve

Usage:
    python journal/journal_compile.py \
        --results-dir journal/results \
        --figures-dir journal/figures \
        --tables-dir journal/tables \
        [--scaling-manifests manifest1.json manifest2.json ...]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# LaTeX Table Generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_ablation_table(ablation_data: dict, output_path: Path):
    """Table 1: Ablation study — metric configs vs perplexity."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation study: effect of metric weight configurations on perplexity. "
        r"Weights for SVD ($w_s$), Kurtosis ($w_k$), Output Sensitivity ($w_o$), "
        r"and Cross-Layer Position ($w_p$). Best SmartQuant result in \textbf{bold}.}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lccccrrr}",
        r"\toprule",
        r"Configuration & $w_s$ & $w_k$ & $w_o$ & $w_p$ & Avg Bits & Size (GB) & PPL $\downarrow$ \\",
        r"\midrule",
    ]

    # Baselines first
    baselines = ablation_data.get("baselines", {})
    for name in ["bf16", "uniform_8bit", "uniform_4bit"]:
        if name in baselines:
            b = baselines[name]
            ppl = b.get("perplexity")
            ppl_str = f"{ppl:.3f}" if ppl else "---"
            bits_str = str(b.get("bits", ""))
            size = b.get("size_gb", 0)
            display_name = {"bf16": "BF16 (baseline)", "uniform_8bit": "Uniform 8-bit",
                           "uniform_4bit": "Uniform 4-bit"}.get(name, name)
            lines.append(f"  {display_name} & --- & --- & --- & --- & {bits_str} & {size:.1f} & {ppl_str} \\\\")

    lines.append(r"\midrule")

    # SmartQuant configs — best = lowest PPL among configs with avg_bits <= 8
    # (excludes configs that are effectively BF16 due to metric saturation)
    configs = ablation_data.get("configs", {})
    best_ppl = float("inf")
    best_name = ""
    for name, data in configs.items():
        ppl = data.get("perplexity")
        avg_bits = data.get("average_bits", 16)
        if ppl is not None and ppl < best_ppl and avg_bits <= 8.0:
            best_ppl = ppl
            best_name = name

    for name, data in configs.items():
        w = data.get("weights", {})
        ppl = data.get("perplexity")
        avg_bits = data.get("average_bits", 0)
        size = data.get("actual_size_gb", data.get("estimated_size_gb", 0))

        if ppl is not None:
            ppl_str = f"\\textbf{{{ppl:.3f}}}" if name == best_name else f"{ppl:.3f}"
        else:
            ppl_str = "---"

        display = name.replace("_", " ").title()
        lines.append(
            f"  {display} & {w.get('svd', 0):.2f} & {w.get('kurtosis', 0):.2f} & "
            f"{w.get('output_sensitivity', 0):.2f} & {w.get('cross_layer', 0):.2f} & "
            f"{avg_bits:.2f} & {size:.1f} & {ppl_str} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Table 1 (ablation) written to {output_path}")


def generate_correlation_table(corr_data: dict, output_path: Path, model_label: str = ""):
    """Table 2: Metric vs reconstruction error correlations.

    Uses raw (unclamped) scores when available, as they provide better
    dynamic range for correlation analysis.
    """
    n_tensors = corr_data.get("tensors_analyzed", 0)
    model_desc = f" on {model_label}" if model_label else ""
    tensor_desc = f" ({n_tensors:,} tensors)" if n_tensors else ""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Correlation between SmartQuant sensitivity metrics and actual "
        r"4-bit quantization reconstruction error (NRMSE)" + model_desc + tensor_desc + ". "
        r"Raw (unclamped) metric values used for correlation analysis. "
        r"Higher $|\rho|$ indicates better predictive validity.}",
        r"\label{tab:correlation}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Metric & Spearman $\rho$ & $p$-value & Pearson $r$ & $p$-value \\",
        r"\midrule",
    ]

    # Prefer raw scores if available
    raw_vs_error = corr_data.get("raw_metric_vs_error", {})
    clamped_vs_error = corr_data.get("metric_vs_error", {})

    # Map from raw metric names to display names
    metrics_to_show = [
        ("svd_concentration", "svd", "SVD Concentration"),
        ("kurtosis_raw", "kurtosis", "Kurtosis"),
        ("output_delta", "output_sensitivity", "Output Sensitivity"),
        ("position_u_score", "cross_layer", "Cross-Layer Position"),
    ]

    for raw_key, clamped_key, display in metrics_to_show:
        # Use raw if available, fallback to clamped
        c = raw_vs_error.get(raw_key, clamped_vs_error.get(clamped_key, {}))
        sp_r = c.get("spearman_r", 0)
        sp_p = c.get("spearman_p", 1)
        pe_r = c.get("pearson_r", 0)
        pe_p = c.get("pearson_p", 1)

        sp_p_str = f"{sp_p:.1e}" if sp_p >= 0.001 else f"$<$0.001"
        pe_p_str = f"{pe_p:.1e}" if pe_p >= 0.001 else f"$<$0.001"

        lines.append(f"  {display} & {sp_r:.4f} & {sp_p_str} & {pe_r:.4f} & {pe_p_str} \\\\")

    # Composite (always from clamped since raw composite isn't computed separately)
    c = clamped_vs_error.get("composite", {})
    sp_r = c.get("spearman_r", 0)
    sp_p = c.get("spearman_p", 1)
    pe_r = c.get("pearson_r", 0)
    pe_p = c.get("pearson_p", 1)
    sp_p_str = f"{sp_p:.1e}" if sp_p >= 0.001 else f"$<$0.001"
    pe_p_str = f"{pe_p:.1e}" if pe_p >= 0.001 else f"$<$0.001"

    lines.append(r"\midrule")
    lines.append(f"  \\textbf{{Composite (SmartQuant)}} & {sp_r:.4f} & {sp_p_str} & {pe_r:.4f} & {pe_p_str} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Table 2 (correlation) written to {output_path}")


def generate_inter_metric_table(corr_data: dict, output_path: Path):
    """Table 3: Inter-metric Spearman correlation matrix."""
    inter = corr_data.get("inter_metric_correlation", {})
    metrics = inter.get("metrics", [])
    matrix = inter.get("spearman", [])

    display_names = {
        "svd": "SVD",
        "kurtosis": "Kurtosis",
        "output_sensitivity": "Output Sens.",
        "cross_layer": "Cross-Layer",
    }

    header_cols = " & ".join(display_names.get(m, m) for m in metrics)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Inter-metric Spearman correlation matrix. Low off-diagonal values "
        r"($|\rho| < 0.7$) confirm that each metric captures distinct information "
        r"about tensor sensitivity.}",
        r"\label{tab:inter_metric}",
        f"\\begin{{tabular}}{{l{'c' * len(metrics)}}}",
        r"\toprule",
        f"  & {header_cols} \\\\",
        r"\midrule",
    ]

    for i, mi in enumerate(metrics):
        row_vals = []
        for j, mj in enumerate(metrics):
            val = matrix[i][j] if i < len(matrix) and j < len(matrix[i]) else 0
            if i == j:
                row_vals.append("1.000")
            else:
                row_vals.append(f"{val:.3f}")
        row_str = " & ".join(row_vals)
        lines.append(f"  {display_names.get(mi, mi)} & {row_str} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Table 3 (inter-metric) written to {output_path}")


def generate_perplexity_table(ppl_data: list, output_path: Path):
    """Table 4: Perplexity comparison across models and quantization methods."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Perplexity comparison across quantization methods and model scales. "
        r"Lower perplexity is better. All measurements use WikiText-2 with "
        r"sequence\_length=2048, num\_samples=256, seed=42.}",
        r"\label{tab:perplexity}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Model & Method & Size (GB) & Avg Bits & PPL $\downarrow$ \\",
        r"\midrule",
    ]

    for entry in ppl_data:
        model = entry.get("model_name", "")
        variant = entry.get("variant", "")
        ppl = entry.get("perplexity")
        size = entry.get("model_info", {}).get("size_gb", 0)
        # Derive avg_bits: explicit field > quantization bits > "---"
        bits = entry.get("avg_bits")
        if bits is None:
            quant = entry.get("model_info", {}).get("quantization", {})
            bits = quant.get("bits")

        ppl_str = f"{ppl:.3f}" if ppl else "---"
        bits_str = f"{bits:.2f}" if isinstance(bits, (int, float)) else "---"

        # Nice display names
        display_model = model.replace("qwen35_397b", "Qwen3.5-397B").replace("qwen3_235b", "Qwen3-235B").replace("qwen3_8b", "Qwen3-8B")
        display_variant = variant.replace("_", " ").title()
        lines.append(f"  {display_model} & {display_variant} & {size:.1f} & {bits_str} & {ppl_str} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Table 4 (perplexity) written to {output_path}")


def generate_scaling_table(scaling_data: dict, output_path: Path):
    """Table 5: Scaling study — bit allocation across models."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{SmartQuant bit allocation across model architectures. "
        r"Consistent patterns: attention layers receive higher precision; "
        r"middle MLP layers are more aggressively quantized.}",
        r"\label{tab:scaling}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Model & Params & Tensors & 16-bit (\%) & 8-bit (\%) & 4-bit (\%) & Avg Bits \\",
        r"\midrule",
    ]

    for model_name, data in scaling_data.items():
        summary = data.get("summary", {})
        dist = summary.get("bits_distribution", {})
        total_params = summary.get("total_params", 0)
        tensor_count = summary.get("tensor_count", 0)
        avg_bits = summary.get("average_bits", 0)

        params_str = _format_params_latex(total_params)
        pct_16 = dist.get("16", {}).get("percentage", 0)
        pct_8 = dist.get("8", {}).get("percentage", 0)
        pct_4 = dist.get("4", {}).get("percentage", 0)

        lines.append(
            f"  {model_name} & {params_str} & {tensor_count} & "
            f"{pct_16:.1f} & {pct_8:.1f} & {pct_4:.1f} & {avg_bits:.2f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Table 5 (scaling) written to {output_path}")


def _format_params_latex(n: int) -> str:
    if n >= 1e9:
        return f"{n/1e9:.1f}B"
    elif n >= 1e6:
        return f"{n/1e6:.0f}M"
    return str(n)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure Generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_ablation_figure(ablation_data: dict, output_path: Path):
    """Figure 1: Ablation bar chart — perplexity vs metric configuration."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configs = ablation_data.get("configs", {})
    baselines = ablation_data.get("baselines", {})

    # Collect data points
    names, ppls, bits, colors = [], [], [], []

    # Baselines
    color_map = {"bf16": "#2ecc71", "uniform_8bit": "#3498db", "uniform_4bit": "#e74c3c"}
    display_map = {"bf16": "BF16", "uniform_8bit": "Unif-8", "uniform_4bit": "Unif-4"}
    for bname in ["bf16", "uniform_8bit", "uniform_4bit"]:
        if bname in baselines:
            b = baselines[bname]
            if b.get("perplexity") is not None:
                names.append(display_map.get(bname, bname))
                ppls.append(b["perplexity"])
                bits.append(b.get("bits", 0))
                colors.append(color_map.get(bname, "#95a5a6"))

    # SmartQuant configs
    for cname, data in configs.items():
        if data.get("perplexity") is not None:
            short_name = cname.replace("_only", "").replace("_", "\n")
            if cname == "composite_default":
                short_name = "Composite\n(default)"
                colors.append("#e67e22")  # Orange for the recommended config
            elif cname == "equal_weights":
                short_name = "Equal\nweights"
                colors.append("#9b59b6")
            else:
                colors.append("#34495e")
            names.append(short_name)
            ppls.append(data["perplexity"])
            bits.append(data.get("average_bits", 0))

    if not names:
        logger.warning("No ablation data to plot")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(names))
    bars = ax.bar(x, ppls, color=colors, edgecolor="white", linewidth=0.5)

    # Add value labels
    for bar, ppl in zip(bars, ppls):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{ppl:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Perplexity", fontsize=11)
    ax.set_title("SmartQuant Ablation: Metric Weight Configurations", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure 1 (ablation) written to {output_path}")


def generate_correlation_figure(corr_data: dict, output_path: Path):
    """Figure 2: Scatter plot — composite score vs reconstruction error."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = corr_data.get("tensor_records", [])
    if not records:
        logger.warning("No tensor records for correlation figure")
        return

    composites = [r["composite_score"] for r in records]
    errors = [r["quant_error_4bit"] for r in records]

    # Color by tensor type
    colors = []
    for r in records:
        name = r.get("name", "")
        if "q_proj" in name or "k_proj" in name:
            colors.append("#e74c3c")  # Red: attention QK
        elif "v_proj" in name or "o_proj" in name:
            colors.append("#3498db")  # Blue: attention VO
        elif "gate" in name or "up_proj" in name or "down_proj" in name:
            colors.append("#2ecc71")  # Green: MLP
        else:
            colors.append("#95a5a6")  # Gray: other

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(composites, errors, c=colors, alpha=0.5, s=12, edgecolor="none")

    # Add trend line
    if len(composites) > 2:
        z = np.polyfit(composites, errors, 1)
        p = np.poly1d(z)
        x_range = np.linspace(min(composites), max(composites), 100)
        ax.plot(x_range, p(x_range), "--", color="#e67e22", linewidth=2, alpha=0.8,
                label=f"Linear fit")

    # Add correlation annotation
    mv = corr_data.get("metric_vs_error", {}).get("composite", {})
    sp_r = mv.get("spearman_r", 0)
    pe_r = mv.get("pearson_r", 0)
    ax.text(0.05, 0.95,
            f"Spearman $\\rho$ = {sp_r:.3f}\nPearson $r$ = {pe_r:.3f}",
            transform=ax.transAxes, fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    # Legend for tensor types
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c", markersize=6, label="Attn QK"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#3498db", markersize=6, label="Attn VO"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71", markersize=6, label="MLP"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#95a5a6", markersize=6, label="Other"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    ax.set_xlabel("SmartQuant Composite Score", fontsize=11)
    ax.set_ylabel("4-bit Reconstruction Error (NRMSE)", fontsize=11)
    ax.set_title("Correlation: SmartQuant Score vs Quantization Error", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure 2 (correlation) written to {output_path}")


def generate_heatmap_figure(corr_data: dict, output_path: Path):
    """Figure 3: Inter-metric correlation heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    inter = corr_data.get("inter_metric_correlation", {})
    metrics = inter.get("metrics", [])
    matrix = np.array(inter.get("spearman", []))

    if matrix.size == 0:
        logger.warning("No inter-metric data for heatmap")
        return

    display_names = {
        "svd": "SVD\nConcentration",
        "kurtosis": "Kurtosis",
        "output_sensitivity": "Output\nSensitivity",
        "cross_layer": "Cross-Layer\nPosition",
    }
    labels = [display_names.get(m, m) for m in metrics]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")

    # Annotations
    for i in range(len(metrics)):
        for j in range(len(metrics)):
            val = matrix[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=10, color=color, fontweight="bold" if i == j else "normal")

    ax.set_xticks(range(len(metrics)))
    ax.set_yticks(range(len(metrics)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Spearman Correlation", fontsize=10)

    ax.set_title("Inter-Metric Correlation Matrix", fontsize=12)

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure 3 (heatmap) written to {output_path}")


def generate_scaling_figure(scaling_data: dict, output_path: Path):
    """Figure 4: Stacked bar chart — bit allocation across models."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_names = list(scaling_data.keys())
    if not model_names:
        logger.warning("No scaling data to plot")
        return

    pct_4, pct_8, pct_16 = [], [], []
    for name in model_names:
        dist = scaling_data[name].get("summary", {}).get("bits_distribution", {})
        pct_4.append(dist.get("4", {}).get("percentage", 0))
        pct_8.append(dist.get("8", {}).get("percentage", 0))
        pct_16.append(dist.get("16", {}).get("percentage", 0))

    x = np.arange(len(model_names))
    width = 0.5

    fig, ax = plt.subplots(figsize=(8, 5))
    p1 = ax.bar(x, pct_4, width, label="4-bit", color="#2ecc71")
    p2 = ax.bar(x, pct_8, width, bottom=pct_4, label="8-bit", color="#3498db")
    p3 = ax.bar(x, pct_16, width, bottom=[a + b for a, b in zip(pct_4, pct_8)],
                label="16-bit", color="#e74c3c")

    # Add percentage labels
    for i, (v4, v8, v16) in enumerate(zip(pct_4, pct_8, pct_16)):
        if v4 > 5:
            ax.text(i, v4 / 2, f"{v4:.0f}%", ha="center", va="center", fontsize=9, color="white")
        if v8 > 5:
            ax.text(i, v4 + v8 / 2, f"{v8:.0f}%", ha="center", va="center", fontsize=9, color="white")
        if v16 > 3:
            ax.text(i, v4 + v8 + v16 / 2, f"{v16:.1f}%", ha="center", va="center", fontsize=9, color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=10)
    ax.set_ylabel("Parameter Percentage", fontsize=11)
    ax.set_title("SmartQuant Bit Allocation Across Model Architectures", fontsize=12)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 105)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure 4 (scaling) written to {output_path}")


def generate_ppl_vs_bits_figure(ppl_data: list, output_path: Path):
    """Figure 5: Perplexity vs average bits curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not ppl_data:
        logger.warning("No perplexity data for bits curve")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    # Group by model
    models = {}
    for entry in ppl_data:
        model = entry.get("model_name", "unknown")
        models.setdefault(model, []).append(entry)

    markers = ["o", "s", "^", "D", "v"]
    colors = ["#2ecc71", "#3498db", "#e74c3c", "#e67e22", "#9b59b6"]

    for idx, (model_name, entries) in enumerate(models.items()):
        bits_vals, ppl_vals, labels = [], [], []
        for e in entries:
            ppl = e.get("perplexity")
            b = e.get("avg_bits")
            if ppl is not None and b is not None:
                bits_vals.append(b)
                ppl_vals.append(ppl)
                labels.append(e.get("variant", ""))

        if bits_vals:
            marker = markers[idx % len(markers)]
            color = colors[idx % len(colors)]
            ax.scatter(bits_vals, ppl_vals, marker=marker, color=color,
                       s=80, label=model_name, zorder=3, edgecolor="white", linewidth=0.5)

            # Connect points with line
            sorted_pairs = sorted(zip(bits_vals, ppl_vals))
            ax.plot([p[0] for p in sorted_pairs], [p[1] for p in sorted_pairs],
                    "-", color=color, alpha=0.4, linewidth=1.5)

            # Label points
            for b, p, l in zip(bits_vals, ppl_vals, labels):
                ax.annotate(l, (b, p), textcoords="offset points",
                            xytext=(5, 5), fontsize=7, alpha=0.8)

    ax.set_xlabel("Average Bits per Parameter", fontsize=11)
    ax.set_ylabel("Perplexity", fontsize=11)
    ax.set_title("Perplexity vs Model Size (Bits per Parameter)", fontsize=12)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure 5 (ppl vs bits) written to {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_json_safe(path: Path) -> dict:
    """Load JSON, return empty dict on failure."""
    if not path.exists():
        logger.warning(f"File not found: {path}")
        return {}
    with open(path) as f:
        return json.load(f)


def find_results(results_dir: Path) -> dict:
    """Discover all result files in the results directory."""
    found = {
        "ablation": [],
        "correlation": [],
        "perplexity": [],
        "scaling_manifests": [],
    }

    for f in sorted(results_dir.glob("ablation_*.json")):
        found["ablation"].append(f)
    for f in sorted(results_dir.glob("correlation_*.json")):
        found["correlation"].append(f)
    for f in sorted(results_dir.glob("perplexity_*.json")):
        found["perplexity"].append(f)

    return found


def load_scaling_data(manifest_paths: list) -> dict:
    """Load manifest files for scaling study comparison."""
    # Known model name mappings
    name_map = {
        "maverick-bf16": "Llama4-Maverick-17B",
        "qwen3-8b-bf16": "Qwen3-8B",
        "qwen3_8b": "Qwen3-8B",
    }
    scaling = {}
    for path in manifest_paths:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            logger.warning(f"Manifest not found: {p}")
            continue
        manifest = load_json_safe(p)
        model_path = manifest.get("model", "")
        model_name = Path(model_path).name if model_path else p.stem

        # Apply known mappings
        model_name = name_map.get(model_name, model_name)

        # Detect Qwen3.5-397B by its characteristics
        total_tensors = manifest.get("total_tensors", 0)
        total_layers = manifest.get("total_layers", 0)
        if total_tensors > 2000 and total_layers == 60:
            model_name = "Qwen3.5-397B"
        elif "qwen3.5" in model_path.lower() or "qwen3_5" in model_path.lower():
            model_name = "Qwen3.5-397B"

        scaling[model_name] = manifest
    return scaling


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def run_compile(args):
    """Main compilation routine."""
    results_dir = Path(args.results_dir).expanduser().resolve()
    figures_dir = Path(args.figures_dir).expanduser().resolve()
    tables_dir = Path(args.tables_dir).expanduser().resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    found = find_results(results_dir)
    logger.info(f"Found: {len(found['ablation'])} ablation, "
                f"{len(found['correlation'])} correlation, "
                f"{len(found['perplexity'])} perplexity files")

    # ── Ablation ──────────────────────────────────────────────────────────
    for ablation_file in found["ablation"]:
        ablation_data = load_json_safe(ablation_file)
        if ablation_data:
            model = ablation_data.get("model_name", "unknown")
            generate_ablation_table(ablation_data, tables_dir / "table1_ablation.tex")
            generate_ablation_figure(ablation_data, figures_dir / "ablation_perplexity.pdf")

    # ── Correlation ───────────────────────────────────────────────────────
    # Use the largest model's correlation as the primary table/figure.
    # Generate model-specific suffixed versions for all files.
    model_display = {
        "qwen3_8b": "Qwen3-8B",
        "qwen35_397b": "Qwen3.5-397B",
        "qwen3_235b": "Qwen3-235B",
    }
    corr_files_sorted = sorted(
        found["correlation"],
        key=lambda f: load_json_safe(f).get("tensors_analyzed", 0),
        reverse=True,
    )
    for i, corr_file in enumerate(corr_files_sorted):
        corr_data = load_json_safe(corr_file)
        if not corr_data:
            continue
        model = corr_data.get("model_name", "unknown")
        label = model_display.get(model, model)
        # Primary (largest model) gets the default filenames
        if i == 0:
            generate_correlation_table(corr_data, tables_dir / "table2_correlation.tex", label)
            generate_inter_metric_table(corr_data, tables_dir / "table3_inter_metric.tex")
            generate_correlation_figure(corr_data, figures_dir / "correlation_scatter.pdf")
            generate_heatmap_figure(corr_data, figures_dir / "inter_metric_heatmap.pdf")
        # All models get suffixed versions
        generate_correlation_table(corr_data, tables_dir / f"table2_correlation_{model}.tex", label)
        generate_inter_metric_table(corr_data, tables_dir / f"table3_inter_metric_{model}.tex")
        generate_correlation_figure(corr_data, figures_dir / f"correlation_scatter_{model}.pdf")
        generate_heatmap_figure(corr_data, figures_dir / f"inter_metric_heatmap_{model}.pdf")

    # ── Perplexity comparison ─────────────────────────────────────────────
    ppl_results = []
    for ppl_file in found["perplexity"]:
        data = load_json_safe(ppl_file)
        if data:
            ppl_results.append(data)

    # Also extract 8B baselines + composite from ablation data for a complete table
    for ablation_file in found["ablation"]:
        abl = load_json_safe(ablation_file)
        if not abl:
            continue
        model_name = abl.get("model_name", "unknown")
        baselines = abl.get("baselines", {})
        for bname, bdata in baselines.items():
            if bdata.get("perplexity") is not None:
                ppl_results.append({
                    "model_name": model_name,
                    "variant": bname,
                    "perplexity": bdata["perplexity"],
                    "avg_bits": bdata.get("bits"),
                    "model_info": {"size_gb": bdata.get("size_gb", 0), "quantization": {"bits": bdata.get("bits")}},
                })
        # Add the default composite config
        composite = abl.get("configs", {}).get("composite_default", {})
        if composite.get("perplexity") is not None:
            ppl_results.append({
                "model_name": model_name,
                "variant": "smartquant",
                "perplexity": composite["perplexity"],
                "avg_bits": composite.get("average_bits"),
                "model_info": {"size_gb": composite.get("actual_size_gb", composite.get("estimated_size_gb", 0))},
            })

    # Sort: by model name, then by avg_bits descending
    def sort_key(e):
        m = e.get("model_name", "")
        b = e.get("avg_bits") or e.get("model_info", {}).get("quantization", {}).get("bits") or 0
        return (m, -b)
    ppl_results.sort(key=sort_key)

    if ppl_results:
        generate_perplexity_table(ppl_results, tables_dir / "table4_perplexity_comparison.tex")
        generate_ppl_vs_bits_figure(ppl_results, figures_dir / "perplexity_vs_bits.pdf")

    # ── Scaling study ─────────────────────────────────────────────────────
    if args.scaling_manifests:
        scaling_data = load_scaling_data(args.scaling_manifests)
        if scaling_data:
            generate_scaling_table(scaling_data, tables_dir / "table5_scaling.tex")
            generate_scaling_figure(scaling_data, figures_dir / "scaling_bit_allocation.pdf")

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info(f"\nCompilation complete.")
    logger.info(f"Figures: {figures_dir}")
    logger.info(f"Tables: {tables_dir}")

    # List generated files
    for d, label in [(figures_dir, "Figures"), (tables_dir, "Tables")]:
        files = sorted(d.glob("*"))
        if files:
            logger.info(f"\n{label}:")
            for f in files:
                logger.info(f"  {f.name} ({f.stat().st_size / 1024:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(description="SmartQuant Results Compiler")
    parser.add_argument("--results-dir", default="journal/results",
                        help="Directory containing experiment result JSONs")
    parser.add_argument("--figures-dir", default="journal/figures",
                        help="Output directory for figures")
    parser.add_argument("--tables-dir", default="journal/tables",
                        help="Output directory for LaTeX tables")
    parser.add_argument("--scaling-manifests", nargs="*",
                        help="Manifest JSON paths for scaling study")
    args = parser.parse_args()
    run_compile(args)


if __name__ == "__main__":
    main()
