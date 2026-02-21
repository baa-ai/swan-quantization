#!/usr/bin/env python3
"""
Generate a summary report of the SmartQuant experiment.

Reads analysis manifest, benchmark results, and optional A/B comparison
to produce a markdown report.

Usage:
    python generate_report.py
    python generate_report.py --output ~/smartquant/results/report.md
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_DIR = Path.home() / "smartquant"


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load a JSON file, returning None if it doesn't exist."""
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def format_params(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    elif n >= 1e6:
        return f"{n / 1e6:.1f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


def generate_analysis_section(manifest: Dict[str, Any]) -> str:
    """Generate the Model Analysis Summary section."""
    lines = ["## 1. Model Analysis Summary", ""]

    summary = manifest.get("summary", {})
    total_params = summary.get("total_params", 0)
    bits_dist = summary.get("bits_distribution", {})

    lines.append(f"**Total parameters**: {format_params(total_params)}")
    lines.append(f"**Total tensors**: {manifest.get('total_tensors', 0)}")
    lines.append(f"**Transformer layers**: {manifest.get('total_layers', 0)}")
    lines.append(f"**Analysis time**: {summary.get('analysis_time_seconds', 0):.0f}s")
    lines.append("")

    # Bit-width distribution table
    lines.append("### Bit-Width Distribution")
    lines.append("")
    lines.append("| Bit-Width | Parameters | Percentage |")
    lines.append("|-----------|-----------|------------|")
    for bits in sorted(bits_dist.keys(), key=int):
        info = bits_dist[bits]
        params = info["params"]
        pct = info["percentage"]
        lines.append(f"| {bits}-bit | {format_params(params)} | {pct:.1f}% |")
    lines.append("")

    avg_bits = summary.get("average_bits", 0)
    est_gb = summary.get("estimated_size_gb", 0)
    lines.append(f"**Average bits per parameter**: {avg_bits:.2f}")
    lines.append(f"**Estimated quantized model size**: {est_gb:.0f} GB")
    lines.append("")

    # Protection analysis: count tensors by decision reason
    reason_counts: Counter = Counter()
    composite_scores = []
    sensitive_by_analysis = []
    for shard_data in manifest["shards"].values():
        for tensor_name, info in shard_data["tensors"].items():
            reason = info["decision"]["reason"]
            reason_counts[reason] += 1
            cs = info.get("composite_score", 0)
            composite_scores.append(cs)
            if "composite_score" in reason and info["decision"]["bits"] == 8:
                sensitive_by_analysis.append((tensor_name, cs))

    lines.append("### Decision Reasons")
    lines.append("")
    lines.append("| Reason | Count |")
    lines.append("|--------|-------|")
    for reason, count in reason_counts.most_common():
        lines.append(f"| {reason} | {count} |")
    lines.append("")

    if sensitive_by_analysis:
        lines.append("### Tensors Flagged Sensitive by Analysis (not by pattern)")
        lines.append("")
        lines.append(
            f"**{len(sensitive_by_analysis)}** tensors were promoted to 8-bit "
            "based on composite score (not name pattern)."
        )
        lines.append("")
        # Show top 10
        for name, score in sorted(sensitive_by_analysis, key=lambda x: -x[1])[:10]:
            lines.append(f"- `{name}` (score: {score:.3f})")
        if len(sensitive_by_analysis) > 10:
            lines.append(f"- ... and {len(sensitive_by_analysis) - 10} more")
        lines.append("")

    # Composite score histogram (text-based)
    if composite_scores:
        lines.append("### Composite Score Distribution")
        lines.append("")
        buckets = [0] * 10
        for s in composite_scores:
            idx = min(9, int(s * 10))
            buckets[idx] += 1
        max_count = max(buckets) if buckets else 1
        for i, count in enumerate(buckets):
            bar = "#" * int(count / max_count * 40) if max_count > 0 else ""
            lines.append(f"  {i * 0.1:.1f}-{(i + 1) * 0.1:.1f}: {bar} ({count})")
        lines.append("")

    return "\n".join(lines)


def generate_performance_section(
    benchmark: Optional[Dict[str, Any]],
    manifest: Optional[Dict[str, Any]],
) -> str:
    """Generate the Memory and Performance section."""
    lines = ["## 2. Memory and Performance", ""]

    if manifest:
        summary = manifest.get("summary", {})
        lines.append(f"**Predicted model size**: {summary.get('estimated_size_gb', 0):.0f} GB")
        lines.append("")

    if not benchmark:
        lines.append("*No benchmark results available. Run test_harness.py --mode benchmark.*")
        lines.append("")
        return "\n".join(lines)

    bsummary = benchmark.get("summary", {})
    mem = bsummary.get("memory", {})

    lines.append(f"**Memory allocated during inference**: {mem.get('allocated_gb', 0):.1f} GB")
    lines.append(f"**Peak memory during inference**: {mem.get('peak_gb', 0):.1f} GB")
    lines.append("")

    lines.append("### Generation Speed")
    lines.append("")
    lines.append(f"- Average: {bsummary.get('avg_tokens_per_second', 0):.1f} tok/s")
    lines.append(f"- Min: {bsummary.get('min_tokens_per_second', 0):.1f} tok/s")
    lines.append(f"- Max: {bsummary.get('max_tokens_per_second', 0):.1f} tok/s")
    lines.append(f"- Total tokens: {bsummary.get('total_tokens', 0)}")
    lines.append(f"- Total time: {bsummary.get('total_time_seconds', 0):.1f}s")
    lines.append("")

    # Per-category speed
    lines.append("### Speed by Category")
    lines.append("")
    lines.append("| Category | Avg tok/s | Prompts |")
    lines.append("|----------|----------|---------|")
    for category, results in benchmark.get("categories", {}).items():
        if results:
            avg = sum(r["tokens_per_second"] for r in results) / len(results)
            lines.append(f"| {category} | {avg:.1f} | {len(results)} |")
    lines.append("")

    return "\n".join(lines)


def generate_quality_section(
    benchmark: Optional[Dict[str, Any]],
    comparison: Optional[Dict[str, Any]],
) -> str:
    """Generate the Quality Assessment section."""
    lines = ["## 3. Quality Assessment", ""]

    if not benchmark:
        lines.append("*No benchmark results available.*")
        lines.append("")
        return "\n".join(lines)

    # Show first response from each category
    lines.append("### Sample Responses")
    lines.append("")
    for category, results in benchmark.get("categories", {}).items():
        if results:
            r = results[0]
            lines.append(f"**{category}**")
            lines.append(f"- Q: {r['prompt']}")
            # Truncate long responses
            resp = r["response"]
            if len(resp) > 300:
                resp = resp[:300] + "..."
            lines.append(f"- A: {resp}")
            lines.append("")

    if comparison:
        lines.append("### A/B Comparison Summary")
        lines.append("")
        csummary = comparison.get("summary", {})
        lines.append(f"- Model A (SmartQuant) speed: {csummary.get('speed_a', 0):.1f} tok/s")
        lines.append(f"- Model B (Baseline) speed: {csummary.get('speed_b', 0):.1f} tok/s")
        lines.append(f"- Speed difference: {csummary.get('speed_diff_pct', 0):+.1f}%")
        lines.append(f"- Model A peak memory: {csummary.get('mem_a_peak_gb', 0):.1f} GB")
        lines.append(f"- Model B peak memory: {csummary.get('mem_b_peak_gb', 0):.1f} GB")
        lines.append("")

    return "\n".join(lines)


def generate_recommendations_section(
    manifest: Optional[Dict[str, Any]],
    benchmark: Optional[Dict[str, Any]],
    comparison: Optional[Dict[str, Any]],
) -> str:
    """Generate the Recommendations section."""
    lines = ["## 4. Recommendations", ""]

    if manifest:
        summary = manifest.get("summary", {})
        avg_bits = summary.get("average_bits", 0)
        bits_dist = summary.get("bits_distribution", {})

        # Check if the mix is too conservative or aggressive
        pct_8bit = bits_dist.get("8", {}).get("percentage", 0)
        pct_16bit = bits_dist.get("16", {}).get("percentage", 0)

        if pct_8bit < 5:
            lines.append(
                "- **Consider lowering `sensitivity_threshold_8bit`**: "
                f"Only {pct_8bit:.1f}% of parameters are at 8-bit. "
                "Lowering the threshold may improve quality for sensitive layers."
            )
        elif pct_8bit > 30:
            lines.append(
                "- **Consider raising `sensitivity_threshold_8bit`**: "
                f"{pct_8bit:.1f}% of parameters are at 8-bit, which increases model size. "
                "Raising the threshold may reduce size with minimal quality loss."
            )

        if avg_bits < 4.2:
            lines.append(
                f"- **Average bits ({avg_bits:.2f}) is close to uniform 4-bit**: "
                "SmartQuant's differentiation may be minimal. Consider whether "
                "the complexity is justified."
            )
        elif avg_bits > 6.0:
            lines.append(
                f"- **Average bits ({avg_bits:.2f}) is high**: "
                "The model is larger than necessary. Review whether all 8-bit/16-bit "
                "allocations are justified."
            )

    if comparison:
        csummary = comparison.get("summary", {})
        speed_diff = csummary.get("speed_diff_pct", 0)
        if abs(speed_diff) < 5:
            lines.append(
                "- **Speed difference is minimal**: SmartQuant has similar performance "
                "to uniform quantization. Focus evaluation on quality differences."
            )

    lines.append("")
    lines.append("### Suggested Next Experiments")
    lines.append("")
    lines.append("1. Test with longer context lengths (8K, 32K) to stress KV cache precision")
    lines.append("2. Run with different sensitivity thresholds to find the quality-size sweet spot")
    lines.append("3. Compare perplexity on a held-out dataset for objective quality measurement")
    lines.append("4. Test on domain-specific tasks (math, code, multilingual) for targeted assessment")
    lines.append("")

    return "\n".join(lines)


def generate_report(
    manifest_path: Path,
    benchmark_path: Path,
    comparison_path: Path,
    output_path: Path,
) -> str:
    """Generate the full experiment report."""
    manifest = load_json(manifest_path)
    benchmark = load_json(benchmark_path)
    comparison = load_json(comparison_path)

    sections = [
        "# SmartQuant Experiment Report",
        "",
        f"Generated: {__import__('time').strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    if manifest:
        sections.append(generate_analysis_section(manifest))
    else:
        sections.append("## 1. Model Analysis Summary\n\n*No manifest found.*\n")

    sections.append(generate_performance_section(benchmark, manifest))
    sections.append(generate_quality_section(benchmark, comparison))
    sections.append(generate_recommendations_section(manifest, benchmark, comparison))

    report = "\n".join(sections)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

    print(f"Report written to: {output_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Generate SmartQuant experiment report")
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_DIR / "analysis" / "manifest.json"),
        help="Path to analysis manifest",
    )
    parser.add_argument(
        "--benchmark",
        default=str(DEFAULT_DIR / "results" / "benchmark_results.json"),
        help="Path to benchmark results",
    )
    parser.add_argument(
        "--comparison",
        default=str(DEFAULT_DIR / "results" / "comparison_report.json"),
        help="Path to A/B comparison results",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_DIR / "results" / "report.md"),
        help="Output path for the report",
    )
    args = parser.parse_args()

    report = generate_report(
        manifest_path=Path(args.manifest),
        benchmark_path=Path(args.benchmark),
        comparison_path=Path(args.comparison),
        output_path=Path(args.output),
    )

    # Also print to stdout
    print("\n" + "=" * 60)
    print(report)


if __name__ == "__main__":
    main()
