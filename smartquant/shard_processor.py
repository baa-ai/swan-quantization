"""
Shard processor for SmartQuant analysis.

Processes safetensor shards one at a time, loading tensors individually
to keep memory usage minimal. Orchestrates the sensitivity analysis
across all shards and produces the analysis manifest.
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from safetensors import safe_open
from tqdm import tqdm

from .analyzer import SensitivityAnalyzer
from .config import QuantConfig
from .utils import (
    classify_tensor,
    discover_tensor_patterns,
    estimate_quantized_size,
    format_params,
    is_1d_tensor,
    parse_tensor_name,
)

logger = logging.getLogger("smartquant.shard_processor")


def _detect_total_layers(weight_map: Dict[str, str]) -> int:
    """Detect the total number of transformer layers from the weight map."""
    layer_indices = set()
    for name in weight_map:
        match = re.search(r"layers\.(\d+)", name)
        if match:
            layer_indices.add(int(match.group(1)))
    return max(layer_indices) + 1 if layer_indices else 0


def _get_layer_idx(name: str) -> Optional[int]:
    """Extract layer index from a tensor name."""
    match = re.search(r"layers\.(\d+)", name)
    return int(match.group(1)) if match else None


def discover_and_print_patterns(model_dir: Path) -> Dict[str, List[str]]:
    """Discover all unique tensor name patterns in the model and print them.

    This should be the first thing run before full analysis so the user
    can verify tensor naming conventions.
    """
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"No model index found at {index_path}")

    with open(index_path) as f:
        index = json.load(f)

    weight_map = index.get("weight_map", {})
    patterns = discover_tensor_patterns(list(weight_map.keys()))

    logger.info(f"Found {len(weight_map)} tensors in {len(set(weight_map.values()))} shards")
    logger.info(f"Discovered {len(patterns)} unique tensor name patterns:")
    for pattern, names in sorted(patterns.items()):
        logger.info(f"  {pattern} ({len(names)} tensors)")

    return patterns


def process_all_shards(
    model_dir: Path,
    config: QuantConfig,
    output_path: Path,
) -> Dict[str, Any]:
    """Process all safetensor shards and produce the analysis manifest.

    Args:
        model_dir: Directory containing the BF16 safetensor files.
        config: SmartQuant configuration.
        output_path: Path to write the manifest JSON.

    Returns:
        The manifest dict (also written to output_path).
    """
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"No model index found at {index_path}")

    with open(index_path) as f:
        index = json.load(f)

    weight_map = index.get("weight_map", {})
    total_layers = _detect_total_layers(weight_map)
    logger.info(f"Detected {total_layers} transformer layers")

    # Discover patterns first
    discover_and_print_patterns(model_dir)

    # Group tensors by shard
    shard_tensors: Dict[str, List[str]] = {}
    for tensor_name, shard_name in weight_map.items():
        shard_tensors.setdefault(shard_name, []).append(tensor_name)

    analyzer = SensitivityAnalyzer(config)

    manifest = {
        "model": str(model_dir),
        "config": {
            "sensitivity_threshold_2bit": getattr(config, "sensitivity_threshold_2bit", None),
            "sensitivity_threshold_8bit": config.sensitivity_threshold_8bit,
            "sensitivity_threshold_16bit": config.sensitivity_threshold_16bit,
            "default_bits": config.default_bits,
            "metric_weights": config.metric_weights,
            "svd_rank": config.svd_rank,
        },
        "total_layers": total_layers,
        "total_tensors": len(weight_map),
        "shards": {},
    }

    total_params = 0
    bits_counts: Dict[int, int] = {2: 0, 4: 0, 8: 0, 16: 0}
    tensor_count = 0
    start_time = time.time()

    sorted_shards = sorted(shard_tensors.keys())
    progress = tqdm(sorted_shards, desc="Processing shards", unit="shard")

    for shard_name in progress:
        tensor_names = shard_tensors[shard_name]
        shard_path = model_dir / shard_name

        if not shard_path.exists():
            logger.warning(f"Shard {shard_name} not found, skipping")
            continue

        shard_results = {"file": shard_name, "tensors": {}}

        with safe_open(str(shard_path), framework="pt") as f:
            for tensor_name in tensor_names:
                tensor_count += 1
                progress.set_postfix(tensor=tensor_name.split(".")[-2], refresh=False)

                tensor = f.get_tensor(tensor_name)
                shape = tuple(tensor.shape)
                num_params = tensor.numel()
                total_params += num_params
                dtype = str(tensor.dtype)

                # Classify by name pattern
                tensor_class = classify_tensor(
                    tensor_name,
                    config.protected_patterns,
                    config.sensitive_patterns,
                )

                # Quick path for 1D tensors (norms, biases)
                if is_1d_tensor(shape):
                    # Use the same metric key as the config
                    fourth_key = "error_proxy" if "error_proxy" in config.metric_weights else "cross_layer"
                    shard_results["tensors"][tensor_name] = {
                        "shape": list(shape),
                        "dtype": dtype,
                        "num_params": num_params,
                        "classification": "protected",
                        "scores": {
                            "svd": 0.0,
                            "kurtosis": 0.0,
                            "output_sensitivity": 0.0,
                            fourth_key: 0.0,
                        },
                        "composite_score": 0.0,
                        "decision": {"bits": 16, "reason": "1d_tensor"},
                    }
                    bits_counts[16] += num_params
                    continue

                layer_idx = _get_layer_idx(tensor_name)

                # Run sensitivity analysis
                analysis = analyzer.analyze_tensor(
                    tensor,
                    name=tensor_name,
                    layer_idx=layer_idx,
                    total_layers=total_layers,
                )

                # Determine bit-width
                bits, reason = analyzer.recommend_bits(
                    analysis["composite_score"],
                    tensor_class,
                )

                shard_results["tensors"][tensor_name] = {
                    "shape": list(shape),
                    "dtype": dtype,
                    "num_params": num_params,
                    "classification": tensor_class,
                    "scores": analysis["scores"],
                    "composite_score": analysis["composite_score"],
                    "decision": {"bits": bits, "reason": reason},
                }
                bits_counts[bits] += num_params

                # Free tensor memory
                del tensor

        manifest["shards"][shard_name] = shard_results

    elapsed = time.time() - start_time

    # Summary statistics
    manifest["summary"] = {
        "total_params": total_params,
        "bits_distribution": {
            str(b): {
                "params": c,
                "percentage": c / total_params * 100 if total_params > 0 else 0,
            }
            for b, c in sorted(bits_counts.items())
        },
        "estimated_size_gb": estimate_quantized_size(
            bits_counts,
            config.group_size_4bit,
            config.group_size_8bit,
        ),
        "average_bits": (
            sum(b * c for b, c in bits_counts.items()) / total_params
            if total_params > 0
            else 0
        ),
        "analysis_time_seconds": elapsed,
        "tensor_count": tensor_count,
    }

    # Write manifest
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"SmartQuant Analysis Complete")
    logger.info(f"{'='*60}")
    logger.info(f"Total parameters: {format_params(total_params)}")
    logger.info(f"Analysis time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    logger.info(f"")
    for bits in sorted(bits_counts):
        count = bits_counts[bits]
        pct = count / total_params * 100 if total_params > 0 else 0
        logger.info(f"  {bits:2d}-bit: {format_params(count):>10s} params ({pct:.1f}%)")
    avg_bits = manifest["summary"]["average_bits"]
    est_gb = manifest["summary"]["estimated_size_gb"]
    logger.info(f"")
    logger.info(f"Average bits per param: {avg_bits:.2f}")
    logger.info(f"Estimated quantized size: {est_gb:.0f} GB")
    logger.info(f"Manifest written to: {output_path}")

    return manifest
