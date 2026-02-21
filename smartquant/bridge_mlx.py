"""
Bridge between SmartQuant analysis manifest and MLX quantization predicate.

Reads the SmartQuant manifest.json and generates either:
1. A quant_predicate function for use with mlx_lm.convert (in-memory)
2. A standalone Python module file that can be imported by mlx-lm (on disk)
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger("smartquant.bridge_mlx")


def load_manifest(manifest_path: Path) -> Dict[str, Any]:
    """Load a SmartQuant analysis manifest."""
    with open(manifest_path) as f:
        return json.load(f)


def build_bits_lookup(manifest: Dict[str, Any]) -> Dict[str, int]:
    """Build a flat lookup from tensor name to recommended bit-width.

    Args:
        manifest: SmartQuant analysis manifest.

    Returns:
        Dict mapping tensor name (e.g., "model.layers.0.self_attn.q_proj.weight")
        to bit-width (4, 8, or 16).
    """
    lookup = {}
    for shard_data in manifest["shards"].values():
        for tensor_name, info in shard_data["tensors"].items():
            bits = info["decision"]["bits"]
            lookup[tensor_name] = bits
    return lookup


def tensor_name_to_module_name(tensor_name: str) -> str:
    """Convert a tensor name to an MLX module name.

    MLX uses module paths without the final '.weight' or '.bias' suffix.
    E.g., "model.layers.0.self_attn.q_proj.weight" -> "model.layers.0.self_attn.q_proj"
    """
    if tensor_name.endswith(".weight") or tensor_name.endswith(".bias"):
        return tensor_name.rsplit(".", 1)[0]
    return tensor_name


def create_quant_predicate(
    manifest: Dict[str, Any],
    default_bits: int = 4,
    default_group_size: int = 128,
) -> Callable:
    """Create an MLX quant_predicate function from a SmartQuant manifest.

    The returned function has the signature expected by mlx_lm.convert:
        def predicate(name, module, config) -> tuple[int, int] | None

    Args:
        manifest: SmartQuant analysis manifest.
        default_bits: Fallback bit-width for unknown tensors.
        default_group_size: Default group size.

    Returns:
        A quant_predicate function.
    """
    bits_lookup = build_bits_lookup(manifest)

    # Build module-level lookup (strip .weight/.bias)
    module_bits: Dict[str, int] = {}
    for tensor_name, bits in bits_lookup.items():
        module_name = tensor_name_to_module_name(tensor_name)
        module_bits[module_name] = bits

    # Summary stats
    from collections import Counter
    bit_counts = Counter(bits_lookup.values())
    total = sum(bit_counts.values())
    logger.info("SmartQuant predicate summary:")
    for b in sorted(bit_counts):
        logger.info(f"  {b:2d}-bit: {bit_counts[b]} tensors ({bit_counts[b]/total*100:.1f}%)")

    def smartquant_predicate(
        path: str,
        module: Any,
    ):
        """MLX quant predicate implementing SmartQuant's per-layer bit allocation.

        Signature matches mlx_lm's expected: (path: str, module: nn.Module) -> bool | dict
        Returns:
            False — skip quantization (keep full precision)
            True — quantize with default settings
            dict — quantize with custom settings, e.g. {"bits": 8, "group_size": 64}
        """
        name = path

        # Never quantize 1D modules (norms, biases)
        if hasattr(module, "weight"):
            w = module.weight
            if hasattr(w, "shape") and len(w.shape) <= 1:
                return False

        # Check if this is a norm or embedding that shouldn't be quantized
        name_lower = name.lower()
        if any(kw in name_lower for kw in ["layernorm", "layer_norm", "rmsnorm"]):
            return False
        if "embed_tokens" in name_lower or "lm_head" in name_lower:
            return False
        if name_lower.endswith("router") or name_lower.endswith("gate"):
            return False
        # model.norm / language_model.model.norm is the final RMSNorm
        if re.search(r"model\.norm$", name):
            return False
        # Vision model and multimodal projector stay at full precision
        if "vision_model" in name_lower or "multi_modal_projector" in name_lower:
            return False

        # Look up in manifest — try multiple name variants
        bits = module_bits.get(name)

        if bits is None:
            # Try with .weight suffix
            bits = bits_lookup.get(name + ".weight")

        if bits is None:
            # MLX may strip the "language_model." prefix; try adding it
            bits = module_bits.get("language_model." + name)

        if bits is None:
            bits = bits_lookup.get("language_model." + name + ".weight")

        if bits is None:
            # Fallback to default
            bits = default_bits

        if bits == 16:
            return False  # Keep full precision

        if bits == 8:
            return {"bits": 8, "group_size": 64}

        if bits == 2:
            return {"bits": 2, "group_size": 32}

        # Default: 4-bit
        return True  # Use the default bits/group_size from convert()

    return smartquant_predicate


def generate_predicate_module(
    manifest_path: Path,
    output_path: Path,
    default_bits: int = 4,
) -> None:
    """Generate a standalone Python module with the quant predicate.

    This creates a file that can be imported independently by mlx-lm.

    Args:
        manifest_path: Path to SmartQuant manifest.json.
        output_path: Path to write the Python module.
        default_bits: Fallback bit-width.
    """
    manifest = load_manifest(manifest_path)
    bits_lookup = build_bits_lookup(manifest)

    # Build the module-level lookup
    module_bits = {}
    for tensor_name, bits in bits_lookup.items():
        module_name = tensor_name_to_module_name(tensor_name)
        module_bits[module_name] = bits

    # Count stats
    from collections import Counter
    param_counts: Dict[int, int] = Counter()
    for shard_data in manifest["shards"].values():
        for tensor_name, info in shard_data["tensors"].items():
            param_counts[info["decision"]["bits"]] += info["num_params"]

    total_params = sum(param_counts.values())

    # Generate the module source
    lines = [
        '"""',
        "SmartQuant quantization predicate for MLX.",
        "",
        "Auto-generated from SmartQuant analysis manifest.",
        f"Total tensors: {len(bits_lookup)}",
        f"Total parameters: {total_params:,}",
    ]
    for b in sorted(param_counts):
        pct = param_counts[b] / total_params * 100 if total_params > 0 else 0
        lines.append(f"  {b}-bit: {param_counts[b]:,} params ({pct:.1f}%)")
    lines.append('"""')
    lines.append("")
    lines.append("import re")
    lines.append("from typing import Any, Optional, Tuple")
    lines.append("")
    lines.append("# Per-module bit-width lookup (module_name -> bits)")
    lines.append(f"_MODULE_BITS = {json.dumps(module_bits, indent=2)}")
    lines.append("")
    lines.append(f"# Per-tensor bit-width lookup (tensor_name -> bits)")
    lines.append(f"_TENSOR_BITS = {json.dumps(bits_lookup, indent=2)}")
    lines.append("")
    lines.append(f"DEFAULT_BITS = {default_bits}")
    lines.append("")
    lines.append("")
    lines.append("def smartquant_predicate(")
    lines.append("    name: str,")
    lines.append("    module: Any,")
    lines.append("    config: dict,")
    lines.append(") -> Optional[Tuple[int, int]]:")
    lines.append('    """MLX quant predicate implementing SmartQuant per-layer bit allocation."""')
    lines.append("    # Never quantize 1D modules")
    lines.append("    if hasattr(module, 'weight'):")
    lines.append("        w = module.weight")
    lines.append("        if hasattr(w, 'shape') and len(w.shape) <= 1:")
    lines.append("            return None")
    lines.append("")
    lines.append("    name_lower = name.lower()")
    lines.append("    # Skip norms, embeddings, routing")
    lines.append("    for kw in ['layernorm', 'layer_norm', 'rmsnorm']:")
    lines.append("        if kw in name_lower:")
    lines.append("            return None")
    lines.append("    if 'embed_tokens' in name_lower or 'lm_head' in name_lower:")
    lines.append("        return None")
    lines.append("    if name_lower.endswith('router') or name_lower.endswith('gate'):")
    lines.append("        return None")
    lines.append("    if re.match(r'^model\\.norm$', name):")
    lines.append("        return None")
    lines.append("")
    lines.append("    # Look up in manifest")
    lines.append("    bits = _MODULE_BITS.get(name)")
    lines.append("    if bits is None:")
    lines.append("        bits = _TENSOR_BITS.get(name + '.weight')")
    lines.append("    if bits is None:")
    lines.append("        bits = DEFAULT_BITS")
    lines.append("")
    lines.append("    if bits == 16:")
    lines.append("        return None")
    lines.append("    if bits == 8:")
    lines.append("        return (8, 64)")
    lines.append("    return (4, 128)")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Predicate module written to: {output_path}")

    # Print summary
    print(f"\nSmartQuant Predicate Summary:")
    print(f"  Total parameters: {total_params:,}")
    for b in sorted(param_counts):
        pct = param_counts[b] / total_params * 100 if total_params > 0 else 0
        print(f"  {b:2d}-bit: {param_counts[b]:>15,} params ({pct:.1f}%)")
    avg = sum(b * c for b, c in param_counts.items()) / total_params if total_params > 0 else 0
    est_gb = manifest.get("summary", {}).get("estimated_size_gb", 0)
    print(f"  Average: {avg:.2f} bits/param")
    print(f"  Estimated model size: {est_gb:.0f} GB")
