"""Utility functions for SmartQuant: tensor name parsing, logging, pattern matching."""

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


def setup_logging(log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    """Configure logging for SmartQuant."""
    logger = logging.getLogger("smartquant")
    logger.setLevel(level)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_file))
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def parse_tensor_name(name: str) -> Dict[str, Optional[str]]:
    """Parse a Llama 4 Maverick tensor name into components.

    Examples:
        "model.layers.5.self_attn.q_proj.weight"
        -> {"prefix": "model", "layer": "5", "block": "self_attn",
            "module": "q_proj", "param": "weight", "expert": None}

        "model.layers.10.feed_forward.experts.42.gate_proj.weight"
        -> {"prefix": "model", "layer": "10", "block": "feed_forward",
            "module": "gate_proj", "param": "weight", "expert": "42"}
    """
    parts = name.split(".")
    result = {
        "prefix": None,
        "layer": None,
        "block": None,
        "module": None,
        "param": None,
        "expert": None,
        "full_name": name,
    }

    # model.layers.{N}.*
    layer_match = re.search(r"layers\.(\d+)", name)
    if layer_match:
        result["layer"] = layer_match.group(1)

    # expert index
    expert_match = re.search(r"experts\.(\d+)", name)
    if expert_match:
        result["expert"] = expert_match.group(1)

    # Block type
    if "self_attn" in name:
        result["block"] = "self_attn"
    elif "feed_forward" in name or "mlp" in name:
        result["block"] = "feed_forward"

    # Module name (last component before .weight/.bias)
    if len(parts) >= 2:
        result["param"] = parts[-1]
        result["module"] = parts[-2]

    # Special cases
    if "embed_tokens" in name:
        result["block"] = "embedding"
        result["module"] = "embed_tokens"
    elif "lm_head" in name:
        result["block"] = "head"
        result["module"] = "lm_head"
    elif "norm" in name.lower() and "layernorm" not in name.lower():
        if result["block"] is None:
            result["block"] = "norm"

    return result


def classify_tensor(name: str, protected_patterns: List[str], sensitive_patterns: List[str]) -> str:
    """Classify a tensor as 'protected', 'sensitive', or 'default'.

    Args:
        name: Full tensor name.
        protected_patterns: Patterns that trigger 16-bit protection.
        sensitive_patterns: Patterns that trigger 8-bit default.

    Returns:
        One of 'protected', 'sensitive', 'default'.
    """
    name_lower = name.lower()

    for pattern in protected_patterns:
        if pattern.lower() in name_lower:
            return "protected"

    for pattern in sensitive_patterns:
        if pattern.lower() in name_lower:
            return "sensitive"

    return "default"


def is_1d_tensor(shape: Tuple[int, ...]) -> bool:
    """Check if a tensor shape is effectively 1D (norms, biases)."""
    return len(shape) <= 1 or (len(shape) == 2 and min(shape) == 1)


def discover_tensor_patterns(tensor_names: List[str]) -> Dict[str, List[str]]:
    """Discover unique tensor name patterns by collapsing numeric indices.

    Replaces layer numbers and expert numbers with {N} and {E} placeholders,
    then groups tensors by their pattern.

    Returns:
        Dict mapping pattern string to list of matching tensor names.
    """
    pattern_groups: Dict[str, List[str]] = defaultdict(list)

    for name in tensor_names:
        # Replace layer indices: layers.0 -> layers.{N}
        pattern = re.sub(r"layers\.\d+", "layers.{N}", name)
        # Replace expert indices: experts.0 -> experts.{E}
        pattern = re.sub(r"experts\.\d+", "experts.{E}", pattern)
        pattern_groups[pattern].append(name)

    return dict(pattern_groups)


def estimate_quantized_size(
    param_counts: Dict[int, int],
    group_size_4bit: int = 128,
    group_size_8bit: int = 64,
) -> float:
    """Estimate the total size in GB for quantized model.

    Args:
        param_counts: Dict mapping bit-width (4, 8, 16) to number of parameters.

    Returns:
        Estimated size in GB.
    """
    total_bytes = 0
    for bits, count in param_counts.items():
        if bits == 16:
            total_bytes += count * 2  # float16 = 2 bytes per param
        elif bits == 8:
            # 8-bit with group scales: 1 byte per param + scale overhead
            group_size = group_size_8bit
            num_groups = count / group_size
            total_bytes += count * 1 + num_groups * 2  # scales are float16
        elif bits == 4:
            # 4-bit with group scales: 0.5 bytes per param + scale overhead
            group_size = group_size_4bit
            num_groups = count / group_size
            total_bytes += count * 0.5 + num_groups * 2
        elif bits == 2:
            # 2-bit with group scales: 0.25 bytes per param + scale overhead
            group_size = 32  # smaller groups for 2-bit
            num_groups = count / group_size
            total_bytes += count * 0.25 + num_groups * 2
        else:
            total_bytes += count * 2  # fallback to float16

    return total_bytes / (1024 ** 3)


def format_params(n: int) -> str:
    """Format parameter count for display."""
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    elif n >= 1e6:
        return f"{n / 1e6:.1f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)
