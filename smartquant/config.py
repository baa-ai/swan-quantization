"""Configuration dataclass for SmartQuant analysis and quantization."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class QuantConfig:
    """Configuration for SmartQuant sensitivity analysis and quantization.

    Attributes:
        sensitivity_threshold_8bit: Composite score above which a tensor gets 8-bit.
        sensitivity_threshold_16bit: Composite score above which a tensor gets 16-bit.
        default_bits: Default quantization bit-width.
        group_size_4bit: Group size for 4-bit quantization.
        group_size_8bit: Group size for 8-bit quantization.
        metric_weights: Weights for combining the four sensitivity metrics
            into a composite score. Keys: svd, kurtosis, output_sensitivity, cross_layer.
        protected_patterns: Tensor name patterns that are always kept at 16-bit.
        sensitive_patterns: Tensor name patterns that default to 8-bit.
        svd_rank: Rank for randomized SVD (higher = more accurate, slower).
        min_tensor_params: Skip analysis for tensors with fewer params than this.
        device: Compute device for analysis ('cpu' or 'mps').
    """

    sensitivity_threshold_2bit: float = 0.10
    sensitivity_threshold_8bit: float = 0.65
    sensitivity_threshold_16bit: float = 0.90
    default_bits: int = 4
    group_size_4bit: int = 128
    group_size_8bit: int = 64

    metric_weights: Dict[str, float] = field(default_factory=lambda: {
        "svd": 0.20,
        "kurtosis": 0.45,
        "output_sensitivity": 0.15,
        "error_proxy": 0.20,
    })

    protected_patterns: List[str] = field(default_factory=lambda: [
        "embed_tokens",
        "lm_head",
        "layernorm",
        "layer_norm",
        "norm.weight",
        "router.weight",
        # Vision model components — keep at full precision
        "vision_model",
        "multi_modal_projector",
        # Positional embeddings
        "positional_embedding",
        "class_embedding",
        "patch_embedding",
    ])

    sensitive_patterns: List[str] = field(default_factory=lambda: [])

    svd_rank: int = 256
    min_tensor_params: int = 1024
    device: str = "cpu"

    def validate(self) -> None:
        """Validate configuration values."""
        assert 0 <= self.sensitivity_threshold_2bit < self.sensitivity_threshold_8bit < self.sensitivity_threshold_16bit <= 1.0, (
            f"Thresholds must satisfy 0 <= 2bit ({self.sensitivity_threshold_2bit}) "
            f"< 8bit ({self.sensitivity_threshold_8bit}) "
            f"< 16bit ({self.sensitivity_threshold_16bit}) <= 1.0"
        )
        assert self.default_bits in (2, 4, 8, 16), f"default_bits must be 2, 4, 8, or 16, got {self.default_bits}"
        total_weight = sum(self.metric_weights.values())
        assert abs(total_weight - 1.0) < 1e-6, (
            f"metric_weights must sum to 1.0, got {total_weight}"
        )
        expected_keys_v2 = {"svd", "kurtosis", "output_sensitivity", "error_proxy"}
        expected_keys_v1 = {"svd", "kurtosis", "output_sensitivity", "cross_layer"}
        actual_keys = set(self.metric_weights.keys())
        assert actual_keys == expected_keys_v2 or actual_keys == expected_keys_v1, (
            f"metric_weights must have keys {expected_keys_v2} or {expected_keys_v1}, got {actual_keys}"
        )
