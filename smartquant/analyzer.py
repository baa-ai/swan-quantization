"""
SmartQuant Sensitivity Analyzer.

Implements four sensitivity metrics for quantization analysis:
1. SVD-based spectral sensitivity — measures information density via singular value distribution
2. Kurtosis — measures weight distribution outlier-heaviness
3. Output sensitivity — estimates how quantization noise propagates through the layer
4. Cross-layer gradient proxy — heuristic for how deep in the network a layer sits

Each metric produces a normalised score in [0, 1]. The composite score is a weighted
combination of all four, used to decide per-tensor bit-width allocation.
"""

import logging
import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from .config import QuantConfig
from .utils import is_1d_tensor

logger = logging.getLogger("smartquant.analyzer")


class SensitivityAnalyzer:
    """Compute sensitivity metrics for a single weight tensor."""

    def __init__(self, config: QuantConfig):
        self.config = config

    def analyze_tensor(
        self,
        tensor: torch.Tensor,
        name: str,
        layer_idx: Optional[int] = None,
        total_layers: Optional[int] = None,
    ) -> Dict:
        """Run all four sensitivity metrics on a tensor.

        Args:
            tensor: Weight tensor (2D for linear layers).
            name: Tensor name for logging.
            layer_idx: Layer index (0-based), for cross-layer metric.
            total_layers: Total number of layers in model, for cross-layer metric.

        Returns:
            Dict with per-metric scores and composite score.
        """
        shape = tuple(tensor.shape)
        num_params = tensor.numel()

        # Skip tiny or 1D tensors
        if is_1d_tensor(shape) or num_params < self.config.min_tensor_params:
            return {
                "name": name,
                "shape": list(shape),
                "num_params": num_params,
                "scores": {
                    "svd": 0.0,
                    "kurtosis": 0.0,
                    "output_sensitivity": 0.0,
                    "cross_layer": 0.0,
                },
                "composite_score": 0.0,
                "skipped": True,
                "skip_reason": "1d_or_tiny",
            }

        # Ensure float32 for analysis
        t = tensor.float()

        # Handle 3D+ tensors (e.g., packed experts [128, 8192, 5120],
        # or 4D conv weights [out, in, kH, kW]) by reshaping to 2D
        if t.dim() == 3:
            t = self._sample_expert_slice(t)
        elif t.dim() > 3:
            # Collapse higher-dimensional tensors to 2D
            t = t.reshape(t.shape[0], -1)

        scores = {}

        # Metric 1: SVD spectral sensitivity
        scores["svd"] = self._svd_sensitivity(t, name)

        # Metric 2: Kurtosis
        scores["kurtosis"] = self._kurtosis_sensitivity(t)

        # Metric 3: Output sensitivity
        scores["output_sensitivity"] = self._output_sensitivity(t)

        # Metric 4: Reconstruction error proxy (replaces cross-layer position)
        # Accept either key name for backward compatibility
        if "error_proxy" in self.config.metric_weights:
            scores["error_proxy"] = self._reconstruction_error_sensitivity(t)
        elif "cross_layer" in self.config.metric_weights:
            scores["cross_layer"] = self._cross_layer_sensitivity(
                layer_idx, total_layers
            )

        # Composite score
        composite = sum(
            self.config.metric_weights[k] * scores[k] for k in scores
        )

        return {
            "name": name,
            "shape": list(shape),
            "num_params": num_params,
            "scores": scores,
            "composite_score": composite,
            "skipped": False,
        }

    def _sample_expert_slice(self, tensor: torch.Tensor, num_samples: int = 4) -> torch.Tensor:
        """Reduce a 3D packed expert tensor to 2D for analysis.

        For tensors shaped [num_experts, rows, cols], sample a few expert
        slices and concatenate them to get a representative 2D matrix.
        """
        num_experts = tensor.shape[0]
        if num_experts <= num_samples:
            # Few enough to just reshape
            return tensor.reshape(-1, tensor.shape[-1])

        # Sample evenly spaced experts (first, last, and middle)
        indices = torch.linspace(0, num_experts - 1, num_samples).long()
        sampled = tensor[indices]  # [num_samples, rows, cols]
        return sampled.reshape(-1, sampled.shape[-1])

    def _svd_sensitivity(self, tensor: torch.Tensor, name: str) -> float:
        """SVD-based spectral sensitivity.

        Measures how concentrated the information is in the top singular values.
        A tensor where most energy is in a few singular values is MORE sensitive
        to quantization (those values carry disproportionate importance).

        Score: ratio of energy in top-k singular values to total energy.
        Higher = more sensitive.
        """
        try:
            rows, cols = tensor.shape[0], tensor.shape[1] if tensor.dim() >= 2 else 1
            if tensor.dim() < 2:
                return 0.0

            # Use randomized SVD for large matrices
            k = min(self.config.svd_rank, min(rows, cols))

            if min(rows, cols) <= self.config.svd_rank * 2:
                # Full SVD for small matrices
                _, s, _ = torch.linalg.svd(tensor, full_matrices=False)
            else:
                # Randomized SVD
                _, s, _ = torch.svd_lowrank(tensor, q=k)

            s = s.abs()
            total_energy = (s ** 2).sum().item()
            if total_energy < 1e-12:
                return 0.0

            # Energy concentration in top 10% of singular values
            top_k = max(1, len(s) // 10)
            top_energy = (s[:top_k] ** 2).sum().item()

            concentration = top_energy / total_energy

            # Normalise: lower floor to capture more of the distribution
            # Maps [0.1, 0.9] to [0, 1]; 8B models have mean ~0.21
            score = min(1.0, max(0.0, (concentration - 0.1) / 0.8))
            return score

        except Exception as e:
            logger.warning(f"SVD failed for {name}: {e}")
            return 0.5  # Default to moderate sensitivity on failure

    def _kurtosis_sensitivity(self, tensor: torch.Tensor) -> float:
        """Kurtosis-based outlier sensitivity.

        High kurtosis means the weight distribution has heavy tails (outliers).
        Quantization is more destructive when there are outliers because the
        quantization grid must accommodate a wider range, reducing precision
        for the majority of values.

        Uses excess kurtosis (normal distribution = 0).
        """
        flat = tensor.flatten().float()
        n = flat.numel()
        if n < 4:
            return 0.0

        mean = flat.mean()
        centered = flat - mean
        var = (centered ** 2).mean()
        if var < 1e-12:
            return 0.0

        std = var.sqrt()
        kurt = ((centered / std) ** 4).mean().item() - 3.0  # excess kurtosis

        # Normal dist: kurt = 0. Outlier-heavy: kurt >> 0. Uniform-like: kurt < 0.
        # Normalise: kurtosis of 10+ is very outlier-heavy
        score = min(1.0, max(0.0, kurt / 10.0))
        return score

    def _output_sensitivity(self, tensor: torch.Tensor) -> float:
        """Output sensitivity: how much quantization noise gets amplified.

        Simulates adding quantization noise to the weights and measures the
        relative change in the layer's output norm. Approximated without
        actual input data by using random Gaussian inputs.

        A layer that amplifies small perturbations is more sensitive to quantization.
        """
        if tensor.dim() < 2:
            return 0.0

        # Ensure 2D for matrix multiplication
        if tensor.dim() > 2:
            tensor = tensor.reshape(tensor.shape[0], -1)

        rows, cols = tensor.shape
        # Use a small number of random input vectors for efficiency
        num_probes = min(32, cols)

        # Random unit-norm input vectors
        x = torch.randn(cols, num_probes, device=tensor.device, dtype=tensor.dtype)
        x = x / (x.norm(dim=0, keepdim=True) + 1e-8)

        # Clean output
        y_clean = tensor @ x  # [rows, num_probes]

        # Simulate quantization noise: uniform noise scaled by weight range
        w_range = tensor.max() - tensor.min()
        # 4-bit quantization has 16 levels, so step size ~ range/16
        noise_scale = w_range / 16.0
        noise = torch.rand_like(tensor) * noise_scale - noise_scale / 2

        y_noisy = (tensor + noise) @ x

        # Relative output change
        delta = (y_noisy - y_clean).norm() / (y_clean.norm() + 1e-8)
        delta = delta.item()

        # Log-scale normalization: maps delta range [0.01, 10.0] to ~[0, 1]
        # Avoids saturation on small models where delta >> 0.1 for all tensors
        if delta < 1e-6:
            return 0.0
        log_delta = math.log10(max(delta, 0.01))
        # Map log10 range [-2, 1] to [0, 1]
        score = min(1.0, max(0.0, (log_delta + 2.0) / 3.0))
        return score

    def _cross_layer_sensitivity(
        self,
        layer_idx: Optional[int],
        total_layers: Optional[int],
    ) -> float:
        """Cross-layer positional sensitivity heuristic.

        Early layers and late layers tend to be more sensitive to quantization:
        - Early layers set the representation foundation
        - Late layers directly affect output quality
        - Middle layers are more redundant

        Uses a U-shaped curve: high sensitivity at both ends, low in the middle.
        """
        if layer_idx is None or total_layers is None or total_layers <= 1:
            return 0.5  # Default to moderate

        # Normalise position to [0, 1]
        pos = layer_idx / (total_layers - 1)

        # U-shaped curve: 1.0 at edges, 0.0 at center
        # f(x) = 4*(x - 0.5)^2
        u_score = 4.0 * (pos - 0.5) ** 2

        # Slightly boost early layers more than late layers
        # (embedding quality matters more)
        if pos < 0.1:
            u_score = max(u_score, 0.8)

        return min(1.0, u_score)

    def _reconstruction_error_sensitivity(
        self, tensor: torch.Tensor, bits: int = 4, group_size: int = 128
    ) -> float:
        """Direct reconstruction error proxy: how much does 4-bit RTN change this tensor?

        Simulates group-wise round-to-nearest quantize→dequantize and measures
        the normalized RMSE. This is the most direct measure of quantization
        sensitivity — no random probes or normalization constants needed.

        Returns a score in [0, 1] based on NRMSE.
        """
        if tensor.dim() < 2:
            return 0.0

        t = tensor.float()
        if t.dim() == 3:
            t = self._sample_expert_slice(t)
        elif t.dim() > 3:
            t = t.reshape(t.shape[0], -1)

        rows, cols = t.shape

        # Pad columns to group_size multiple
        if cols % group_size != 0:
            pad = group_size - (cols % group_size)
            t = torch.nn.functional.pad(t, (0, pad))
            cols = t.shape[1]

        t_grouped = t.reshape(rows, cols // group_size, group_size)
        g_min = t_grouped.min(dim=-1, keepdim=True).values
        g_max = t_grouped.max(dim=-1, keepdim=True).values
        n_levels = (1 << bits) - 1
        scale = ((g_max - g_min) / n_levels).clamp(min=1e-12)

        quantized = torch.round((t_grouped - g_min) / scale).clamp(0, n_levels)
        dequantized = quantized * scale + g_min

        diff = dequantized - t_grouped
        rmse = diff.pow(2).mean().sqrt().item()
        rms_orig = t_grouped.pow(2).mean().sqrt().item()

        if rms_orig < 1e-12:
            return 0.0

        nrmse = rmse / rms_orig

        # Normalize: typical NRMSE range [0.005, 0.05] for 4-bit
        # Use wide range to avoid saturation
        score = min(1.0, max(0.0, (nrmse - 0.005) / 0.045))
        return score

    def recommend_bits(
        self,
        composite_score: float,
        tensor_class: str,
    ) -> Tuple[int, str]:
        """Recommend bit-width based on composite score and tensor classification.

        Args:
            composite_score: Weighted combination of all metric scores.
            tensor_class: One of 'protected', 'sensitive', 'default'.

        Returns:
            (bits, reason) tuple.
        """
        if tensor_class == "protected":
            return 16, "protected_pattern"

        if composite_score >= self.config.sensitivity_threshold_16bit:
            return 16, f"composite_score={composite_score:.3f} >= 16bit_threshold"

        if tensor_class == "sensitive":
            return 8, "sensitive_pattern"

        if composite_score >= self.config.sensitivity_threshold_8bit:
            return 8, f"composite_score={composite_score:.3f} >= 8bit_threshold"

        if hasattr(self.config, 'sensitivity_threshold_2bit') and \
           composite_score <= self.config.sensitivity_threshold_2bit:
            return 2, f"composite_score={composite_score:.3f} <= 2bit_threshold"

        return self.config.default_bits, "default"
