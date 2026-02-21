"""Unit tests for quantization roundtrip verification."""

import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from smartquant.analyzer import SensitivityAnalyzer
from smartquant.config import QuantConfig


@pytest.fixture
def config():
    return QuantConfig()


class TestQuantizationSimulation:
    """Test that simulated quantization behaves as expected."""

    def _simulate_quantize(self, tensor: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
        """Simulate symmetric quantization and dequantization."""
        flat = tensor.flatten()
        n = flat.numel()
        # Pad to group_size multiple
        padded_len = ((n + group_size - 1) // group_size) * group_size
        padded = torch.zeros(padded_len, dtype=flat.dtype)
        padded[:n] = flat

        groups = padded.reshape(-1, group_size)
        scales = groups.abs().max(dim=1).values / (2 ** (bits - 1) - 1)
        scales = scales.clamp(min=1e-8)

        # Quantize
        quantized = torch.round(groups / scales.unsqueeze(1))
        quantized = quantized.clamp(-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)

        # Dequantize
        dequantized = quantized * scales.unsqueeze(1)
        return dequantized.flatten()[:n].reshape(tensor.shape)

    def test_4bit_roundtrip_error(self):
        """4-bit quantization should have measurable but bounded error."""
        tensor = torch.randn(256, 128)
        dequant = self._simulate_quantize(tensor, bits=4, group_size=128)
        error = (tensor - dequant).abs().mean() / tensor.abs().mean()
        # Relative error for 4-bit should be roughly 5-15%
        assert error < 0.20, f"4-bit relative error too high: {error:.3f}"
        assert error > 0.001, f"4-bit error suspiciously low: {error:.3f}"

    def test_8bit_less_error_than_4bit(self):
        """8-bit should have less quantization error than 4-bit."""
        tensor = torch.randn(256, 128)
        dequant_4 = self._simulate_quantize(tensor, bits=4, group_size=128)
        dequant_8 = self._simulate_quantize(tensor, bits=8, group_size=64)

        error_4 = (tensor - dequant_4).abs().mean().item()
        error_8 = (tensor - dequant_8).abs().mean().item()

        assert error_8 < error_4, (
            f"8-bit error ({error_8:.4f}) should be less than 4-bit ({error_4:.4f})"
        )

    def test_16bit_minimal_error(self):
        """16-bit (float16 roundtrip) should have negligible error."""
        tensor = torch.randn(256, 128)
        roundtrip = tensor.half().float()
        error = (tensor - roundtrip).abs().mean() / tensor.abs().mean()
        assert error < 0.001, f"16-bit relative error too high: {error:.3f}"

    def test_quantize_preserves_shape(self):
        """Quantization should preserve tensor shape."""
        for shape in [(512, 256), (128, 1024), (64, 64)]:
            tensor = torch.randn(*shape)
            dequant = self._simulate_quantize(tensor, bits=4, group_size=128)
            assert dequant.shape == tensor.shape

    def test_zero_tensor(self):
        """Quantization of zero tensor should return zeros."""
        tensor = torch.zeros(256, 128)
        dequant = self._simulate_quantize(tensor, bits=4, group_size=128)
        assert dequant.abs().max() < 1e-6

    def test_outlier_impact(self):
        """Tensors with outliers should have larger quantization error."""
        normal = torch.randn(256, 128)
        outlier = normal.clone()
        outlier[0, 0] = 100.0  # Extreme outlier

        error_normal = (normal - self._simulate_quantize(normal, 4, 128)).abs().mean().item()
        error_outlier = (outlier - self._simulate_quantize(outlier, 4, 128)).abs().mean().item()

        assert error_outlier > error_normal, (
            f"Outlier error ({error_outlier:.4f}) should exceed normal ({error_normal:.4f})"
        )


class TestSensitivityCorrelation:
    """Test that sensitivity scores correlate with actual quantization impact."""

    def test_sensitive_tensor_higher_error(self, config):
        """Tensors scored as more sensitive should have larger quantization error."""
        analyzer = SensitivityAnalyzer(config)

        # Create a low-rank (sensitive) and random (less sensitive) tensor
        u = torch.randn(512, 1)
        v = torch.randn(1, 256)
        low_rank = u @ v + torch.randn(512, 256) * 0.01
        random = torch.randn(512, 256)

        r_lr = analyzer.analyze_tensor(low_rank, "low_rank")
        r_rand = analyzer.analyze_tensor(random, "random")

        # The low-rank tensor should score higher on SVD sensitivity
        assert r_lr["scores"]["svd"] > r_rand["scores"]["svd"], (
            f"Low-rank SVD ({r_lr['scores']['svd']:.3f}) should exceed "
            f"random SVD ({r_rand['scores']['svd']:.3f})"
        )
