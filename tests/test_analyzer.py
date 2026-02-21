"""Unit tests for SmartQuant sensitivity analyzer."""

import sys
from pathlib import Path

import torch
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from smartquant.analyzer import SensitivityAnalyzer
from smartquant.config import QuantConfig


@pytest.fixture
def config():
    return QuantConfig()


@pytest.fixture
def analyzer(config):
    return SensitivityAnalyzer(config)


class TestSVDSensitivity:
    def test_low_rank_matrix_high_sensitivity(self, analyzer):
        """A low-rank matrix should have high SVD sensitivity (energy concentrated)."""
        # Rank-1 matrix: all energy in first singular value
        u = torch.randn(512, 1)
        v = torch.randn(1, 256)
        tensor = u @ v
        result = analyzer.analyze_tensor(tensor, "test.low_rank")
        assert result["scores"]["svd"] > 0.5, (
            f"Low-rank matrix should have high SVD score, got {result['scores']['svd']}"
        )

    def test_random_matrix_lower_sensitivity(self, analyzer):
        """A random (full-rank) matrix should have lower SVD sensitivity."""
        tensor = torch.randn(512, 256)
        result = analyzer.analyze_tensor(tensor, "test.random")
        # Random matrices have more spread singular values
        assert result["scores"]["svd"] < 0.8, (
            f"Random matrix should have moderate SVD score, got {result['scores']['svd']}"
        )

    def test_1d_tensor_skipped(self, analyzer):
        """1D tensors should be skipped."""
        tensor = torch.randn(512)
        result = analyzer.analyze_tensor(tensor, "test.norm")
        assert result["skipped"] is True


class TestKurtosisSensitivity:
    def test_normal_distribution_low_kurtosis(self, analyzer):
        """Normal distribution should have near-zero excess kurtosis -> low score."""
        tensor = torch.randn(512, 256)
        result = analyzer.analyze_tensor(tensor, "test.normal")
        assert result["scores"]["kurtosis"] < 0.3, (
            f"Normal distribution should have low kurtosis score, got {result['scores']['kurtosis']}"
        )

    def test_heavy_tailed_high_kurtosis(self, analyzer):
        """Distribution with outliers should have high kurtosis score."""
        tensor = torch.randn(512, 256)
        # Add extreme outliers
        tensor[0, 0] = 100.0
        tensor[1, 1] = -100.0
        tensor[2, 2] = 80.0
        result = analyzer.analyze_tensor(tensor, "test.outlier")
        assert result["scores"]["kurtosis"] > 0.1, (
            f"Heavy-tailed distribution should have higher kurtosis, got {result['scores']['kurtosis']}"
        )


class TestOutputSensitivity:
    def test_large_weights_more_sensitive(self, analyzer):
        """Layers with large weight magnitudes should be more output-sensitive."""
        small = torch.randn(256, 128) * 0.01
        large = torch.randn(256, 128) * 10.0

        result_small = analyzer.analyze_tensor(small, "test.small")
        result_large = analyzer.analyze_tensor(large, "test.large")

        # Larger weights amplify quantization noise more
        assert result_large["scores"]["output_sensitivity"] >= result_small["scores"]["output_sensitivity"], (
            f"Large weights ({result_large['scores']['output_sensitivity']:.3f}) should be >= "
            f"small weights ({result_small['scores']['output_sensitivity']:.3f})"
        )


class TestCrossLayerSensitivity:
    def test_early_layers_high(self, analyzer):
        """Early layers should have high cross-layer sensitivity."""
        tensor = torch.randn(256, 128)
        result = analyzer.analyze_tensor(tensor, "test.early", layer_idx=0, total_layers=60)
        assert result["scores"]["cross_layer"] > 0.6

    def test_middle_layers_low(self, analyzer):
        """Middle layers should have low cross-layer sensitivity."""
        tensor = torch.randn(256, 128)
        result = analyzer.analyze_tensor(tensor, "test.middle", layer_idx=30, total_layers=60)
        assert result["scores"]["cross_layer"] < 0.3

    def test_late_layers_high(self, analyzer):
        """Late layers should have high cross-layer sensitivity."""
        tensor = torch.randn(256, 128)
        result = analyzer.analyze_tensor(tensor, "test.late", layer_idx=59, total_layers=60)
        assert result["scores"]["cross_layer"] > 0.6


class TestCompositeScore:
    def test_composite_is_weighted_sum(self, analyzer):
        """Composite score should be the weighted sum of individual metrics."""
        tensor = torch.randn(256, 128)
        result = analyzer.analyze_tensor(tensor, "test.composite", layer_idx=5, total_layers=60)

        scores = result["scores"]
        weights = analyzer.config.metric_weights
        expected = sum(weights[k] * scores[k] for k in scores)

        assert abs(result["composite_score"] - expected) < 1e-6, (
            f"Composite {result['composite_score']} != expected {expected}"
        )

    def test_score_range(self, analyzer):
        """All scores should be in [0, 1]."""
        for _ in range(10):
            tensor = torch.randn(256, 128) * torch.rand(1).item() * 10
            result = analyzer.analyze_tensor(tensor, "test.range", layer_idx=5, total_layers=60)
            for name, score in result["scores"].items():
                assert 0.0 <= score <= 1.0, f"{name} score {score} out of [0,1] range"
            assert 0.0 <= result["composite_score"] <= 1.0


class TestBitRecommendation:
    def test_protected_gets_16bit(self, analyzer):
        bits, reason = analyzer.recommend_bits(0.0, "protected")
        assert bits == 16
        assert "protected" in reason

    def test_high_score_gets_16bit(self, analyzer):
        bits, reason = analyzer.recommend_bits(0.90, "default")
        assert bits == 16

    def test_sensitive_pattern_gets_8bit(self, analyzer):
        bits, reason = analyzer.recommend_bits(0.30, "sensitive")
        assert bits == 8
        assert "sensitive" in reason

    def test_medium_score_gets_8bit(self, analyzer):
        bits, reason = analyzer.recommend_bits(0.50, "default")
        assert bits == 8

    def test_low_score_gets_4bit(self, analyzer):
        bits, reason = analyzer.recommend_bits(0.20, "default")
        assert bits == 4
        assert "default" in reason


class TestConfigValidation:
    def test_valid_config(self):
        config = QuantConfig()
        config.validate()

    def test_invalid_threshold_order(self):
        config = QuantConfig(sensitivity_threshold_8bit=0.9, sensitivity_threshold_16bit=0.5)
        with pytest.raises(AssertionError):
            config.validate()

    def test_invalid_metric_weights(self):
        config = QuantConfig(metric_weights={
            "svd": 0.5, "kurtosis": 0.5, "output_sensitivity": 0.5, "cross_layer": 0.5,
        })
        with pytest.raises(AssertionError):
            config.validate()

    def test_missing_metric_weight(self):
        config = QuantConfig(metric_weights={
            "svd": 0.5, "kurtosis": 0.5,
        })
        with pytest.raises(AssertionError):
            config.validate()
