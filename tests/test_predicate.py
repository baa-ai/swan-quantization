"""Unit tests for SmartQuant -> MLX predicate bridge."""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from smartquant.bridge_mlx import (
    build_bits_lookup,
    create_quant_predicate,
    generate_predicate_module,
    tensor_name_to_module_name,
)


@pytest.fixture
def sample_manifest():
    """Create a minimal SmartQuant manifest for testing."""
    return {
        "model": "/tmp/test-model",
        "config": {
            "sensitivity_threshold_8bit": 0.45,
            "sensitivity_threshold_16bit": 0.85,
            "default_bits": 4,
            "metric_weights": {
                "svd": 0.30,
                "kurtosis": 0.20,
                "output_sensitivity": 0.30,
                "cross_layer": 0.20,
            },
            "svd_rank": 256,
        },
        "total_layers": 2,
        "total_tensors": 12,
        "shards": {
            "model-00001.safetensors": {
                "file": "model-00001.safetensors",
                "tensors": {
                    "model.embed_tokens.weight": {
                        "shape": [128256, 5120],
                        "dtype": "torch.bfloat16",
                        "num_params": 656179200,
                        "classification": "protected",
                        "scores": {"svd": 0.0, "kurtosis": 0.0, "output_sensitivity": 0.0, "cross_layer": 0.0},
                        "composite_score": 0.0,
                        "decision": {"bits": 16, "reason": "protected_pattern"},
                    },
                    "model.layers.0.self_attn.q_proj.weight": {
                        "shape": [5120, 5120],
                        "dtype": "torch.bfloat16",
                        "num_params": 26214400,
                        "classification": "sensitive",
                        "scores": {"svd": 0.6, "kurtosis": 0.3, "output_sensitivity": 0.5, "cross_layer": 0.8},
                        "composite_score": 0.55,
                        "decision": {"bits": 8, "reason": "sensitive_pattern"},
                    },
                    "model.layers.0.self_attn.k_proj.weight": {
                        "shape": [1024, 5120],
                        "dtype": "torch.bfloat16",
                        "num_params": 5242880,
                        "classification": "sensitive",
                        "scores": {"svd": 0.5, "kurtosis": 0.2, "output_sensitivity": 0.4, "cross_layer": 0.8},
                        "composite_score": 0.47,
                        "decision": {"bits": 8, "reason": "sensitive_pattern"},
                    },
                    "model.layers.0.self_attn.v_proj.weight": {
                        "shape": [1024, 5120],
                        "dtype": "torch.bfloat16",
                        "num_params": 5242880,
                        "classification": "default",
                        "scores": {"svd": 0.3, "kurtosis": 0.1, "output_sensitivity": 0.2, "cross_layer": 0.8},
                        "composite_score": 0.30,
                        "decision": {"bits": 4, "reason": "default"},
                    },
                    "model.layers.0.feed_forward.experts.0.gate_proj.weight": {
                        "shape": [8192, 5120],
                        "dtype": "torch.bfloat16",
                        "num_params": 41943040,
                        "classification": "default",
                        "scores": {"svd": 0.2, "kurtosis": 0.1, "output_sensitivity": 0.15, "cross_layer": 0.5},
                        "composite_score": 0.22,
                        "decision": {"bits": 4, "reason": "default"},
                    },
                    "model.layers.0.input_layernorm.weight": {
                        "shape": [5120],
                        "dtype": "torch.bfloat16",
                        "num_params": 5120,
                        "classification": "protected",
                        "scores": {"svd": 0.0, "kurtosis": 0.0, "output_sensitivity": 0.0, "cross_layer": 0.0},
                        "composite_score": 0.0,
                        "decision": {"bits": 16, "reason": "1d_tensor"},
                    },
                    "model.layers.0.feed_forward.router.weight": {
                        "shape": [128, 5120],
                        "dtype": "torch.bfloat16",
                        "num_params": 655360,
                        "classification": "protected",
                        "scores": {"svd": 0.0, "kurtosis": 0.0, "output_sensitivity": 0.0, "cross_layer": 0.0},
                        "composite_score": 0.0,
                        "decision": {"bits": 16, "reason": "protected_pattern"},
                    },
                },
            },
        },
        "summary": {
            "total_params": 735482880,
            "bits_distribution": {
                "4": {"params": 47185920, "percentage": 6.4},
                "8": {"params": 31457280, "percentage": 4.3},
                "16": {"params": 656839680, "percentage": 89.3},
            },
            "estimated_size_gb": 1.2,
            "average_bits": 15.1,
            "analysis_time_seconds": 10.0,
            "tensor_count": 7,
        },
    }


class TestTensorNameToModuleName:
    def test_strip_weight(self):
        assert tensor_name_to_module_name("model.layers.0.self_attn.q_proj.weight") == "model.layers.0.self_attn.q_proj"

    def test_strip_bias(self):
        assert tensor_name_to_module_name("model.layers.0.self_attn.q_proj.bias") == "model.layers.0.self_attn.q_proj"

    def test_no_suffix(self):
        assert tensor_name_to_module_name("model.embed_tokens") == "model.embed_tokens"


class TestBuildBitsLookup:
    def test_correct_mapping(self, sample_manifest):
        lookup = build_bits_lookup(sample_manifest)
        assert lookup["model.embed_tokens.weight"] == 16
        assert lookup["model.layers.0.self_attn.q_proj.weight"] == 8
        assert lookup["model.layers.0.self_attn.v_proj.weight"] == 4
        assert lookup["model.layers.0.feed_forward.experts.0.gate_proj.weight"] == 4
        assert lookup["model.layers.0.input_layernorm.weight"] == 16
        assert lookup["model.layers.0.feed_forward.router.weight"] == 16

    def test_all_tensors_present(self, sample_manifest):
        lookup = build_bits_lookup(sample_manifest)
        total_in_manifest = sum(
            len(shard["tensors"])
            for shard in sample_manifest["shards"].values()
        )
        assert len(lookup) == total_in_manifest


class TestQuantPredicate:
    def test_embedding_returns_none(self, sample_manifest):
        """Embedding should not be quantized (return None)."""
        predicate = create_quant_predicate(sample_manifest)
        module = MagicMock()
        module.weight.shape = [128256, 5120]
        result = predicate("model.embed_tokens", module, {})
        assert result is None

    def test_q_proj_returns_8bit(self, sample_manifest):
        """Q projection should be 8-bit per manifest."""
        predicate = create_quant_predicate(sample_manifest)
        module = MagicMock()
        module.weight.shape = [5120, 5120]
        result = predicate("model.layers.0.self_attn.q_proj", module, {})
        assert result == (8, 64)

    def test_v_proj_returns_4bit(self, sample_manifest):
        """V projection should be 4-bit per manifest."""
        predicate = create_quant_predicate(sample_manifest)
        module = MagicMock()
        module.weight.shape = [1024, 5120]
        result = predicate("model.layers.0.self_attn.v_proj", module, {})
        assert result == (4, 128)

    def test_expert_returns_4bit(self, sample_manifest):
        """Expert MLP should be 4-bit per manifest."""
        predicate = create_quant_predicate(sample_manifest)
        module = MagicMock()
        module.weight.shape = [8192, 5120]
        result = predicate("model.layers.0.feed_forward.experts.0.gate_proj", module, {})
        assert result == (4, 128)

    def test_layernorm_returns_none(self, sample_manifest):
        """LayerNorm should not be quantized."""
        predicate = create_quant_predicate(sample_manifest)
        module = MagicMock()
        module.weight.shape = [5120]
        result = predicate("model.layers.0.input_layernorm", module, {})
        assert result is None

    def test_router_returns_none(self, sample_manifest):
        """Router should not be quantized."""
        predicate = create_quant_predicate(sample_manifest)
        module = MagicMock()
        module.weight.shape = [128, 5120]
        result = predicate("model.layers.0.feed_forward.router", module, {})
        assert result is None

    def test_unknown_tensor_gets_default(self, sample_manifest):
        """Unknown tensor name should get default 4-bit."""
        predicate = create_quant_predicate(sample_manifest)
        module = MagicMock()
        module.weight.shape = [256, 256]
        result = predicate("model.layers.99.unknown_module", module, {})
        assert result == (4, 128)

    def test_lm_head_returns_none(self, sample_manifest):
        """lm_head should not be quantized."""
        predicate = create_quant_predicate(sample_manifest)
        module = MagicMock()
        module.weight.shape = [128256, 5120]
        result = predicate("lm_head", module, {})
        assert result is None

    def test_model_norm_returns_none(self, sample_manifest):
        """model.norm (final RMSNorm) should not be quantized."""
        predicate = create_quant_predicate(sample_manifest)
        module = MagicMock()
        module.weight.shape = [5120]
        result = predicate("model.norm", module, {})
        assert result is None


class TestGeneratePredicateModule:
    def test_generates_valid_python(self, sample_manifest):
        """Generated predicate module should be valid Python."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(sample_manifest, f)

            output_path = Path(tmpdir) / "predicate.py"
            generate_predicate_module(manifest_path, output_path)

            assert output_path.exists()

            # Should be valid Python
            with open(output_path) as f:
                source = f.read()
            compile(source, str(output_path), "exec")

    def test_generated_module_has_predicate(self, sample_manifest):
        """Generated module should define smartquant_predicate function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(sample_manifest, f)

            output_path = Path(tmpdir) / "predicate.py"
            generate_predicate_module(manifest_path, output_path)

            # Import and verify
            import importlib.util
            spec = importlib.util.spec_from_file_location("predicate", str(output_path))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            assert hasattr(mod, "smartquant_predicate")
            assert callable(mod.smartquant_predicate)
