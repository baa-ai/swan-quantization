# SWAN: Running the Pipeline on a New Model

This guide walks through running SWAN's intelligent mixed-precision quantization pipeline on any HuggingFace model with safetensor weights. The pipeline analyzes per-tensor sensitivity using four metrics (SVD, kurtosis, output sensitivity, cross-layer position) and assigns each tensor an optimal bit-width (4, 8, or 16-bit).

## Prerequisites

- **Hardware**: Apple Silicon Mac with sufficient unified memory. The model must fit in memory at ~4-5 bits per parameter after quantization. For reference, a 70B model needs ~40 GB, a 400B model needs ~230 GB.
- **macOS**: Apple Silicon (M1/M2/M3/M4 series) with Metal GPU support.
- **Python**: 3.9+
- **HuggingFace account**: Required for gated models (Llama, Mistral, etc.). Accept the model license on the model's HuggingFace page before downloading.

## Quick Start (5 commands)

```bash
# 1. Setup environment
bash ~/smartquant/setup.sh

# 2. Download model
source ~/smartquant-env/bin/activate
python ~/smartquant/download_model.py \
    --repo-id <org/model-name> \
    --local-dir ~/smartquant/models/<model-name>-bf16

# 3. Analyze tensors
python -m smartquant analyze \
    --input-dir ~/smartquant/models/<model-name>-bf16 \
    --output ~/smartquant/analysis/manifest.json

# 4. Convert to MLX with mixed precision
python ~/smartquant/convert_model.py \
    --hf-path ~/smartquant/models/<model-name>-bf16 \
    --mlx-path ~/smartquant/models/<model-name>-smartquant \
    --manifest ~/smartquant/analysis/manifest.json

# 5. Benchmark
python ~/smartquant/test_harness.py \
    --model ~/smartquant/models/<model-name>-smartquant \
    --mode benchmark
```

---

## Step-by-Step Guide

### Phase 1: Environment Setup

Run the setup script once to create the virtual environment and install dependencies:

```bash
bash ~/smartquant/setup.sh
```

This installs: `mlx`, `mlx-lm`, `torch` (CPU), `safetensors`, `transformers`, `huggingface_hub`.

Activate the environment for all subsequent commands:

```bash
source ~/smartquant-env/bin/activate
```

If you need to log in to HuggingFace (for gated models):

```bash
huggingface-cli login
```

### Phase 2: Download the Model

```bash
python ~/smartquant/download_model.py \
    --repo-id <org/model-name> \
    --local-dir ~/smartquant/models/<model-name>-bf16
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--repo-id` | `meta-llama/Llama-4-Maverick-17B-128E-Instruct` | HuggingFace repository ID |
| `--local-dir` | `~/smartquant/models/maverick-bf16` | Local directory to save the model |
| `--verify-only` | (flag) | Only verify an existing download without downloading |

**Notes:**
- Downloads are resumable. You can interrupt with Ctrl+C and re-run the same command.
- Only safetensor weights, configs, and tokenizer files are downloaded (no .bin, optimizer states, etc.).
- After download, each shard is spot-checked by loading one tensor.

**Example for Llama 3.1 70B:**
```bash
python ~/smartquant/download_model.py \
    --repo-id meta-llama/Llama-3.1-70B-Instruct \
    --local-dir ~/smartquant/models/llama-3.1-70b-bf16
```

### Phase 3: Sensitivity Analysis

First, discover the tensor naming patterns in your model (optional but recommended for new architectures):

```bash
python -m smartquant discover \
    --input-dir ~/smartquant/models/<model-name>-bf16
```

This prints all unique tensor name patterns without running the full analysis. Review the output to verify that SWAN's protected/sensitive patterns match your model's naming convention. If your model uses non-standard names, update `smartquant/config.py` (see [Customizing for New Architectures](#customizing-for-new-architectures) below).

Run the full analysis:

```bash
python -m smartquant analyze \
    --input-dir ~/smartquant/models/<model-name>-bf16 \
    --output ~/smartquant/analysis/manifest.json
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--input-dir` | (required) | Path to the BF16 model directory |
| `--output` | (required) | Path to write the analysis manifest JSON |
| `--device` | `cpu` | Compute device: `cpu` or `mps` |
| `--threshold-8bit` | `0.45` | Composite score >= this gets 8-bit |
| `--threshold-16bit` | `0.85` | Composite score >= this gets 16-bit |
| `--svd-rank` | `256` | Rank for randomized SVD (higher = more accurate, slower) |
| `--discover-only` | (flag) | Only print tensor patterns, skip full analysis |

**Tuning thresholds:**
- **Lower `--threshold-8bit`** (e.g., 0.35): More tensors get 8-bit. Larger model, better quality.
- **Higher `--threshold-8bit`** (e.g., 0.55): Fewer tensors get 8-bit. Smaller model, more aggressive.
- **`--threshold-16bit`**: Only the most sensitive tensors (embeddings, final layer norms) should reach this. Rarely needs tuning.

**Output:** A `manifest.json` file containing per-tensor analysis results:
```json
{
  "shards": {
    "model-00001-of-00055.safetensors": {
      "tensors": {
        "model.layers.0.self_attn.q_proj.weight": {
          "shape": [8192, 5120],
          "dtype": "bfloat16",
          "params": 41943040,
          "metrics": {
            "svd": 0.42,
            "kurtosis": 0.31,
            "output_sensitivity": 0.55,
            "cross_layer": 0.80
          },
          "composite_score": 0.51,
          "decision": {
            "bits": 8,
            "reason": "composite_score=0.510 >= 8bit_threshold"
          }
        }
      }
    }
  },
  "summary": {
    "total_params": 401580000000,
    "bits_distribution": { ... },
    "average_bits": 4.78,
    "estimated_size_gb": 230
  }
}
```

### Phase 4: MLX Conversion

Convert the model to MLX format with the SWAN mixed-precision predicate:

```bash
python ~/smartquant/convert_model.py \
    --hf-path ~/smartquant/models/<model-name>-bf16 \
    --mlx-path ~/smartquant/models/<model-name>-smartquant \
    --manifest ~/smartquant/analysis/manifest.json
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--hf-path` | (required) | Path to the BF16 model |
| `--mlx-path` | (required) | Output path for the quantized MLX model |
| `--manifest` | (required) | Path to the SWAN manifest.json |
| `--dtype` | `float16` | Data type for non-quantized params: `float16` or `bfloat16` |
| `--default-bits` | `4` | Default bit-width for tensors not in manifest |
| `--default-group-size` | `128` | Default group size for quantization |

**Notes:**
- If the output directory already exists, it is automatically removed before conversion (mlx_lm requires a clean output path).
- After conversion, a quick load test is performed to verify the output.
- 8-bit tensors use group_size=64, 4-bit tensors use group_size=128 by default.

### Phase 5: Run Tests

```bash
python -m pytest ~/smartquant/tests/ -v
```

All 43 tests should pass. These validate:
- Sensitivity metric computation (SVD, kurtosis, output sensitivity, cross-layer)
- Quantization roundtrip error bounds
- Bridge/predicate logic (manifest to MLX predicate mapping)

### Phase 6: Benchmark

Run the automated benchmark suite:

```bash
python ~/smartquant/test_harness.py \
    --model ~/smartquant/models/<model-name>-smartquant \
    --mode benchmark
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `~/smartquant/models/maverick-smartquant` | Path to quantized model |
| `--mode` | `interactive` | Mode: `interactive`, `benchmark`, or `compare` |
| `--max-tokens` | `2048` | Maximum tokens to generate per prompt |
| `--temp` | `0.7` | Generation temperature |
| `--output` | `~/smartquant/results/benchmark_results.json` | Output path for results |

**Modes:**

- **`interactive`**: Chat REPL with live streaming. Commands: `/stats`, `/memory`, `/temp N`, `/bench`, `/quit`
- **`benchmark`**: Runs 19 prompts across 6 categories (factual recall, reasoning, coding, creative writing, instruction following, long form). Saves results to JSON.
- **`compare`**: A/B comparison between two models (e.g., SWAN vs uniform 4-bit). Requires `--model-a` and `--model-b`.

**A/B Comparison example:**

```bash
python ~/smartquant/test_harness.py \
    --mode compare \
    --model-a ~/smartquant/models/<model-name>-smartquant \
    --model-b ~/smartquant/models/<model-name>-uniform4bit
```

### Phase 7: Generate Report

```bash
python ~/smartquant/generate_report.py \
    --manifest ~/smartquant/analysis/manifest.json \
    --benchmark ~/smartquant/results/benchmark_results.json \
    --output ~/smartquant/results/report.md
```

Produces a markdown report with analysis summary, performance metrics, sample responses, and recommendations.

### Phase 8: Serve as API (Optional)

Start an OpenAI-compatible API server:

```bash
python -m mlx_lm.server \
    --model ~/smartquant/models/<model-name>-smartquant \
    --host 0.0.0.0 \
    --port 8080
```

Serve the web chat UI:

```bash
python -m http.server 8081 --directory ~/smartquant/web
```

Then open `http://localhost:8081` in a browser.

The API supports the standard OpenAI chat completions format:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default_model",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

> **Important:** Use `"model": "default_model"` as the model name when sending requests. Using the actual model name will cause a 404 error as mlx_lm.server tries to look it up on HuggingFace.

---

## Customizing for New Architectures

If your model uses non-standard tensor naming (not `model.layers.{N}.self_attn.q_proj.weight` style), you may need to update SWAN's pattern configuration.

### 1. Discover tensor patterns

```bash
python -m smartquant discover --input-dir ~/smartquant/models/<model-name>-bf16
```

### 2. Update `smartquant/config.py`

Edit the `QuantConfig` class:

```python
@dataclass
class QuantConfig:
    # Tensors matching these patterns are kept at 16-bit (never quantized)
    protected_patterns: List[str] = field(default_factory=lambda: [
        "embed_tokens",
        "lm_head",
        "layernorm",
        "layer_norm",
        "norm.weight",
        "router.weight",
        # Add your model-specific patterns:
        "vision_model",           # multimodal models
        "multi_modal_projector",  # multimodal models
        "positional_embedding",
        "class_embedding",
        "patch_embedding",
    ])

    # Tensors matching these patterns default to 8-bit
    sensitive_patterns: List[str] = field(default_factory=lambda: [
        "q_proj",
        "k_proj",
    ])
```

### 3. Update `smartquant/bridge_mlx.py` if needed

The bridge maps tensor names from the manifest to MLX module paths. If your model has a prefix (e.g., `language_model.model.layers...` instead of `model.layers...`), the bridge already handles this by trying lookups with and without common prefixes. For unusual prefixes, add them to the lookup logic in `smartquant_predicate()`.

### Common architectures and their patterns

| Architecture | Prefix | Notes |
|-------------|--------|-------|
| Llama / Mistral | `model.layers.{N}...` | Standard, works out of the box |
| Llama 4 Maverick | `language_model.model.layers.{N}...` | Has `language_model.` prefix + vision components |
| Qwen | `model.layers.{N}...` | Standard-style, uses `mlp` instead of `feed_forward` |
| Phi | `model.layers.{N}...` | Uses `Wqkv` fused projections |
| DeepSeek MoE | `model.layers.{N}...` | Packed expert tensors similar to Maverick |

### Handling MoE (Mixture of Experts) models

MoE models often pack all expert weights into a single 3D tensor (e.g., `[num_experts, hidden_dim, ffn_dim]`). SWAN handles this automatically by sampling a subset of expert slices for SVD analysis. No configuration changes needed.

---

## Tuning Guide

### Threshold tuning for size vs quality

| Goal | `--threshold-8bit` | `--threshold-16bit` | Expected avg bits |
|------|-------------------|---------------------|-------------------|
| Maximum compression | 0.55 | 0.90 | ~4.2 |
| Balanced (default) | 0.45 | 0.85 | ~4.8 |
| Quality-focused | 0.35 | 0.80 | ~5.5 |
| Conservative | 0.30 | 0.75 | ~6.0+ |

### Metric weight tuning

Edit `metric_weights` in `config.py` to emphasize different aspects:

```python
metric_weights: Dict[str, float] = field(default_factory=lambda: {
    "svd": 0.30,              # Spectral sensitivity (information concentration)
    "kurtosis": 0.20,         # Outlier detection (heavy-tail distributions)
    "output_sensitivity": 0.30, # Simulated quantization impact
    "cross_layer": 0.20,      # Position heuristic (U-shaped curve)
})
```

- **Increase `svd`** weight for models where you suspect information is concentrated in few singular values.
- **Increase `kurtosis`** weight for models with known outlier issues.
- **Increase `output_sensitivity`** for the most empirically-grounded metric.
- **Decrease `cross_layer`** if the U-shaped hypothesis doesn't apply to your model.

Weights must sum to 1.0.

---

## Troubleshooting

### Common errors and fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `401 Unauthorized` during download | Not logged in to HuggingFace | Run `huggingface-cli login` |
| `403 Forbidden` during download | Haven't accepted model license | Visit the model page on HuggingFace and accept the license |
| `ValueError: Cannot save to path ... as it already exists` | mlx_lm.convert output dir exists | The script handles this automatically by removing the directory. If running manually, delete the output dir first. |
| `GPU Timeout Error` during conversion | Model too large for Metal command buffer | This shouldn't happen with the pipeline — conversion streams through shards. If it does, try setting `MLX_GPU_MEMORY_LIMIT` env var. |
| `AttributeError: module 'mlx.core.metal' has no attribute 'get_memory_info'` | Wrong MLX memory API | Already fixed in test_harness.py. Use `mx.get_active_memory()` / `mx.get_peak_memory()`. |
| `TypeError: generate_step() got an unexpected keyword argument 'temp'` | mlx_lm sampler API change | Use `sampler=make_sampler(temp=..., top_p=...)` instead of passing `temp` directly. |
| API returns 404 for chat completions | Wrong model name in request | Use `"model": "default_model"` in API requests. |
| 3D tensor shape error during SVD | Packed expert tensors (MoE) | Already handled — SWAN samples expert slices automatically. |

### Memory estimation

Before running, estimate whether your model will fit:

```
Quantized size (GB) ≈ (total_params × avg_bits) / 8 / 1e9
```

Examples:
- 7B model at 4.8 avg bits: ~4.2 GB
- 70B model at 4.8 avg bits: ~42 GB
- 400B model at 4.8 avg bits: ~240 GB

You need roughly 1.05x the quantized size for inference (KV cache overhead is minimal for short contexts).

---

## Full Example: Llama 3.1 70B

```bash
# Setup (once)
bash ~/smartquant/setup.sh
source ~/smartquant-env/bin/activate
huggingface-cli login

# Download (~140 GB, ~30 min on fast connection)
python ~/smartquant/download_model.py \
    --repo-id meta-llama/Llama-3.1-70B-Instruct \
    --local-dir ~/smartquant/models/llama70b-bf16

# Discover patterns (optional, ~10s)
python -m smartquant discover \
    --input-dir ~/smartquant/models/llama70b-bf16

# Analyze (~5-10 min for 70B)
python -m smartquant analyze \
    --input-dir ~/smartquant/models/llama70b-bf16 \
    --output ~/smartquant/analysis/manifest.json

# Convert (~2-5 min)
python ~/smartquant/convert_model.py \
    --hf-path ~/smartquant/models/llama70b-bf16 \
    --mlx-path ~/smartquant/models/llama70b-smartquant \
    --manifest ~/smartquant/analysis/manifest.json

# Benchmark
python ~/smartquant/test_harness.py \
    --model ~/smartquant/models/llama70b-smartquant \
    --mode benchmark

# Generate report
python ~/smartquant/generate_report.py

# Interactive chat
python ~/smartquant/test_harness.py \
    --model ~/smartquant/models/llama70b-smartquant \
    --mode interactive

# Serve as API
python -m mlx_lm.server \
    --model ~/smartquant/models/llama70b-smartquant \
    --port 8080 &
python -m http.server 8081 --directory ~/smartquant/web &
open http://localhost:8081
```
