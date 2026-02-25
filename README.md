# SWAN: Statistical Weight Analysis for N-bit Allocation

**Data-free mixed-precision quantization for large language models via multi-metric sensitivity analysis.**

SWAN computes four complementary sensitivity metrics directly on weight tensors — no calibration data, no gradients, no fine-tuning — to drive per-tensor bit-width allocation (2, 4, 8, or 16-bit). The entire analysis pipeline completes in under 13 minutes for 400B+ parameter models on commodity hardware.

## Key Results

| Model | Method | Size (GB) | Avg Bits | PPL |
|-------|--------|-----------|----------|-----|
| Qwen3.5-397B | **SWAN** | 199 | 4.31 | **4.283** |
| Qwen3.5-397B | Uniform 4-bit RTN | 196 | 4.25 | 4.298 |
| Qwen3.5-397B | BF16 (baseline) | 807 | 16.00 | — |

SWAN outperforms uniform 4-bit quantization at matched group sizes while requiring zero calibration data.

**Academic benchmarks** (Qwen3.5-397B, SWAN quantized, 0-shot, greedy decoding):
- MMLU-Pro: **77.1%** (thinking enabled) | 72.1% (thinking disabled)
- ARC-Challenge: 96.0% | GSM8K: 88.7% | HumanEval: 78.7%

## How It Works

SWAN combines four data-free sensitivity metrics into a weighted composite score:

1. **SVD Spectral Concentration** (w=0.20) — measures information concentration in top singular values
2. **Excess Kurtosis** (w=0.45) — quantifies outlier heaviness in weight distributions
3. **Output Noise Amplification** (w=0.15) — estimates how perturbations propagate through the layer
4. **Reconstruction Error Proxy** (w=0.20) — directly measures change under simulated 4-bit quantization

The composite score drives automatic bit-width allocation:
- Score >= 0.90 → 16-bit (protected)
- Score >= 0.65 → 8-bit (sensitive)
- Score <= 0.10 → 2-bit (insensitive)
- Otherwise → 4-bit (default)

## Requirements

- Apple Silicon Mac with sufficient unified memory for your target model
- Python 3.9+
- MLX framework

## Installation

```bash
git clone https://github.com/baa-ai/swan-quantization.git
cd swan-quantization
pip install -r requirements.txt
```

## Quick Start

### 1. Analyze a model

```bash
python -m smartquant analyze \
    --input-dir /path/to/model \
    --output ./analysis
```

This produces a JSON manifest with per-tensor sensitivity scores and bit-width decisions.

### 2. Convert with mixed precision

```bash
python convert_model.py \
    --model-dir /path/to/bf16-model \
    --manifest ./analysis/smartquant_manifest.json \
    --output-dir ./models/quantized
```

### 3. Run inference

```python
import mlx_lm

model, tokenizer = mlx_lm.load("./models/quantized")
response = mlx_lm.generate(model, tokenizer, prompt="Hello, world!")
```

## Supported Models

SWAN has been validated on:
- **Qwen3-8B** (dense, 8.2B parameters)
- **Llama4-Maverick-17B-128E** (MoE, 401.6B parameters, 128 experts)
- **Qwen3.5-397B-A17B** (MoE, 403.4B parameters, 512 experts)

The method is architecture-agnostic and should work on any transformer-based LLM stored in safetensors format.

## Project Structure

```
swan-quantization/
├── smartquant/              # Core Python package
│   ├── analyzer.py          # Four sensitivity metrics
│   ├── config.py            # QuantConfig dataclass
│   ├── shard_processor.py   # Safetensor shard analysis pipeline
│   ├── bridge_mlx.py        # MLX quantization predicate bridge
│   ├── utils.py             # Tensor classification utilities
│   └── main.py              # CLI entry point
├── tests/                   # Unit tests
├── scripts/                 # Baseline measurement scripts
├── paper/                   # SWAN paper (LaTeX + HTML)
├── journal/                 # Experimental data and tables
├── convert_model.py         # Model conversion script
├── setup.sh                 # Environment setup
└── requirements.txt
```

## Paper

**SWAN: Data-Free Mixed-Precision Quantization for Large Language Models via Multi-Metric Sensitivity Analysis**

Black Sheep AI Research

The paper is available in `paper/smartquant.tex` (LaTeX) and `paper/smartquant.html` (HTML).

## Citation

```bibtex
@article{baa2026swan,
  title={SWAN: Data-Free Mixed-Precision Quantization for Large Language Models via Multi-Metric Sensitivity Analysis},
  author={Black Sheep AI Research},
  year={2026}
}
```

## License

PolyForm Noncommercial License 1.0.0. See [LICENSE](LICENSE) for details.
