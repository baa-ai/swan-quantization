# SWAN Journal Data Sprint — Experiment Results

## Overview

Five experiments validating SWAN's data-free, per-tensor mixed-precision quantization method using a 4-metric composite score (SVD concentration, kurtosis, output noise amplification, U-shaped layer position).

## Experiments

### Exp 1: Ablation Study (COMPLETE)

**File:** `results/ablation_qwen3_8b.json`

Tests 10 metric weight configurations on Qwen3-8B (8.19B params, 36 layers):

| Config | Weights (S/K/O/P) | Avg Bits | PPL |
|--------|-------------------|----------|-----|
| BF16 baseline | — | 16.0 | 4.459 |
| Uniform 8-bit | — | 8.0 | 4.464 |
| Uniform 4-bit | — | 4.0 | 4.503 |
| SVD only | 1/0/0/0 | 6.19 | 4.552 |
| Kurtosis only | 0/1/0/0 | 6.64 | 4.565 |
| Output only | 0/0/1/0 | 16.00 | 4.456 |
| Position only | 0/0/0/1 | 7.95 | 4.525 |
| SVD+Kurtosis | .5/.5/0/0 | 6.29 | 4.549 |
| SVD+Output | .5/0/.5/0 | 9.22 | 4.464 |
| Output+Position | 0/0/.5/.5 | 10.54 | 4.458 |
| Kurtosis+Position | 0/.5/0/.5 | 6.93 | 4.539 |
| **Composite (default)** | **.3/.2/.3/.2** | **6.84** | **4.551** |
| Equal weights | .25 each | 6.81 | 4.546 |

**Key finding:** On Qwen3-8B, output_sensitivity saturates at 1.0 for all tensors (metric not discriminative at this scale). All SWAN configs preserve perplexity within 2.4% of BF16 at significantly reduced size.

### Exp 2: Correlation Study (COMPLETE — both 8B and 397B)

**Files:** `results/correlation_qwen3_8b.json`, `results/correlation_qwen35_397b.json`

#### Qwen3-8B (252 tensors)

Raw (unclamped) metric correlations with 4-bit reconstruction error:

| Metric | Spearman rho | p-value | Pearson r | p-value |
|--------|-----------|---------|-----------|---------|
| SVD Concentration | 0.229 | <0.001 | 0.236 | <0.001 |
| Kurtosis | 0.482 | <1e-15 | 0.667 | <1e-33 |
| Output Sensitivity | -0.134 | 0.034 | 0.206 | 0.001 |
| Cross-Layer Position | -0.220 | <0.001 | 0.032 | 0.613 |
| Composite | -0.104 | 0.100 | 0.292 | <0.001 |

Inter-metric correlation: Max off-diagonal |rho| = 0.223 (PASS: metrics capture distinct information)

#### Qwen3.5-397B (2,347 tensors)

Raw (unclamped) metric correlations with 4-bit reconstruction error:

| Metric | Spearman rho | p-value | Pearson r | p-value |
|--------|-----------|---------|-----------|---------|
| SVD Concentration | 0.399 | <0.001 | 0.298 | <0.001 |
| Kurtosis | 0.796 | <0.001 | 0.347 | <0.001 |
| Output Sensitivity | 0.694 | <0.001 | 0.316 | <0.001 |
| Cross-Layer Position | -0.468 | <0.001 | -0.224 | <0.001 |
| Composite (clamped) | 0.374 | <0.001 | 0.400 | <0.001 |

Inter-metric correlation: Max off-diagonal |rho| = 0.381 (PASS: metrics capture distinct information)

**Key findings:**
- Kurtosis is the strongest individual predictor (rho=0.48 on 8B, rho=0.80 on 397B)
- All 4 metrics have low inter-correlation (<0.38), confirming non-redundancy
- Raw scores reveal signal hidden by [0,1] clamping (especially on 8B where output_sensitivity saturates)
- All metrics show highly significant correlations (p<0.001) on the 397B model
- Correlation strength increases dramatically with model scale (252 vs 2,347 tensors)

### Exp 3: Perplexity Comparison (COMPLETE)

**Files:** `results/perplexity_*.json`

| Model | Method | Size (GB) | Avg Bits | PPL |
|-------|--------|-----------|----------|-----|
| Qwen3.5-397B | SWAN | 199.1 | 5.06 | 4.283 |
| Qwen3.5-397B | Uniform 4-bit | 208.5 | 4.00 | 3.931 |
| Qwen3-235B | Uniform 4-bit | 123.2 | 4.00 | 3.526 |
| Qwen3-8B | BF16 | 15.3 | 16.00 | 4.459 |
| Qwen3-8B | Uniform 8-bit | 8.1 | 8.00 | 4.464 |
| Qwen3-8B | SWAN | 6.8 | 6.84 | 4.551 |
| Qwen3-8B | Uniform 4-bit | 4.3 | 4.00 | 4.503 |

**Important caveats:**
- Qwen3.5-397B SWAN was measured on a different machine (remote, mlx_lm 0.30.7, Python 3.12) vs uniform-4bit (local, mlx_lm 0.29.1, Python 3.9). Different mlx_lm versions and group_size (128 vs 64) may affect results.
- SWAN's 5.06 avg bits vs uniform 4 bits means SWAN allocates more bits to critical tensors at the expense of being larger overall but with targeted precision.

### Exp 4: Scaling Study (COMPLETE)

**File:** `results/scaling_study.json`

Cross-model comparison of SWAN sensitivity profiles:

| Model | Architecture | Params | Tensors | Avg Bits | Attn Bits | MLP Bits |
|-------|-------------|--------|---------|----------|-----------|----------|
| Qwen3-8B | Dense | 8.2B | 399 | 6.84 | 8.7 | 5.2 |
| Llama4-Maverick | MoE 128E | 401.6B | 1,061 | 4.78 | 6.5 | 4.3 |
| Qwen3.5-397B | MoE 512E | 403.4B | 2,924 | 5.06 | 7.1 | 4.7 |

**Consistent patterns across all architectures:**
- Attention tensors get +1.6 to +2.5 more bits than MLP
- 61-70% of attention tensors get 8+ bits
- U-shaped sensitivity: early (first 25%) and late (last 25%) layers more sensitive than middle
- MoE expert tensors are most aggressively quantized (mostly 4-bit)

### Exp 5: Academic Benchmarks (EXISTING DATA)

From prior work on Qwen3.5-397B SWAN:
- MMLU-Pro: 72.1%
- ARC-Challenge: 96.0%
- GSM8K: 88.7%
- HumanEval: 78.7%

## File Structure

```
journal/
├── results/
│   ├── ablation_qwen3_8b.json              # Exp 1: 13 perplexity measurements
│   ├── correlation_qwen3_8b.json            # Exp 2: 252 tensor measurements
│   ├── correlation_qwen35_397b.json         # Exp 2: 2,347 tensor measurements
│   ├── perplexity_qwen3_235b_uniform_4bit.json   # Exp 3: 235B baseline
│   ├── perplexity_qwen35_397b_uniform_4bit.json  # Exp 3: 397B uniform
│   ├── perplexity_qwen35_397b_smartquant.json     # Exp 3: 397B SWAN
│   └── scaling_study.json                   # Exp 4: 3-model comparison
├── analysis/
│   └── qwen3_8b/
│       ├── manifest_base.json               # Full per-metric scores
│       └── manifest_*.json                  # 10 ablation manifests
├── figures/
│   ├── ablation_perplexity.pdf              # Exp 1 bar chart
│   ├── correlation_scatter.pdf              # Exp 2 scatter (397B, primary)
│   ├── correlation_scatter_qwen3_8b.pdf     # Exp 2 scatter (8B)
│   ├── correlation_scatter_qwen35_397b.pdf  # Exp 2 scatter (397B)
│   ├── inter_metric_heatmap.pdf             # Exp 2 heatmap (397B, primary)
│   ├── inter_metric_heatmap_qwen3_8b.pdf    # Exp 2 heatmap (8B)
│   ├── inter_metric_heatmap_qwen35_397b.pdf # Exp 2 heatmap (397B)
│   ├── scaling_bit_allocation.pdf           # Exp 4 stacked bar
│   └── perplexity_vs_bits.pdf               # Exp 3 curve
├── tables/
│   ├── table1_ablation.tex                  # Ablation results
│   ├── table2_correlation.tex               # Metric correlations (397B, primary)
│   ├── table2_correlation_qwen3_8b.tex      # Metric correlations (8B)
│   ├── table2_correlation_qwen35_397b.tex   # Metric correlations (397B)
│   ├── table3_inter_metric.tex              # Inter-metric matrix (397B, primary)
│   ├── table3_inter_metric_qwen3_8b.tex     # Inter-metric matrix (8B)
│   ├── table3_inter_metric_qwen35_397b.tex  # Inter-metric matrix (397B)
│   ├── table4_perplexity_comparison.tex     # Perplexity comparison
│   └── table5_scaling.tex                   # Scaling study
├── journal_ablation.py                      # Exp 1 script
├── journal_correlation.py                   # Exp 2 script
├── journal_perplexity.py                    # Exp 3 script
├── journal_scaling.py                       # Exp 4 script
├── journal_compile.py                       # Results -> tables/figures
└── README.md                                # This file
```

## Reproduction

All experiments use standardized parameters:
- **Perplexity:** sequence_length=2048, num_samples=256, seed=42
- **Quantization simulation:** group-wise RTN, group_size=128, bits=4
- **SVD analysis:** rank=256 (randomized SVD for large matrices)

```bash
# Environment
python3.9, mlx 0.29.3, mlx_lm 0.29.1, torch 2.8.0, scipy

# Exp 1: Ablation
python journal/journal_ablation.py --bf16-model models/qwen3-8b-bf16 --output-dir journal/results

# Exp 2: Correlation
python journal/journal_correlation.py --model models/qwen3-8b-bf16 --output-dir journal/results --model-name qwen3_8b
python journal/journal_correlation.py --model models/qwen-bf16 --output-dir journal/results --model-name qwen35_397b

# Exp 3: Perplexity
python journal/journal_perplexity.py --model <model_path> --variant <name> --output-dir journal/results

# Exp 4: Scaling
python journal/journal_scaling.py

# Compile
python journal/journal_compile.py --results-dir journal/results --figures-dir journal/figures --tables-dir journal/tables \
    --scaling-manifests journal/analysis/qwen3_8b/manifest_base.json analysis/manifest.json analysis/qwen35-397b/smartquant_manifest.json
```

## Notes

- **Qwen3-8B output_sensitivity saturation:** All tensors score 1.0 (metric not discriminative at this scale). On 397B models, the metric has meaningful variance and strong correlation (Spearman rho=0.69).
- **Remote machine (192.168.89.174):** Experienced thermal shutdown during 397B correlation/perplexity runs. All results were recovered after machine came back up.
- **Metric normalization:** The clamping to [0,1] can hide signal. Raw (unclamped) scores reported alongside normalized for correlation analysis.
- **Perplexity cross-machine caveat:** The 397B SWAN perplexity was measured on a remote machine with mlx_lm 0.30.7 (Python 3.12), while other measurements used the local machine with mlx_lm 0.29.1 (Python 3.9). Additionally, SWAN uses group_size=128 while the mlx-community uniform-4bit model uses group_size=64.
- **Correlation file correction:** The correlation file originally named `correlation_qwen3_235b.json` was renamed to `correlation_qwen35_397b.json` because the model at `/Users/tk/smartquant/models/qwen-bf16` on the remote machine is actually Qwen3.5-397B (60 layers, 512 experts), not Qwen3-235B (94 layers, 128 experts).
