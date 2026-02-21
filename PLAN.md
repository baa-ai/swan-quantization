# Plan: Official Benchmark Evaluation

## Goal
Run standard academic benchmarks on all three SWAN models (thresholds 0.35, 0.45, 0.55) and compare against Meta's official Llama 4 Maverick BF16 scores.

## Benchmarks Selected

| Benchmark | Questions | Meta's Score | Type | Why |
|-----------|-----------|-------------|------|-----|
| MMLU-Pro | ~1000 (sampled) | 80.5 | MCQ reasoning | Meta's headline knowledge benchmark |
| GPQA Diamond | 198 | 69.8 | MCQ reasoning | Graduate-level, Meta's headline reasoning benchmark |
| GSM8K | 1319 | not reported | Chain-of-thought math | Industry-standard math reasoning |
| HumanEval | 164 | ~86.4 (3rd party) | Code generation | Industry-standard coding benchmark |

## Approach: Custom Evaluation Script

Rather than fighting lm-evaluation-harness compatibility with MLX (it doesn't support loglikelihood via chat API, which most MCQ tasks need), we'll write a custom evaluation script that:

1. **Loads the MLX model directly** (no API overhead, fastest possible)
2. **Downloads datasets** from HuggingFace `datasets` library
3. **Formats prompts** matching Meta's methodology (0-shot, temp=0)
4. **Parses answers** from generated text for MCQ (regex for "answer is X" patterns)
5. **Executes code** in a sandbox for HumanEval (subprocess with timeout)
6. **Computes accuracy** per benchmark and overall

## Execution Order (sequential — only one model fits in memory)

For each model (0.55 first — smallest/fastest, then 0.45, then 0.35):
1. Load model
2. Run all 4 benchmarks
3. Save results JSON
4. Unload model

Estimated time per model:
- MMLU-Pro (1000 questions × ~3s each): ~50 min
- GPQA Diamond (198 questions × ~5s each): ~16 min
- GSM8K (1319 questions × ~5s each): ~110 min
- HumanEval (164 problems × ~8s each): ~22 min
- Total per model: ~3.3 hours
- Total for 3 models: ~10 hours

## Files to Create

1. `~/smartquant/official_benchmarks.py` — Main evaluation script
2. `~/smartquant/web/benchmark-report.html` — Final comparison report

## Report Content

- Per-benchmark accuracy for each threshold
- Comparison table: SWAN scores vs Meta's official BF16 scores
- Delta analysis: how much quantization degrades each benchmark
- Cross-threshold comparison: does 0.35 actually help vs 0.55?
- Conclusions and recommendations
