# We Just Fit a 400B Parameter Model on a Single Mac — and Made It Smarter Than Uniform Quantization

## How SWAN Brings Frontier-Scale AI to Apple Silicon Without a Single GPU

There's a quiet revolution happening in the Apple ML ecosystem, and it doesn't involve a data center.

Last week, I ran Qwen3.5-397B — a 403-billion-parameter Mixture-of-Experts model with 512 experts — on a single Mac Studio. Not a cluster. Not a cloud instance. One machine, one M3 Ultra chip, 512 GB of unified memory. The model scored 96% on ARC-Challenge, 77.1% on MMLU-Pro (with thinking enabled), and generated coherent responses at interactive speeds.

The key was **SWAN** (Statistical Weight Analysis for N-bit allocation), a new quantization method I built specifically for the MLX ecosystem. And I think it matters for everyone building on Apple Silicon.

---

### The Problem: Calibration-Based Quantization Doesn't Scale on Mac

The dominant quantization methods — GPTQ, AWQ, SqueezeLLM — all share a hidden dependency: they need calibration data. You run representative samples through the model, measure which weights matter most, then quantize accordingly. This works fine when you have an 8-GPU node with 640 GB of VRAM.

But on a Mac? Running calibration on a 400B model means full forward passes through 800+ GB of parameters. Even with 512 GB of unified memory, you're swapping to disk, waiting hours, and hoping the calibration distribution matches your actual use case. For most practitioners, this is a non-starter.

So the community defaults to uniform 4-bit quantization — every tensor gets the same treatment. It's fast, it works, but it's leaving quality on the table. Some tensors need more precision. Some need less. Uniform quantization can't tell the difference.

### SWAN: Let the Weights Speak for Themselves

SWAN takes a different approach: instead of running data through the model, it analyzes the weight tensors directly. Four metrics, each capturing a different aspect of how a tensor will respond to quantization:

1. **SVD spectral concentration** — is the tensor's information concentrated in a few critical directions?
2. **Excess kurtosis** — does the weight distribution have outliers that will blow up quantization error?
3. **Output noise amplification** — how much does this layer amplify perturbations?
4. **Reconstruction error proxy** — what actually happens when you quantize and dequantize this tensor?

These four scores combine into a composite that drives per-tensor bit-width allocation: 2, 4, 8, or 16-bit. No calibration data. No gradients. No fine-tuning.

The entire analysis of a 400B+ parameter model completes in **under 13 minutes** on a Mac Studio.

### The Results: Better Than Uniform, Zero Data Required

In controlled experiments on Qwen3.5-397B at matched group sizes:

| Method | Size | Avg Bits | Perplexity |
|--------|------|----------|------------|
| **SWAN** | **199 GB** | **4.31** | **4.283** |
| Uniform 4-bit RTN | 196 GB | 4.25 | 4.298 |

SWAN outperforms uniform quantization — and it figured out which tensors to protect without seeing a single token of training data. The method identified that 4.3% of tensors (primarily shared expert gates, MTP layers, and linear attention projections) genuinely benefit from 8-bit precision, while 95.2% compress cleanly to 4-bit.

The academic benchmarks on the quantized model tell the rest of the story:

- **MMLU-Pro**: **77.1%** (thinking enabled) | 72.1% (thinking disabled)
- **ARC-Challenge**: 96.0%
- **GSM8K**: 88.7%
- **HumanEval**: 78.7%

Qwen3.5 is a thinking model — enabling its native chain-of-thought reasoning improves MMLU-Pro by +5 points. These are frontier-class scores, running locally on consumer hardware.

### Why This Matters for the Apple AI Community

The MLX ecosystem is at an inflection point. Apple Silicon's unified memory architecture is uniquely suited for large model inference — no PCIe bottleneck, no GPU memory wall, just a flat address space shared between CPU and GPU. But the software tooling needs to match the hardware's ambition.

Right now, the MLX community relies heavily on `mlx-community` uploads of uniformly quantized models. SWAN adds a layer of intelligence to that pipeline:

- **No infrastructure barrier**: If you can load a model, you can analyze it. No calibration dataset to curate, no GPU hours to rent.
- **Per-tensor precision**: The same framework that drives uniform quantization in `mlx_lm.convert` now accepts SWAN's sensitivity manifest as a predicate function. It's a drop-in enhancement.
- **Architecture-agnostic**: SWAN has been validated on dense transformers (Qwen3-8B), 128-expert MoE (Llama4-Maverick), and 512-expert MoE (Qwen3.5-397B). The sensitivity patterns it discovers are consistent across all three.
- **Reproducible**: Same weights in, same analysis out. No seed sensitivity, no calibration distribution mismatch.

For teams building local AI assistants, on-device reasoning, or private inference pipelines on Apple hardware — this is a practical tool that works today.

### The Bigger Picture

We validated SWAN's metrics rigorously. Kurtosis alone predicts quantization error with Spearman correlation of 0.80 across 2,347 tensors. The four metrics are non-redundant (max inter-metric correlation 0.38). And an interesting finding: **group size matters more than bit allocation** — halving the quantization group size from 128 to 64 improves perplexity by 0.23, while SWAN's selective 8-bit allocation adds 0.015. Both matter, but the community should be paying more attention to group size.

There's also an honest limitation to share: SWAN's reconstruction error metric saturates on very large models. The normalization range is too narrow for 512-expert MoE layers. This is discussed in the paper and is the first target for v3.

### Try It

Everything is open source under PolyForm Noncommercial 1.0.0:

- **Paper**: [SWAN: Data-Free Mixed-Precision Quantization for LLMs](https://huggingface.co/spaces/baa-ai/swan-paper)
- **Code**: [github.com/baa-ai/swan-quantization](https://github.com/baa-ai/swan-quantization)
- **Quantized Model**: [Qwen3.5-397B-A17B-SWAN-4bit](https://huggingface.co/baa-ai/Qwen3.5-397B-A17B-SWAN-4bit) (199 GB, ready for `mlx_lm.load`)

If you're running large models on Apple Silicon, I'd love to hear what you think. The method is designed to be extended — better metrics, adaptive normalization, joint group-size optimization — and contributions are welcome.

---

*Black Sheep AI Research*
*[GitHub](https://github.com/baa-ai) | [Website](https://baa.ai)) | [GitHub](https://github.com/baa-ai)*
