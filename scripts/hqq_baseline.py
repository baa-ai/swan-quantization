#!/usr/bin/env python3
"""
HQQ 4-bit Baseline — Perplexity Measurement for SmartQuant Comparison.

Quantizes Qwen3-8B BF16 with HQQ (Half-Quadratic Quantization) at 4-bit,
group_size=128, then measures perplexity using the SAME dataset and parameters
as mlx_lm.perplexity for direct comparability.

Algorithm (matches mlx_lm.perplexity exactly):
  1. Load dataset (allenai/tulu-3-sft-mixture, train split)
  2. Tokenize with chat template, concatenate all tokens
  3. Shuffle samples with fixed seed, take first N tokens
  4. Reshape into (num_samples, sequence_length) chunks
  5. For each chunk: compute per-token cross-entropy loss
  6. Perplexity = exp(mean of all token losses)
  7. Standard error via delta method: ppl * std(losses) / sqrt(n_tokens)

Usage:
    python scripts/hqq_baseline.py \
        --model /Users/tk/smartquant/models/qwen3-8b-bf16 \
        --output-dir /Users/tk/smartquant/scripts/results \
        [--save-model /Users/tk/smartquant/models/qwen3-8b-hqq-4bit] \
        [--seq-len 2048] [--num-samples 256] [--seed 42] \
        [--nbits 4] [--group-size 128] [--batch-size 4]

Requirements:
    pip install hqq transformers torch datasets
    Python env: /Users/tk/smartquant-env/bin/python3
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Dataset loading — replicates mlx_lm.perplexity.load_data exactly
# ---------------------------------------------------------------------------

def load_data_for_ppl(tokenizer, data_path: str, num_samples: int,
                      sequence_length: int, seed: int) -> torch.Tensor:
    """Load and tokenize dataset identically to mlx_lm.perplexity.load_data.

    The mlx_lm pipeline:
      1. Loads HF dataset with train split
      2. Creates ChatDataset (applies chat template to each sample)
      3. Shuffles sample order with np.random.permutation (seeded)
      4. Concatenates tokens from shuffled samples until we have enough
      5. Trims to exact multiple of sequence_length
      6. Reshapes to (num_samples, sequence_length)

    We replicate this exactly for comparability.
    """
    from datasets import load_dataset

    np.random.seed(seed)

    print(f"  Loading dataset: {data_path}")
    ds = load_dataset(data_path, split="train")
    print(f"  Dataset size: {len(ds)} samples")

    # Permute sample order (same as mlx_lm)
    perm = np.random.permutation(len(ds)).tolist()

    num_tokens_needed = sequence_length * num_samples

    # Tokenize samples in permuted order, concatenating tokens
    data = []
    i = 0
    while len(data) < num_tokens_needed:
        sample = ds[perm[i]]
        messages = sample["messages"]
        tokens = tokenizer.apply_chat_template(messages)
        data.extend(tokens)
        i += 1
        if i % 500 == 0:
            print(f"    Tokenized {i} samples, {len(data)} tokens so far...")

    print(f"  Total tokens collected: {len(data)} (from {i} samples)")

    # Trim to exact multiple of sequence_length and reshape
    total = (len(data) // sequence_length) * sequence_length
    data = data[:total]
    data = torch.tensor(data, dtype=torch.long).reshape(-1, sequence_length)

    if num_samples > 0:
        data = data[:num_samples]

    print(f"  Final data shape: {data.shape} "
          f"({data.shape[0]} samples x {data.shape[1]} tokens)")
    return data


# ---------------------------------------------------------------------------
# Perplexity evaluation — replicates mlx_lm.perplexity.eval_ppl exactly
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_ppl(model, data: torch.Tensor, batch_size: int = 4,
             device: str = "mps") -> tuple:
    """Evaluate perplexity matching mlx_lm.perplexity.eval_ppl algorithm.

    Returns:
        (perplexity, standard_error_of_perplexity)
    """
    model.eval()
    all_losses = []

    num_batches = (len(data) + batch_size - 1) // batch_size

    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size].to(device)

        # Forward pass: get logits for all tokens except last
        # Input: tokens[:-1], Target: tokens[1:]
        outputs = model(input_ids=batch[:, :-1])
        logits = outputs.logits.float()  # (B, L-1, V)

        # Per-token cross-entropy (no reduction, matching mlx_lm)
        # logits: (B, L-1, V), targets: (B, L-1)
        targets = batch[:, 1:]
        B, L, V = logits.shape
        losses = F.cross_entropy(
            logits.reshape(B * L, V),
            targets.reshape(B * L),
            reduction="none",
        )
        all_losses.append(losses.cpu())

        batch_idx = i // batch_size + 1
        if batch_idx % 1 == 0 or batch_idx == num_batches:
            current_mean = torch.cat(all_losses).mean().item()
            current_ppl = math.exp(current_mean)
            print(f"  Batch {batch_idx}/{num_batches} "
                  f"(running ppl: {current_ppl:.3f})", end="\r")

        # Free GPU memory
        del outputs, logits, losses, batch
        if device == "mps":
            torch.mps.empty_cache()

    print()  # newline after progress

    # Concatenate all losses
    all_losses = torch.cat(all_losses)

    # Calculate perplexity and standard error (same as mlx_lm)
    mean_loss = all_losses.mean().item()
    ppl = math.exp(mean_loss)

    std_dev = all_losses.std(correction=1).item()  # ddof=1
    num_tokens = all_losses.numel()
    standard_error = std_dev / math.sqrt(num_tokens)
    standard_error_ppl = ppl * standard_error  # delta method

    return ppl, standard_error_ppl


# ---------------------------------------------------------------------------
# Model loading and quantization
# ---------------------------------------------------------------------------

def load_and_quantize(model_path: str, nbits: int, group_size: int,
                      device: str = "mps"):
    """Load BF16 model and quantize with HQQ via transformers integration.

    Uses transformers' native HQQ integration which applies quantization
    on-the-fly during model loading, avoiding full BF16 materialization.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, HqqConfig

    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print(f"Loading and quantizing model with HQQ "
          f"(nbits={nbits}, group_size={group_size})...")
    print(f"  Target device: {device}")

    quant_config = HqqConfig(
        nbits=nbits,
        group_size=group_size,
        axis=1,  # column-wise (faster backends)
    )

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float32,  # HQQ dequantizes to float32; bfloat16 causes dtype mismatch
        quantization_config=quant_config,
        device_map=device,
    )
    quant_time = time.time() - t0

    print(f"  Quantization completed in {quant_time:.1f}s")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params / 1e6:.1f}M")

    return model, tokenizer, quant_time


def save_quantized_model(model, tokenizer, save_path: str):
    """Save quantized model and tokenizer."""
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving quantized model to {save_dir}...")
    t0 = time.time()
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    save_time = time.time() - t0

    # Report size
    total_bytes = sum(
        f.stat().st_size for f in save_dir.rglob("*") if f.is_file()
    )
    print(f"  Saved in {save_time:.1f}s, total size: {total_bytes / 1e9:.2f} GB")
    return total_bytes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="HQQ 4-bit baseline perplexity for SmartQuant comparison"
    )
    parser.add_argument(
        "--model", type=str,
        default="/Users/tk/smartquant/models/qwen3-8b-bf16",
        help="Path to BF16 model directory",
    )
    parser.add_argument(
        "--save-model", type=str, default=None,
        help="Path to save HQQ-quantized model (optional)",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="/Users/tk/smartquant/scripts/results",
        help="Directory for results JSON",
    )
    parser.add_argument(
        "--data-path", type=str,
        default="allenai/tulu-3-sft-mixture",
        help="HF dataset path (must match mlx_lm.perplexity default)",
    )
    parser.add_argument("--nbits", type=int, default=4, help="Quantization bits")
    parser.add_argument("--group-size", type=int, default=128, help="Group size")
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length")
    parser.add_argument("--num-samples", type=int, default=256, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for perplexity eval")
    parser.add_argument("--device", type=str, default="mps",
                        help="PyTorch device (mps, cpu)")

    args = parser.parse_args()

    print("=" * 70)
    print("HQQ BASELINE — PERPLEXITY MEASUREMENT")
    print("=" * 70)
    print(f"Model:          {args.model}")
    print(f"Quantization:   HQQ {args.nbits}-bit, group_size={args.group_size}")
    print(f"Dataset:        {args.data_path}")
    print(f"Seq length:     {args.seq_len}")
    print(f"Num samples:    {args.num_samples}")
    print(f"Seed:           {args.seed}")
    print(f"Batch size:     {args.batch_size}")
    print(f"Device:         {args.device}")
    print()

    # Seed everything
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Step 1: Load and quantize
    print("Step 1: Loading and quantizing model...")
    print("-" * 50)
    model, tokenizer, quant_time = load_and_quantize(
        args.model, args.nbits, args.group_size, args.device,
    )
    print()

    # Step 2: Optionally save
    model_size_bytes = None
    if args.save_model:
        print("Step 2: Saving quantized model...")
        print("-" * 50)
        model_size_bytes = save_quantized_model(model, tokenizer, args.save_model)
        print()

    # Step 3: Load evaluation data
    # Re-seed to match mlx_lm (it seeds AFTER model load)
    np.random.seed(args.seed)

    print("Step 3: Loading evaluation data...")
    print("-" * 50)
    data = load_data_for_ppl(
        tokenizer, args.data_path, args.num_samples,
        args.seq_len, args.seed,
    )
    print()

    # Step 4: Evaluate perplexity
    print("Step 4: Evaluating perplexity...")
    print("-" * 50)
    t0 = time.time()
    ppl, se = eval_ppl(model, data, batch_size=args.batch_size, device=args.device)
    eval_time = time.time() - t0
    tokens_evaluated = data.shape[0] * (data.shape[1] - 1)

    # Results
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Perplexity:       {ppl:.4f} +/- {se:.4f}")
    print(f"Quant time:       {quant_time:.1f}s")
    print(f"Eval time:        {eval_time:.1f}s")
    print(f"Tokens evaluated: {tokens_evaluated:,}")
    print(f"Tokens/sec:       {tokens_evaluated / eval_time:,.0f}")
    if model_size_bytes:
        print(f"Model size:       {model_size_bytes / 1e9:.2f} GB")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variant = f"hqq_{args.nbits}bit_g{args.group_size}"
    model_name = Path(args.model).name

    results = {
        "model_name": model_name,
        "variant": variant,
        "method": "hqq",
        "quantization": {
            "nbits": args.nbits,
            "group_size": args.group_size,
            "axis": 1,
            "hqq_version": None,
        },
        "parameters": {
            "sequence_length": args.seq_len,
            "num_samples": args.num_samples,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "dataset": args.data_path,
        },
        "results": {
            "perplexity": round(ppl, 4),
            "standard_error": round(se, 4),
            "tokens_evaluated": tokens_evaluated,
        },
        "timing": {
            "quantization_seconds": round(quant_time, 1),
            "evaluation_seconds": round(eval_time, 1),
            "tokens_per_second": round(tokens_evaluated / eval_time, 0),
        },
        "environment": {
            "torch_version": torch.__version__,
            "device": args.device,
            "python_version": sys.version.split()[0],
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # Add HQQ version
    try:
        import hqq
        results["quantization"]["hqq_version"] = hqq.__version__
    except Exception:
        pass

    # Add model size if saved
    if model_size_bytes:
        results["model_size_gb"] = round(model_size_bytes / 1e9, 2)

    output_path = output_dir / f"perplexity_{model_name}_{variant}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
