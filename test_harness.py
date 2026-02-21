#!/usr/bin/env python3
"""
Test Harness for SmartQuant Maverick.

Three testing modes:
  1. Interactive chat (REPL)
  2. Automated benchmark suite
  3. A/B comparison with uniform 4-bit model

Usage:
  python test_harness.py --model ~/smartquant/models/maverick-smartquant
  python test_harness.py --model ~/smartquant/models/maverick-smartquant --mode benchmark
  python test_harness.py --mode compare \
      --model-a ~/smartquant/models/maverick-smartquant \
      --model-b ~/smartquant/models/maverick-uniform-4bit
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx


# === Benchmark Prompts ===

BENCHMARK_PROMPTS = {
    "factual_recall": [
        "What is the capital of New Zealand?",
        "Who wrote Romeo and Juliet?",
        "What is the speed of light in meters per second?",
        "Name the planets in our solar system in order.",
        "What year did World War II end?",
    ],
    "reasoning": [
        "If a train travels at 60 mph for 2.5 hours, how far does it go?",
        "A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much does the ball cost?",
        "If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly?",
        (
            "I have 3 boxes: one has only apples, one has only oranges, and one has both. "
            "They are all mislabeled. If I pick one fruit from the box labeled 'Apples and Oranges', "
            "and it's an apple, what's in each box?"
        ),
    ],
    "coding": [
        "Write a Python function to find the longest common subsequence of two strings.",
        "Implement a simple LRU cache in Python with get() and put() methods.",
        "Write a SQL query to find the second highest salary from an employees table.",
    ],
    "creative_writing": [
        "Write a haiku about machine learning.",
        "Describe a sunset as if you were a medieval knight writing in their journal.",
        "Create a short dialogue between two AIs debating whether consciousness requires a body.",
    ],
    "instruction_following": [
        "List exactly 5 programming languages created before 1980. Number them 1-5.",
        "Summarize the concept of quantum entanglement in exactly 3 sentences.",
        "Write the word 'hello' backwards, then capitalize every other letter.",
    ],
    "long_form": [
        (
            "Explain the transformer architecture in detail, covering attention mechanisms, "
            "positional encoding, and the differences between encoder and decoder blocks."
        ),
    ],
}


def get_memory_info() -> Dict[str, float]:
    """Get MLX Metal memory usage in GB."""
    try:
        return {
            "allocated_gb": mx.get_active_memory() / 1e9,
            "peak_gb": mx.get_peak_memory() / 1e9,
        }
    except Exception:
        return {"allocated_gb": 0.0, "peak_gb": 0.0}


def load_model(model_path: str) -> Tuple[Any, Any]:
    """Load model and tokenizer from path."""
    from mlx_lm import load

    print(f"Loading model from {model_path} ...")
    start = time.time()
    model, tokenizer = load(model_path)
    elapsed = time.time() - start
    mem = get_memory_info()
    print(f"Model loaded in {elapsed:.1f}s")
    print(f"Memory: {mem['allocated_gb']:.1f} GB allocated, {mem['peak_gb']:.1f} GB peak")
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 2048,
    temp: float = 0.7,
    top_p: float = 0.9,
    stream: bool = False,
) -> Dict[str, Any]:
    """Generate a response and collect metrics."""
    from mlx_lm import generate as mlx_generate
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=temp, top_p=top_p)

    start = time.time()
    first_token_time = None

    if stream:
        tokens = []
        for response in stream_generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
        ):
            token_text = response.text if hasattr(response, "text") else str(response)
            if first_token_time is None:
                first_token_time = time.time()
            print(token_text, end="", flush=True)
            tokens.append(token_text)
        print()
        response_text = "".join(tokens)
        token_count = len(tokens)
    else:
        response_text = mlx_generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
        )
        first_token_time = start
        token_count = len(tokenizer.encode(response_text))

    elapsed = time.time() - start
    ttft = (first_token_time - start) if first_token_time else 0.0
    tok_per_sec = token_count / elapsed if elapsed > 0 else 0.0
    mem = get_memory_info()

    return {
        "prompt": prompt,
        "response": response_text,
        "token_count": token_count,
        "elapsed_seconds": elapsed,
        "ttft_seconds": ttft,
        "tokens_per_second": tok_per_sec,
        "memory_allocated_gb": mem["allocated_gb"],
        "memory_peak_gb": mem["peak_gb"],
    }


# === MODE 1: Interactive Chat ===


def run_interactive(model_path: str, max_tokens: int, temp: float) -> None:
    """Run an interactive chat REPL."""
    model, tokenizer = load_model(model_path)

    print("\n=== SmartQuant Interactive Chat ===")
    print("Commands: /stats  /context  /clear  /temp N  /bench  /memory  /quit")
    print()

    conversation_tokens = 0

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        if user_input.lower() in ["/quit", "/exit", "quit", "exit"]:
            print("Goodbye.")
            break

        if user_input == "/stats":
            mem = get_memory_info()
            print(f"Memory: {mem['allocated_gb']:.1f} GB allocated, {mem['peak_gb']:.1f} GB peak")
            continue

        if user_input == "/context":
            print(f"Approximate conversation tokens: {conversation_tokens}")
            continue

        if user_input == "/clear":
            conversation_tokens = 0
            print("Conversation cleared.")
            continue

        if user_input.startswith("/temp"):
            parts = user_input.split()
            if len(parts) == 2:
                try:
                    temp = float(parts[1])
                    print(f"Temperature set to {temp}")
                except ValueError:
                    print("Usage: /temp 0.7")
            else:
                print(f"Current temperature: {temp}")
            continue

        if user_input == "/memory":
            mem = get_memory_info()
            print(f"  MLX allocated: {mem['allocated_gb']:.1f} GB")
            print(f"  MLX peak:      {mem['peak_gb']:.1f} GB")
            continue

        if user_input == "/bench":
            print("Running quick benchmark (5 prompts) ...")
            quick_prompts = BENCHMARK_PROMPTS["factual_recall"]
            for p in quick_prompts:
                result = generate_response(model, tokenizer, p, max_tokens=200, temp=0.0)
                print(f"  Q: {p}")
                print(f"  A: {result['response'][:100]}...")
                print(f"  [{result['token_count']} tok, {result['tokens_per_second']:.1f} tok/s]")
                print()
            continue

        # Normal message
        print("Assistant: ", end="")
        result = generate_response(
            model, tokenizer, user_input, max_tokens=max_tokens, temp=temp, stream=True
        )
        conversation_tokens += result["token_count"]
        print(
            f"[{result['token_count']} tokens, "
            f"{result['tokens_per_second']:.1f} tok/s, "
            f"{result['elapsed_seconds']:.1f}s]"
        )
        print()


# === MODE 2: Automated Benchmark ===


def run_benchmark(
    model_path: str,
    max_tokens: int,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the full benchmark suite."""
    model, tokenizer = load_model(model_path)

    results = {
        "model_path": model_path,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "categories": {},
        "summary": {},
    }

    total_tokens = 0
    total_time = 0.0
    all_speeds = []

    for category, prompts in BENCHMARK_PROMPTS.items():
        print(f"\n=== {category.upper()} ({len(prompts)} prompts) ===")
        cat_results = []

        for i, prompt in enumerate(prompts):
            print(f"  [{i + 1}/{len(prompts)}] {prompt[:60]}...")
            result = generate_response(
                model, tokenizer, prompt, max_tokens=max_tokens, temp=0.0
            )
            cat_results.append(result)
            total_tokens += result["token_count"]
            total_time += result["elapsed_seconds"]
            all_speeds.append(result["tokens_per_second"])
            print(
                f"    → {result['token_count']} tok, "
                f"{result['tokens_per_second']:.1f} tok/s, "
                f"{result['elapsed_seconds']:.1f}s"
            )

        results["categories"][category] = cat_results

    # Summary
    results["summary"] = {
        "total_tokens": total_tokens,
        "total_time_seconds": total_time,
        "avg_tokens_per_second": sum(all_speeds) / len(all_speeds) if all_speeds else 0,
        "min_tokens_per_second": min(all_speeds) if all_speeds else 0,
        "max_tokens_per_second": max(all_speeds) if all_speeds else 0,
        "memory": get_memory_info(),
    }

    # Print summary table
    print(f"\n{'='*60}")
    print(f"BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average speed: {results['summary']['avg_tokens_per_second']:.1f} tok/s")
    print(f"Speed range: {results['summary']['min_tokens_per_second']:.1f} - "
          f"{results['summary']['max_tokens_per_second']:.1f} tok/s")
    mem = results["summary"]["memory"]
    print(f"Memory: {mem['allocated_gb']:.1f} GB allocated, {mem['peak_gb']:.1f} GB peak")

    # Save results
    if output_path is None:
        output_path = str(Path.home() / "smartquant" / "results" / "benchmark_results.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    return results


# === MODE 3: A/B Comparison ===


def run_comparison(
    model_a_path: str,
    model_b_path: str,
    max_tokens: int,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare two models on the same benchmark prompts.

    Models are loaded sequentially (one at a time) to fit in memory.
    """
    comparison = {
        "model_a": model_a_path,
        "model_b": model_b_path,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "categories": {},
        "summary": {},
    }

    # Run Model A
    print(f"\n{'='*60}")
    print(f"RUNNING MODEL A: {model_a_path}")
    print(f"{'='*60}")
    results_a = run_benchmark(model_a_path, max_tokens)

    # Force cleanup
    import gc
    gc.collect()
    mx.reset_peak_memory()

    # Run Model B
    print(f"\n{'='*60}")
    print(f"RUNNING MODEL B: {model_b_path}")
    print(f"{'='*60}")
    results_b = run_benchmark(model_b_path, max_tokens)

    # Compare
    print(f"\n{'='*60}")
    print(f"A/B COMPARISON")
    print(f"{'='*60}")
    print(f"Model A (SmartQuant): {model_a_path}")
    print(f"Model B (Baseline):   {model_b_path}")
    print()

    speed_a = results_a["summary"]["avg_tokens_per_second"]
    speed_b = results_b["summary"]["avg_tokens_per_second"]
    print(f"Average speed:")
    print(f"  Model A: {speed_a:.1f} tok/s")
    print(f"  Model B: {speed_b:.1f} tok/s")
    if speed_b > 0:
        diff = (speed_a - speed_b) / speed_b * 100
        print(f"  Difference: {diff:+.1f}%")

    mem_a = results_a["summary"]["memory"]
    mem_b = results_b["summary"]["memory"]
    print(f"\nMemory usage:")
    print(f"  Model A: {mem_a['peak_gb']:.1f} GB peak")
    print(f"  Model B: {mem_b['peak_gb']:.1f} GB peak")

    # Per-category comparison
    print(f"\nPer-category speed comparison:")
    for category in BENCHMARK_PROMPTS:
        cat_a = results_a["categories"].get(category, [])
        cat_b = results_b["categories"].get(category, [])
        if cat_a and cat_b:
            avg_a = sum(r["tokens_per_second"] for r in cat_a) / len(cat_a)
            avg_b = sum(r["tokens_per_second"] for r in cat_b) / len(cat_b)
            print(f"  {category:25s}  A: {avg_a:6.1f} tok/s  B: {avg_b:6.1f} tok/s")

    # Side-by-side responses for manual review
    print(f"\n{'='*60}")
    print("SIDE-BY-SIDE RESPONSES (first prompt per category)")
    print(f"{'='*60}")
    for category in BENCHMARK_PROMPTS:
        cat_a = results_a["categories"].get(category, [])
        cat_b = results_b["categories"].get(category, [])
        if cat_a and cat_b:
            print(f"\n--- {category} ---")
            print(f"Q: {cat_a[0]['prompt'][:80]}")
            print(f"A (SmartQuant): {cat_a[0]['response'][:200]}...")
            print(f"B (Baseline):   {cat_b[0]['response'][:200]}...")

    comparison["results_a"] = results_a
    comparison["results_b"] = results_b
    comparison["summary"] = {
        "speed_a": speed_a,
        "speed_b": speed_b,
        "speed_diff_pct": (speed_a - speed_b) / speed_b * 100 if speed_b > 0 else 0,
        "mem_a_peak_gb": mem_a["peak_gb"],
        "mem_b_peak_gb": mem_b["peak_gb"],
    }

    # Save
    if output_path is None:
        output_path = str(Path.home() / "smartquant" / "results" / "comparison_report.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"\nComparison saved to: {output_path}")

    return comparison


def main():
    parser = argparse.ArgumentParser(description="SmartQuant Test Harness")
    parser.add_argument(
        "--model",
        default=str(Path.home() / "smartquant" / "models" / "maverick-smartquant"),
        help="Path to quantized model (for interactive/benchmark mode)",
    )
    parser.add_argument(
        "--mode",
        choices=["interactive", "benchmark", "compare"],
        default="interactive",
        help="Test mode (default: interactive)",
    )
    parser.add_argument(
        "--model-a",
        help="Model A path (for compare mode, SmartQuant)",
    )
    parser.add_argument(
        "--model-b",
        help="Model B path (for compare mode, baseline)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate (default: 2048)",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.7,
        help="Temperature for generation (default: 0.7)",
    )
    parser.add_argument(
        "--output",
        help="Output path for results JSON",
    )
    args = parser.parse_args()

    if args.mode == "interactive":
        run_interactive(args.model, args.max_tokens, args.temp)
    elif args.mode == "benchmark":
        run_benchmark(args.model, args.max_tokens, args.output)
    elif args.mode == "compare":
        if not args.model_a or not args.model_b:
            parser.error("--model-a and --model-b required for compare mode")
        run_comparison(args.model_a, args.model_b, args.max_tokens, args.output)


if __name__ == "__main__":
    main()
