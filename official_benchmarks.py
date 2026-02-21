#!/usr/bin/env python3
"""
Official Benchmark Evaluation for SmartQuant Models.

Runs standard academic benchmarks and compares results against
Meta's official Llama 4 Maverick BF16 scores.

Benchmarks:
  - MMLU-Pro (0-shot MCQ, sampled 1000 questions)
  - ARC-Challenge (0-shot MCQ, all 1172 questions)
  - GSM8K (0-shot chain-of-thought math, all 1319 questions)
  - HumanEval (code generation, all 164 problems)

Usage:
    python official_benchmarks.py --model ~/smartquant/models/maverick-smartquant
    python official_benchmarks.py --model ~/smartquant/models/maverick-smartquant --benchmarks mmlu_pro arc gsm8k
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Meta's official BF16 scores for Llama 4 Maverick Instruct
META_OFFICIAL_SCORES = {
    "mmlu_pro": {"score": 80.5, "metric": "accuracy", "shots": 0, "note": "macro_avg/acc"},
    "arc_challenge": {"score": None, "metric": "accuracy", "shots": 0, "note": "not reported by Meta; community ~96%"},
    "gsm8k": {"score": None, "metric": "accuracy", "shots": 0, "note": "not reported by Meta"},
    "humaneval": {"score": 86.4, "metric": "pass@1", "shots": 0, "note": "3rd party report"},
}


def get_memory_info():
    import mlx.core as mx
    try:
        return {"allocated_gb": mx.get_active_memory() / 1e9, "peak_gb": mx.get_peak_memory() / 1e9}
    except Exception:
        return {"allocated_gb": 0.0, "peak_gb": 0.0}


def load_model(model_path: str):
    from mlx_lm import load
    print(f"\nLoading model from {model_path} ...")
    start = time.time()
    model, tokenizer = load(model_path)
    elapsed = time.time() - start
    mem = get_memory_info()
    print(f"Model loaded in {elapsed:.1f}s, memory: {mem['allocated_gb']:.1f} GB")
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_tokens: int = 1024) -> str:
    from mlx_lm import generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler
    sampler = make_sampler(temp=0.0)
    response = mlx_generate(
        model, tokenizer, prompt=prompt, max_tokens=max_tokens, sampler=sampler,
    )
    return response


def format_chat_prompt(tokenizer, system: str, user: str) -> str:
    """Format a prompt using the model's chat template."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Fallback if chat template fails
        if system:
            return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


def strip_thinking(response: str) -> str:
    """Strip <think>...</think> blocks from model response."""
    return re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()


def extract_mcq_answer(response: str, options: List[str]) -> Optional[str]:
    """Extract a multiple-choice answer from a generated response.

    Handles patterns like:
    - "The answer is (A)"
    - "The answer is A"
    - "A)"
    - "**A**"
    - Just "A" at the start
    """
    response = strip_thinking(response)
    response = response.strip()
    option_letters = [chr(65 + i) for i in range(len(options))]  # A, B, C, D, ...

    # Pattern 1: "The answer is (X)" or "the best answer is X"
    match = re.search(r'(?:the\s+)?(?:best\s+)?answer\s+is\s*[:\s]*\(?([A-Z])\)?', response, re.IGNORECASE)
    if match and match.group(1).upper() in option_letters:
        return match.group(1).upper()

    # Pattern 2: "Answer: X" or "Answer: (X)"
    match = re.search(r'answer\s*:\s*\(?([A-Z])\)?', response, re.IGNORECASE)
    if match and match.group(1).upper() in option_letters:
        return match.group(1).upper()

    # Pattern 3: Starts with a letter option like "A." or "A)" or "A:"
    match = re.match(r'^([A-Z])\s*[.):]\s', response)
    if match and match.group(1).upper() in option_letters:
        return match.group(1).upper()

    # Pattern 4: **X** (bold letter)
    match = re.search(r'\*\*([A-Z])\*\*', response)
    if match and match.group(1).upper() in option_letters:
        return match.group(1).upper()

    # Pattern 5: Last resort — find the last standalone letter mentioned
    matches = re.findall(r'\b([A-Z])\b', response)
    for m in reversed(matches):
        if m in option_letters:
            return m

    return None


def extract_number(text: str) -> Optional[float]:
    """Extract the final numerical answer from a math response."""
    # Look for "#### NUMBER" pattern (GSM8K standard)
    match = re.search(r'####\s*([-\d,\.]+)', text)
    if match:
        return parse_number(match.group(1))

    # Look for "the answer is NUMBER"
    match = re.search(r'(?:the\s+)?answer\s+is\s*[:\s]*([-\d,\.]+)', text, re.IGNORECASE)
    if match:
        return parse_number(match.group(1))

    # Look for "= NUMBER" at end of lines
    match = re.search(r'=\s*([-\d,\.]+)\s*$', text, re.MULTILINE)
    if match:
        return parse_number(match.group(1))

    # Last resort: find the last number in the text
    matches = re.findall(r'([-\d,]+\.?\d*)', text)
    if matches:
        return parse_number(matches[-1])

    return None


def parse_number(s: str) -> Optional[float]:
    """Parse a number string, handling commas."""
    try:
        return float(s.replace(",", ""))
    except (ValueError, TypeError):
        return None


# =============================================================================
# MMLU-Pro Benchmark
# =============================================================================

def run_mmlu_pro(model, tokenizer, max_questions: int = 1000) -> Dict[str, Any]:
    """Run MMLU-Pro benchmark (sampled subset)."""
    from datasets import load_dataset

    print("\n" + "=" * 60)
    print("BENCHMARK: MMLU-Pro (0-shot)")
    print("=" * 60)

    print("Loading MMLU-Pro dataset ...")
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

    # Sample if needed
    total = len(ds)
    if max_questions < total:
        # Stratified sampling by category
        categories = {}
        for i, item in enumerate(ds):
            cat = item.get("category", "unknown")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(i)

        # Sample proportionally from each category
        sampled_indices = []
        for cat, indices in categories.items():
            n_sample = max(1, int(len(indices) * max_questions / total))
            import random
            random.seed(42)
            sampled_indices.extend(random.sample(indices, min(n_sample, len(indices))))
        sampled_indices = sorted(sampled_indices[:max_questions])
        print(f"Sampled {len(sampled_indices)} of {total} questions across {len(categories)} categories")
    else:
        sampled_indices = list(range(total))
        print(f"Running all {total} questions")

    correct = 0
    total_run = 0
    category_results = {}
    start_time = time.time()

    for idx_num, idx in enumerate(sampled_indices):
        item = ds[idx]
        question = item["question"]
        options = item["options"]
        answer_idx = item["answer_index"]
        answer_letter = chr(65 + answer_idx)
        category = item.get("category", "unknown")

        # Format options
        options_text = "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(options))

        user_prompt = (
            f"The following is a multiple choice question. Think step by step and then "
            f"output the answer in the format of \"The answer is (X)\" at the end.\n\n"
            f"Question: {question}\n\n{options_text}"
        )

        prompt = format_chat_prompt(tokenizer, "", user_prompt)
        response = generate(model, tokenizer, prompt, max_tokens=2048)
        predicted = extract_mcq_answer(response, options)

        is_correct = predicted == answer_letter
        if is_correct:
            correct += 1
        total_run += 1

        if category not in category_results:
            category_results[category] = {"correct": 0, "total": 0}
        category_results[category]["total"] += 1
        if is_correct:
            category_results[category]["correct"] += 1

        if (idx_num + 1) % 50 == 0 or idx_num == 0:
            acc = correct / total_run * 100
            elapsed = time.time() - start_time
            rate = total_run / elapsed
            eta = (len(sampled_indices) - total_run) / rate if rate > 0 else 0
            msg = (f"  [{idx_num+1}/{len(sampled_indices)}] Running acc: {acc:.1f}% "
                   f"({correct}/{total_run}) | {rate:.1f} q/s | ETA: {eta/60:.0f}m")
            print(msg)
        # Write progress every 10 questions to a separate file (atomic write)
        if (idx_num + 1) % 10 == 0 or idx_num == 0:
            acc = correct / total_run * 100
            elapsed = time.time() - start_time
            rate = total_run / elapsed if elapsed > 0 else 0
            with open(str(Path.home() / "smartquant" / "results" / "progress.txt"), "w") as pf:
                pf.write(f"mmlu_pro: {idx_num+1}/{len(sampled_indices)} acc={acc:.1f}% rate={rate:.2f}q/s\n")

    elapsed = time.time() - start_time
    accuracy = correct / total_run * 100 if total_run > 0 else 0

    # Per-category results
    cat_accs = {}
    for cat, res in sorted(category_results.items()):
        cat_acc = res["correct"] / res["total"] * 100 if res["total"] > 0 else 0
        cat_accs[cat] = {"accuracy": cat_acc, "correct": res["correct"], "total": res["total"]}

    print(f"\nMMLU-Pro Results: {accuracy:.1f}% ({correct}/{total_run})")
    print(f"Time: {elapsed:.0f}s ({total_run/elapsed:.1f} q/s)")
    for cat, res in sorted(cat_accs.items(), key=lambda x: -x[1]["accuracy"]):
        print(f"  {cat}: {res['accuracy']:.1f}% ({res['correct']}/{res['total']})")

    return {
        "benchmark": "mmlu_pro",
        "accuracy": accuracy,
        "correct": correct,
        "total": total_run,
        "elapsed_seconds": elapsed,
        "questions_per_second": total_run / elapsed,
        "category_results": cat_accs,
    }


# =============================================================================
# ARC-Challenge Benchmark
# =============================================================================

def run_arc_challenge(model, tokenizer) -> Dict[str, Any]:
    """Run ARC-Challenge benchmark (all 1172 questions)."""
    from datasets import load_dataset

    print("\n" + "=" * 60)
    print("BENCHMARK: ARC-Challenge (0-shot)")
    print("=" * 60)

    print("Loading ARC-Challenge dataset ...")
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    total = len(ds)
    print(f"Loaded {total} questions")

    correct = 0
    total_run = 0
    start_time = time.time()

    for idx, item in enumerate(ds):
        question = item["question"]
        choices = item["choices"]
        answer_key = item["answerKey"]

        labels = choices["label"]
        texts = choices["text"]
        options = list(zip(labels, texts))

        options_text = "\n".join(f"{label}. {text}" for label, text in options)

        user_prompt = (
            f"The following is a multiple choice question. Think step by step and then "
            f"output the answer in the format of \"The answer is (X)\" at the end.\n\n"
            f"Question: {question}\n\n{options_text}"
        )

        prompt = format_chat_prompt(tokenizer, "", user_prompt)
        response = generate(model, tokenizer, prompt, max_tokens=2048)
        predicted = extract_mcq_answer(response, texts)

        # ARC uses labels like "A", "B", "C", "D" or "1", "2", "3", "4"
        if predicted is not None:
            # Map predicted letter back to label
            pred_idx = ord(predicted) - 65
            if 0 <= pred_idx < len(labels):
                predicted_label = labels[pred_idx]
            else:
                predicted_label = predicted
        else:
            predicted_label = None

        is_correct = predicted_label == answer_key
        if is_correct:
            correct += 1
        total_run += 1

        if (idx + 1) % 50 == 0 or idx == 0:
            acc = correct / total_run * 100
            elapsed = time.time() - start_time
            rate = total_run / elapsed
            eta = (total - total_run) / rate if rate > 0 else 0
            print(f"  [{idx+1}/{total}] Running acc: {acc:.1f}% "
                  f"({correct}/{total_run}) | {rate:.1f} q/s | ETA: {eta/60:.0f}m")
        if (idx + 1) % 10 == 0 or idx == 0:
            acc = correct / total_run * 100
            elapsed = time.time() - start_time
            rate = total_run / elapsed if elapsed > 0 else 0
            with open(str(Path.home() / "smartquant" / "results" / "progress.txt"), "w") as pf:
                pf.write(f"arc: {idx+1}/{total} acc={acc:.1f}% rate={rate:.2f}q/s\n")

    elapsed = time.time() - start_time
    accuracy = correct / total_run * 100 if total_run > 0 else 0

    print(f"\nARC-Challenge Results: {accuracy:.1f}% ({correct}/{total_run})")
    print(f"Time: {elapsed:.0f}s ({total_run/elapsed:.1f} q/s)")

    return {
        "benchmark": "arc_challenge",
        "accuracy": accuracy,
        "correct": correct,
        "total": total_run,
        "elapsed_seconds": elapsed,
        "questions_per_second": total_run / elapsed,
    }


# =============================================================================
# GSM8K Benchmark
# =============================================================================

def run_gsm8k(model, tokenizer) -> Dict[str, Any]:
    """Run GSM8K benchmark (0-shot chain-of-thought)."""
    from datasets import load_dataset

    print("\n" + "=" * 60)
    print("BENCHMARK: GSM8K (0-shot CoT)")
    print("=" * 60)

    print("Loading GSM8K dataset ...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    total = len(ds)
    print(f"Loaded {total} questions")

    correct = 0
    total_run = 0
    start_time = time.time()

    for idx, item in enumerate(ds):
        question = item["question"]
        answer_text = item["answer"]

        # Extract ground truth number from "#### NUMBER" in answer
        gt_match = re.search(r'####\s*([-\d,\.]+)', answer_text)
        if not gt_match:
            continue
        ground_truth = parse_number(gt_match.group(1))
        if ground_truth is None:
            continue

        user_prompt = (
            f"Solve the following math problem step by step. "
            f"At the end, provide the final numerical answer after \"####\".\n\n"
            f"Problem: {question}"
        )

        prompt = format_chat_prompt(tokenizer, "", user_prompt)
        response = generate(model, tokenizer, prompt, max_tokens=2048)
        predicted = extract_number(strip_thinking(response))

        is_correct = predicted is not None and abs(predicted - ground_truth) < 0.01
        if is_correct:
            correct += 1
        total_run += 1

        if (idx + 1) % 100 == 0 or idx == 0:
            acc = correct / total_run * 100
            elapsed = time.time() - start_time
            rate = total_run / elapsed
            eta = (total - total_run) / rate if rate > 0 else 0
            print(f"  [{idx+1}/{total}] Running acc: {acc:.1f}% "
                  f"({correct}/{total_run}) | {rate:.1f} q/s | ETA: {eta/60:.0f}m")

    elapsed = time.time() - start_time
    accuracy = correct / total_run * 100 if total_run > 0 else 0

    print(f"\nGSM8K Results: {accuracy:.1f}% ({correct}/{total_run})")
    print(f"Time: {elapsed:.0f}s ({total_run/elapsed:.1f} q/s)")

    return {
        "benchmark": "gsm8k",
        "accuracy": accuracy,
        "correct": correct,
        "total": total_run,
        "elapsed_seconds": elapsed,
        "questions_per_second": total_run / elapsed,
    }


# =============================================================================
# HumanEval Benchmark
# =============================================================================

def run_humaneval(model, tokenizer) -> Dict[str, Any]:
    """Run HumanEval benchmark (pass@1 with greedy decoding)."""
    from datasets import load_dataset

    print("\n" + "=" * 60)
    print("BENCHMARK: HumanEval (pass@1)")
    print("=" * 60)

    print("Loading HumanEval dataset ...")
    ds = load_dataset("openai/openai_humaneval", split="test")
    total = len(ds)
    print(f"Loaded {total} problems")

    passed = 0
    total_run = 0
    start_time = time.time()

    for idx, item in enumerate(ds):
        task_id = item["task_id"]
        prompt_code = item["prompt"]
        test_code = item["test"]
        entry_point = item["entry_point"]

        user_prompt = (
            f"Complete the following Python function. Return ONLY the completed function code, "
            f"no explanations or markdown formatting.\n\n{prompt_code}"
        )

        prompt = format_chat_prompt(tokenizer, "You are a Python coding assistant. Output only valid Python code.", user_prompt)
        response = generate(model, tokenizer, prompt, max_tokens=2048)

        # Extract code from response (strip thinking first)
        code = extract_code(strip_thinking(response), prompt_code)

        # Run tests in sandbox
        test_passed = run_code_test(code, test_code, entry_point, timeout=10)
        if test_passed:
            passed += 1
        total_run += 1

        if (idx + 1) % 20 == 0 or idx == 0:
            acc = passed / total_run * 100
            elapsed = time.time() - start_time
            rate = total_run / elapsed
            eta = (total - total_run) / rate if rate > 0 else 0
            print(f"  [{idx+1}/{total}] pass@1: {acc:.1f}% "
                  f"({passed}/{total_run}) | {rate:.1f} p/s | ETA: {eta/60:.0f}m")

    elapsed = time.time() - start_time
    pass_at_1 = passed / total_run * 100 if total_run > 0 else 0

    print(f"\nHumanEval Results: pass@1 = {pass_at_1:.1f}% ({passed}/{total_run})")
    print(f"Time: {elapsed:.0f}s ({total_run/elapsed:.1f} p/s)")

    return {
        "benchmark": "humaneval",
        "accuracy": pass_at_1,
        "correct": passed,
        "total": total_run,
        "elapsed_seconds": elapsed,
        "questions_per_second": total_run / elapsed,
    }


def extract_code(response: str, original_prompt: str) -> str:
    """Extract Python code from a model response."""
    # Try to find code in markdown code block
    match = re.search(r'```python\s*\n(.*?)```', response, re.DOTALL)
    if match:
        code = match.group(1)
        # If the code doesn't include the function signature, prepend the prompt
        if "def " not in code.split("\n")[0]:
            return original_prompt + code
        return code

    match = re.search(r'```\s*\n(.*?)```', response, re.DOTALL)
    if match:
        code = match.group(1)
        if "def " not in code.split("\n")[0]:
            return original_prompt + code
        return code

    # No code block — try to use the response directly
    # If it starts with the function body (indented), prepend the prompt
    lines = response.strip().split("\n")
    if lines and (lines[0].startswith("    ") or lines[0].startswith("\t")):
        return original_prompt + response

    # If it contains a def statement, use from there
    for i, line in enumerate(lines):
        if line.strip().startswith("def "):
            return "\n".join(lines[i:])

    # Last resort: prepend prompt and hope for the best
    return original_prompt + response


def run_code_test(code: str, test_code: str, entry_point: str, timeout: int = 10) -> bool:
    """Run code + tests in a subprocess sandbox."""
    full_code = code + "\n\n" + test_code + f"\n\ncheck({entry_point})\n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        f.flush()
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            timeout=timeout,
            text=True,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        return False
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# =============================================================================
# Main
# =============================================================================

def run_all_benchmarks(
    model_path: str,
    benchmarks: List[str],
    mmlu_max_questions: int = 1000,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run all requested benchmarks on a model."""

    model, tokenizer = load_model(model_path)

    results = {
        "model_path": model_path,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "benchmarks": {},
        "memory": get_memory_info(),
    }

    total_start = time.time()

    def save_partial():
        """Save partial results after each benchmark."""
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, default=str)

    if "mmlu_pro" in benchmarks:
        try:
            results["benchmarks"]["mmlu_pro"] = run_mmlu_pro(model, tokenizer, mmlu_max_questions)
            save_partial()
        except Exception as e:
            print(f"\nERROR in MMLU-Pro: {e}")
            results["benchmarks"]["mmlu_pro"] = {"error": str(e)}

    if "arc" in benchmarks or "arc_challenge" in benchmarks:
        try:
            results["benchmarks"]["arc_challenge"] = run_arc_challenge(model, tokenizer)
            save_partial()
        except Exception as e:
            print(f"\nERROR in ARC-Challenge: {e}")
            results["benchmarks"]["arc_challenge"] = {"error": str(e)}

    if "gsm8k" in benchmarks:
        try:
            results["benchmarks"]["gsm8k"] = run_gsm8k(model, tokenizer)
            save_partial()
        except Exception as e:
            print(f"\nERROR in GSM8K: {e}")
            results["benchmarks"]["gsm8k"] = {"error": str(e)}

    if "humaneval" in benchmarks:
        try:
            results["benchmarks"]["humaneval"] = run_humaneval(model, tokenizer)
            save_partial()
        except Exception as e:
            print(f"\nERROR in HumanEval: {e}")
            results["benchmarks"]["humaneval"] = {"error": str(e)}

    results["total_time_seconds"] = time.time() - total_start
    results["memory_peak"] = get_memory_info()

    # Print summary
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Total time: {results['total_time_seconds']/60:.1f} min")
    print()
    print(f"{'Benchmark':<20} {'Score':>8} {'Meta BF16':>10} {'Delta':>8}")
    print("-" * 50)
    for name, res in results["benchmarks"].items():
        score = res["accuracy"]
        meta = META_OFFICIAL_SCORES.get(name, {})
        meta_score = meta.get("score")
        if meta_score is not None:
            delta = score - meta_score
            print(f"{name:<20} {score:>7.1f}% {meta_score:>9.1f}% {delta:>+7.1f}%")
        else:
            print(f"{name:<20} {score:>7.1f}% {'N/A':>9} {'':>8}")

    # Save
    if output_path is None:
        output_path = str(Path.home() / "smartquant" / "results" / "official_benchmarks.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Official Benchmark Evaluation for SmartQuant")
    parser.add_argument("--model", required=True, help="Path to quantized MLX model")
    parser.add_argument(
        "--benchmarks", nargs="+",
        default=["mmlu_pro", "arc", "gsm8k", "humaneval"],
        help="Benchmarks to run (default: all)",
    )
    parser.add_argument(
        "--mmlu-max-questions", type=int, default=1000,
        help="Max MMLU-Pro questions to sample (default: 1000)",
    )
    parser.add_argument("--output", help="Output path for results JSON")
    parser.add_argument("--log", help="Path to write log output (line-buffered)")
    args = parser.parse_args()

    # Redirect stdout to a log file with line buffering for monitoring
    if args.log:
        log_fh = open(args.log, "w")
        log_fh.reconfigure(line_buffering=True)
        sys.stdout = log_fh
        sys.stderr = log_fh

    run_all_benchmarks(
        model_path=args.model,
        benchmarks=args.benchmarks,
        mmlu_max_questions=args.mmlu_max_questions,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
