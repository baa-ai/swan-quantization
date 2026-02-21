#!/usr/bin/env python3
"""
API-based Benchmark Evaluation for SmartQuant Models.
Runs benchmarks against an MLX-LM API server endpoint.

Usage:
    python api_benchmarks_qwen3.py --api http://192.168.89.174:8080
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

# Qwen3.5-397B official/community scores (bfloat16 baseline)
OFFICIAL_SCORES = {
    "mmlu_pro": {"score": None, "metric": "accuracy", "note": "not yet reported"},
    "arc_challenge": {"score": None, "metric": "accuracy", "note": "not yet reported"},
    "gsm8k": {"score": None, "metric": "accuracy", "note": "not yet reported"},
    "humaneval": {"score": None, "metric": "pass@1", "note": "not yet reported"},
}


def api_chat(api_url: str, messages: List[Dict], max_tokens: int = 512, temperature: float = 0.0) -> Dict:
    """Call the chat completions API with retry and backoff for machine crashes."""
    url = f"{api_url}/v1/chat/completions"
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    data = json.dumps(payload).encode("utf-8")

    max_retries = 20
    backoff_delays = [10, 20, 30, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60]

    for attempt in range(max_retries):
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=1800) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            if attempt < max_retries - 1:
                delay = backoff_delays[attempt]
                print(f"  API error (attempt {attempt+1}/{max_retries}): {e} — retrying in {delay}s", flush=True)
                time.sleep(delay)
            else:
                print(f"  API error (final attempt): {e}", flush=True)
                return {"error": str(e)}


def api_generate(api_url: str, user_prompt: str, system_prompt: str = "", max_tokens: int = 512) -> Optional[str]:
    """Generate a response via the API. Returns None if API completely failed (skip question)."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    result = api_chat(api_url, messages, max_tokens=max_tokens, temperature=0.0)
    if "error" in result:
        return None  # Signal to skip this question
    try:
        return result["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return ""


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from response."""
    return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()


def extract_mcq_answer(response: str, option_letters: List[str]) -> Optional[str]:
    """Extract MCQ answer from response."""
    response = response.strip()

    # Direct letter answer
    if len(response) <= 3:
        letter = response.strip(".):").upper()
        if letter in option_letters:
            return letter

    # "The answer is (X)" pattern
    match = re.search(r'answer\s+is\s*[:\s]*\(?([A-Z])\)?', response, re.IGNORECASE)
    if match and match.group(1).upper() in option_letters:
        return match.group(1).upper()

    # "Answer: X"
    match = re.search(r'answer\s*:\s*\(?([A-Z])\)?', response, re.IGNORECASE)
    if match and match.group(1).upper() in option_letters:
        return match.group(1).upper()

    # Bold **X**
    match = re.search(r'\*\*([A-Z])\*\*', response)
    if match and match.group(1).upper() in option_letters:
        return match.group(1).upper()

    # Last standalone letter
    matches = re.findall(r'\b([A-Z])\b', response)
    for m in reversed(matches):
        if m in option_letters:
            return m

    return None


def extract_number(text: str) -> Optional[float]:
    """Extract numerical answer from math response."""
    # #### NUMBER pattern
    match = re.search(r'####\s*([-\d,\.]+)', text)
    if match:
        return parse_number(match.group(1))

    # "the answer is NUMBER"
    match = re.search(r'answer\s+is\s*[:\s]*([-\d,\.]+)', text, re.IGNORECASE)
    if match:
        return parse_number(match.group(1))

    # "= NUMBER" at end
    match = re.search(r'=\s*([-\d,\.]+)\s*$', text, re.MULTILINE)
    if match:
        return parse_number(match.group(1))

    # Last number
    matches = re.findall(r'([-\d,]+\.?\d*)', text)
    if matches:
        return parse_number(matches[-1])

    return None


def parse_number(s: str) -> Optional[float]:
    try:
        return float(s.replace(",", ""))
    except (ValueError, TypeError):
        return None


# =============================================================================
# MMLU-Pro
# =============================================================================

def run_mmlu_pro(api_url: str, max_questions: int = 300, thinking: bool = False) -> Dict[str, Any]:
    from datasets import load_dataset

    mode_label = "0-shot, thinking" if thinking else "0-shot, concise"
    print("\n" + "=" * 60, flush=True)
    print(f"BENCHMARK: MMLU-Pro ({mode_label})", flush=True)
    print("=" * 60, flush=True)

    print("Loading MMLU-Pro dataset ...", flush=True)
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

    total = len(ds)
    if max_questions < total:
        categories = {}
        for i, item in enumerate(ds):
            cat = item.get("category", "unknown")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(i)

        import random
        random.seed(42)
        sampled_indices = []
        for cat, indices in categories.items():
            n_sample = max(1, int(len(indices) * max_questions / total))
            sampled_indices.extend(random.sample(indices, min(n_sample, len(indices))))
        sampled_indices = sorted(sampled_indices[:max_questions])
        print(f"Sampled {len(sampled_indices)} of {total} questions", flush=True)
    else:
        sampled_indices = list(range(total))

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

        option_letters = [chr(65 + i) for i in range(len(options))]
        options_text = "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(options))

        if thinking:
            prompt = (
                f"Answer the following multiple choice question. "
                f"Think through it carefully, then give your final answer as a single letter.\n\n"
                f"Question: {question}\n\n{options_text}"
            )
            tok_limit = 2048
        else:
            prompt = (
                f"Answer the following multiple choice question. "
                f"Output ONLY the letter of the correct answer.\n\n"
                f"Question: {question}\n\n{options_text}"
            )
            tok_limit = 10

        response = api_generate(api_url, prompt, max_tokens=tok_limit)
        if response is None:
            continue  # Skip questions where API completely failed

        if thinking:
            response = strip_thinking(response)

        predicted = extract_mcq_answer(response, option_letters)

        is_correct = predicted == answer_letter
        if is_correct:
            correct += 1
        total_run += 1

        if category not in category_results:
            category_results[category] = {"correct": 0, "total": 0}
        category_results[category]["total"] += 1
        if is_correct:
            category_results[category]["correct"] += 1

        if (idx_num + 1) % 10 == 0 or idx_num == 0:
            acc = correct / total_run * 100
            elapsed = time.time() - start_time
            rate = total_run / elapsed
            eta = (len(sampled_indices) - idx_num - 1) / rate if rate > 0 else 0
            print(f"  [{idx_num+1}/{len(sampled_indices)}] acc: {acc:.1f}% "
                  f"({correct}/{total_run}) | {rate:.2f} q/s | ETA: {eta/60:.0f}m", flush=True)

    elapsed = time.time() - start_time
    accuracy = correct / total_run * 100 if total_run > 0 else 0

    cat_accs = {}
    for cat, res in sorted(category_results.items()):
        cat_acc = res["correct"] / res["total"] * 100 if res["total"] > 0 else 0
        cat_accs[cat] = {"accuracy": cat_acc, "correct": res["correct"], "total": res["total"]}

    print(f"\nMMLU-Pro Results: {accuracy:.1f}% ({correct}/{total_run})", flush=True)
    print(f"Time: {elapsed:.0f}s ({total_run/elapsed:.2f} q/s)", flush=True)

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
# ARC-Challenge
# =============================================================================

def run_arc_challenge(api_url: str, max_questions: int = 300) -> Dict[str, Any]:
    from datasets import load_dataset

    print("\n" + "=" * 60, flush=True)
    print("BENCHMARK: ARC-Challenge (0-shot, concise)", flush=True)
    print("=" * 60, flush=True)

    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    total = len(ds)

    if max_questions < total:
        import random
        random.seed(42)
        indices = random.sample(range(total), max_questions)
        indices.sort()
    else:
        indices = list(range(total))

    print(f"Running {len(indices)} of {total} questions", flush=True)

    correct = 0
    total_run = 0
    start_time = time.time()

    for idx_num, idx in enumerate(indices):
        item = ds[idx]
        question = item["question"]
        choices = item["choices"]
        answer_key = item["answerKey"]

        labels = choices["label"]
        texts = choices["text"]

        options_text = "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))
        option_letters = [chr(65 + i) for i in range(len(labels))]

        prompt = (
            f"Answer the following multiple choice question. "
            f"Output ONLY the letter of the correct answer.\n\n"
            f"Question: {question}\n\n{options_text}"
        )

        response = api_generate(api_url, prompt, max_tokens=10)
        if response is None:
            continue  # Skip questions where API completely failed

        predicted = extract_mcq_answer(response, option_letters)

        # Map predicted letter back to label
        if predicted is not None:
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

        if (idx_num + 1) % 10 == 0 or idx_num == 0:
            acc = correct / total_run * 100
            elapsed = time.time() - start_time
            rate = total_run / elapsed
            eta = (len(indices) - total_run) / rate if rate > 0 else 0
            print(f"  [{idx_num+1}/{len(indices)}] acc: {acc:.1f}% "
                  f"({correct}/{total_run}) | {rate:.2f} q/s | ETA: {eta/60:.0f}m", flush=True)

    elapsed = time.time() - start_time
    accuracy = correct / total_run * 100 if total_run > 0 else 0

    print(f"\nARC-Challenge Results: {accuracy:.1f}% ({correct}/{total_run})", flush=True)
    print(f"Time: {elapsed:.0f}s ({total_run/elapsed:.2f} q/s)", flush=True)

    return {
        "benchmark": "arc_challenge",
        "accuracy": accuracy,
        "correct": correct,
        "total": total_run,
        "elapsed_seconds": elapsed,
        "questions_per_second": total_run / elapsed,
    }


# =============================================================================
# GSM8K
# =============================================================================

def run_gsm8k(api_url: str, max_questions: int = 300) -> Dict[str, Any]:
    from datasets import load_dataset

    print("\n" + "=" * 60, flush=True)
    print("BENCHMARK: GSM8K (0-shot CoT)", flush=True)
    print("=" * 60, flush=True)

    ds = load_dataset("openai/gsm8k", "main", split="test")
    total = len(ds)

    if max_questions < total:
        import random
        random.seed(42)
        indices = random.sample(range(total), max_questions)
        indices.sort()
    else:
        indices = list(range(total))

    print(f"Running {len(indices)} of {total} questions", flush=True)

    correct = 0
    total_run = 0
    skipped = 0
    start_time = time.time()

    for idx_num, idx in enumerate(indices):
        item = ds[idx]
        question = item["question"]
        answer_text = item["answer"]

        gt_match = re.search(r'####\s*([-\d,\.]+)', answer_text)
        if not gt_match:
            continue
        ground_truth = parse_number(gt_match.group(1))
        if ground_truth is None:
            continue

        prompt = (
            f"Solve the following math problem step by step. "
            f"At the end, provide the final numerical answer after \"####\".\n\n"
            f"Problem: {question}"
        )

        response = api_generate(api_url, prompt, max_tokens=512)
        if response is None:
            skipped += 1
            continue  # Skip questions where API completely failed

        predicted = extract_number(response)

        is_correct = predicted is not None and abs(predicted - ground_truth) < 0.01
        if is_correct:
            correct += 1
        total_run += 1

        if (idx_num + 1) % 10 == 0 or idx_num == 0:
            acc = correct / total_run * 100
            elapsed = time.time() - start_time
            rate = total_run / elapsed
            eta = (len(indices) - idx_num - 1) / rate if rate > 0 else 0
            skip_str = f" (skipped: {skipped})" if skipped else ""
            print(f"  [{idx_num+1}/{len(indices)}] acc: {acc:.1f}% "
                  f"({correct}/{total_run}) | {rate:.2f} q/s | ETA: {eta/60:.0f}m{skip_str}", flush=True)

    elapsed = time.time() - start_time
    accuracy = correct / total_run * 100 if total_run > 0 else 0

    print(f"\nGSM8K Results: {accuracy:.1f}% ({correct}/{total_run})", flush=True)
    print(f"Time: {elapsed:.0f}s ({total_run/elapsed:.2f} q/s)", flush=True)

    return {
        "benchmark": "gsm8k",
        "accuracy": accuracy,
        "correct": correct,
        "total": total_run,
        "elapsed_seconds": elapsed,
        "questions_per_second": total_run / elapsed,
    }


# =============================================================================
# HumanEval
# =============================================================================

def run_humaneval(api_url: str) -> Dict[str, Any]:
    from datasets import load_dataset

    print("\n" + "=" * 60, flush=True)
    print("BENCHMARK: HumanEval (pass@1)", flush=True)
    print("=" * 60, flush=True)

    ds = load_dataset("openai/openai_humaneval", split="test")
    total = len(ds)
    print(f"Running all {total} problems", flush=True)

    passed = 0
    total_run = 0
    skipped = 0
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

        response = api_generate(api_url, user_prompt,
                                system_prompt="You are a Python coding assistant. Output only valid Python code.",
                                max_tokens=512)

        if response is None:
            skipped += 1
            continue  # Skip questions where API completely failed

        code = extract_code(response, prompt_code)
        test_passed = run_code_test(code, test_code, entry_point, timeout=10)
        if test_passed:
            passed += 1
        total_run += 1

        if (idx + 1) % 10 == 0 or idx == 0:
            acc = passed / total_run * 100
            elapsed = time.time() - start_time
            rate = total_run / elapsed
            eta = (total - idx - 1) / rate if rate > 0 else 0
            skip_str = f" (skipped: {skipped})" if skipped else ""
            print(f"  [{idx+1}/{total}] pass@1: {acc:.1f}% "
                  f"({passed}/{total_run}) | {rate:.2f} p/s | ETA: {eta/60:.0f}m{skip_str}", flush=True)

    elapsed = time.time() - start_time
    pass_at_1 = passed / total_run * 100 if total_run > 0 else 0

    print(f"\nHumanEval Results: pass@1 = {pass_at_1:.1f}% ({passed}/{total_run})", flush=True)
    print(f"Time: {elapsed:.0f}s ({total_run/elapsed:.2f} p/s)", flush=True)

    return {
        "benchmark": "humaneval",
        "accuracy": pass_at_1,
        "correct": passed,
        "total": total_run,
        "elapsed_seconds": elapsed,
        "questions_per_second": total_run / elapsed,
    }


def extract_code(response: str, original_prompt: str) -> str:
    match = re.search(r'```python\s*\n(.*?)```', response, re.DOTALL)
    if match:
        code = match.group(1)
        if "def " not in code.split("\n")[0]:
            return original_prompt + code
        return code

    match = re.search(r'```\s*\n(.*?)```', response, re.DOTALL)
    if match:
        code = match.group(1)
        if "def " not in code.split("\n")[0]:
            return original_prompt + code
        return code

    lines = response.strip().split("\n")
    if lines and (lines[0].startswith("    ") or lines[0].startswith("\t")):
        return original_prompt + response

    for i, line in enumerate(lines):
        if line.strip().startswith("def "):
            return "\n".join(lines[i:])

    return original_prompt + response


def run_code_test(code: str, test_code: str, entry_point: str, timeout: int = 10) -> bool:
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

def main():
    parser = argparse.ArgumentParser(description="API-based Benchmark Evaluation")
    parser.add_argument("--api", default="http://192.168.89.174:8080", help="API base URL")
    parser.add_argument("--benchmarks", nargs="+",
                        default=["mmlu_pro", "arc", "gsm8k", "humaneval"],
                        help="Benchmarks to run")
    parser.add_argument("--mmlu-max", type=int, default=300, help="Max MMLU-Pro questions")
    parser.add_argument("--arc-max", type=int, default=300, help="Max ARC questions")
    parser.add_argument("--gsm8k-max", type=int, default=300, help="Max GSM8K questions")
    parser.add_argument("--thinking", action="store_true", help="Enable thinking mode (longer generation)")
    parser.add_argument("--output", default=str(Path.home() / "smartquant" / "results" / "official_benchmarks_qwen3.json"))
    args = parser.parse_args()

    print(f"API endpoint: {args.api}", flush=True)
    if args.thinking:
        print("Thinking mode: ENABLED", flush=True)
    print(f"Benchmarks: {args.benchmarks}", flush=True)

    results = {
        "model": "Qwen3.5-397B-A17B (SmartQuant mixed-precision)",
        "api_endpoint": args.api,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "benchmarks": {},
    }

    total_start = time.time()

    def save_partial():
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)

    if "mmlu_pro" in args.benchmarks:
        try:
            results["benchmarks"]["mmlu_pro"] = run_mmlu_pro(args.api, args.mmlu_max, thinking=args.thinking)
            save_partial()
        except Exception as e:
            print(f"\nERROR in MMLU-Pro: {e}", flush=True)
            results["benchmarks"]["mmlu_pro"] = {"error": str(e)}

    if "arc" in args.benchmarks or "arc_challenge" in args.benchmarks:
        try:
            results["benchmarks"]["arc_challenge"] = run_arc_challenge(args.api, args.arc_max)
            save_partial()
        except Exception as e:
            print(f"\nERROR in ARC-Challenge: {e}", flush=True)
            results["benchmarks"]["arc_challenge"] = {"error": str(e)}

    if "gsm8k" in args.benchmarks:
        try:
            results["benchmarks"]["gsm8k"] = run_gsm8k(args.api, args.gsm8k_max)
            save_partial()
        except Exception as e:
            print(f"\nERROR in GSM8K: {e}", flush=True)
            results["benchmarks"]["gsm8k"] = {"error": str(e)}

    if "humaneval" in args.benchmarks:
        try:
            results["benchmarks"]["humaneval"] = run_humaneval(args.api)
            save_partial()
        except Exception as e:
            print(f"\nERROR in HumanEval: {e}", flush=True)
            results["benchmarks"]["humaneval"] = {"error": str(e)}

    results["total_time_seconds"] = time.time() - total_start

    # Summary
    print(f"\n{'='*60}", flush=True)
    print("BENCHMARK SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Model: {results['model']}", flush=True)
    print(f"Total time: {results['total_time_seconds']/60:.1f} min", flush=True)
    print(f"\n{'Benchmark':<20} {'Score':>8} {'Questions':>10}", flush=True)
    print("-" * 42, flush=True)
    for name, res in results["benchmarks"].items():
        if "error" in res:
            print(f"{name:<20} {'ERROR':>8} {'':>10}", flush=True)
        else:
            print(f"{name:<20} {res['accuracy']:>7.1f}% {res['total']:>9}", flush=True)

    save_partial()
    print(f"\nResults saved to: {args.output}", flush=True)


if __name__ == "__main__":
    main()
