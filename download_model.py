#!/usr/bin/env python3
"""
Download Llama 4 Maverick BF16 weights from HuggingFace.

Supports resumable downloads, parallel file downloads, and integrity verification.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download, HfApi
from safetensors import safe_open

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path.home() / "smartquant" / "logs" / "download.log"),
    ],
)
logger = logging.getLogger(__name__)

DEFAULT_REPO_ID = "meta-llama/Llama-4-Maverick-17B-128E-Instruct"
DEFAULT_LOCAL_DIR = str(Path.home() / "smartquant" / "models" / "maverick-bf16")

ALLOW_PATTERNS = [
    "*.safetensors",
    "*.json",
    "tokenizer*",
    "special_tokens_map*",
    "generation_config*",
]

IGNORE_PATTERNS = [
    "optimizer*",
    "training_args*",
    "*.bin",
    "*.pt",
    "*.ckpt",
]


def download_model(repo_id: str, local_dir: str) -> Path:
    """Download the model with resume support."""
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {repo_id} to {local_dir}")
    logger.info("This will take several hours for the ~800 GB BF16 model.")
    logger.info("Downloads are resumable — you can safely interrupt and restart.")

    start = time.time()

    result_path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_path),
        allow_patterns=ALLOW_PATTERNS,
        ignore_patterns=IGNORE_PATTERNS,
        resume_download=True,
        max_workers=4,
    )

    elapsed = time.time() - start
    logger.info(f"Download completed in {elapsed / 3600:.1f} hours")

    return Path(result_path)


def verify_download(model_dir: Path) -> bool:
    """Verify the downloaded model is complete and readable."""
    logger.info("Verifying download ...")

    # Check for index file
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        logger.error("model.safetensors.index.json not found")
        return False

    with open(index_path) as f:
        index = json.load(f)

    weight_map = index.get("weight_map", {})
    expected_shards = set(weight_map.values())
    logger.info(f"Index lists {len(weight_map)} tensors across {len(expected_shards)} shards")

    # Check all shards exist
    missing = []
    for shard_name in expected_shards:
        shard_path = model_dir / shard_name
        if not shard_path.exists():
            missing.append(shard_name)

    if missing:
        logger.error(f"Missing {len(missing)} shard(s): {missing[:5]}...")
        return False

    # Calculate total size
    total_bytes = sum(
        (model_dir / s).stat().st_size for s in expected_shards
    )
    total_gb = total_bytes / (1024 ** 3)
    logger.info(f"Total safetensor size: {total_gb:.1f} GB")

    # Spot-check: load one tensor from each shard
    logger.info("Spot-checking shards (loading one tensor per shard) ...")
    errors = []
    for shard_name in sorted(expected_shards):
        shard_path = model_dir / shard_name
        try:
            with safe_open(str(shard_path), framework="pt") as f:
                keys = list(f.keys())
                if keys:
                    tensor = f.get_tensor(keys[0])
                    _ = tensor.shape
        except Exception as e:
            errors.append((shard_name, str(e)))
            logger.error(f"Failed to read {shard_name}: {e}")

    if errors:
        logger.error(f"{len(errors)} shard(s) failed verification")
        return False

    logger.info("All shards verified successfully")
    logger.info(f"  Shards: {len(expected_shards)}")
    logger.info(f"  Tensors: {len(weight_map)}")
    logger.info(f"  Total size: {total_gb:.1f} GB")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download Llama 4 Maverick BF16 from HuggingFace"
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"HuggingFace repo ID (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--local-dir",
        default=DEFAULT_LOCAL_DIR,
        help=f"Local download directory (default: {DEFAULT_LOCAL_DIR})",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify an existing download, don't download",
    )
    args = parser.parse_args()

    if args.verify_only:
        ok = verify_download(Path(args.local_dir))
        sys.exit(0 if ok else 1)

    download_model(args.repo_id, args.local_dir)
    ok = verify_download(Path(args.local_dir))
    if ok:
        logger.info("Download and verification complete. Ready for analysis.")
    else:
        logger.error("Verification failed. Re-run to resume download.")
        sys.exit(1)


if __name__ == "__main__":
    main()
