#!/usr/bin/env python3
"""
Convert Llama 4 Maverick to MLX format with SmartQuant mixed-precision quantization.

Wraps mlx_lm.convert to use the custom SmartQuant quant predicate generated
from the analysis manifest.

Usage:
    python convert_model.py \
        --hf-path ~/smartquant/models/maverick-bf16 \
        --mlx-path ~/smartquant/models/maverick-smartquant \
        --manifest ~/smartquant/analysis/manifest.json
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path.home() / "smartquant" / "logs" / "conversion.log"),
    ],
)
logger = logging.getLogger(__name__)


def convert_with_predicate(
    hf_path: str,
    mlx_path: str,
    manifest_path: str,
    dtype: str = "float16",
    default_bits: int = 4,
    default_group_size: int = 128,
) -> None:
    """Convert a HuggingFace model to MLX with SmartQuant mixed quantization.

    Attempts the primary mlx_lm.convert API path first. Falls back to manual
    quantization using MLX primitives if the API doesn't support custom predicates.
    """
    from smartquant.bridge_mlx import create_quant_predicate, load_manifest

    manifest = load_manifest(Path(manifest_path))
    predicate = create_quant_predicate(
        manifest, default_bits=default_bits, default_group_size=default_group_size
    )

    logger.info(f"Converting model: {hf_path}")
    logger.info(f"Output: {mlx_path}")
    logger.info(f"Using SmartQuant predicate from: {manifest_path}")

    # Ensure output path does not exist (mlx_lm.convert refuses if it does)
    output_dir = Path(mlx_path)
    if output_dir.exists():
        logger.info(f"Removing existing output path: {mlx_path}")
        import shutil
        shutil.rmtree(str(output_dir))

    start = time.time()

    _convert_via_api(hf_path, mlx_path, predicate, dtype, default_bits, default_group_size)

    elapsed = time.time() - start
    logger.info(f"Conversion completed in {elapsed / 60:.1f} minutes")

    _verify_output(mlx_path)


def _convert_via_api(
    hf_path: str,
    mlx_path: str,
    predicate,
    dtype: str,
    default_bits: int,
    default_group_size: int,
) -> None:
    """Primary path: use mlx_lm.convert with custom quant_predicate."""
    from mlx_lm import convert

    logger.info("Using mlx_lm.convert with custom quant_predicate ...")

    convert(
        hf_path=hf_path,
        mlx_path=mlx_path,
        quantize=True,
        q_bits=default_bits,
        q_group_size=default_group_size,
        quant_predicate=predicate,
        dtype=dtype,
    )


def _convert_manual(
    hf_path: str,
    mlx_path: str,
    manifest: dict,
    dtype: str,
    default_bits: int,
    default_group_size: int,
) -> None:
    """Fallback: manual quantization using MLX primitives.

    Loads the model at full precision, then iterates over each module,
    quantizing according to the SmartQuant manifest.
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load
    from mlx_lm.utils import save_model

    from smartquant.bridge_mlx import build_bits_lookup, tensor_name_to_module_name

    logger.info("Loading model at full precision ...")
    model, tokenizer = load(hf_path)

    bits_lookup = build_bits_lookup(manifest)
    module_bits = {}
    for tensor_name, bits in bits_lookup.items():
        module_name = tensor_name_to_module_name(tensor_name)
        module_bits[module_name] = bits

    # Quantize each linear layer
    quantized_count = 0
    skipped_count = 0

    def quantize_module(parent_name: str, module: nn.Module) -> nn.Module:
        nonlocal quantized_count, skipped_count

        if isinstance(module, nn.Linear):
            full_name = parent_name
            bits = module_bits.get(full_name, default_bits)

            # Skip if 16-bit or if module is a norm/embedding
            name_lower = full_name.lower()
            should_skip = (
                bits == 16
                or any(kw in name_lower for kw in ["layernorm", "layer_norm", "rmsnorm"])
                or "embed_tokens" in name_lower
                or "lm_head" in name_lower
                or name_lower.endswith("router")
                or name_lower.endswith("gate")
            )

            if should_skip:
                skipped_count += 1
                return module

            group_size = 64 if bits == 8 else default_group_size
            try:
                q_module = nn.QuantizedLinear.from_linear(
                    module, group_size=group_size, bits=bits
                )
                quantized_count += 1
                return q_module
            except Exception as e:
                logger.warning(f"Failed to quantize {full_name}: {e}")
                skipped_count += 1
                return module

        return module

    def walk_and_quantize(name: str, module: nn.Module):
        """Recursively walk the module tree and quantize linear layers."""
        children = {}
        if hasattr(module, "children"):
            child_items = []
            if hasattr(module, "_modules"):
                child_items = module._modules.items()
            elif hasattr(module, "__dict__"):
                for k, v in module.__dict__.items():
                    if isinstance(v, nn.Module):
                        child_items.append((k, v))
                    elif isinstance(v, list):
                        for i, item in enumerate(v):
                            if isinstance(item, nn.Module):
                                child_items.append((f"{k}.{i}", item))

            for child_name, child_module in child_items:
                full_name = f"{name}.{child_name}" if name else child_name
                walk_and_quantize(full_name, child_module)

        # Quantize this module if it's a linear layer
        if isinstance(module, nn.Linear):
            new_module = quantize_module(name, module)
            if new_module is not module:
                # Replace in parent
                parts = name.rsplit(".", 1)
                if len(parts) == 2:
                    parent_name, attr_name = parts
                    parent = model
                    for p in parent_name.split("."):
                        if p.isdigit():
                            parent = parent[int(p)]
                        else:
                            parent = getattr(parent, p)
                    if attr_name.isdigit():
                        parent[int(attr_name)] = new_module
                    else:
                        setattr(parent, attr_name, new_module)

    logger.info("Quantizing model ...")
    # Use MLX's built-in quantization traversal
    nn.quantize(
        model,
        bits=default_bits,
        group_size=default_group_size,
        class_predicate=lambda p, m: isinstance(m, nn.Linear) and _should_quantize(p, module_bits, default_bits),
    )

    logger.info(f"Saving quantized model to {mlx_path} ...")
    output_path = Path(mlx_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save using mlx_lm's save utility
    save_model(output_path, model=model, tokenizer=tokenizer)

    logger.info(f"Manual quantization complete.")


def _should_quantize(name: str, module_bits: dict, default_bits: int) -> bool:
    """Check if a module should be quantized based on SmartQuant manifest."""
    bits = module_bits.get(name, default_bits)
    name_lower = name.lower()

    if bits == 16:
        return False
    if any(kw in name_lower for kw in ["layernorm", "layer_norm", "rmsnorm", "embed_tokens", "lm_head"]):
        return False
    if name_lower.endswith("router") or name_lower.endswith("gate"):
        return False

    return True


def _verify_output(mlx_path: str) -> None:
    """Verify the converted model output."""
    output_dir = Path(mlx_path)

    if not output_dir.exists():
        logger.error(f"Output directory does not exist: {output_dir}")
        return

    # Check for expected files
    safetensor_files = list(output_dir.glob("*.safetensors"))
    config_file = output_dir / "config.json"

    logger.info(f"Output verification:")
    logger.info(f"  Directory: {output_dir}")
    logger.info(f"  Safetensor files: {len(safetensor_files)}")
    logger.info(f"  config.json present: {config_file.exists()}")

    # Total size
    total_bytes = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    total_gb = total_bytes / (1024 ** 3)
    logger.info(f"  Total output size: {total_gb:.1f} GB")

    # Quick load test
    try:
        from mlx_lm import load
        logger.info("  Attempting quick load test ...")
        model, tokenizer = load(str(output_dir))
        logger.info("  Load test: PASSED")
        del model, tokenizer
    except Exception as e:
        logger.warning(f"  Load test: FAILED — {e}")
        logger.warning("  The model may still work; try loading manually.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert model to MLX with SmartQuant mixed-precision quantization"
    )
    parser.add_argument(
        "--hf-path",
        required=True,
        help="Path to the HuggingFace model (BF16 safetensors)",
    )
    parser.add_argument(
        "--mlx-path",
        required=True,
        help="Output path for the MLX quantized model",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to SmartQuant analysis manifest.json",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16"],
        help="Data type for non-quantized parameters (default: float16)",
    )
    parser.add_argument(
        "--default-bits",
        type=int,
        default=4,
        choices=[4, 8],
        help="Default bit-width for tensors not in manifest (default: 4)",
    )
    parser.add_argument(
        "--default-group-size",
        type=int,
        default=128,
        help="Default group size for quantization (default: 128)",
    )
    args = parser.parse_args()

    convert_with_predicate(
        hf_path=args.hf_path,
        mlx_path=args.mlx_path,
        manifest_path=args.manifest,
        dtype=args.dtype,
        default_bits=args.default_bits,
        default_group_size=args.default_group_size,
    )


if __name__ == "__main__":
    main()
