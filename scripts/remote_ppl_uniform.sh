#!/bin/bash
# Download and measure perplexity of uniform 4-bit Qwen3.5-397B on remote
export PATH=~/smartquant-env312/bin:$PATH

MODEL_DIR="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.5-397B-A17B-4bit"

if [ ! -d "$MODEL_DIR" ] || [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "Downloading Qwen3.5-397B-A17B-4bit..."
    python3 -c "
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id='mlx-community/Qwen3.5-397B-A17B-4bit',
    local_dir='$MODEL_DIR'
)
print(f'Downloaded to: {path}')
"
else
    echo "Model already exists at $MODEL_DIR"
fi

echo "Running perplexity measurement..."
python3 -m mlx_lm.perplexity \
    --model "$MODEL_DIR" \
    --sequence-length 2048 \
    --num-samples 256 \
    --seed 42

echo "Done at $(date)"
