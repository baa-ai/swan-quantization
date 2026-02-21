#!/bin/bash
set -e

VENV_DIR="$HOME/smartquant-env"
PROJECT_DIR="$HOME/smartquant"

echo "=== SmartQuant Environment Setup ==="

# 1. Create a dedicated Python virtual environment
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
else
    echo "Creating virtual environment at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# 2. Install core dependencies
echo "Installing Python dependencies ..."
pip install --upgrade pip

pip install mlx mlx-lm
pip install safetensors
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy tqdm pyyaml
pip install huggingface_hub
pip install transformers

# 3. Login to HuggingFace (needed for gated Llama 4 model)
echo ""
echo "You must accept Meta's license at:"
echo "  https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct"
echo ""
if huggingface-cli whoami &>/dev/null; then
    echo "Already logged in to HuggingFace."
else
    echo "Please log in to HuggingFace:"
    huggingface-cli login
fi

# 4. Verify MLX is working
echo ""
echo "Verifying MLX installation ..."
python3 -c "
import mlx.core as mx
print(f'MLX device: {mx.default_device()}')
print('Metal available: True')
print(f'MLX version: {mx.__version__}')
"

# 5. Create working directories
mkdir -p "$PROJECT_DIR/models/maverick-bf16"
mkdir -p "$PROJECT_DIR/models/maverick-smartquant"
mkdir -p "$PROJECT_DIR/analysis"
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/results"

echo ""
echo "=== Environment ready ==="
echo "Activate with: source $VENV_DIR/bin/activate"
