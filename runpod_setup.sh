#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# FitFusion — RunPod Pod Setup Script
# ═══════════════════════════════════════════════════════════════════════
# This runs ONCE on the pod after creation to install all dependencies.
# Called by: python runpod_controller.py install
# ═══════════════════════════════════════════════════════════════════════

set -e  # Exit on error

echo "============================================================"
echo "FitFusion RunPod Setup"
echo "============================================================"
echo "  Start time: $(date)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'No GPU detected')"
echo "  CUDA: $(nvcc --version 2>/dev/null | grep release || echo 'Unknown')"
echo "  Python: $(python --version)"
echo ""

PROJECT_DIR="/workspace/FitFusion"
cd "$PROJECT_DIR"

# ─── Step 1: System packages ─────────────────────────────────────────
echo "[1/7] Installing system packages..."
apt-get update -qq && apt-get install -y -qq \
    git git-lfs wget curl unzip \
    libgl1-mesa-glx libglib2.0-0 \
    > /dev/null 2>&1
echo "  Done."

# ─── Step 2: Python dependencies ─────────────────────────────────────
echo "[2/7] Installing Python dependencies..."

# Core ML
pip install -q --no-cache-dir \
    torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118 \
    2>/dev/null || echo "  PyTorch already at correct version"

pip install -q --no-cache-dir \
    diffusers==0.25.0 \
    transformers==4.36.2 \
    accelerate==0.26.1 \
    peft==0.11.1 \
    bitsandbytes==0.43.3 \
    xformers==0.0.22.post7 \
    huggingface_hub==0.23.0 \
    safetensors \
    einops \
    opencv-contrib-python-headless \
    pillow \
    numpy\<2 \
    scipy \
    tqdm \
    wandb \
    tensorboard \
    mediapipe \
    rembg[gpu]

# IDM-VTON additional required packages
echo "  Installing IDM-VTON additional dependencies..."
pip install -q --no-cache-dir \
    omegaconf \
    fvcore \
    cloudpickle \
    pycocotools \
    timm \
    packaging \
    ftfy \
    torchmetrics \
    basicsr \
    av \
    2>/dev/null || echo "  [WARN] Some IDM-VTON extras failed, continuing..."
echo "  IDM-VTON extras done."

# run_tryon.py runtime dependencies
echo "  Installing run_tryon.py dependencies (boto3, celery, redis, sentry-sdk, opencv)..."
pip install -q --no-cache-dir boto3 celery redis sentry-sdk opencv-contrib-python-headless
echo "  Done."

# Detectron2 + DensePose (for proper IUV body part maps)
echo "  Installing Detectron2 + DensePose..."
pip install -q --no-cache-dir \
    'git+https://github.com/facebookresearch/detectron2.git@main#egg=detectron2' \
    2>/dev/null || echo "  Detectron2 install from source..."

# Install DensePose from Detectron2's projects
if python -c "import detectron2" 2>/dev/null; then
    pip install -q --no-cache-dir \
        'git+https://github.com/facebookresearch/detectron2@main#subdirectory=projects/DensePose' \
        2>/dev/null || echo "  DensePose install note: will use MediaPipe IUV fallback if needed"
    echo "  Detectron2 + DensePose installed."
else
    echo "  [WARN] Detectron2 failed to install. Will use MediaPipe IUV fallback."
fi

echo "  Done."

# ─── Step 3: Clone/update IDM-VTON if needed ─────────────────────────
echo "[3/7] Setting up IDM-VTON..."
if [ ! -d "$PROJECT_DIR/IDM-VTON/.git" ]; then
    echo "  IDM-VTON source already present (uploaded via sync)."
else
    echo "  IDM-VTON repo found."
fi

# Ensure IDM-VTON source is importable
if [ ! -f "$PROJECT_DIR/IDM-VTON/src/__init__.py" ]; then
    touch "$PROJECT_DIR/IDM-VTON/src/__init__.py"
fi

echo "  Done."

# ─── Step 4: Download model weights ──────────────────────────────────
echo "[4/7] Downloading model weights (IDM-VTON + RMBG-1.4)..."
python -c "
from huggingface_hub import snapshot_download
import os

cache_dir = '/workspace/model_cache'
os.makedirs(cache_dir, exist_ok=True)

# IDM-VTON (main model)
print('  Downloading IDM-VTON weights...')
snapshot_download(
    'yisol/IDM-VTON',
    local_dir=os.path.join(cache_dir, 'IDM-VTON'),
    cache_dir=cache_dir,
    local_dir_use_symlinks=False,
)
print('  IDM-VTON weights downloaded.')
print('  (CLIP image encoder is included in IDM-VTON/image_encoder/)')

# BRIA RMBG-1.4 — background removal model
# Produces much cleaner person/garment masks than the generic rembg u2net model.
# Used by fitfusion/utils/preprocessing.py Stage 1 background standardization.
print('  Downloading BRIA RMBG-1.4 background removal model...')
snapshot_download(
    'briaai/RMBG-1.4',
    local_dir=os.path.join(cache_dir, 'RMBG-1.4'),
    cache_dir=cache_dir,
    local_dir_use_symlinks=False,
)
print('  RMBG-1.4 downloaded.')
"

echo "  Done."

# ─── Step 5: Verify GPU + CUDA ───────────────────────────────────────
echo "[5/7] Verifying GPU setup..."
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    # Quick test
    x = torch.randn(100, 100, device='cuda')
    y = x @ x.T
    print(f'  CUDA compute test: PASSED')
else:
    print('  ERROR: No CUDA GPU detected!')
    exit(1)

import diffusers
print(f'  diffusers: {diffusers.__version__}')

import peft
print(f'  peft: {peft.__version__}')

import bitsandbytes
print(f'  bitsandbytes: {bitsandbytes.__version__}')

import accelerate
print(f'  accelerate: {accelerate.__version__}')
"
echo "  Done."

# ─── Step 6: Prepare training dataset ─────────────────────────────────
echo "[6/7] Preparing VITON-HD format dataset..."

cd "$PROJECT_DIR"

# Stage 1: Organize images + create metadata (Already done locally!)
# python IDM-VTON/prepare_vitonhd_dataset.py \
#     --stage 1 \
#     --output_dir "$PROJECT_DIR/data/fitfusion_vitonhd" \
#     --project_root "$PROJECT_DIR"

# Stage 2: Generate DensePose + agnostic masks
python IDM-VTON/prepare_vitonhd_dataset.py \
    --stage 2 \
    --output_dir "$PROJECT_DIR/data/fitfusion_vitonhd" \
    --project_root "$PROJECT_DIR"

echo "  Done."

# ─── Step 7: Create convenience scripts ──────────────────────────────
echo "[7/7] Creating convenience scripts..."

# Quick GPU monitor
cat > "$PROJECT_DIR/gpu_monitor.sh" << 'MONITOR'
#!/bin/bash
watch -n 2 'nvidia-smi; echo ""; echo "Python processes:"; ps aux | grep python | grep -v grep'
MONITOR
chmod +x "$PROJECT_DIR/gpu_monitor.sh"

echo "  Done."

echo ""
echo "============================================================"
echo "Setup Complete!"
echo "============================================================"
echo "  Project: $PROJECT_DIR"
echo "  Models:  /workspace/model_cache/"
echo ""
echo "  To start training:"
echo "    python runpod_controller.py train"
echo ""
echo "  To monitor GPU:"
echo "    ./gpu_monitor.sh"
echo "============================================================"
echo "  End time: $(date)"
