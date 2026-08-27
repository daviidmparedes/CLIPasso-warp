#!/usr/bin/env bash
# Stage 1 of environment setup: venv + PyTorch (cu128, Blackwell sm_120) + python deps.
# diffvg is built separately by bench/build_diffvg.sh once this completes.
set -euo pipefail
ROOT=/home/dmiranda/CLIPasso
VENV=$ROOT/.venv

echo "### [$(date +%T)] creating venv (python $(python3 --version))"
python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install -q --upgrade pip wheel setuptools

echo "### [$(date +%T)] installing torch 2.9.1+cu128 (sm_120 support)"
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128

echo "### [$(date +%T)] installing build tooling (cmake/ninja) + CUDA compiler wheels"
pip install cmake ninja
# nvcc + CCCL (thrust/cub) headers as pip wheels: no root, and newer than Ubuntu's 12.0 apt toolkit
pip install nvidia-cuda-nvcc-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-cccl-cu12 nvidia-cuda-nvrtc-cu12

echo "### [$(date +%T)] installing CLIPasso python deps (unpinned where the 2021 pins are dead)"
pip install numpy scipy scikit-image scikit-learn pandas matplotlib seaborn \
            opencv-python-headless "Pillow<11" imageio imageio-ffmpeg moviepy \
            svgwrite svgpathtools cssutils ftfy regex tqdm gdown visdom torch-tools \
            pyyaml psutil rich tabulate

echo "### [$(date +%T)] DONE stage 1"
python -c "
import torch, torchvision
print('torch', torch.__version__, '| torchvision', torchvision.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0))
    print('capability:', torch.cuda.get_device_capability(0))
    print('torch arch list:', torch.cuda.get_arch_list())
"
