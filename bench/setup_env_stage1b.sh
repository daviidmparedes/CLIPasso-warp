#!/usr/bin/env bash
# Stage 1b: python deps, minus the dead 2021 pins.
#   dropped: visdom (needs pkg_resources, unused by repo), torch-tools (unused), numba (unused)
#   kept:    wandb (imported at module scope in 4 files even when --use_wandb 0)
set -euo pipefail
source /home/dmiranda/CLIPasso/.venv/bin/activate
echo "### [$(date +%T)] stage 1b deps"
pip install -q setuptools wheel
pip install numpy scipy scikit-image scikit-learn pandas matplotlib seaborn \
            opencv-python-headless "Pillow<11" imageio imageio-ffmpeg \
            svgwrite svgpathtools cssutils ftfy regex tqdm gdown wandb \
            pyyaml psutil rich tabulate
echo "### [$(date +%T)] DONE stage 1b"
python - <<'PY'
import torch, torchvision, PIL, numpy, scipy, skimage
print("torch       ", torch.__version__)
print("torchvision ", torchvision.__version__)
print("numpy       ", numpy.__version__, "| scipy", scipy.__version__, "| skimage", skimage.__version__, "| Pillow", PIL.__version__)
print("cuda avail  ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device      ", torch.cuda.get_device_name(0))
    print("capability  ", torch.cuda.get_device_capability(0))
    print("arch list   ", torch.cuda.get_arch_list())
    print("cuda(build) ", torch.version.cuda)
PY
