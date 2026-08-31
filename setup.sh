#!/usr/bin/env bash
# Set up CLIPasso: virtualenv, PyTorch, dependencies, and diffvg built for your GPU.
#
# Linux with an NVIDIA GPU and a working driver. Takes about ten minutes, most of
# it compiling diffvg.
#
#   ./setup.sh                  # build for the GPU in this machine
#   CUDA_TAG=cu121 ./setup.sh   # pin a different PyTorch CUDA build
#
# diffvg is the differentiable rasteriser CLIPasso draws with. It has no wheel, so
# it is compiled here against the CUDA architecture of the card you actually have.
# That is the step that fails on a stock checkout: upstream pins C++11, which the
# CUDA 12 headers no longer accept, and lists no architecture newer than Ampere.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv"
CUDA_TAG="${CUDA_TAG:-cu128}"

command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found. This needs an NVIDIA GPU."; exit 1; }

echo "== creating virtualenv ($(python3 --version))"
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -q --upgrade pip wheel setuptools

echo "== installing PyTorch ($CUDA_TAG)"
pip install -q torch torchvision --index-url "https://download.pytorch.org/whl/$CUDA_TAG"

echo "== installing dependencies"
pip install -q -r "$ROOT/requirements.txt"

# cmake 4 removed FindCUDA and FindPythonLibs, which diffvg's build still uses.
echo "== installing build tools"
pip install -q "cmake==3.31.6" ninja

ARCH="$(python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see the GPU; check your driver.")
major, minor = torch.cuda.get_device_capability(0)
print(f"{major}{minor}")
PY
)"
echo "== GPU compute capability: sm_$ARCH"

DIFFVG="$ROOT/third_party/diffvg"
if [ ! -d "$DIFFVG" ]; then
  echo "== cloning diffvg"
  git clone -q --recursive https://github.com/BachiLi/diffvg "$DIFFVG"
fi

echo "== patching diffvg for C++17 and sm_$ARCH"
cd "$DIFFVG/pybind11"
# the pinned pybind11 predates Python 3.11 and will not compile against it
git fetch -q --depth 1 origin tag v2.13.6 2>/dev/null || git fetch -q --tags
git checkout -q v2.13.6
cd "$DIFFVG"
python - "$ARCH" <<'PY'
import pathlib, sys
arch = sys.argv[1]
p = pathlib.Path("CMakeLists.txt"); s = p.read_text()
subs = [
    # CUDA 12's bundled thrust/cub require C++17
    ("set(CMAKE_CUDA_STANDARD 11)", "set(CMAKE_CUDA_STANDARD 17)"),
    ('set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")',
     'set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++17 '
     f'-gencode arch=compute_{arch},code=sm_{arch} '
     f'-gencode arch=compute_{arch},code=compute_{arch} '
     '--expt-relaxed-constexpr")'),
    ("set_property(TARGET diffvg PROPERTY CXX_STANDARD 11)",
     "set_property(TARGET diffvg PROPERTY CXX_STANDARD 17)"),
]
for old, new in subs:
    if new in s:
        continue
    if old not in s:
        raise SystemExit(f"could not find in CMakeLists.txt: {old!r}")
    s = s.replace(old, new)
p.write_text(s)
print("  CMakeLists.txt patched")
PY

echo "== building diffvg (a few minutes)"
rm -rf build
DIFFVG_CUDA=1 CMAKE_CUDA_ARCHITECTURES="$ARCH" python setup.py install

cd "$ROOT"
echo "== verifying"
python - <<'PY'
import torch, pydiffvg
print("  torch     ", torch.__version__)
print("  gpu       ", torch.cuda.get_device_name(0))
pydiffvg.set_use_gpu(True)
pydiffvg.set_device(torch.device("cuda"))
pts = torch.tensor([[20.,20.],[60.,80.],[100.,20.],[140.,80.]], device="cuda", requires_grad=True)
path = pydiffvg.Path(num_control_points=torch.tensor([2], dtype=torch.int32),
                     points=pts, is_closed=False, stroke_width=torch.tensor(2.0))
grp = pydiffvg.ShapeGroup(shape_ids=torch.tensor([0]), fill_color=None,
                          stroke_color=torch.tensor([0.,0.,0.,1.]))
scene = pydiffvg.RenderFunction.serialize_scene(160, 100, [path], [grp])
img = pydiffvg.RenderFunction.apply(160, 100, 2, 2, 0, None, *scene)
img.sum().backward()
assert pts.grad is not None and torch.isfinite(pts.grad).all()
print("  diffvg     forward and backward OK")
PY
echo
echo "Setup complete. Activate with:  source .venv/bin/activate"
echo "Then try:                       python run_sketch.py --target target_images/camel.png"
