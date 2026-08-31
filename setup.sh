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

# diffvg compiles CUDA sources, so it needs nvcc -- not just the driver that
# nvidia-smi reports. PyTorch ships its own CUDA runtime and does not provide one.
if command -v nvcc >/dev/null; then
  CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
elif [ -n "${CUDA_HOME:-}" ] && [ -x "$CUDA_HOME/bin/nvcc" ]; then
  :
else
  for c in /usr/local/cuda /usr/local/cuda-*; do
    [ -x "$c/bin/nvcc" ] && { CUDA_HOME="$c"; break; }
  done
fi
if [ -z "${CUDA_HOME:-}" ] || [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
  echo
  echo "No CUDA toolkit found: nvcc is not on PATH and CUDA_HOME is not set."
  echo "diffvg compiles CUDA kernels, so a driver alone is not enough."
  echo
  echo "    sudo apt install nvidia-cuda-toolkit     # Debian / Ubuntu"
  echo "    or install from https://developer.nvidia.com/cuda-downloads"
  echo
  echo "Already have one somewhere? Point at it:"
  echo "    CUDA_HOME=/path/to/cuda ./setup.sh"
  exit 1
fi
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
echo "== CUDA toolkit: $CUDA_HOME ($("$CUDA_HOME/bin/nvcc" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p' | tail -1))"

# diffvg is a C++ extension and needs the Python development headers. Check now
# rather than after a 2.5 GB PyTorch download: without them the build dies much
# later with an unhelpful "Could NOT find Python (missing: Python_INCLUDE_DIRS)".
PYVER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PYINC_SYS="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
LOCAL_HEADERS=""
if [ ! -f "$PYINC_SYS/Python.h" ]; then
  if [ "${ALLOW_LOCAL_PYTHON_HEADERS:-0}" = "1" ] && command -v apt-get >/dev/null; then
    echo "== Python headers missing; fetching python$PYVER-dev locally (no root needed)"
    LOCAL_HEADERS="$ROOT/third_party/pyhdr"
    mkdir -p "$LOCAL_HEADERS/debs"
    ( cd "$LOCAL_HEADERS/debs" && apt-get download "libpython$PYVER-dev" "python$PYVER-dev" >/dev/null 2>&1 || true )
    ls "$LOCAL_HEADERS"/debs/*.deb >/dev/null 2>&1 || {
      echo "could not download python$PYVER-dev; install it with your package manager:"
      echo "    sudo apt install python$PYVER-dev"; exit 1; }
    for d in "$LOCAL_HEADERS"/debs/*.deb; do dpkg -x "$d" "$LOCAL_HEADERS"; done
    PYINC_SYS="$LOCAL_HEADERS/usr/include/python$PYVER"
    # Debian keeps pyconfig.h in a multiarch directory that Python.h includes by a
    # bare name, so it has to be reachable from the include root.
    for MA in "$LOCAL_HEADERS"/usr/include/*-linux-gnu; do
      [ -d "$MA" ] && ln -sfn "$MA" "$PYINC_SYS/$(basename "$MA")"
    done
    [ -f "$PYINC_SYS/Python.h" ] || { echo "extraction produced no Python.h"; exit 1; }
    echo "   headers: $PYINC_SYS"
  else
    echo
    echo "Python development headers are missing: no Python.h under $PYINC_SYS"
    echo "diffvg is a C++ extension and cannot be compiled without them."
    echo
    echo "    sudo apt install python$PYVER-dev      # Debian / Ubuntu"
    echo "    sudo dnf install python3-devel        # Fedora / RHEL"
    echo
    echo "No root on this machine? Re-run as:"
    echo "    ALLOW_LOCAL_PYTHON_HEADERS=1 ./setup.sh"
    exit 1
  fi
fi

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

# diffvg's setup.py assembles its own cmake args and honours no environment
# variable, so teach it one. CMake otherwise searches only default prefixes and
# cannot see a locally extracted header tree or a non-standard CUDA location.
grep -q DIFFVG_EXTRA_CMAKE_ARGS setup.py || sed -i \
  "s|^\( *\)subprocess.check_call(\['cmake', ext.sourcedir\]|\1cmake_args += os.environ.get('DIFFVG_EXTRA_CMAKE_ARGS', '').split()\n\1subprocess.check_call(['cmake', ext.sourcedir]|" \
  setup.py
grep -q DIFFVG_EXTRA_CMAKE_ARGS setup.py || { echo "failed to patch diffvg/setup.py"; exit 1; }

EXTRA="-DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc"
if [ -n "$LOCAL_HEADERS" ]; then
  EXTRA="$EXTRA -DPython_INCLUDE_DIR=$PYINC_SYS -DPython_INCLUDE_DIRS=$PYINC_SYS"
  EXTRA="$EXTRA -DPYTHON_INCLUDE_DIR=$PYINC_SYS -DPYTHON_INCLUDE_DIRS=$PYINC_SYS"
  EXTRA="$EXTRA -DPython_EXECUTABLE=$VENV/bin/python"
fi
export DIFFVG_EXTRA_CMAKE_ARGS="$EXTRA"

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
