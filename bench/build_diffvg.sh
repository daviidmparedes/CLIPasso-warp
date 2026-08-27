#!/usr/bin/env bash
# Build diffvg for Blackwell (sm_120) against the local CUDA 12.8.1 toolkit.
#
# Why each step is needed (all verified against this tree, not guessed):
#   1. pybind11 submodule is 2.5.dev1 -> no Python 3.12 support. Bump to 2.13.6.
#   2. diffvg/CMakeLists.txt uses find_package(CUDA) + cuda_add_library + FindPythonLibs.
#      All three are the DEPRECATED modules that CMake 4.0 REMOVED. pip gave us cmake
#      4.4.2, so we pin cmake 3.31.x, where they still exist.
#   3. -std=c++11 is hardcoded, but CUDA 12.8 ships CCCL/thrust 2.x which requires
#      C++17. diffvg.cpp includes <thrust/sort.h>, so this is a real conflict.
#   4. nvcc defaults to sm_52; Blackwell needs an explicit compute_120 gencode.
set -euo pipefail
ROOT=/home/dmiranda/CLIPasso
DIFFVG=$ROOT/third_party/diffvg
export CUDA_HOME=${CUDA_HOME:-$ROOT/third_party/cuda}
source "$ROOT/.venv/bin/activate"

echo "### [$(date +%T)] CUDA_HOME=$CUDA_HOME"
"$CUDA_HOME/bin/nvcc" --version | tail -2

echo "### [$(date +%T)] pinning cmake 3.31 (4.x removed FindCUDA/FindPythonLibs)"
pip install -q "cmake==3.31.6"
cmake --version | head -1

echo "### [$(date +%T)] bumping pybind11 submodule to v2.13.6 (py3.12 support)"
cd "$DIFFVG/pybind11"
git fetch --depth 1 origin tag v2.13.6 2>/dev/null || git fetch --tags
git checkout -q v2.13.6
grep -m1 "cmake_minimum_required" CMakeLists.txt

echo "### [$(date +%T)] patching diffvg CMakeLists for C++17 + sm_120"
cd "$DIFFVG"
python - <<'PY'
import pathlib
p = pathlib.Path("CMakeLists.txt")
s = p.read_text()
subs = [
    # CCCL/thrust 2.x in CUDA 12.8 requires C++17; diffvg hardcodes C++11 in 3 places.
    ("set(CMAKE_CUDA_STANDARD 11)", "set(CMAKE_CUDA_STANDARD 17)"),
    ('set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")',
     'set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++17 '
     '-gencode arch=compute_120,code=sm_120 '
     '-gencode arch=compute_120,code=compute_120 '
     '--expt-relaxed-constexpr")'),
    ("set_property(TARGET diffvg PROPERTY CXX_STANDARD 11)",
     "set_property(TARGET diffvg PROPERTY CXX_STANDARD 17)"),
    # -Wall on nvcc-compiled TUs floods the log; keep it for host only.
    ("add_compile_options(-Wall -g -O3 -fvisibility=hidden -Wno-unknown-pragmas)",
     "add_compile_options(-g -O3 -fvisibility=hidden -Wno-unknown-pragmas)"),
]
for old, new in subs:
    if new in s:
        print(f"  already: {old[:50]}")
    elif old in s:
        s = s.replace(old, new); print(f"  patched: {old[:50]}")
    else:
        raise SystemExit(f"  FAILED to find: {old!r}")
p.write_text(s)
PY

echo "### [$(date +%T)] locating Python development headers"
# Ubuntu 24.04 ships no python3.12-dev and we have no sudo, so the headers were
# pulled with `apt-get download` + `dpkg -x` into third_party/pyhdr.
PYHDR=$ROOT/third_party/pyhdr
PYINC=$PYHDR/usr/include/python3.12
PYLIB=$PYHDR/usr/lib/x86_64-linux-gnu/libpython3.12.so
test -f "$PYINC/Python.h" || { echo "FATAL: no Python.h at $PYINC"; exit 1; }
echo "  include: $PYINC"
echo "  library: $PYLIB"
# Debian pyconfig.h is a stub that does #include <x86_64-linux-gnu/python3.12/pyconfig.h>,
# so usr/include must be on the search path too, not just usr/include/python3.12.
test -f "$PYHDR/usr/include/x86_64-linux-gnu/python3.12/pyconfig.h" || { echo "FATAL: multiarch pyconfig.h missing"; exit 1; }
# Debian pyconfig.h is a stub doing #include <x86_64-linux-gnu/python3.12/pyconfig.h>.
# Rather than plumb a second -I through CMake (find_package(PythonLibs) clobbers
# PYTHON_INCLUDE_DIRS anyway), make the stub resolve relative to $PYINC itself.
ln -sfn "$PYHDR/usr/include/x86_64-linux-gnu" "$PYINC/x86_64-linux-gnu"

# diffvg calls BOTH find_package(Python COMPONENTS Development) and the older
# find_package(PythonLibs), so both variable spellings have to be satisfied.
export DIFFVG_EXTRA_CMAKE_ARGS="-DPython_INCLUDE_DIR=$PYINC -DPython_LIBRARY=$PYLIB -DPython_EXECUTABLE=$ROOT/.venv/bin/python -DPYTHON_INCLUDE_DIR=$PYINC -DPYTHON_INCLUDE_DIRS=$PYINC -DPYTHON_INCLUDE_PATH=$PYINC -DPYTHON_LIBRARY=$PYLIB -DPYTHON_LIBRARIES=$PYLIB -DCMAKE_CUDA_ARCHITECTURES=120"

echo "### [$(date +%T)] building (this takes a few minutes)"
rm -rf build
export PATH="$CUDA_HOME/bin:$PATH"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
DIFFVG_CUDA=1 python setup.py install 2>&1 | tail -40

echo "### [$(date +%T)] verifying import"
cd "$ROOT"
python - <<'PY'
import torch, pydiffvg, diffvg
print("pydiffvg imported from:", pydiffvg.__file__)
pydiffvg.set_use_gpu(True)
pydiffvg.set_device(torch.device("cuda"))
# minimal render: one 4-point cubic on a 224x224 canvas, exactly as Painter does
pts = torch.tensor([[50.,50.],[80.,90.],[120.,60.],[170.,160.]], device="cuda")
path = pydiffvg.Path(num_control_points=torch.zeros(1, dtype=torch.int32)+2,
                     points=pts, stroke_width=torch.tensor(1.5), is_closed=False)
grp = pydiffvg.ShapeGroup(shape_ids=torch.tensor([0]), fill_color=None,
                          stroke_color=torch.tensor([0.,0.,0.,1.]))
args = pydiffvg.RenderFunction.serialize_scene(224, 224, [path], [grp])
img = pydiffvg.RenderFunction.apply(224, 224, 2, 2, 0, None, *args)
print("render OK:", img.shape, img.dtype, img.device, "| alpha sum:", img[:,:,3].sum().item())
# and confirm gradients flow back to the control points
pts.requires_grad_(True)
path.points = pts
args = pydiffvg.RenderFunction.serialize_scene(224, 224, [path], [grp])
img = pydiffvg.RenderFunction.apply(224, 224, 2, 2, 0, None, *args)
img.sum().backward()
print("backward OK: grad norm =", pts.grad.norm().item())
PY
echo "### [$(date +%T)] DIFFVG BUILD COMPLETE"
