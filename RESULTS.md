# CLIPasso Speed-Up — Results Log

Single source of truth for what has been tried. Every change gets a
**(speedup, quality delta)** pair. Negative results stay in the table.

Reproduce anything here with the scripts in [`bench/`](bench/). All numbers below
come from the harnesses, not from estimates.

---

## 0. Environment

Built from scratch — the shipped stack (Python 3.7 / torch 1.7.1+cu101 / CUDA 11.0,
and the author's `yaelvinker/clipasso_docker` image) **cannot run on this machine at all**:
the GPU is Blackwell (sm_120) and neither CUDA 11.0 nor any torch below 2.7 emits code for it.

| Component | Version | Note |
|---|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q | 97 GB, **sm_120**, driver 595.84 (CUDA 13.2) |
| CPU / RAM | 64 cores / 188 GB | |
| Python | 3.12.3 | system; `python3.12-dev` **not** installed (see below) |
| PyTorch | 2.9.1+cu128 | arch list includes `sm_120` |
| torchvision | 0.24.1+cu128 | |
| CUDA toolkit | 12.8.93 | extracted from NVIDIA runfile, **no root** |
| diffvg | BachiLi/diffvg @ master, patched | built for `compute_120`, C++17 |
| pybind11 | 2.13.6 | bumped from vendored 2.5.dev1 |
| CMake | 3.31.6 | pinned **down** from 4.4.2 |

### Environment obstacles and how each was resolved

| Obstacle | Resolution |
|---|---|
| No `nvcc` anywhere; no passwordless sudo; Ubuntu's apt CUDA is 12.0 (too old for sm_120) | PyPI's `nvidia-cuda-nvcc-cu12` ships **only `ptxas`**, and `cuda-toolkit[nvcc]` resolves to that same trimmed wheel — both dead ends. Downloaded the CUDA 12.8.1 runfile and `--extract`ed it (needs no root), then symlinked the per-component dirs into one toolkit root. |
| CMake 4.4.2 **removed** `FindCUDA`, `cuda_add_library` and `FindPythonLibs`, all of which diffvg uses | Pinned cmake back to 3.31.6 rather than rewriting diffvg's build. |
| Vendored pybind11 2.5.dev1 has no Python 3.12 support | Bumped the submodule to v2.13.6. |
| diffvg hardcodes `-std=c++11`, but CUDA 12.8 ships CCCL/thrust 2.x which requires C++17 (`diffvg.cpp` includes `<thrust/sort.h>`) | Patched `CMAKE_CUDA_STANDARD`, `CUDA_NVCC_FLAGS` and `CXX_STANDARD` to 17. |
| nvcc defaults to sm_52 | Added `-gencode arch=compute_120,code=sm_120` (+ PTX). |
| **`python3.12-dev` not installed** — no `Python.h`, and no sudo to install it | `apt-get download libpython3.12-dev libpython3.12t64` (works unprivileged) + `dpkg -x` into `third_party/pyhdr`. Version 3.12.3 matches the interpreter exactly. |
| Debian's `pyconfig.h` is a stub that `#include`s `<x86_64-linux-gnu/python3.12/pyconfig.h>` from a *different* directory, and `find_package(PythonLibs)` clobbers `PYTHON_INCLUDE_DIRS` so a second `-I` won't survive | Symlinked the multiarch dir *inside* the include dir so the stub resolves relatively. |
| `u2net.pth` missing — and `get_target()` calls `get_mask_u2net` **unconditionally**, so it is required even at `--mask_object 0` | Pre-downloaded via `gdown` (176 MB). |

### Source compat patches (`bench/apply_compat_patches.py`, idempotent)

| File | Change | Why |
|---|---|---|
| `CLIP_/clip/auxilary.py` | `F._pad` → `F.pad` | `F._pad` removed after torch 1.9. **Blocked every import.** |
| `models/painter_params.py` | `scipy.ndimage.filters` → `scipy.ndimage` | namespace removed in SciPy 2.0 |
| `models/painter_params.py` | `np.int` → `int` | removed in NumPy 1.24 (latent: only on the `dino` saliency path) |
| `sketch_utils.py` | `torch.load(..., weights_only=True)` | torch 2.6 flipped the default |

### Two corrections to the project brief, from reading the source

1. **`--aug_scale_min` is dead config.** `models/loss.py` hardcodes
   `RandomResizedCrop(224, scale=(0.8, 0.8), ratio=(1.0,1.0))` + `RandomPerspective(distortion_scale=0.5, p=1.0)`.
   `CLIPConvLoss` never reads `args.aug_scale_min`. Idea **1.4**'s affine bank must reproduce
   *perspective + a fixed 0.8 crop*, not a 0.7–1.0 scale range.
2. **The target branch is already `.detach()`ed** (`loss.py:377`). The waste is a redundant
   5-image CLIP *forward*, not a forward+backward — which caps idea **1.4** far below the
   brief's 1.4–1.8× estimate. Measured ceiling: **9.6% of wall-clock** (§2).

---

## 1. Dataset

The original `image_test/` (1441 JPEGs in folders `10/15/20/25`) was **not usable**: the folder
names carry no recoverable class label, and the images are *scene* photographs (cathedrals,
sunsets, landscapes) rather than the object-centric photos CLIPasso targets. Guardrail metric #2
needs class names.

Replaced with **Sketchy** (`/home/shared_data/sketches/sketchy`): 125 classes × exactly 100
photos, all 256×256, object-centric, with paired human sketches.
`SketchyCoco/Object.tar` on this server is **0 bytes**, so the paper's exact corpus was never available.

| Artifact | Contents |
|---|---|
| `data/sketchy2000/` | 2000 photos, 16/class × 125 classes (deterministic, seed 1234) |
| `data/manifest.json` | every image + class label + zero-shot prompt name |
| `data/eval_set.json` | **fixed 5-image guardrail set** — horse, bicycle, teapot, giraffe, butterfly |
| `data/paper_protocol.json` | 200 images / 10 categories, mirroring the paper's protocol |

Rebuild: `python bench/build_dataset.py`

---

## 2. Profile — where the time actually goes

`python bench/profile_iter.py --num-paths 16 --iters 50`
→ `bench/results/profile/profile_n16.json`

Measured three ways, because no single pass is trustworthy alone: **FUSED** (no intra-iteration
syncs — ground truth wall clock), **SPLIT** (synced phase boundaries — attribution), and
**PROFILE** (`torch.profiler` — GPU kernel time and launch counts). The backward is split
*exactly*, not estimated: the graph is a chain `points → diffvg → img → CLIP → loss`, so
`autograd.grad(loss, img)` is precisely the CLIP backward and `img.backward(g)` precisely diffvg's.

### Per-iteration breakdown, n=16 (128 free parameters)

| Phase | ms | % of wall |
|---|---:|---:|
| diffvg fwd (total) | 3.62 | 10.2% |
| &nbsp;&nbsp;· `serialize_scene` (pure Python) | 0.80 | 2.3% |
| &nbsp;&nbsp;· rasterise + composite | 2.82 | 7.9% |
| CLIP fwd (total) | 10.04 | 28.2% |
| &nbsp;&nbsp;· normalize | 0.21 | 0.6% |
| &nbsp;&nbsp;· augment ×4 | 2.02 | 5.7% |
| &nbsp;&nbsp;· encode **sketch** branch | 3.90 | 11.0% |
| &nbsp;&nbsp;· encode **target** branch *(redundant)* | 3.42 | 9.6% |
| &nbsp;&nbsp;· loss reduction | 0.49 | 1.4% |
| CLIP bwd | 5.56 | 15.7% |
| **diffvg bwd** | **7.77** | **21.9%** |
| optimiser | 0.37 | 1.0% |
| **FUSED TOTAL** | **35.53** | **100%** |
| &nbsp;&nbsp;· GPU busy | 20.76 | 58.4% |
| &nbsp;&nbsp;· **Python / launch idle** | **14.77** | **41.6%** |

### Verdict: neither CLIP nor diffvg "dominates" — launch overhead does

- **CLIP total 15.6 ms (57%) vs diffvg total 11.4 ms (42%), ratio 1.37× → BALANCED.**
  The brief's prior ("at n=16, CLIP + launch overhead dominates") is **half right**: launch
  overhead is indeed huge, but CLIP does *not* dominate diffvg.
- **`diffvg bwd` is the single largest phase** at 21.9% of wall — more than 2× diffvg's forward.
  The brief assumed the vector side was cheap at n=16. It is cheap to *rasterise*; it is not
  cheap to *differentiate*.
- **41.6% of wall-clock is GPU idle**, with **3786 kernels/iter averaging 5.5 µs**. This is a
  launch-bound workload. That is the single biggest lever, and it is what makes batching
  (1.1/1.2) and CUDA graphs (1.3) valuable — not the FLOPs.
- The redundant target-branch encode is **9.6%** — a real but modest ceiling for idea 1.4.

### Stroke count is nearly free

| n strokes | ms/iter | GPU busy | launch idle | CLIP/diffvg |
|---:|---:|---:|---:|---:|
| 4 | 35.38 | 59.5% | 40.5% | 1.44× |
| 8 | 35.33 | 58.7% | 41.3% | 1.44× |
| 16 | 35.53 | 58.4% | 41.6% | 1.37× |
| 32 | 37.73 | 56.2% | 43.8% | 1.17× |

**8× the strokes costs +6.7% wall-clock.** Cost is essentially fixed overhead, independent of
scene complexity. Two consequences: the 4/8/16/32 abstraction ladder is almost free to produce,
and idea 2.3 (progressive stroke addition) cannot pay off through *fewer strokes* — only through
faster convergence.

---

## 3. Baseline (unmodified CLIPasso)

`python bench/run_baseline.py` → `bench/results/baseline/baseline_shipped.json`
Runs `painterly_rendering.py` as a subprocess exactly as `run_object_sketching.py` does, so
timings include real process startup and model loading.

Fixed eval set: 5 Sketchy images × {8,16,32} strokes × 3 seeds (0/1000/2000), `--num_iter 2001`.

> **Status: running.** Preliminary single run (horse, n=16, seed 0):
> **125.6 s/seed**, 62.5 ms/iter, 2010 iterations (no early stop), `loss_eval` 0.56613.

> **Note the gap:** 62.5 ms/iter end-to-end vs **35.5 ms/iter** of actual optimisation (§2).
> ~43% of baseline wall-clock is *not* optimisation — it is process startup plus the
> `save_interval=10` logging (200 SVG writes + 200 matplotlib figures per run). The `nolog`
> variant quantifies this separately.

| n | s/seed | ms/iter | iters | early-stop % | mean loss_eval | ± | s/image (3 seeds) |
|---|---|---|---|---|---|---|---|
| 8 | _pending_ | | | | | | |
| 16 | _pending_ | | | | | | |
| 32 | _pending_ | | | | | | |

---

## 4. (Speedup, quality delta) table

Quality is **not** reported until the guardrails run. `n/a` means not yet measured — never
"assumed unchanged".

| # | Change | Speedup | Δ loss_eval | Δ retrieval | Status | Evidence |
|---|---|---|---|---|---|---|
| — | Baseline (unmodified, as shipped) | 1.00× | — | — | running | `bench/results/baseline/` |
| 0.1 | **Freeze CLIP encoder** (`requires_grad_(False)`) | **1.36×** | n/a | n/a | verified, quality pending | `bench/results/freeze_clip.json` |

### 0.1 — Freeze the CLIP encoder (not in the brief's backlog)

`models/loss.py` never sets `requires_grad=False` on the RN101 used for the perceptual loss.
So **every** `loss.backward()` computes and accumulates weight gradients for **119.7 M
parameters** in order to optimise **128** control points — 935,000× more gradient than needed.
The optimiser only ever steps the points, so all of it is discarded.

`python bench/verify_freeze_clip.py`

| | as shipped | frozen |
|---|---:|---:|
| ms/iter | 35.84 | 25.85 |
| peak alloc | 2757 MB | 2459 MB |
| 2001-iter seed | 71.7 s | 52.8 s |

**1.36× (26% of wall-clock removed), ~300 MB saved.**

Numerical equivalence, checked carefully (this is *not* bit-identical, and the reason matters):

- Loss is **bit-identical** — the forward is untouched.
- Point gradients differ by relL2 **1.4e-2**, against a run-to-run control of **1.3e-6**. Real, reproducible, confound-free (verified by toggling `requires_grad` within a single pipeline).
- **Not** TF32 — identical delta with TF32 disabled.
- **Cause: fp16.** OpenAI's `clip.load` converts weights to half on CUDA, so CLIPasso's whole
  perceptual loss runs in fp16. Forcing fp32 collapses the frozen-vs-unfrozen delta to **8.2e-4**
  (17× smaller) — a precision artifact from a different cuDNN backward algorithm, not a
  semantic change.
- **Context:** the baseline's own fp16 gradients differ from fp32 by relL2 **1.15e-1**. Freezing
  perturbs the gradient **~8× less than the arithmetic CLIPasso already tolerates.**

Mathematically exact (weight gradients cannot influence input gradients); numerically within
the method's existing fp16 noise floor. **Still must clear the §5 quality guardrails before it
counts as free** — 2000 Adam steps can amplify small differences.

---

## 5. Quality guardrails

Metric #2 was re-specified: the paper's zero-shot `"A sketch of a(n) {class}"` is now available
again (Sketchy has class names), and **label-free sketch→photo retrieval** is added as a harder,
semi-independent check. Both use ViT-B/32, which is *not* the RN101 driving the loss.

1. **`loss_eval`** — the repo's own un-augmented eval loss. Primary, cheap, directly comparable.
2. **Zero-shot top-1** — CLIP ViT-B/32, prompt `"A sketch of a(n) {class}"`, on `paper_protocol.json`.
   Paper reference: ~78% @ 16 strokes, ~91% @ 32.
3. **Sketch→photo retrieval** — does sketch *i* rank its own source photo top-1 against the rest? Label-free, strictly harder than 10-way classification.
4. **Control-point trajectory divergence** vs baseline — catches behavioural changes the loss hides.
5. **Visual contact sheet** — non-negotiable. Loss numbers hide perceptual failure, especially at 4 strokes.

*(Not yet implemented — next task.)*

---

## 6. Open questions

- **§4 of the brief is unresolved** (deferred pending profile numbers — now available).
  The profile argues the answer matters less than expected for Tier 1: the workload is
  launch-bound at every stroke count, so batching helps *all three* goals.
- Does the 41.6% launch idle survive `torch.compile` on the CLIP branch? diffvg will graph-break;
  the question is whether the CLIP subgraph alone can be captured by CUDA graphs.
- `diffvg bwd` at 21.9% is the largest single phase and has no entry in the brief's backlog.
