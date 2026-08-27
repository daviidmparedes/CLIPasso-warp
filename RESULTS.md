# CLIPasso Speed-Up — Results Log

Single source of truth for what has been tried. Every change gets a
**(speedup, quality delta)** pair. Negative results stay in the table.

Reproduce anything here with the scripts in [`bench/`](bench/). All numbers below
come from the harnesses, not from estimates.

---

## Summary — current state

**3.44× cumulative, no measurable quality cost**, on n=16 over the fixed 5-image eval set × 3 seeds.

| # | change | marginal | cumulative | s/seed | quality |
|---|---|---:|---:|---:|---|
| — | baseline as shipped | — | 1.00× | 126.3 | — |
| 0.2 | release the per-iteration autograd graph | 1.48× | 1.48× | 85.3 | inside noise floor |
| 1.1 | batch the 3 seeds into one process | 1.67× | 2.47× | 51.1 | inside noise floor |
| 0.1 + 0.3 | freeze the CLIP encoder + skip the unused `CLIPLoss` | 1.28× | 3.16× | 40.0 | inside noise floor |
| 1.2 | batch across images (M=5) | 1.09× | **3.44×** | 36.7 | inside noise floor |

Measured against the like-for-like unpatched arm (`nolog`, 117.7 s/seed) rather than the
fully-logged default, the cumulative figure is **3.21×**. Both are stated because the
default ships with logging on.

Four findings that shaped everything after them:

1. **The workload is launch-bound, not compute-bound.** 41.6% of wall-clock is GPU-idle,
   ~3800 kernels/iteration averaging 5.5 µs. CLIP/diffvg is 1.37× — balanced, *not*
   CLIP-dominated as the brief's prior expected. (§2)
2. **CLIPasso does not reproduce itself.** Identical code, identical seed, run twice →
   48 px mean control-point divergence on a 224 px canvas. This sets the noise floor for
   every quality claim and invalidates guardrail #4 as the brief specifies it. (§4.3)
3. **Three free speedups the brief's backlog never listed** (0.1, 0.2, 0.3), all found by
   profiling rather than by working down the tier list. Together they are 1.90×.
4. **diffvg does not batch, and that caps Tier 1.** Idea 1.2 was projected at 10–20×
   throughput; it delivers **1.09×**. (§1.2)

Corrections to earlier claims in this document, kept visible on purpose: an estimate that
"~43% of wall-clock is startup + logging" was wrong (measured: 6.9% + 7.2%); a hypothesis
that per-iteration cost grows as strokes lengthen was wrong (measured: 1.01× drift); and
the brief's 10–20× estimate for 1.2 was wrong for a structural reason given below.

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

### Per-iteration cost is flat across the run

`python bench/profile_over_time.py` → `bench/results/profile/over_time_n16.json`

A 50-iteration profile taken from initialisation could in principle understate diffvg,
since strokes start as ~0.05-radius squiggles and lengthen as they fit the target.
Measured at six checkpoints through a full 2001-iteration run, it does not:

| iter | fused ms | diffvg | CLIP | ink coverage | ctrl-polygon len |
|---:|---:|---:|---:|---:|---:|
| 25 | 36.25 | 11.14 | 15.34 | 2.1% | 411 px |
| 500 | 36.14 | 11.28 | 14.70 | 6.3% | 1492 px |
| 1000 | 36.62 | 11.69 | 15.52 | 9.1% | 2208 px |
| 1950 | 36.44 | 11.63 | 15.32 | 9.4% | 2911 px |

**1.01× drift over the whole run**, while ink coverage rises 4.5× and control-polygon
length 7×. Stroke growth is effectively free — consistent with the stroke-count sweep.
The §2 breakdown is therefore representative of the entire run, not just its start.

---

## 3. Baseline (unmodified CLIPasso)

`python bench/run_baseline.py` → `bench/results/baseline/baseline_shipped.json`
45/45 runs succeeded, **95.4 min** total. Fixed eval set: 5 Sketchy images ×
{8,16,32} strokes × 3 seeds, `--num_iter 2001`.

| n | s/seed | ms/iter | iters | early-stop % | mean loss_eval | ± | s/image (3 seeds) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 122.1 | 60.7 | 2010 | **0%** | 0.59140 | 0.03055 | 366.2 |
| 16 | 126.3 | 62.9 | 2010 | **0%** | 0.55880 | 0.03625 | 379.0 |
| 32 | 133.2 | 66.2 | 2010 | **0%** | 0.53748 | 0.04460 | 399.5 |

### The early-stopping rule never fires

**0% of 45 runs stopped early.** The `min_delta=1e-5` criterion did not trigger once
in 2001 iterations at any stroke count. Idea **2.1** therefore has the *entire*
budget to cut into — the existing rule contributes nothing.

### Wall-clock decomposition at n=16 (measured, not inferred)

| component | s/seed | how measured |
|---|---:|---|
| fixed startup (imports, RN101 + ViT-B/32 + U2Net load, saliency init) | 9.1 | extrapolated from `--num_iter` 11 vs 111 |
| optimisation loop | 104.4 | full run, all instrumentation disabled |
| eval block (`eval_interval=10`) | 5.2 | eval on vs off, save off |
| SVG + matplotlib logging (`save_interval=10`) | 8.7 | `shipped` vs `nolog` tags |
| **total** | **126.3** | |

An earlier estimate that "~43% of wall-clock is startup + logging" was **wrong**:
logging is only 6.9% and startup 7.2%. The loop itself dominates — and §4.1 explains
why it was slower than the profiler predicted.

## 4. (Speedup, quality delta) table

Quality is **not** reported until the guardrails run. `n/a` means not yet measured — never
"assumed unchanged".

| # | Change | Speedup | Δ loss_eval | Δ retrieval | Status | Evidence |
|---|---|---|---|---|---|---|
| — | Baseline (unmodified, as shipped) | 1.00× | — | — | running | `bench/results/baseline/` |
| 0.1 | **Freeze the CLIP encoder** | **1.28×** with 0.3 | +0.0025 | median rank 397→395 | **verified** | `guardrails_batched_freeze_n16.json` |
| 0.3 | **Skip the unused `CLIPLoss`** (`Loss.__init__`) | included in 0.1 row | none (never called) | — | **verified** | `models/loss.py` |
| 0.2 | **Release the autograd graph each iteration** (`del sketches, losses_dict, loss`) | **1.34×** end-to-end (1.40× loop-only) | +0.0016 (noise floor ±0.013) | R@1 0.0%→0.0%, median rank 500→492 | **verified** | `bench/results/guardrails/guardrails_fixed_n16.json` |
| **1.1** | **Batch the 3 seeds into one process** | **1.67×** | −0.0060 (slightly better) | median rank 525→397 | **verified** | `guardrails_batched_n16.json` |
| **1.2** | **Batch across images** (M=5 × 3 seeds) | **1.09×** — see §1.2 | +0.0046 | median rank 395→420 | **verified (mostly negative)** | `guardrails_tier12_M5_n16.json` |

### 0.1 / 0.3 — Freeze the CLIP encoder, and stop building the loss nobody uses

Two independent defects in `models/loss.py`, both now fixed in `CLIPConvLoss.__init__`
and `Loss.__init__` so they hold for every entry point.

**0.1 — the encoder was never frozen.** `PainterOptimizer` only ever steps the Bézier
control points, but nothing set `requires_grad=False` on RN101. Every `loss.backward()`
therefore computed and accumulated weight gradients for **119.7 M parameters** in order to
update **128** — 935,000× more gradient than needed, all discarded.

**0.3 — `Loss.__init__` built both losses unconditionally**, so at default settings
(`train_with_clip=0`) every process loaded a full ViT-B/32 for a `CLIPLoss` that is never
called. Now only the selected losses are constructed, with lazy construction retained in
`update_losses_to_apply` for the case where `clip` is appended mid-run. `Loss()`
construction drops to **1.19 s**.

Measured together on top of 1.1, over the 5-image eval set × 3 seeds:
**51.1 → 40.0 s/seed = 1.28×.**

Numerical equivalence was checked carefully, because freezing is *not* bit-identical:

- Loss is **bit-identical** — the forward is untouched.
- Point gradients differ by relL2 **1.4e-2** against a run-to-run control of 1.3e-6.
- **Not** TF32 — identical delta with TF32 disabled.
- **Cause: fp16.** OpenAI's `clip.load` converts weights to half on CUDA, so the whole
  perceptual loss runs in fp16. Forcing fp32 collapses the delta 17× to **8.2e-4** — a
  precision artifact from a different cuDNN backward algorithm, not a semantic change.
- **Context:** the baseline's own fp16 gradients differ from fp32 by relL2 **1.15e-1**, so
  freezing perturbs the gradient **~8× less than the arithmetic CLIPasso already tolerates.**

End-to-end quality: Δ mean `loss_eval` **+0.0025** against the ±0.0128 noise floor (§4.3);
zero-shot 125-way unchanged at 13.3%; retrieval median rank 397 → 395. Accepted.

### 0.2 — Release the per-iteration autograd graph (not in the brief's backlog)

`painterly_rendering.py` binds `sketches`, `losses_dict` and `loss` in the enclosing
loop scope, so they stay alive until they are **reassigned partway through the next
iteration**. The previous iteration's autograd graph — including diffvg's
`RenderFunction` context, which holds the serialized scene — is therefore still live
while the next forward and backward run.

This surfaced as a discrepancy, not a hunch: the in-process profiler measured
36 ms/iter but a real 2001-iteration subprocess ran at 52 ms/iter, and the
subprocess cost *drifted upward* (34.6 → 48.4 → 54.0 → 54.6 ms/iter) while the
in-process loop stayed flat. Bisecting the two loop bodies isolated the difference to
variable lifetime.

Ruled out along the way, each by direct measurement: tqdm's per-iteration
`epoch_range.refresh()` (113.79 s vs 113.48 s with `--display 1`), a missing
per-iteration sync (52.11 vs 52.24 ms with an added `loss.item()`), and GPU thermal
throttling (the in-process loop holds 36 ms under the same sustained load).

The cost lands on exactly one phase (measured at iteration ~800):

| phase | retaining | `del` |
|---|---:|---:|
| diffvg fwd | 4.36 ms | 3.61 ms |
| CLIP fwd | 10.65 ms | 9.80 ms |
| CLIP bwd | 5.99 ms | 5.70 ms |
| **diffvg bwd** | **26.45 ms** | **7.74 ms** |

**diffvg's backward pays a 3.4× penalty.** diffvg allocates outside PyTorch's caching
allocator, so when its context is pinned by the retained graph it falls back to raw
`cudaMalloc`/`cudaFree`, which synchronise. This also explains why
`torch.cuda.max_memory_allocated` reports the *opposite* of the intuitive story
(1716 MB retaining vs 1983 MB with `del`) — diffvg's allocations never appear in
torch's allocator statistics at all.

End-to-end on the real script, 2001 iterations, instrumentation off:
**113.48 s → 81.18 s = 1.40×.** The drift disappears: 36.1 ms/iter, flat.

**Verified on the n=16 guardrail set** (15 runs, same 5 images × 3 seeds):

| | shipped | patched |
|---|---:|---:|
| s/seed | 126.3 | **94.2** (1.34×) |
| ms/iter | 62.9 | **46.9** |
| mean loss_eval | 0.55880 ± 0.03625 | 0.56036 ± 0.03713 |
| zero-shot top-1 (5-way) | 26.7% | 53.3% |
| retrieval R@1 / median rank | 0.0% / 500 | 0.0% / 492 |

I expected the `loss_eval` curves to be **bit-identical** — the change frees memory after
`optimizer.step_()` and alters no computation. They are not (max |Δ| = 4.6e-2). That turned
out to say nothing about the change and everything about the method: see §4.3.


### 4.3 — CLIPasso does not reproduce itself (affects every future comparison)

Re-running the **unmodified** code with the **same seed** produces materially different
sketches. Control: 3 runs re-executed from commit `2c08de5` and compared against the
stored `shipped` results for the same image and seeds.

| comparison | max \|Δ loss_eval\| | final ctrl-pt L2 | Δ best_loss_eval |
|---|---:|---:|---:|
| **unpatched vs unpatched** (identical code + seed) | 3.49e-2 | **48.38 px** | 1.28e-2 |
| unpatched vs patched (`del`) | 4.61e-2 | 54.38 px | 2.09e-2 |

On a 224 px canvas, the baseline diverges from *itself* by ~48 px of mean control-point
displacement. The cause is diffvg's backward, which accumulates gradients with
`atomicAdd` — a non-deterministic summation order — amplified through 2001 Adam steps.

**Consequences for the project's methodology:**

1. **Quality guardrail #4 as specified in the brief cannot work.** "Control-point
   trajectory divergence vs the baseline — catches silent behavioural changes that leave
   the loss unchanged" presumes a reproducible baseline. The noise floor is ~48 px, so
   the metric has no resolution below that. It is kept in `bench/guardrails.py` but must
   be read against this floor, never as an absolute.
2. **No per-run comparison is meaningful.** Every (speedup, quality delta) pair has to be
   a statistic over many runs and seeds. Single-run A/B is noise.
3. **A change is "quality-neutral" only if its effect is inside this floor.** For 0.2:
   Δ mean loss_eval = +0.0016 against a per-run nondeterminism floor of ±0.0128 and a
   between-seed sd of ±0.036 — an order of magnitude below the noise. Accepted.
4. Reproducible runs would need deterministic diffvg kernels (upstream change) or
   `torch.use_deterministic_algorithms` plus a diffvg atomics rewrite. Worth knowing
   before anyone tries to debug a "regression" that is really just variance.

*Control is n=3; the ~48 vs ~54 px comparison is same-order rather than a tight bound.
Widening it is cheap and worth doing before any change with a subtler expected effect.*


### 1.1 — Batch the seeds (brief Tier 1, first entry)

`bench/batch_seeds.py`. `run_object_sketching.py` spawns one **process** per seed over the
same target. Each reloads RN101, ViT-B/32 (twice, inside `Painter`) and U2Net, recomputes
the saliency map, then runs a 2001-iteration loop launching ~3800 kernels of ~5.5 µs. The
profile says 41.6% of wall-clock is GPU-idle launch overhead, so three small launch-bound
loops should become one larger one.

**Shared:** model loading, target tensor, U2Net mask, and one CLIP encoder call over all
seeds' views (15 sketch + 15 target images per iteration, instead of 3 × (5+5)).
**Not shared, deliberately:** each seed keeps its own augmentation draws, and each seed's
loss is reduced over only its own views before summing. The seeds have disjoint parameters,
so `d(Σᵢ Lᵢ)/dθⱼ = dLⱼ/dθⱼ` — one backward gives every seed the gradient it would get
alone, and one Adam over the concatenated parameters equals three separate Adams because
Adam's state is per-parameter.

**diffvg is not batched** — separate scenes cannot share a canvas, so rendering stays one
call per seed. Only the CLIP half (57% of per-iteration cost) and startup amortise, which
is why this lands at 1.67× rather than the brief's estimated 2–2.5×.

#### Speedup ladder (n=16, 5 images × 3 seeds, all measured)

| configuration | s/seed | cumulative | loss_eval | sd |
|---|---:|---:|---:|---:|
| baseline as shipped | 126.3 | 1.00× | 0.55880 | 0.03625 |
| + no save logging (unpatched) | 117.7 | 1.07× | 0.55629 | 0.03503 |
| + graph release 0.2 (full logging) | 94.2 | 1.34× | 0.56036 | 0.03713 |
| + graph release 0.2 (no save logging) | 85.3 | 1.48× | 0.56085 | 0.03534 |
| **+ batched seeds 1.1** | **51.1** | **2.47×** | 0.55487 | 0.03929 |

Boot cost drops from ~9.1 s × 3 to **2.97 s shared**; loop cost from 42.5 to
**~24.8 ms/iter/seed**.

#### Equivalence, verified rather than assumed (`bench/verify_batched_equiv.py`)

1. With N=1, `batched_conv_loss` reproduces `CLIPConvLoss.forward` **bit-identically** on
   every loss term.
2. Given the same augmented views, seed *i*'s loss inside the 15-row batch equals its loss
   in its own 5-row batch **exactly (relative change 0.000e+00)**. BatchNorm is in eval
   mode across all 106 layers, so the rows are genuinely independent.

An earlier version of test 2 compared two RNG replays across differently shaped batches and
reported deviations as large as the gradients themselves. That was a defect in the test, not
the code — replaying augmentation draws across different batch shapes is not well defined.
The rewritten test compares the quantity that actually matters.

#### Quality

Δ mean `loss_eval` = **−0.0060** (batched is marginally *better*), against a per-run
nondeterminism floor of ±0.0128 (§4.3) and a between-seed sd of ±0.035 — comfortably inside
the noise. Zero-shot 125-way is unchanged at 13.3%; retrieval median rank improves 525 → 397
and R@10 0% → 13.3%. Trajectory divergence vs the per-process arm is 63.6 px, the same order
as the 48.4 px the baseline shows against *itself*. Accepted.


### 1.2 — Batch across images (brief Tier 1) — **largely a negative result**

`bench/batch_images.py`. Extends 1.1 from "N seeds of one image" to "N seeds of M images",
all optimised concurrently with one CLIP encoder call per iteration over M·N·(1+A) sketch
views and the matching target views. The equivalence argument and verification are the same
as 1.1 (the shared `batched_conv_loss` was generalised to per-sketch targets and re-verified
bit-identical).

**The brief projected 10–20× throughput. Measured: 1.09×** (599.5 s → 551.2 s for
5 images × 3 seeds, i.e. 40.0 → 36.7 s per sketch).

#### Throughput saturates immediately

201-iteration probe, 3 seeds per image:

| M images | ms/iter | **ms/iter/sketch** | peak alloc |
|---:|---:|---:|---:|
| 1 | 55.6 | 18.5 | 1.80 GB |
| 2 | 107.2 | 17.9 | 2.94 GB |
| 5 | 267.0 | **17.8** | 6.38 GB |
| 10 | 561.2 | 18.7 | 12.11 GB |
| 16 | 936.3 | 19.5 | 18.95 GB |

Per-sketch cost is flat, with a shallow optimum near M=5 and mild *regression* past M=10.
Total time scales almost perfectly linearly (M=16 is 16.8× M=1), which means essentially
nothing amortises beyond M=1. Memory is not the constraint — 19 GB of 97 GB at M=16.

#### Why: diffvg does not batch

Phase attribution at M=1 vs M=5 (25 measured iterations, synchronised boundaries):

| phase | M=1 (3 sketches) | M=5 (15 sketches) | scaling for 5× work |
|---|---:|---:|---:|
| diffvg render | 9.84 ms | 60.42 ms | **6.1× (superlinear)** |
| CLIP forward | 15.04 ms | 66.82 ms | 4.4× (sublinear — this part *does* amortise) |
| backward | 28.02 ms | 133.85 ms | 4.8× |
| **per sketch** | **17.88 ms** | **17.54 ms** | **1.02×** |

The CLIP forward is the only phase that amortises, and its gain is cancelled by rendering,
which scales *worse* than linearly. Separate scenes cannot share a canvas, so diffvg is
called M·N times per iteration and its backward likewise. By M=5 the vector side is roughly
60–70% of the iteration, and adding images adds it proportionally.

**This caps Tier 1 as a whole.** The CLIP half was already saturated at M=1 (a 15-row
encoder batch is enough to fill this GPU), so 1.1 captured nearly all the available
amortisation and 1.2 has almost nothing left to take.

Quality: Δ mean `loss_eval` **+0.0046** against the ±0.0128 noise floor; zero-shot 5-way
unchanged at 46.7%; 125-way moved 13.3% → 6.7%, which on 15 runs is 2 correct vs 1 and is
not a signal. Trajectory divergence 62.8 px vs the 48.4 px self-divergence floor. Accepted
as neutral — it simply does not buy much.


---

## 5. Quality guardrails

Metric #2 was re-specified: the paper's zero-shot `"A sketch of a(n) {class}"` is now available
again (Sketchy has class names), and **label-free sketch→photo retrieval** is added as a harder,
semi-independent check. Both use ViT-B/32, which is *not* the RN101 driving the loss.

**Measured baseline quality** (`bench/results/guardrails/`), and a validation of the harness itself:

| metric | shipped, all 45 runs | shipped, n=16 only | **photos (control)** |
|---|---:|---:|---:|
| loss_eval | 0.56256 ± 0.04363 | 0.55880 ± 0.03625 | — |
| zero-shot top-1, 125-way | 15.6% | 0.0% | **80.0%** |
| zero-shot top-1, 5-way | 42.2% | 26.7% | — |
| retrieval R@1 (2000 gallery) | 0.0% | 0.0% | **100.0%** |
| retrieval median rank | 380 | 500 | 1 |

The photo column is a plumbing control, not a result: the same `encode_tensor` path that
scores sketches gives 80.0% on the source photos and 100% self-retrieval, and agrees with
CLIP's official preprocess to cos 0.998. **The harness is correct; the sketches really are
only weakly recognisable to ViT-B/32.** Recognisability is carried almost entirely by the
32-stroke runs (125-way drops 15.6% → 0.0% when restricted to n=16), which is consistent
in direction with the paper's 78% @ 16 / 91% @ 32, though not directly comparable — the
paper's label space and corpus differ, and our eval set is only 5 distinct classes.

1. **`loss_eval`** — the repo's own un-augmented eval loss. Primary, cheap, directly comparable.
2. **Zero-shot top-1** — CLIP ViT-B/32, prompt `"A sketch of a(n) {class}"`, on `paper_protocol.json`.
   Paper reference: ~78% @ 16 strokes, ~91% @ 32.
3. **Sketch→photo retrieval** — does sketch *i* rank its own source photo top-1 against the rest? Label-free, strictly harder than 10-way classification.
4. **Control-point trajectory divergence** vs baseline — catches behavioural changes the loss hides.
5. **Visual contact sheet** — non-negotiable. Loss numbers hide perceptual failure, especially at 4 strokes.

Implemented in `bench/guardrails.py`:

```
python bench/guardrails.py --runs bench/results/baseline/shipped --tag shipped
python bench/guardrails.py --runs bench/results/baseline/fixed_graphfree --tag fixed \
                           --compare-to bench/results/baseline/shipped
```

Emits `loss_eval`, 125-way and subset zero-shot top-1, sketch→photo R@1/5/10 with
median rank, control-point trajectory divergence vs a reference run, and a contact sheet.

---

## 6. Open questions

- **§4 of the brief remains formally open**, but the profile has largely defused it for
  Tier 1: the workload is launch-bound at every stroke count, so batching serves batch
  generation, interactive latency and a paper equally. The question only starts to bite
  when choosing between Tiers 2, 3 and 4.
- **diffvg cannot be batched** — separate scenes cannot share a canvas. This capped 1.1 at
  1.67× and, as §1.2 then measured, capped 1.2 at 1.09×. Any further large win on the
  vector side has to come from diffvg itself (see §7 P2, which proposes rendering many
  scenes into one tiled raster so the launch cost is paid once).
- **`diffvg bwd` is 21.9% of wall-clock**, the largest single phase, and has no entry in
  the brief's backlog.
- Does the 41.6% launch idle survive `torch.compile` on the CLIP branch (1.3)? diffvg will
  graph-break; the question is whether the CLIP subgraph alone can be CUDA-graph captured.
- **The early-stopping rule never fires** (0/45 runs), so idea 2.1 has the entire
  2001-iteration budget available.
- Reproducible runs would need deterministic diffvg kernels. Worth knowing before anyone
  debugs a "regression" that is really variance.

---

## 7. What to do next

Tier 1 is effectively exhausted: 1.1 captured nearly all available amortisation and 1.2
proved there is little left (§1.2). The bottleneck has moved. Priorities below are ordered
by evidence, not by the brief's tier numbering.

### P1 — Idea 2.1, but measure *perceptual* saturation, not loss saturation (cheapest, do first)

The early-stopping rule never fires (0/45 runs), so the whole 2001-iteration budget is
available. But how much of it is dead weight is now quantifiable from the stored curves:

| criterion | mean iter reached | median | p90 | implied speedup (p90) |
|---|---:|---:|---:|---:|
| within 5% of final best `loss_eval` | 470 | 430 | 744 | **2.7×** |
| within 2% | 811 | 680 | 1472 | 1.4× |
| within 1% | 1080 | 1070 | 1660 | 1.2× |

Mean best iteration is 1230–1493 depending on stroke count. **This tempers the brief's
"up to ~5×" estimate for 2.1**: on the raw objective, 5× would mean stopping around iter
400, which costs well over 5% of final loss.

The open question is whether *recognisability* saturates earlier than the loss does — the
brief's Figure-7 intuition is about perceptual quality, not `loss_eval`. That is directly
testable **with data already on disk**: the 45 `shipped` runs saved `svg_logs/svg_iter{k}.svg`
every 10 iterations. Running `bench/guardrails.py` over those snapshots gives a
quality-vs-iteration curve for zero-shot accuracy and retrieval rank with **zero new
optimisation runs**. If recognisability plateaus at iter ~400 while loss keeps creeping,
2.1 is worth 2.7–5×; if it tracks the loss, 2.1 is worth ~1.4× and should be de-prioritised.
Do this before writing any new optimiser code.

### P2 — Attack diffvg; it is now the bottleneck

At M=5, rendering plus its share of backward is roughly 60–70% of each iteration, scales
linearly with work, and amortises not at all. Nothing in the brief's backlog targets it.

- **Tile the scenes onto one canvas.** M·N scenes cannot share a canvas *semantically*, but
  they can share one *raster*: lay them out on a grid (e.g. 4×4 of 224², one 896² render),
  call diffvg once, then slice. Rasterisation cost scales with area, but the ~3800-kernel
  launch overhead is paid once instead of M·N times. This is the single most promising
  untested idea, and it is a contained experiment.
- **Reduce `num_samples_x/y` from 2 to 1** in `render_warp` — 4× fewer rasteriser samples.
  Pure quality trade; measure with the guardrails.
- **Profile diffvg's backward specifically.** It costs 2× its forward (7.77 vs 3.62 ms),
  which is unusual and may indicate a fixable inefficiency rather than intrinsic cost.

### P3 — Idea 1.3 (`torch.compile` / CUDA graphs on the CLIP branch), payoff now smaller

Worth doing, but the profile has already devalued it. Batching absorbed much of the launch
overhead the idea targets, and CLIP forward is now only ~25% of a batched iteration.
Realistic expectation is ~1.1×, not the brief's 1.5–2×. diffvg will graph-break; the
question is whether the CLIP subgraph alone can be CUDA-graph captured with static shapes.

### P4 — Idea 1.4 (cache the augmented target), modest but it unlocks 2.4

Measured ceiling was 9.6% of unbatched wall-clock (§2), lower than the brief's 1.4–1.8×
estimate because the target branch is already `.detach()`ed — the waste is a redundant
forward, not a forward+backward. In the batched setting the target rows are half the encoder
batch, so caching could halve encoder work (~12% of total). The real reason to do it is that
**a fixed augmentation bank makes the objective deterministic**, which is the precondition
for idea 2.4 (L-BFGS) — the brief's most under-explored idea.

### P5 — Then the research tier

2.5 (structure-aware init) and 2.4 (L-BFGS on 128 parameters) remain the highest-upside
items. 2.3 (progressive stroke addition) should be re-scoped: the stroke-count sweep showed
8× the strokes costs only +6.7%, so it cannot pay off through *fewer strokes* — only through
faster convergence.

### Methodology debt to clear first

1. **The eval set is too small for the recognition metrics.** 5 images / 5 classes means
   zero-shot 125-way moves in 6.7% steps — one sketch. `data/paper_protocol.json` (200
   images, 10 categories) already exists and should replace it for any quality claim that
   turns on metric 2 or 3. `loss_eval` is unaffected and stays usable on the small set.
2. **Widen the nondeterminism control** (§4.3), currently n=3. Every quality delta is judged
   against that floor, so it deserves more than three runs.
3. **Decide the §4 question.** It no longer reorders Tier 1 (that work is done and helps all
   three goals equally), but P1 vs P2 vs P5 does depend on it: batch generation wants P2,
   interactive latency wants Tier 4, a paper wants P1 + P5.
