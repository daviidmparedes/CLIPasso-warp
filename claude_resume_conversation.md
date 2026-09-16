# CLIPasso speed-up project — handoff

Written for an LLM picking this up on a new server. Read this file first, then
`RESULTS.md`, which is the full measurement log and the single source of truth for
numbers. This file tells you what the project is, what state it is in, what will break
during migration, and what to do next.

---

## 1. What this project is

Reduce the wall-clock cost of CLIPasso's test-time optimisation **without degrading sketch
quality**. CLIPasso draws a sketch by optimising Bézier control points with Adam against a
frozen CLIP perceptual loss, rendered differentiably by diffvg, for 2001 iterations per
sketch.

The user is a full-time CV researcher. The working agreement, which has held all project:

- **Measure before changing.** No optimisation ships on a prediction.
- **One change at a time, on its own branch**, with a measured `(speedup, quality delta)` pair.
- **Negative results stay in the document.** Several major ideas failed; that is the record.
- **Every quality claim is judged against CLIPasso's own run-to-run noise**, never against zero.
- Prefer small reproducible scripts in `bench/`.

---

## 2. Status at a glance

**3.44× cumulative**, n=16, 5 images × 3 seeds, seconds per sketch:

| # | change | marginal | cumulative | s/seed |
|---|---|---:|---:|---:|
| — | baseline as shipped | — | 1.00× | 126.3 |
| 0.2 | release the per-iteration autograd graph | 1.48× | 1.48× | 85.3 |
| 1.1 | batch the 3 seeds into one process | 1.67× | 2.47× | 51.1 |
| 0.1+0.3 | freeze CLIP encoder + skip unused `CLIPLoss` | 1.28× | 3.16× | 40.0 |
| 1.2 | batch across images (M=5) | 1.09× | **3.44×** | 36.7 |

Measured, verified, **not** in the ladder:

| # | change | result |
|---|---|---|
| 0.4 | strip diffvg's per-shape `isfinite` asserts | 1.03× idle / 1.13× shared, **bit-identical**; opt-in via `bench/fast_serialize.py` |
| 0.5 | tile M scenes onto one raster | 1.10–1.50× on diffvg fwd+bwd; prototype only, not integrated |
| 2.1 | early stopping | **worth ~1.67×** at a fixed 1200-iteration budget — validated, not yet implemented (this is N2, the top next task) |
| N1 | learning-rate schedule | **negative** — works, but loses to plain truncation at every budget |
| A2 | dedupe the clean target view | **abandoned** — not bit-identical, and worth ~0.56%, not the ~1.02× estimated |

⚠️ **All timings were taken on a GPU shared with another researcher's job holding 83–95 GB of
97 GB.** Ratios are paired and trustworthy; absolute milliseconds are upper bounds.
`bench/run_night.sh` exists to re-measure them on an idle card and has never been run.

---

## 3. MIGRATION — read this before anything else

### Must be copied. Cannot be regenerated without ~25 GPU-hours.

| path | size | what it is |
|---|---:|---|
| `bench/results/` | **463 MB** | every measurement in `RESULTS.md`. 45 baseline runs with 9045 SVG snapshots, 5 LR arms, 2 replicate controls, the 200-image wide set, all guardrail and profile JSON |
| `data/` | **130 MB** | the sampled Sketchy corpus + `manifest.json`, `eval_set.json`, `paper_protocol.json`. Every eval set is defined by these files |
| `bench/logs/` | 140 KB | run logs |

**`bench/results/` and `data/` are in `.gitignore`. They are NOT on GitHub. If they are not
copied, roughly 25 GPU-hours of measurement is lost and none of `RESULTS.md` is reproducible.**

    rsync -a bench/results bench/logs data  user@newserver:/path/to/CLIPasso/

### Rebuildable, do not copy

| path | size | how to rebuild |
|---|---:|---|
| `.venv/` | 7.9 GB | `./setup.sh` |
| `third_party/` | 15 GB | `./setup.sh` (diffvg + a local CUDA toolkit) |
| `U2Net_/saved_models/` | 170 MB | downloaded on first run |
| `output_sketches/` | — | generated output |

### The code is safe

Everything is committed and pushed to `git@github.com:daviidmparedes/CLIPasso-warp.git`.

---

## 4. Environment

What this ran on, and what the stack must satisfy:

- GPU: **NVIDIA RTX PRO 6000 Blackwell**, compute capability **12.0 (sm_120)**, 97 GB
- Python 3.12.3, torch 2.9.1+cu128 (a fresh install pulled 2.11.0+cu128 and worked)
- CUDA toolkit 12.8, hand-extracted to `third_party/cuda` (this machine had **no** system CUDA)

**The original 2021 stack (Python 3.7 / torch 1.7.1+cu101 / CUDA 11.0) cannot be installed on a
current machine and cannot run on anything newer than Ampere.** Rebuilding it was the first
week of this project. `setup.sh` on the fork now automates it and is verified from a clean clone.

Non-obvious requirements `setup.sh` checks up front:

- **Python development headers** (`Python.h`). Without them diffvg dies in CMake with
  `Could NOT find Python (missing: Python_INCLUDE_DIRS)` — after a 2.5 GB download.
- **A CUDA toolkit providing `nvcc`.** PyTorch ships a CUDA *runtime*, not a compiler. A working
  driver is not enough.

No root on the new server? This path is tested and works:

    CUDA_HOME=/path/to/cuda ALLOW_LOCAL_PYTHON_HEADERS=1 ./setup.sh

Source-level breakages that had to be fixed (all committed): `F._pad`→`F.pad` in
`CLIP_/clip/auxilary.py`; an attention hook registered unconditionally, which fails under
`no_grad`; `scipy.ndimage.filters` and `np.int` removed; `torch.load` weights_only default.

---

## 5. Repo and branch map

Remotes: `origin` = **upstream** `yael-vinker/CLIPasso` (never push here).
`fork` = `daviidmparedes/CLIPasso-warp` (ours).

| branch | where | what |
|---|---|---|
| `fork-main` | pushed as `main` | the lean public fork: 26 files, 3.0 MB, 10 curated commits |
| `opt/lr-schedule` | local, current HEAD | the live working branch with everything |
| `experiments` | pushed as `experiments` | same content, commit messages scrubbed of tool attribution |
| `main` | local | pristine upstream, do not commit here |
| others | local | superseded historical branches |

**The commit messages on the published branches deliberately contain no AI/assistant
attribution — the user asked for this. Keep new commits in that style.**

The fork ships: `run_sketch.py` (batched entry point, the 1.67× win), `setup.sh`,
patched source, a plain README. It does **not** ship `bench/` or `RESULTS.md` — those live on
`experiments`, because the harness is the evidence for the fork's claims and must stay
version-controlled.

`run_sketch.py` is verified end to end from a clean clone: 3 sketches of `camel.png` at 16
strokes in 56.5 s, valid 16-path SVG.

---

## 6. The findings that shape every decision

1. **The workload is launch-bound, not compute-bound.** 41.6% of wall-clock is GPU-idle,
   ~3800 kernels/iteration at 5.5 µs mean. CLIP/diffvg is 1.37× — balanced, not CLIP-dominated.
2. **CLIPasso does not reproduce itself.** Identical code, identical seed, run twice →
   **57.5 ± 7.9 px** mean control-point drift on a 224 px canvas, because diffvg's backward
   accumulates with `atomicAdd`. This sets the floor for every quality claim.
3. **40% of 5-way zero-shot decisions flip between two identical runs** (13.3% for 125-way).
   Any accuracy delta smaller than that is measuring the random seed.
4. **diffvg does not batch.** Separate scenes cannot share a canvas, which capped 1.1 at 1.67×
   and 1.2 at 1.09×.
5. **83% of diffvg's forward kernel launches are a debug assertion.** `serialize_scene` runs
   `assert(torch.isfinite(points).all())` per shape per iteration — a GPU reduction plus a host
   sync. Stripping it: 124 → 21 kernels, bit-identical.
6. **diffvg's backward is four large kernels doing real work** — `render_kernel` ×2 (54%),
   `sample_boundary_kernel` (18%), `render_edge_kernel` (14%). No overhead left to remove.
   This is the remaining wall.
7. **`loss_eval` goes blind after ~iteration 500.** It reaches 95% of its fall by 480, but every
   perceptual metric keeps improving to ~1900 — because training minimises over 5 augmented
   views while `loss_eval` scores 1 clean view (`models/loss.py:176-182`). This is why the
   repo's own early-stopping rule never fires in 45 runs.
8. **Nothing makes CLIPasso converge.** `lr=1.0`, Adam, no decay: control points move at a
   constant ~0.19 px/iter for 1800 iterations. The repo's `--lr_scheduler` called
   `utils.get_epoch_lr()`, **a function that did not exist** — the flag always crashed. Now
   implemented, but turning it on does not beat plain truncation.
9. **The 5-image eval set was unrepresentative, not just small.** 125-way zero-shot is 29.0% on
   the 200-image paper protocol vs 13.3% on the 5-image set, *at a shorter budget*. Paired
   deltas in `RESULTS.md` §4 still hold (bias cancels within a pair), but absolute
   recognisability numbers on the small set understated the method.

---

## 7. Methodology you must preserve

- **Judge every delta against the measured noise floor**, in
  `bench/results/quality_curve/noise_floor_rep15.json`: `loss_eval` ±0.00905, replicate CLIP
  agreement 0.9368 ± 0.0251, drift 57.5 px. A change "inside the floor" is a change you cannot
  distinguish from rerunning the unmodified code.
- **Time with paired, interleaved A/B.** Run A then B within each rep so contention hits both.
  Report the min-ratio (best estimate of uncontended cost) *and* the paired median. See
  `bench/tiled_render.py:bench_paired`.
- **Correctness gates must compare against the renderer's own noise, not an epsilon.** diffvg
  jitters sub-pixel samples by absolute pixel index, so changing canvas size changes stroke
  edges. Tiling "differs" from separate rendering by exactly as much as re-seeding diffvg does.
- **Compare like summaries with like.** A minimum over M scenes against a single-sample baseline
  is not a comparison; the minimum of a noisy statistic drifts down as M grows.
- **The quality metrics are coarse.** Use the continuous companions (zero-shot margin, cosine
  similarity to the true photo, mean log retrieval rank), which have a standard error.

---

## 8. Traps that have already bitten

- **diffvg silently returns wrong output when the GPU is near its memory ceiling.** No
  exception. A guardrails run at 2.9 GB free reported 0.0% zero-shot where the truth is 13.3%,
  and reproduced the *same* wrong numbers on a second run. `common.require_free_gpu_memory()`
  now warns at the top of every analysis script. **Any number taken on a full GPU is suspect.**
- **`bench/run_tier11.sh` and `bench/run_freeze_batched.sh` have hardcoded output paths** and
  will overwrite `bench/results/{batched,batched_freeze}` and `baseline/patched_nolog` — live
  comparison arms, one of which is half the noise floor. They carry warning headers. Use
  `bench/run_night.sh`, which writes everything under `bench/results/night/`.
- **Never edit `bench/batch_seeds.py` (or anything it imports) while a job is running.** Each
  image spawns a fresh subprocess that re-reads the file mid-run.
- **`config.parse_arguments()` appends the run name to `output_dir` and calls `os.mkdir`, not
  `makedirs`** — the parent must exist first.
- **CLIP runs in fp16**, so changing an encoder batch's shape changes GEMM tiling and perturbs
  results at ~1e-5. Any "bit-identical" claim must be verified, not assumed.

---

## 9. Corrections made during this work — do not repeat them

Recorded because each looked convincing and was wrong:

1. **An n=3 noise floor moved a conclusion by 1.6×.** The early-stopping verdict computed
   against it put the stopping point at iteration 810; against the correct n=15 floor it is 500.
   Under-powered controls silently move answers.
2. **Measuring truncation cost against each arm's own endpoint** made LR decay look 1.62× better
   with 2.6× tighter variance. Both vanish under the matched-budget absolute comparison
   (absolute sd 0.03723 vs 0.03709 — identical). Compare arms at matched cost, in absolute terms.
3. **A2's saving was overestimated ~20×** by counting encoder *rows* as encoder *cost* and
   ignoring that the target branch is forward-only. **A1's "40% of encoder rows" estimate in
   `RESULTS.md` §8 has the identical flaw — redo that arithmetic before running it.**
4. Earlier wrong claims, all corrected by measurement: "~43% of wall-clock is startup + logging"
   (real: 6.9% + 7.2%); "per-iteration cost grows as strokes lengthen" (real: 1.01× drift);
   the plan's rationale for tiling ("launch cost paid once") — tiling works, but because it
   collapses allocation cycles, not launches.

---

## 10. What to do next

### N2 — implement early stopping. No GPU needed. Top priority.
Validated in `RESULTS.md` §6, implementation only. Stop at **1200 iterations**, which keeps 96%
of runs at or above the reproducibility floor, and **scale with stroke count**: the floor is
reached at 180 / 400 / 800 iterations for n=8 / 16 / 32, so one constant wastes the cheap cases.
Do **not** build an adaptive rule — every velocity threshold tested scored +0 points against a
fixed cut, because nothing decays. Worth ~1.67×, the last easily-claimed win.

### `bash bench/run_night.sh` — needs an idle GPU. Guarded at ≥80 GB free; aborts otherwise.
Phase 1 (clean ladder re-measurement, ~105 min) is the valuable one: it converts every s/seed in
`RESULTS.md` and the fork's README from a contended upper bound into a measurement. It does not
discover anything; it makes what is already claimed defensible. Then 0.4 end-to-end, a clean
tiling sweep, and `num_samples` timing.

### Then, in order
- **Integrate 0.4 and 0.5** into the batched harnesses and re-measure the ladder *with both
  enabled* — they overlap (both attack per-render overhead), so do not multiply their gains.
- **A1** (`num_aug_clip` 4→2) — recompute its cost estimate first, see correction 3.
- **N5** `torch.compile` / CUDA graphs on the CLIP branch — realistic ~1.1×, devalued by batching.
- **The research tier**: 2.5 structure-aware init, 2.4 L-BFGS (needs 1.4's fixed augmentation
  bank to make the objective deterministic first).

### Open question worth resolving before anything is published
Against the paper protocol we measure **29.0%** (125-way) and **49.5%** (10-way) zero-shot,
versus the paper's **~78%** at 16 strokes. This is no longer explainable as a sampling artefact.
Likely causes: our 125-class label space is harder than theirs, and prompt/gallery construction
differs. Resolve before quoting these numbers anywhere.

---

## 11. Script reference

| script | what it does |
|---|---|
| `bench/run_baseline.py` | per-process baseline runs |
| `bench/batch_seeds.py` | Tier 1.1 batched-seeds harness (the main workhorse) |
| `bench/batch_images.py` | Tier 1.2 batch-across-images |
| `bench/profile_iter.py` | per-iteration phase breakdown (diffvg/CLIP fwd/bwd) |
| `bench/profile_diffvg.py` | diffvg per-kernel attribution |
| `bench/fast_serialize.py` | 0.4 — assert-free `serialize_scene`; `enable()` to opt in |
| `bench/tiled_render.py` | 0.5 — tiled multi-scene raster prototype |
| `bench/guardrails.py` | the 5 quality metrics + continuous companions |
| `bench/noise_floor.py` | run-to-run floor from two replicate directories |
| `bench/quality_vs_iter.py` | quality at every truncation point from stored SVG snapshots |
| `bench/analyze_p1.py` | early-stopping verdict + statistical power table |
| `bench/stopping_rule.py` | fixed vs adaptive stopping rules |
| `bench/verify_batched_equiv.py`, `verify_freeze_clip.py`, `verify_a2.py` | equivalence proofs |
| `bench/run_night.sh` | everything needing an idle GPU, guarded |
| `bench/run_wide_evalset.sh` | the 200-image protocol |

Key result files: `bench/results/quality_curve/` (noise floors, P1 verdict, stopping rules),
`bench/results/guardrails/` (per-arm quality), `bench/results/profile/` (phase breakdowns).

---

## 12. First things to do on the new server

1. `git clone git@github.com:daviidmparedes/CLIPasso-warp.git && git checkout experiments`
2. Copy `bench/results/`, `bench/logs/`, `data/` from the old server — **they are not in git**
3. `./setup.sh` (add `CUDA_HOME=... ALLOW_LOCAL_PYTHON_HEADERS=1` if no root)
4. Sanity check: `python bench/guardrails.py --runs bench/results/batched_freeze --tag sanity --strokes 16`
   → expect `loss_eval 0.55732`, 125-way **13.3%**, median rank **395**.
   If you get 0.0% and 669, the GPU is out of memory and diffvg is silently lying (§8).
5. Read `RESULTS.md`. Then do N2.
