# CLIPasso Speed-Up Project — Session Brief

**Purpose of this file:** hand-off document to start a fresh Claude session focused on *reducing the wall-clock cost of CLIPasso's test-time optimization* without destroying sketch quality. Pair this file with `CLIPasso Explained: Test-Time Optimization of Bezier Strokes with CLIP-Guided Perceptual Loss` (the technical report from the previous session). That report is the reference for *how the method works*; this file is the reference for *what we are changing and why*.

---

## 0. Who / where / what

- **Researcher:** full-time CV lab researcher, CS background. No need to simplify explanations. Assume comfort with PyTorch internals, autograd, CUDA profiling, and vector graphics.
- **Repo:** official `yael-vinker/CLIPasso`, already cloned on the server.
- **Test images:** `/image_test` (path as given by the user — confirm the absolute path in the first session turn).
- **Goal:** make CLIPasso faster. Secondary goal: understand *which* part of the cost is actually buying quality.
- **Non-goal (for now):** improving sketch aesthetics for its own sake. Quality only matters as a constraint we must not violate.

### First actions for a new session

1. Confirm the repo path, the image folder path, GPU model, driver/CUDA version, PyTorch version, and whether `pydiffvg` imports cleanly.
2. Run the unmodified baseline on 3–5 images from `/image_test` and record timings. **Do not change anything before there is a baseline.**
3. Profile. Do not accept the cost model in §2 on faith — measure it.

---

## 1. The core insight driving this work

CLIPasso is per-image (test-time) optimization: no trained network, just Adam on Bézier control points against a frozen CLIP perceptual loss. The naive framing is "every quality improvement costs optimization time." That framing is wrong in a useful way:

> Initialization quality, loss design, and encoder choice do not merely trade against optimization time — they **determine** it. A better init doesn't just find a better local minimum, it reaches it in 200 steps instead of 2000. Speed and quality share the same lever.

So the question is not "how do I go faster despite the optimization," it's **"which parts of the 3 seeds × 2000 iterations budget are actually buying anything?"**

Two facts from the paper that motivate everything below:

- Figure 7 shows a recognizable sketch by **iteration ~100**. Iterations 300→2000 are asymptotic polish.
- The stopping rule is `min_delta = 1e-5` on the **raw loss**, i.e. it measures *loss saturation*, not *perceptual saturation*. These are very different points on the curve.

And one structural observation:

- At 16 strokes there are only **128 free parameters** (16 strokes × 4 control points × 2 coords). We are running 2000 Adam steps on a 128-dimensional problem. The dimensionality is trivial; **cost per gradient evaluation is the entire problem.**

---

## 2. Cost model (hypothesis — verify by profiling)

```
T = n_seeds × n_iters × [ diffvg_fwd + diffvg_bwd
                        + CLIP_fwd( 2·(1+A) images )
                        + CLIP_bwd( sketch branch only ) ]
```

Defaults: `n_seeds = 3`, `n_iters ≈ 2001`, `A = num_aug_clip = 4`, canvas 224², RN101 geometric loss on layers 3+4, `w_s = clip_fc_loss_weight = 0.1`.

Suspected inefficiencies:

- **The target branch is recomputed ~2000× for an image that never changes.** It only changes because the same random affine is applied to both sketch and target. That is a design choice, not a necessity.
- **diffvg re-serializes the scene in Python every iteration** (`serialize_scene` rebuilds shape lists each step). At n=16 the rasterization itself is cheap; Python overhead + kernel-launch latency may dominate the vector side.
- **Three seeds are three separate processes** over the same target, each under-utilizing the GPU.

**Profiling protocol (do this first):** `torch.profiler` over ~50 iterations, split into: diffvg-fwd / diffvg-bwd / CLIP-fwd / CLIP-bwd / Python-and-launch overhead. Also record GPU utilization and memory. The right strategy differs a lot depending on whether CLIP or diffvg dominates. Prior: at n=16, CLIP + launch overhead dominates.

---

## 3. Experiment backlog, in priority order

### Tier 1 — free speedups, zero quality cost (do these first)

| # | Idea | Mechanism | Expected |
|---|---|---|---|
| 1.1 | **Batch the seeds** | Render all 3 scenes in one loop, concatenate into a single CLIP batch instead of 3 processes | ~2–2.5×, bit-identical output |
| 1.2 | **Batch across images** | For dataset generation, run 16–32 targets concurrently with a batched CLIP forward | 10–20× throughput |
| 1.3 | **AMP + channels_last + `torch.compile`** | Compile the **CLIP branch only** — diffvg is a custom autograd op and will graph-break. Static shapes make CUDA graphs viable, killing launch overhead | 1.5–2× |
| 1.4 | **Cache the augmented target** | Pre-sample a bank of M fixed affines (M ≈ 64–256), precompute CLIP features for all M augmented target views once; each iteration sample 4 from the bank and apply the matching transform to the sketch | ~1.4–1.8× + enables 2.4 |

Stacked, Tier 1 is plausibly **4–8× with no change to output quality.** Most of this is engineering, not research.

### Tier 2 — fewer iterations, same objective

| # | Idea | Mechanism | Expected |
|---|---|---|---|
| 2.1 | **Fix the stopping criterion** | Replace loss-delta with perceptual saturation: stop when mean control-point displacement over the last 50 iters < ~0.5 px, or when the semantic term plateaus independently of the geometric term | up to ~5× if 400 iters ≈ 98% of quality |
| 2.2 | **Coarse-to-fine** | First few hundred iters at 112² (RN101 is fine with interpolated attnpool positional embeddings), finish at 224². Conv cost ~quadratic in resolution | early iters ~4× cheaper |
| 2.3 | **Progressive stroke addition** | Optimize 4 strokes → converge → add 4 more initialized on the *residual* saliency → repeat. Fewer params early = faster convergence, and you get the whole 4/8/16/32 ladder from one run instead of four | ~4× when the ladder is needed anyway |
| 2.4 | **L-BFGS instead of Adam** | 128 params is ideal for quasi-Newton. Stochastic augmentations normally break curvature estimates and line search — **but the fixed augmentation bank from 1.4 makes the objective deterministic**, which makes L-BFGS legitimate | potentially large; **most under-explored idea here, cheap to test** |
| 2.5 | **Structure-aware initialization** | Instead of sampling points from saliency and jittering the other 3 control points within `radius = 0.05`, extract contour fragments from the saliency-weighted XDoG map, cluster them, and **least-squares fit a cubic Bézier per fragment**. Strokes start *along* structure, not as squiggles near it | fewer iters **and** lower seed variance → possibly drop to 1–2 seeds (compounding) |

### Tier 3 — change the teacher (riskier)

| # | Idea | Mechanism | Risk |
|---|---|---|---|
| 3.1 | **Distill the guidance network** | Train a small student CNN offline to match RN101 layer-3/4 feature maps on the sketch+photo distribution (generate that distribution with CLIPasso itself). Per-image optimization then uses a 5–10× cheaper encoder | Distribution shift: strokes pass through weird intermediate states early. Mitigate by including intermediate optimization snapshots in the distillation set |
| 3.2 | **Warp features instead of images** | For the geometric term, augment-then-encode ≈ encode-then-warp, since conv features are roughly covariant to translation/scale. Replaces a CLIP forward with a `grid_sample` on a cached feature map | Speculative. **Cheap validation:** measure the L2 gap between the two paths over a few hundred random transforms *before* building anything on it |

### Tier 4 — stop optimizing at test time (the paper)

- Train a feed-forward predictor image → control points, supervised on a corpus of CLIPasso outputs, then fine-tuned **end-to-end through diffvg with the same CLIP loss** (the pipeline is differentiable, so the amortized optimizer can be trained against the true objective, not just distilled targets). This is what SwiftSketch does.
- **Preferred framing — the hybrid:** use the amortized predictor purely as an **initializer**, then run 50–200 refinement steps of the real objective. Keeps everything test-time optimization buys (arbitrary categories, arbitrary stroke counts, no dataset dependence) while starting inside the right basin. Cleanest contribution framing: *amortization as initialization, not replacement.*
- Further out: meta-learned initialization (MAML-style) or a learned optimizer with per-parameter step sizes so k≈50 steps suffice; condition on `n` via a hypernetwork so one model covers the whole abstraction ladder.

### Expected cumulative outcome

| Track | Realistic speedup | Quality cost |
|---|---|---|
| Tier 1 engineering | 4–8× | none |
| + Tier 2 | 15–40× cumulative | small, measurable |
| + Tier 3 distilled teacher | 30–80× | needs validation |
| Tier 4 amortized + refine | ~100× to interactive | small if refinement kept |

---

## 4. The question that reorders all of this

**What is the actual bottleneck?**

- *"I need 10,000 sketches for a benchmark"* → per-image latency is irrelevant, GPU utilization is everything. Do **1.2** and stop. None of the algorithmic work matters.
- *"I want an interactive sketching tool"* → only **Tier 4** gets there. Tiers 1–2 are necessary but not sufficient.
- *"I want a paper on efficient vector sketch synthesis"* → Tiers 1–2 are the baseline hygiene; Tier 4 hybrid is the contribution; Tier 3 is the risky middle.

Resolve this in the first session turn. It changes the priority order completely.

---

## 5. Quality guardrails — how we prove we didn't break it

Every speedup must be reported as a **(speedup, quality delta)** pair. Fixed evaluation set: the same 5–10 images from `/image_test`, same seeds, same stroke counts (4 / 8 / 16 / 32).

Metrics:

1. **Final eval loss** (the un-augmented `loss_eval` CLIPasso already computes) — primary, cheap, directly comparable.
2. **CLIP ViT-B/32 zero-shot classification** of the output sketch with prompt `"A sketch of a(n) {class}"` — the paper's own metric. Note ViT-B/32 is *not* the RN101 used for the loss, so this is a semi-independent check.
3. **Control-point trajectory divergence** vs. the baseline run — catches silent behavioral changes that leave the loss unchanged.
4. **Visual side-by-side contact sheet** — non-negotiable. Loss numbers hide perceptual failures, especially at 4 strokes.

Reference numbers from the paper: CLIP ViT-B/32 top-1 ≈ **78% @ 16 strokes**, **91% @ 32 strokes**. Category-level human recognition collapses to **36% at 4 strokes** — the documented "breaking point." Expect 4-stroke results to be noisy and unreliable as a quality signal; weight 8/16/32 more heavily.

---

## 6. Files that matter

| File | Role | Relevance |
|---|---|---|
| `run_object_sketching.py` | Spawns one process per seed, then picks the best via `min(loss_eval)` from each run's `config.npy` | **Target of 1.1, 1.2** |
| `painterly_rendering.py` | `main(args)` — the optimization loop, `save_interval`/`eval_interval = 10`, `min_delta = 1e-5` early stop | **Target of 2.1, 2.2, 2.3** |
| `models/painter_params.py` | `Painter` (stroke construction, `get_path()`, `radius = 0.05` jitter, `init_image()`, `render_warp()`/`get_image()`, `save_svg()`), `PainterOptimizer` (single Adam, `lr=1.0`, `betas=(0.9,0.9)`, `eps=1e-6`), and the ViT relevancy `interpret()` | **Target of 2.4, 2.5** |
| `models/loss.py` | `Loss` dispatcher + `CLIPConvLoss` — augmentations, RN101 layer hooks, `forward_inspection_clip_resnet` | **Target of 1.3, 1.4, 3.1, 3.2** |
| `config.py` | All defaults | Read first; treat as ground truth |
| `U2Net_/` + `utils.get_mask_u2net` | Background masking (`--mask_object 1`) | One-time cost, low priority |

Key defaults to keep in mind: `--lr 1.0`, `--num_paths 16`, `--width 1.5`, `--control_points_per_seg 4`, `--num_segments 1`, `--num_aug_clip 4`, `--augment_both 1`, `--aug_scale_min 0.7`, `--clip_model_name RN101`, `--clip_conv_layer_weights 0,0,1.0,1.0,0`, `--clip_conv_loss_type L2`, `--clip_fc_loss_weight 0.1`, `--softmax_temp 0.3`, `--saliency_clip_model ViT-B/32`, `--xdog_intersec 1`, `--image_scale 224`. Runner overrides `--num_iter` to 2001 and `--num_sketches` to 3.

---

## 7. Known environment gotchas

- The repo targets **Python 3.7 / PyTorch 1.7.1+cu101 / torchvision 0.8.2+cu101**. The authors ship a Docker image (`yaelvinker/clipasso_docker`) — prefer it if the local build fights back.
- **diffvg is the usual failure point.** Dominant error: CUDA version mismatch between the CUDA that built PyTorch and the local toolkit. Also `ninja` build errors and, on Windows, `ptxas fatal: Unresolved extern function`. Build on Linux. **Verify `import pydiffvg` works before touching any CLIPasso code.**
- Modernizing the stack (newer PyTorch for `torch.compile` / AMP) may require patching diffvg. Budget time for this; it is a prerequisite for 1.3 and possibly a blocker.
- `u2net.pth` is fetched from Google Drive via `gdown` on first run — it may fail silently on a headless server. Pre-download it.
- One caveat carried over from the technical report: the exact body of `models/loss.py` was never fetched verbatim in the prior session. The hyperparameters are confirmed from `config.py`, but **the precise `RandomResizedCrop` scale tuple and the per-layer L2 reduction should be read directly from the source file** before building 1.4 or 3.2 on top of them.

---

## 8. Working agreement for the new session

- **Measure before changing.** No optimization lands without a before/after timing on the same images and seeds.
- **One change at a time**, each on its own git branch, each with its own (speedup, quality delta) row in a running results table.
- **Report negative results.** "L-BFGS diverged because X" is as valuable as a win and prevents re-litigating it later.
- Prefer **small reproducible scripts** in a `bench/` folder over edits scattered through the repo, until an idea proves out.
- Keep a `RESULTS.md` table updated as the single source of truth for what has been tried.

---

## 9. Starter prompt for the new session

Copy-paste this to open the next conversation (attach this file and the CLIPasso technical report alongside it):

> I'm a CV researcher working on speeding up CLIPasso's test-time optimization. I've attached two documents: a deep technical explanation of how CLIPasso works, and a project brief laying out the speedup ideas we developed (cost model, tiered experiment backlog, quality guardrails).
>
> Setup: the official `yael-vinker/CLIPasso` repo is cloned on my server, and I have test images in `/image_test`. I have GPU access.
>
> I want to start executing the plan in the brief. Please begin at the top: help me (1) verify the environment — `pydiffvg` imports, PyTorch/CUDA versions, GPU model; (2) establish a clean baseline by running unmodified CLIPasso on a few images from `/image_test` and recording per-image wall-clock, per-seed timing, and final `loss_eval`; and (3) write a `torch.profiler` harness that runs ~50 iterations and breaks the per-iteration cost into diffvg-fwd, diffvg-bwd, CLIP-fwd, CLIP-bwd, and Python/launch overhead.
>
> Before we start, ask me anything you need about my setup — and in particular, help me settle the §4 question in the brief (batch dataset generation vs. interactive latency vs. a paper contribution), because the brief says that reorders the whole priority list. Don't skip ahead to the optimizations until we have real profile numbers; I want to know whether CLIP or diffvg actually dominates at n=16 strokes before we commit to a direction.
>
> Give me actual runnable code for the benchmark and profiling harness, put it in a `bench/` folder, and set up a `RESULTS.md` table so we can track every (speedup, quality delta) pair as we go.
