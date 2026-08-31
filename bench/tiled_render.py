#!/usr/bin/env python3
"""
P2: can M sketches be rasterised in ONE diffvg call instead of M?

diffvg is now 60-70% of a batched iteration and has no batched entry point, so
bench/batch_seeds.py and bench/batch_images.py still loop `for i in range(M):
render(scene_i)`. Tier 1.2 measured that loop going *superlinear* -- 5 scenes cost
6.1x one scene -- which is the signature of per-call overhead, not per-pixel work:
diffvg allocates outside PyTorch's caching allocator, so every call pays raw
cudaMalloc/cudaFree, and cudaFree synchronises the device.

If that diagnosis is right, laying the M scenes out side by side on one larger
canvas and issuing a single render should collapse M allocation cycles into one.
This script tests the idea in the order that matters: correctness first (does the
tiled raster produce the same pixels and the same gradients?), then cost.

Tiles carry a gutter because CLIPasso never clamps control points to the canvas --
a stroke may wander outside its viewport, which a single-scene render simply clips
but a tiled render would draw into the neighbour. The gutter absorbs that, and the
correctness check below reports whatever bleed survives it.

  python bench/tiled_render.py --num-scenes 5 --strokes 16
  python bench/tiled_render.py --num-scenes 16 --strokes 16 --profile
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import common
from common import ROOT, env_fingerprint, make_args


def _grid(m):
    cols = int(math.ceil(math.sqrt(m)))
    return cols, int(math.ceil(m / cols))


def render_separate(painters, canvas, samples=2):
    """What the code does today: one diffvg call per scene."""
    import pydiffvg
    out = []
    for p in painters:
        scene = pydiffvg.RenderFunction.serialize_scene(canvas, canvas, p.shapes, p.shape_groups)
        img = pydiffvg.RenderFunction.apply(canvas, canvas, samples, samples, 0, None, *scene)
        out.append(img)
    return out


def render_tiled(painters, canvas, gutter=32, samples=2, device=None):
    """One diffvg call for every scene, then slice the result back apart.

    Each painter's control points are translated by a constant tile offset, so the
    gradient reaching points is unchanged -- d(points + c)/d(points) = I. Nothing
    else about the scene is touched.
    """
    import pydiffvg
    m = len(painters)
    cols, rows = _grid(m)
    pitch = canvas + gutter
    W, H = cols * pitch, rows * pitch

    shapes, groups = [], []
    for i, p in enumerate(painters):
        ox = float((i % cols) * pitch)
        oy = float((i // cols) * pitch)
        off = torch.tensor([ox, oy], device=p.shapes[0].points.device,
                           dtype=p.shapes[0].points.dtype)
        base = len(shapes)
        for s in p.shapes:
            shapes.append(pydiffvg.Path(
                num_control_points=s.num_control_points,
                points=s.points + off,               # <- keeps the graph intact
                is_closed=s.is_closed,
                stroke_width=s.stroke_width))
        for g in p.shape_groups:
            groups.append(pydiffvg.ShapeGroup(
                shape_ids=g.shape_ids + base,
                fill_color=g.fill_color,
                use_even_odd_rule=g.use_even_odd_rule,
                stroke_color=g.stroke_color,
                shape_to_canvas=g.shape_to_canvas))

    scene = pydiffvg.RenderFunction.serialize_scene(W, H, shapes, groups)
    big = pydiffvg.RenderFunction.apply(W, H, samples, samples, 0, None, *scene)

    out = []
    for i in range(m):
        y0 = (i // cols) * pitch
        x0 = (i % cols) * pitch
        out.append(big[y0:y0 + canvas, x0:x0 + canvas, :])
    return out, big


def to_rgb(img, device):
    a = img[:, :, 3:4]
    rgb = a * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device=device) * (1 - a)
    return rgb.permute(2, 0, 1).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-scenes", type=int, default=5)
    ap.add_argument("--strokes", type=int, default=16)
    ap.add_argument("--gutter", type=int, default=32)
    ap.add_argument("--samples", type=int, default=2, help="num_samples_x/y (repo uses 2)")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--profile", action="store_true", help="count kernels per approach")
    ap.add_argument("--out", default=str(ROOT / "bench" / "results" / "tiled"))
    a = ap.parse_args()

    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    import pydiffvg

    ev = json.loads((ROOT / "data" / "eval_set.json").read_text())
    work = outdir / "_work"; work.mkdir(parents=True, exist_ok=True)

    # Build M independent painters, cycling the eval images so scenes differ.
    painters = []
    for i in range(a.num_scenes):
        tgt = ev[i % len(ev)]["path"]
        d = work / f"scene{i}"; d.mkdir(parents=True, exist_ok=True)
        args = make_args(tgt, d, f"tile{i}", num_paths=a.strokes, seed=1000 * i,
                         num_iter=1, save_interval=10 ** 9, eval_interval=10 ** 9)
        import painterly_rendering as pr
        inputs, mask = pr.get_target(args)
        r = pr.load_renderer(args, inputs, mask)
        r.set_random_noise(0)
        r.init_image(stage=0)
        r.parameters()                      # marks points as requiring grad
        painters.append(r)
    canvas = painters[0].canvas_width
    cols, rows = _grid(a.num_scenes)
    print(f"{a.num_scenes} scenes x {a.strokes} strokes, canvas {canvas}, "
          f"grid {cols}x{rows}, gutter {a.gutter} -> "
          f"{cols*(canvas+a.gutter)}x{rows*(canvas+a.gutter)}")

    # ---------------------------------------------------------- correctness
    # An absolute epsilon is the wrong gate. diffvg jitters its sub-pixel samples
    # by absolute pixel index, so enlarging the canvas re-indexes the jitter and
    # stroke edges land on different sample patterns -- with 2x2 sampling a single
    # edge pixel can move by up to 3/4. That is not an error in the tiling; it is
    # the renderer's own sampling noise, and it is bounded by the same quantity as
    # re-running the identical scene with a different diffvg seed. So the gate is:
    # is separate-vs-tiled any worse than seed-vs-seed on an unchanged scene?
    # (Verified separately: with gutter=0 and M=1, tiled output is bit-identical.)
    import pydiffvg as _pdv
    sc = _pdv.RenderFunction.serialize_scene(canvas, canvas,
                                             painters[0].shapes, painters[0].shape_groups)
    ref_a = _pdv.RenderFunction.apply(canvas, canvas, a.samples, a.samples, 0, None, *sc)
    sc = _pdv.RenderFunction.serialize_scene(canvas, canvas,
                                             painters[0].shapes, painters[0].shape_groups)
    ref_b = _pdv.RenderFunction.apply(canvas, canvas, a.samples, a.samples, 7, None, *sc)
    seed_max = float((ref_a - ref_b).abs().max())
    seed_mean = float((ref_a - ref_b).abs().mean())

    sep = render_separate(painters, canvas, a.samples)
    til, _ = render_tiled(painters, canvas, a.gutter, a.samples, device)
    diffs = [float((s - t).abs().max()) for s, t in zip(sep, til)]
    means = [float((s - t).abs().mean()) for s, t in zip(sep, til)]
    print(f"\nrenderer's own sampling noise (same scene, diffvg seed 0 vs 7):")
    print(f"    max {seed_max:.4f}   mean {seed_mean:.2e}")
    print(f"separate vs tiled:")
    print(f"    max {max(diffs):.4f}   mean {np.mean(means):.2e}")

    # gradients: push the same upstream signal through both paths
    g_sep, g_til = [], []
    for mode in ("sep", "til"):
        for p in painters:
            for s in p.shapes:
                if s.points.grad is not None:
                    s.points.grad = None
        imgs = (render_separate(painters, canvas, a.samples) if mode == "sep"
                else render_tiled(painters, canvas, a.gutter, a.samples, device)[0])
        loss = sum((to_rgb(im, device) ** 2).sum() for im in imgs)
        loss.backward()
        store = g_sep if mode == "sep" else g_til
        for p in painters:
            store.append(torch.cat([s.points.grad.flatten() for s in p.shapes]).clone())
    gd = [float((x - y).abs().max()) for x, y in zip(g_sep, g_til)]
    gn = [float(x.abs().max()) for x in g_sep]
    cos = [float(torch.nn.functional.cosine_similarity(x, y, dim=0))
           for x, y in zip(g_sep, g_til)]

    # Same baseline as the pixel check: how much does the gradient move when only
    # the diffvg seed changes? The sampling noise sits on stroke edges, which is
    # exactly where the gradient lives, so an absolute threshold would reject the
    # renderer's own reruns too.
    #
    # The baseline is computed on EVERY scene, not just the first. Comparing a
    # minimum over M scenes against a single-sample baseline is not a comparison:
    # the minimum of a noisy statistic drifts down as M grows, so a large M would
    # look like a regression purely as an order-statistic effect. Both sides are
    # summarised the same way -- worst case against worst case, median against median.
    seed_g = [[], []]
    for k, sd in enumerate((0, 7)):
        for p in painters:
            for sh in p.shapes:
                sh.points.grad = None
        for p in painters:
            sc = _pdv.RenderFunction.serialize_scene(canvas, canvas, p.shapes, p.shape_groups)
            im = _pdv.RenderFunction.apply(canvas, canvas, a.samples, a.samples,
                                           sd, None, *sc)
            (to_rgb(im, device) ** 2).sum().backward()
        for p in painters:
            seed_g[k].append(torch.cat([sh.points.grad.flatten()
                                        for sh in p.shapes]).clone())
    seed_cos = [float(torch.nn.functional.cosine_similarity(x, y, dim=0))
                for x, y in zip(seed_g[0], seed_g[1])]
    g_seed_cos, g_seed_med = min(seed_cos), float(np.median(seed_cos))
    print(f"gradient cosine        worst scene    median")
    print(f"  separate vs tiled    {min(cos):.6f}    {np.median(cos):.6f}")
    print(f"  seed 0 vs seed 7     {g_seed_cos:.6f}    {g_seed_med:.6f}"
          f"   <- renderer's own noise")

    # Pass if the pixel difference is no larger than the renderer's own seed noise
    # (2x margin) and the gradients still point the same way.
    # Pass if tiling perturbs pixels and gradients no more than a seed change does,
    # comparing like summaries with like.
    ok = (max(diffs) <= 2 * seed_max
          and min(cos) >= g_seed_cos - 0.02
          and float(np.median(cos)) >= g_seed_med - 0.02)
    print(f"correctness: {'PASS' if ok else 'FAIL'} "
          f"(tiled perturbs no more than the renderer perturbs itself)")

    # ---------------------------------------------------------------- cost
    def bench_paired(fn_a, fn_b):
        """Interleaved A/B timing on a shared GPU.

        Two things matter when another job owns most of the device. First, a
        contending kernel can only ever make a rep slower, never faster, so the
        minimum over many reps is the best available estimate of the uncontended
        cost -- the mean is unbounded above and says more about the neighbour than
        about the code. Second, timing all of A and then all of B lets a burst of
        contention land entirely on one arm and show up as a speedup; alternating
        A and B within each rep forces both to see the same interference.

        Returns per-arm stats plus the distribution of the per-rep paired ratio,
        which is the honest comparison: its median is robust to bursts that the
        min would only see if they happened to spare one arm.
        """
        for _ in range(a.warmup):
            fn_a(); fn_b()
        torch.cuda.synchronize()
        ta, tb = [], []
        for _ in range(a.reps):
            t0 = time.perf_counter(); fn_a(); torch.cuda.synchronize()
            t1 = time.perf_counter(); fn_b(); torch.cuda.synchronize()
            t2 = time.perf_counter()
            ta.append((t1 - t0) * 1e3); tb.append((t2 - t1) * 1e3)
        ta, tb = np.array(ta), np.array(tb)
        ratio = ta / tb
        return ({"min": float(ta.min()), "median": float(np.median(ta)),
                 "mean": float(ta.mean())},
                {"min": float(tb.min()), "median": float(np.median(tb)),
                 "mean": float(tb.mean())},
                {"min_ratio": float(ta.min() / tb.min()),
                 "median_paired_ratio": float(np.median(ratio)),
                 "p25_paired_ratio": float(np.percentile(ratio, 25)),
                 "p75_paired_ratio": float(np.percentile(ratio, 75))})

    def fwd_sep():
        render_separate(painters, canvas, a.samples)

    def fwd_til():
        render_tiled(painters, canvas, a.gutter, a.samples, device)

    def fb(mode):
        for p in painters:
            for s in p.shapes:
                s.points.grad = None
        imgs = (render_separate(painters, canvas, a.samples) if mode == "sep"
                else render_tiled(painters, canvas, a.gutter, a.samples, device)[0])
        (sum((to_rgb(im, device) ** 2).sum() for im in imgs)).backward()

    fsep, ftil, fr = bench_paired(fwd_sep, fwd_til)
    bsep, btil, br = bench_paired(lambda: fb("sep"), lambda: fb("til"))
    res = {
        "num_scenes": a.num_scenes, "strokes": a.strokes, "gutter": a.gutter,
        "samples": a.samples, "canvas": canvas, "grid": [cols, rows],
        "max_fwd_diff": max(diffs), "mean_fwd_diff": float(np.mean(means)),
        "seed_noise_max": seed_max, "seed_noise_mean": seed_mean,
        "max_grad_diff": max(gd), "grad_magnitude": max(gn),
        "min_grad_cosine": min(cos), "median_grad_cosine": float(np.median(cos)),
        "seed_grad_cosine_min": g_seed_cos, "seed_grad_cosine_median": g_seed_med,
        "correct": bool(ok),
        "fwd_separate_ms": fsep, "fwd_tiled_ms": ftil, "fwd_ratio": fr,
        "fwdbwd_separate_ms": bsep, "fwdbwd_tiled_ms": btil, "fwdbwd_ratio": br,
        "free_gpu_gb": common.require_free_gpu_memory(),
        "env": env_fingerprint(),
    }
    res["fwd_speedup"] = fr["median_paired_ratio"]
    res["fwdbwd_speedup"] = br["median_paired_ratio"]

    def row(label, sep, til, r):
        print(f"  {label:<12} separate {sep['min']:7.2f} ms   tiled {til['min']:7.2f} ms"
              f"   min-ratio {r['min_ratio']:5.2f}x   "
              f"paired median {r['median_paired_ratio']:5.2f}x "
              f"[{r['p25_paired_ratio']:.2f}-{r['p75_paired_ratio']:.2f}]")
    print()
    row("forward", fsep, ftil, fr)
    row("forward+bwd", bsep, btil, br)
    print(f"  per sketch   separate {bsep['min']/a.num_scenes:7.2f} ms   "
          f"tiled {btil['min']/a.num_scenes:7.2f} ms   (min-based)")
    spread = bsep["median"] / bsep["min"]
    if spread > 1.3:
        print(f"  NOTE: median/min = {spread:.1f} on the separate arm -- the GPU is shared.")
        print(f"        Trust the paired median ratio; the absolute ms are upper bounds.")

    if a.profile:
        from torch.profiler import ProfilerActivity, profile
        from torch.autograd import DeviceType
        for name, fn in (("separate", lambda: fb("sep")), ("tiled", lambda: fb("til"))):
            fn()
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                fn()
                torch.cuda.synchronize()
            evs = [e for e in prof.key_averages() if e.device_type == DeviceType.CUDA]
            n = sum(e.count for e in evs)
            dt = sum(e.self_device_time_total for e in evs) / 1e3
            res[f"{name}_kernels"] = int(n)
            res[f"{name}_gpu_ms"] = float(dt)
            print(f"  {name:<10} {n:6d} kernels   {dt:8.2f} ms on-GPU")
        if res.get("separate_kernels"):
            print(f"  kernel reduction: "
                  f"{res['separate_kernels']/max(res['tiled_kernels'],1):.2f}x")

    fp = outdir / f"tiled_M{a.num_scenes}_n{a.strokes}_s{a.samples}.json"
    fp.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {fp}")
    print("NOTE: another job may be sharing this GPU -- check before trusting timings.")


if __name__ == "__main__":
    main()
