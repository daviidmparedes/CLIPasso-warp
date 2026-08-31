#!/usr/bin/env python3
"""
Kernel-level breakdown of diffvg's forward and backward (project plan P2).

diffvg's backward costs about 2x its forward, which the plan flagged as possibly a
fixable inefficiency rather than intrinsic cost. This attributes both halves to
individual CUDA kernels so the question can be answered from evidence.

It also answers the structural question the tiling experiment raised: does diffvg's
kernel count scale with the number of *render calls* or with the number of *shapes*?
The tiling result says shapes, which is why merging M scenes into one raster removes
no launches. Sweeping the stroke count here confirms it directly.

Kernel timings are meaningful under GPU contention only in relative terms; kernel
*counts* are exact regardless. Both are reported, counts first.

  python bench/profile_diffvg.py --strokes 16
  python bench/profile_diffvg.py --sweep 4,8,16,32,64
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

import common
from common import ROOT, env_fingerprint, make_args


def build_painter(target, outdir, strokes, seed=0):
    import painterly_rendering as pr
    args = make_args(target, outdir, "prof", num_paths=strokes, seed=seed,
                     num_iter=1, save_interval=10 ** 9, eval_interval=10 ** 9)
    inputs, mask = pr.get_target(args)
    r = pr.load_renderer(args, inputs, mask)
    r.set_random_noise(0)
    r.init_image(stage=0)
    r.parameters()
    return r


def kernels_for(fn, warmup=3):
    """Named CUDA kernels launched by fn, with counts and self time."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    agg = defaultdict(lambda: [0, 0.0])
    for e in prof.key_averages():
        # CPU ops carry attributed device time; summing everything double-counts.
        if e.device_type != DeviceType.CUDA:
            continue
        agg[e.key][0] += e.count
        agg[e.key][1] += e.self_device_time_total / 1e3
    return {k: {"count": v[0], "ms": v[1]} for k, v in agg.items()}


def summarise(name, ks, top=8):
    total_n = sum(v["count"] for v in ks.values())
    total_ms = sum(v["ms"] for v in ks.values())
    print(f"\n  {name}: {total_n} kernels, {total_ms:.2f} ms on-GPU")
    rows = sorted(ks.items(), key=lambda kv: -kv[1]["ms"])[:top]
    for k, v in rows:
        short = k if len(k) <= 58 else k[:55] + "..."
        print(f"    {short:<58} {v['count']:5d} x   {v['ms']:7.3f} ms "
              f"({100*v['ms']/max(total_ms,1e-9):4.1f}%)")
    return total_n, total_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strokes", type=int, default=16)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--sweep", default=None, help="comma-separated stroke counts")
    ap.add_argument("--out", default=str(ROOT / "bench" / "results" / "diffvg"))
    a = ap.parse_args()

    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    common.require_free_gpu_memory()
    import pydiffvg

    ev = json.loads((ROOT / "data" / "eval_set.json").read_text())
    target = ev[0]["path"]
    work = outdir / "_work"; work.mkdir(parents=True, exist_ok=True)

    counts = [int(x) for x in a.sweep.split(",")] if a.sweep else [a.strokes]
    res = {"samples": a.samples, "sweep": {}, "env": env_fingerprint()}

    for n in counts:
        r = build_painter(target, work, n)
        canvas = r.canvas_width

        def fwd():
            sc = pydiffvg.RenderFunction.serialize_scene(canvas, canvas, r.shapes, r.shape_groups)
            return pydiffvg.RenderFunction.apply(canvas, canvas, a.samples, a.samples,
                                                 0, None, *sc)

        def fwd_bwd():
            for s in r.shapes:
                s.points.grad = None
            img = fwd()
            (img ** 2).sum().backward()

        print("=" * 78)
        print(f"n = {n} strokes, canvas {canvas}, {a.samples}x{a.samples} samples")
        kf = kernels_for(fwd)
        kfb = kernels_for(fwd_bwd)
        nf, mf = summarise("forward", kf)
        nfb, mfb = summarise("forward + backward", kfb)
        # backward = the difference between the two passes
        nb, mb = nfb - nf, mfb - mf
        print(f"\n  => forward  {nf:5d} kernels  {mf:7.3f} ms")
        print(f"     backward {nb:5d} kernels  {mb:7.3f} ms   "
              f"({mb/max(mf,1e-9):.2f}x the forward, "
              f"{nb/max(nf,1):.2f}x its kernels)")
        print(f"     per stroke: {nfb/n:.1f} kernels")
        res["sweep"][n] = {"fwd_kernels": nf, "fwd_ms": mf,
                           "bwd_kernels": nb, "bwd_ms": mb,
                           "kernels_per_stroke": nfb / n,
                           "fwd_top": dict(sorted(kf.items(), key=lambda kv: -kv[1]["ms"])[:10]),
                           "fwdbwd_top": dict(sorted(kfb.items(), key=lambda kv: -kv[1]["ms"])[:10])}

    if len(counts) > 1:
        print("\n" + "=" * 78)
        print("does the kernel count scale with shapes?")
        print(f"{'strokes':>9}{'fwd kern':>10}{'bwd kern':>10}{'kern/stroke':>13}"
              f"{'fwd ms':>9}{'bwd ms':>9}")
        for n in counts:
            d = res["sweep"][n]
            print(f"{n:>9}{d['fwd_kernels']:>10}{d['bwd_kernels']:>10}"
                  f"{d['kernels_per_stroke']:>13.1f}{d['fwd_ms']:>9.3f}{d['bwd_ms']:>9.3f}")

    fp = outdir / f"diffvg_kernels_s{a.samples}.json"
    fp.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {fp}")


if __name__ == "__main__":
    main()
