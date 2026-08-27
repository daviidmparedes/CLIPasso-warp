#!/usr/bin/env python3
"""
Per-iteration cost breakdown for CLIPasso's test-time optimisation loop.

Splits one optimisation step into diffvg-fwd / diffvg-bwd / CLIP-fwd / CLIP-bwd /
optimiser / Python-and-launch overhead, and answers the question the project brief
puts first: at n=16 strokes, is this workload CLIP-bound or diffvg-bound?

Three measurement passes, because no single one is trustworthy on its own:

  1. FUSED   - the real iteration, no intra-iteration syncs. Ground truth for
               wall-clock. Everything else is validated against this.
  2. SPLIT   - the same iteration with torch.cuda.synchronize() at phase
               boundaries. Gives clean attribution, but the syncs serialise
               CPU/GPU overlap, so the total runs slower than FUSED. The gap
               between them is itself reported (it is the pipelining benefit).
  3. PROFILE - torch.profiler over the fused loop. Gives GPU kernel time and
               kernel count per iteration. Python/launch overhead is then
               FUSED wall-clock minus GPU busy time.

The backward pass is split exactly, not estimated: the graph is a chain
points -> diffvg -> img -> CLIP -> loss, so autograd.grad(loss, img) is precisely
the CLIP backward and img.backward(g) is precisely the diffvg backward.
"""
import argparse
import contextlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

import common
from common import ROOT, GpuSampler, build_pipeline, env_fingerprint, make_args, sync


class PhaseTimer:
    """Accumulates per-iteration, per-phase durations (ms).

    Repeated calls to the same phase inside one iteration are summed, which is
    what we want for e.g. the 4 augmentation calls or the 2 encoder calls.
    """

    def __init__(self):
        self.cur = {}
        self.rows = []

    @contextlib.contextmanager
    def __call__(self, name, gpu=True):
        if gpu:
            sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if gpu:
                sync()
            self.cur[name] = self.cur.get(name, 0.0) + (time.perf_counter() - t0) * 1e3

    def flush(self):
        self.rows.append(self.cur)
        self.cur = {}

    def stats(self):
        keys = sorted({k for r in self.rows for k in r})
        return {k: {
            "mean_ms": float(np.mean([r.get(k, 0.0) for r in self.rows])),
            "median_ms": float(np.median([r.get(k, 0.0) for r in self.rows])),
            "std_ms": float(np.std([r.get(k, 0.0) for r in self.rows])),
        } for k in keys}


class _Timed:
    """Wraps a callable so its runtime lands in a PhaseTimer bucket."""

    def __init__(self, fn, pt, name, gpu=True):
        self.fn, self.pt, self.name, self.gpu = fn, pt, name, gpu

    def __call__(self, *a, **kw):
        with self.pt(self.name, gpu=self.gpu):
            return self.fn(*a, **kw)


def instrument_loss(loss_func, pt):
    """Sub-split CLIPConvLoss.forward without touching repo source.

    Isolates the target-branch encoder call specifically, because the brief's
    idea 1.4 (cache the augmented target) is worth exactly that number and no more.
    """
    cl = loss_func.loss_mapper["clip_conv_loss"]
    cl.normalize_transform = _Timed(cl.normalize_transform, pt, "clip_fwd.normalize")
    cl.augment_trans = _Timed(cl.augment_trans, pt, "clip_fwd.augment")

    orig_enc = cl.forward_inspection_clip_resnet
    state = {"n": 0}

    def enc(x):
        # forward() calls this for the sketch batch first, then the target batch.
        name = "clip_fwd.encode_sketch" if state["n"] % 2 == 0 else "clip_fwd.encode_target"
        state["n"] += 1
        with pt(name):
            return orig_enc(x)

    cl.forward_inspection_clip_resnet = enc
    return cl


def instrument_diffvg(pt):
    """Separate diffvg's pure-Python scene rebuild from the actual rasterisation."""
    import pydiffvg
    orig = pydiffvg.RenderFunction.serialize_scene

    def timed(*a, **kw):
        # CPU-only work: do NOT sync, or we charge it for pending GPU work.
        with pt("diffvg_fwd.serialize_scene", gpu=False):
            return orig(*a, **kw)

    pydiffvg.RenderFunction.serialize_scene = staticmethod(timed)
    return orig


def one_iteration(renderer, loss_func, optimizer, inputs, epoch, pt=None, split=False):
    """A single optimisation step, matching painterly_rendering.main()'s inner loop.

    split=True inserts phase boundaries; split=False runs it fused (ground truth).
    """
    renderer.set_random_noise(epoch)
    if not split:
        optimizer.zero_grad_()
        img = renderer.get_image()
        losses = loss_func(img, inputs.detach(), renderer.get_color_parameters(),
                           renderer, epoch, optimizer)
        loss = sum(losses.values())
        loss.backward()
        optimizer.step_()
        return float(loss.item())

    with pt("optim.zero_grad"):
        optimizer.zero_grad_()
    with pt("diffvg_fwd.total"):
        img = renderer.get_image()
    with pt("clip_fwd.total"):
        losses = loss_func(img, inputs.detach(), renderer.get_color_parameters(),
                           renderer, epoch, optimizer)
        loss = sum(losses.values())
    # The graph is a chain, so the backward splits exactly at `img`.
    with pt("clip_bwd"):
        grad_img = torch.autograd.grad(loss, img, retain_graph=False)[0]
    with pt("diffvg_bwd"):
        img.backward(grad_img)
    with pt("optim.step"):
        optimizer.step_()
    return float(loss.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=None, help="image path (default: first of eval_set.json)")
    ap.add_argument("--num-paths", type=int, default=16)
    ap.add_argument("--iters", type=int, default=50, help="measured iterations per pass")
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "bench" / "results" / "profile"))
    ap.add_argument("--trace", action="store_true", help="also write a chrome trace")
    args_cli = ap.parse_args()

    if args_cli.target is None:
        eval_set = json.loads((ROOT / "data" / "eval_set.json").read_text())
        args_cli.target = eval_set[0]["path"]

    outdir = Path(args_cli.out)
    outdir.mkdir(parents=True, exist_ok=True)
    workdir = outdir / "_work"
    workdir.mkdir(exist_ok=True)

    print(f"target     : {args_cli.target}")
    print(f"num_paths  : {args_cli.num_paths}  (= {args_cli.num_paths*4*2} free parameters)")
    print(f"iters      : {args_cli.iters} measured, {args_cli.warmup} warmup\n")

    args = make_args(args_cli.target, workdir, f"prof_n{args_cli.num_paths}",
                     num_paths=args_cli.num_paths, seed=args_cli.seed,
                     num_iter=args_cli.iters + args_cli.warmup + 10,
                     save_interval=10 ** 9, eval_interval=10 ** 9)

    print("building pipeline (CLIP RN101 + ViT-B/32 saliency + U2Net) ...")
    with common.quiet():
        renderer, loss_func, optimizer, inputs, _ = build_pipeline(args)
    n_aug = args.num_aug_clip
    print(f"  augmentations/iter : {n_aug}  -> CLIP sees {2*(1+n_aug)} images/iter "
          f"({1+n_aug} sketch + {1+n_aug} target)\n")

    results = {"config": {
        "target": args_cli.target, "num_paths": args_cli.num_paths,
        "iters": args_cli.iters, "warmup": args_cli.warmup, "seed": args_cli.seed,
        "num_aug_clip": n_aug, "image_scale": args.image_scale,
        "clip_model": args.clip_model_name,
        "clip_conv_layer_weights": args.clip_conv_layer_weights,
        "n_free_params": args_cli.num_paths * args.control_points_per_seg * 2,
    }, "env": env_fingerprint()}

    epoch = 0
    # ---------------------------------------------------------------- pass 1: FUSED
    print("[1/3] FUSED pass (ground-truth wall clock, no intra-iteration syncs)")
    for _ in range(args_cli.warmup):
        one_iteration(renderer, loss_func, optimizer, inputs, epoch); epoch += 1
    sync()
    # Timing pass runs CLEAN. An nvidia-smi poll costs ~50 ms of CPU and stalls the
    # driver, so sampling during this loop would inflate the very number we want.
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(args_cli.iters):
        one_iteration(renderer, loss_func, optimizer, inputs, epoch); epoch += 1
    sync()
    fused_ms = (time.perf_counter() - t0) * 1e3 / args_cli.iters
    torch_peak_mb = torch.cuda.max_memory_allocated() / 2**20
    # Separate, throwaway pass purely to sample utilisation//memory.
    with GpuSampler(interval=0.1) as gs:
        for _ in range(args_cli.iters):
            one_iteration(renderer, loss_func, optimizer, inputs, epoch); epoch += 1
        sync()
    results["fused"] = {"iter_ms": fused_ms, "iters_per_s": 1e3 / fused_ms,
                        "torch_peak_alloc_mb": torch_peak_mb, **gs.summary()}
    print(f"      {fused_ms:.2f} ms/iter  ({1e3/fused_ms:.1f} it/s)   "
          f"GPU util mean {gs.summary()['gpu_util_mean']}%  "
          f"peak mem {gs.summary()['gpu_mem_used_mb_max']} MB")
    print(f"      -> a full 2001-iteration seed = {fused_ms*2001/1e3:.1f} s\n")

    # ---------------------------------------------------------------- pass 2: SPLIT
    print("[2/3] SPLIT pass (synchronised phase attribution)")
    pt = PhaseTimer()
    instrument_loss(loss_func, pt)
    orig_serialize = instrument_diffvg(pt)
    for _ in range(5):
        one_iteration(renderer, loss_func, optimizer, inputs, epoch, pt, split=True)
        pt.flush(); epoch += 1
    pt.rows.clear()
    for _ in range(args_cli.iters):
        one_iteration(renderer, loss_func, optimizer, inputs, epoch, pt, split=True)
        pt.flush(); epoch += 1
    ps = pt.stats()
    results["split"] = ps

    # ---------------------------------------------------------- pass 3: torch.profiler
    print("[3/3] PROFILE pass (torch.profiler: GPU kernel time + launch counts)")
    import pydiffvg
    pydiffvg.RenderFunction.serialize_scene = staticmethod(orig_serialize)  # un-instrument
    from torch.profiler import ProfilerActivity, profile

    prof_iters = min(args_cli.iters, 30)
    for _ in range(5):
        one_iteration(renderer, loss_func, optimizer, inputs, epoch); epoch += 1
    sync()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False, with_stack=False) as prof:
        for _ in range(prof_iters):
            one_iteration(renderer, loss_func, optimizer, inputs, epoch); epoch += 1
        sync()

    # key_averages() returns BOTH cpu-side op entries and device kernel entries, and
    # the cpu-side ones carry device time attributed from their children. Summing
    # everything double-counts (it produced gpu_busy > wall-clock). Only DeviceType.CUDA
    # entries are actual kernels.
    from torch.autograd import DeviceType
    ka = prof.key_averages()
    kern = [e for e in ka if e.device_type == DeviceType.CUDA]
    cpu = [e for e in ka if e.device_type == DeviceType.CPU]
    gpu_ms = sum(e.self_device_time_total for e in kern) / 1e3 / prof_iters
    cpu_ms = sum(e.self_cpu_time_total for e in cpu) / 1e3 / prof_iters
    n_kernels = sum(e.count for e in kern) / prof_iters
    n_cpu_ops = sum(e.count for e in cpu) / prof_iters
    overhead_ms = fused_ms - gpu_ms
    results["profile"] = {
        "gpu_busy_ms_per_iter": gpu_ms,
        "cpu_self_ms_per_iter": cpu_ms,
        "kernels_per_iter": n_kernels,
        "cpu_ops_per_iter": n_cpu_ops,
        "python_launch_overhead_ms": overhead_ms,
        "overhead_frac_of_wall": overhead_ms / fused_ms,
        "mean_kernel_us": (gpu_ms * 1e3 / n_kernels) if n_kernels else None,
    }
    top = [{"name": e.key[:110], "self_cuda_ms_per_iter": e.self_device_time_total / 1e3 / prof_iters,
            "calls_per_iter": e.count / prof_iters}
           for e in sorted(kern, key=lambda x: -x.self_device_time_total)[:15]]
    results["profile"]["top_kernels"] = top

    if args_cli.trace:
        tp = outdir / f"trace_n{args_cli.num_paths}.json"
        prof.export_chrome_trace(str(tp))
        print(f"      chrome trace -> {tp}")

    # ---------------------------------------------------------------- report
    def g(k):
        return ps.get(k, {}).get("mean_ms", 0.0)

    diffvg_fwd = g("diffvg_fwd.total")
    serialize = g("diffvg_fwd.serialize_scene")
    clip_fwd = g("clip_fwd.total")
    clip_bwd = g("clip_bwd")
    diffvg_bwd = g("diffvg_bwd")
    optim = g("optim.zero_grad") + g("optim.step")
    split_total = diffvg_fwd + clip_fwd + clip_bwd + diffvg_bwd + optim

    diffvg_all = diffvg_fwd + diffvg_bwd
    clip_all = clip_fwd + clip_bwd

    rows = [
        ("diffvg fwd  (total)", diffvg_fwd),
        ("   . serialize_scene (python)", serialize),
        ("   . rasterise + composite", diffvg_fwd - serialize),
        ("CLIP fwd    (total)", clip_fwd),
        ("   . normalize", g("clip_fwd.normalize")),
        ("   . augment x%d" % n_aug, g("clip_fwd.augment")),
        ("   . encode SKETCH branch", g("clip_fwd.encode_sketch")),
        ("   . encode TARGET branch", g("clip_fwd.encode_target")),
        ("   . loss reduction (residual)", clip_fwd - g("clip_fwd.normalize")
            - g("clip_fwd.augment") - g("clip_fwd.encode_sketch") - g("clip_fwd.encode_target")),
        ("CLIP bwd", clip_bwd),
        ("diffvg bwd", diffvg_bwd),
        ("optimiser (zero_grad + step)", optim),
    ]
    print("\n" + "=" * 74)
    print(f"PER-ITERATION BREAKDOWN  (n={args_cli.num_paths} strokes, "
          f"{results['config']['n_free_params']} free params)")
    print("=" * 74)
    print(f"{'phase':<34}{'ms':>9}{'% of split':>12}{'% of wall':>12}")
    print("-" * 74)
    for name, ms in rows:
        lead = "   " if name.startswith("   ") else ""
        print(f"{lead}{name.strip():<{34-len(lead)}}{ms:>9.3f}"
              f"{100*ms/split_total:>11.1f}%{100*ms/fused_ms:>11.1f}%")
    print("-" * 74)
    print(f"{'SPLIT TOTAL (with syncs)':<34}{split_total:>9.3f}{100:>11.1f}%"
          f"{100*split_total/fused_ms:>11.1f}%")
    print(f"{'FUSED TOTAL (ground truth)':<34}{fused_ms:>9.3f}{'':>12}{100:>11.1f}%")
    print(f"{'  of which GPU busy':<34}{gpu_ms:>9.3f}{'':>12}{100*gpu_ms/fused_ms:>11.1f}%")
    print(f"{'  of which python/launch idle':<34}{overhead_ms:>9.3f}{'':>12}"
          f"{100*overhead_ms/fused_ms:>11.1f}%")
    print("=" * 74)

    print(f"\nCLIP total   : {clip_all:8.3f} ms  ({100*clip_all/split_total:.1f}% of split)")
    print(f"diffvg total : {diffvg_all:8.3f} ms  ({100*diffvg_all/split_total:.1f}% of split)")
    ratio = clip_all / diffvg_all if diffvg_all else float("inf")
    verdict = ("CLIP-BOUND" if ratio > 1.5 else
               "diffvg-BOUND" if ratio < 0.67 else "BALANCED")
    print(f"ratio CLIP/diffvg = {ratio:.2f}x  ->  {verdict}")
    print(f"\nkernels/iter : {n_kernels:.0f}  (mean {results['profile']['mean_kernel_us']:.1f} us each)")
    print(f"cpu ops/iter : {n_cpu_ops:.0f}")
    print(f"redundant target-branch encode: {g('clip_fwd.encode_target'):.3f} ms/iter "
          f"= {100*g('clip_fwd.encode_target')/fused_ms:.1f}% of wall  "
          f"(ceiling for idea 1.4)")

    results["summary"] = {
        "clip_total_ms": clip_all, "diffvg_total_ms": diffvg_all,
        "clip_over_diffvg": ratio, "verdict": verdict,
        "split_total_ms": split_total, "fused_iter_ms": fused_ms,
        "target_branch_ms": g("clip_fwd.encode_target"),
        "target_branch_frac_of_wall": g("clip_fwd.encode_target") / fused_ms,
        "serialize_scene_ms": serialize,
        "serialize_frac_of_wall": serialize / fused_ms,
    }
    fp = outdir / f"profile_n{args_cli.num_paths}.json"
    fp.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {fp}")
    shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
