#!/usr/bin/env python3
"""
Verify a finding from the profile: CLIPasso never sets requires_grad=False on the
RN101 used for the perceptual loss, so every loss.backward() also computes and
accumulates weight gradients for the frozen encoder. The optimiser only ever steps
the Bezier control points, so that work is pure waste.

Checks three things:
  1. how many encoder parameters are collecting gradients,
  2. that freezing leaves the control-point gradients BIT-IDENTICAL,
  3. what freezing is worth in ms/iter and in peak memory.
"""
import time

import torch

import common
from common import ROOT, build_pipeline, make_args, sync
import json


def bench(renderer, loss_func, optimizer, inputs, iters=40, warmup=10):
    ep = 0
    for _ in range(warmup + iters):
        if _ == warmup:
            sync(); torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
        renderer.set_random_noise(ep)
        optimizer.zero_grad_()
        img = renderer.get_image()
        losses = loss_func(img, inputs.detach(), renderer.get_color_parameters(),
                           renderer, ep, optimizer)
        loss = sum(losses.values())
        loss.backward()
        optimizer.step_()
        ep += 1
    sync()
    return (time.perf_counter() - t0) * 1e3 / iters, torch.cuda.max_memory_allocated() / 2**20


def grads_after_one_step(args, freeze, seed=0):
    """Run exactly one backward from a fixed init and return the point grads.

    Reseeds python/numpy/torch BEFORE building, because stroke initialisation draws
    from python random (get_path jitter) and numpy (saliency index sampling). Seeding
    only torch afterwards gives the two runs different initial control points, which
    makes their gradients differ for reasons that have nothing to do with freezing.
    """
    import config
    config.set_seed(seed)
    with common.quiet():
        renderer, loss_func, optimizer, inputs, _ = build_pipeline(args)
    if freeze:
        for p in loss_func.loss_mapper["clip_conv_loss"].parameters():
            p.requires_grad_(False)
    config.set_seed(seed)
    renderer.set_random_noise(0)
    optimizer.zero_grad_()
    img = renderer.get_image()
    losses = loss_func(img, inputs.detach(), renderer.get_color_parameters(), renderer, 0, optimizer)
    sum(losses.values()).backward()
    g = [p.grad.detach().clone() for p in renderer.get_points_parans()]
    return g, renderer, loss_func, optimizer, inputs


def main():
    eval_set = json.loads((ROOT / "data" / "eval_set.json").read_text())
    target = eval_set[0]["path"]
    workdir = ROOT / "bench" / "results" / "_freeze_work"
    workdir.mkdir(parents=True, exist_ok=True)

    args = make_args(target, workdir, "freeze", num_paths=16, seed=0,
                     num_iter=200, save_interval=10**9, eval_interval=10**9)

    print("=== 1. how many encoder params are collecting gradients? ===")
    with common.quiet():
        r0, lf0, o0, inp0, _ = build_pipeline(args)
    cl = lf0.loss_mapper["clip_conv_loss"]
    tot = sum(p.numel() for p in cl.parameters())
    trainable = sum(p.numel() for p in cl.parameters() if p.requires_grad)
    print(f"  CLIPConvLoss params      : {tot/1e6:.1f} M")
    print(f"  with requires_grad=True  : {trainable/1e6:.1f} M   <-- all of RN101, unfrozen")
    n_points = sum(p.numel() for p in r0.get_points_parans())
    print(f"  actual optimised params  : {n_points}  (the Bezier control points)")
    print(f"  ratio                    : {trainable/max(n_points,1):,.0f}x more grads than needed\n")
    del r0, lf0, o0, inp0
    torch.cuda.empty_cache()

    print("=== 2. does freezing change the control-point gradients? ===")
    # Control: run the UNFROZEN path twice. cuDNN algorithm selection and atomic
    # accumulation order make float32 backward non-deterministic run-to-run, so the
    # frozen-vs-unfrozen delta is only meaningful relative to this baseline noise.
    g_a, *_ = grads_after_one_step(args, freeze=False); torch.cuda.empty_cache()
    g_b, *_ = grads_after_one_step(args, freeze=False); torch.cuda.empty_cache()
    g_frozen, *_ = grads_after_one_step(args, freeze=True); torch.cuda.empty_cache()

    def cmp(x, y):
        md = max((a - b).abs().max().item() for a, b in zip(x, y))
        scale = max(a.abs().max().item() for a in x)
        return md, md / scale if scale else float("nan"), all(torch.equal(a, b) for a, b in zip(x, y))

    md_self, rel_self, id_self = cmp(g_a, g_b)
    maxdiff, rel_fr, identical = cmp(g_a, g_frozen)
    gmax = max(a.abs().max().item() for a in g_a)
    print(f"  gradient magnitude (max) : {gmax:.4f}")
    print(f"  unfrozen vs unfrozen     : bit-identical={id_self}  max|d|={md_self:.3e}  rel={rel_self:.2e}   <- baseline noise")
    print(f"  unfrozen vs frozen       : bit-identical={identical}  max|d|={maxdiff:.3e}  rel={rel_fr:.2e}")
    verdict = ("within run-to-run nondeterminism" if maxdiff <= max(md_self * 3, 1e-9)
               else "LARGER than run-to-run noise -- investigate")
    print(f"  verdict                  : {verdict}\n")

    print("=== 3. what is freezing worth? ===")
    with common.quiet():
        r1, lf1, o1, inp1, _ = build_pipeline(args)
    ms_unfrozen, mem_unfrozen = bench(r1, lf1, o1, inp1)
    del r1, lf1, o1, inp1
    torch.cuda.empty_cache()

    with common.quiet():
        r2, lf2, o2, inp2, _ = build_pipeline(args)
    for p in lf2.loss_mapper["clip_conv_loss"].parameters():
        p.requires_grad_(False)
    ms_frozen, mem_frozen = bench(r2, lf2, o2, inp2)

    print(f"  unfrozen (as shipped)    : {ms_unfrozen:7.2f} ms/iter   peak alloc {mem_unfrozen:8.1f} MB")
    print(f"  frozen                   : {ms_frozen:7.2f} ms/iter   peak alloc {mem_frozen:8.1f} MB")
    print(f"  speedup                  : {ms_unfrozen/ms_frozen:.3f}x "
          f"({100*(1-ms_frozen/ms_unfrozen):.1f}% of wall-clock removed)")
    print(f"  memory saved             : {mem_unfrozen-mem_frozen:.1f} MB")
    print(f"\n  full 2001-iter seed: {ms_unfrozen*2001/1e3:.1f}s -> {ms_frozen*2001/1e3:.1f}s")

    out = ROOT / "bench" / "results" / "freeze_clip.json"
    out.write_text(json.dumps({
        "clip_params_M": tot / 1e6, "trainable_M": trainable / 1e6, "point_params": n_points,
        "grads_bit_identical": identical, "max_abs_grad_diff": maxdiff,
        "max_abs_grad_diff_selfnoise": md_self, "grad_verdict": verdict,
        "ms_unfrozen": ms_unfrozen, "ms_frozen": ms_frozen,
        "speedup": ms_unfrozen / ms_frozen,
        "peak_mb_unfrozen": mem_unfrozen, "peak_mb_frozen": mem_frozen,
        "env": common.env_fingerprint(),
    }, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
