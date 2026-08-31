#!/usr/bin/env python3
"""
Verify A2 (target-view deduplication) is bit-identical.

The dedup fires when several batched items share a target tensor -- it keys on
data_ptr. Cloning the target produces byte-identical pixels at a *different*
address, which forces the original un-deduped path. So the same function, given
the same numbers, gives a reference and a test in one process:

    shared target   -> dedup path   (M clean rows encoded once)
    cloned targets  -> original path (M clean rows encoded M times)

Any difference is a bug in A2. Equality is not "close enough" here: the rows are
literally the same pixels through the same frozen encoder, so the only correct
result is exact.

  python bench/verify_a2.py --num-seeds 3
"""
import argparse
import json
from pathlib import Path

import torch

import common
from common import ROOT, make_args
from batch_seeds import batched_conv_loss, cache_clip_load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-seeds", type=int, default=3)
    ap.add_argument("--num-paths", type=int, default=16)
    a = ap.parse_args()

    common.require_free_gpu_memory()
    ev = json.loads((ROOT / "data" / "eval_set.json").read_text())
    work = ROOT / "bench" / "results" / "_a2"; work.mkdir(parents=True, exist_ok=True)

    import painterly_rendering as pr
    from models.loss import Loss
    from models.painter_params import Painter
    import config

    cache_clip_load()
    args = make_args(ev[0]["path"], work, "a2", num_paths=a.num_paths, seed=0,
                     num_iter=1, save_interval=10 ** 9, eval_interval=10 ** 9)
    with common.quiet():
        loss_func = Loss(args)
        inputs, mask = pr.get_target(args)
        painters = []
        for s in range(a.num_seeds):
            config.set_seed(s * 1000)
            p = Painter(num_strokes=args.num_paths, args=args,
                        num_segments=args.num_segments, imsize=args.image_scale,
                        device=args.device, target_im=inputs, mask=mask).to(args.device)
            p.set_random_noise(0)
            p.init_image(stage=0)
            p.parameters()
            painters.append(p)
    cl = loss_func.loss_mapper["clip_conv_loss"]
    imgs = [p.get_image() for p in painters]

    shared = inputs.detach()
    cloned = [inputs.detach().clone() for _ in painters]   # same pixels, new addresses
    assert all(torch.equal(shared, c) for c in cloned)
    assert len({c.data_ptr() for c in cloned}) == len(cloned)

    ok = True
    for mode in ("train", "eval"):
        torch.manual_seed(0)
        dedup = batched_conv_loss(cl, imgs, shared, mode=mode)
        torch.manual_seed(0)                 # same augmentations on both paths
        ref = batched_conv_loss(cl, imgs, cloned, mode=mode)
        for i, (d, r) in enumerate(zip(dedup, ref)):
            assert set(d) == set(r), f"{mode}: different loss keys"
            for k in d:
                same = torch.equal(d[k], r[k])
                if not same:
                    ok = False
                    print(f"  {mode} item {i} {k}: MISMATCH "
                          f"|Δ|={float((d[k]-r[k]).abs().max()):.3e}")
        print(f"  {mode:<6} {len(dedup)} items: "
              f"{'identical' if ok else 'DIFFERENT'}   "
              f"per-item losses "
              f"{[round(float(sum(x.values())), 6) for x in dedup]}")

    print("\nA2 verification:", "PASS (bit-identical)" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
