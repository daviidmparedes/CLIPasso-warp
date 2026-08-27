#!/usr/bin/env python3
"""
Tier 1.2 -- batch across images (dataset-generation throughput).

Extends 1.1 from "N seeds of one image" to "N seeds of M images", all optimised
concurrently in one process with a single CLIP encoder call per iteration over
M*N*(1+num_augs) sketch views and the matching target views.

The same equivalence argument as 1.1 applies and is verified the same way: every
(image, seed) pair keeps its own augmentation draws and its own loss reduction, the
losses are summed, and because all M*N parameter sets are disjoint one backward gives
each pair exactly the gradient it would receive alone. BatchNorm is in eval mode, so
rows do not interact.

diffvg still does not batch -- rendering remains M*N separate calls -- so the ceiling
here is the same as 1.1: only the CLIP half and startup amortise. What 1.2 adds over
1.1 is GPU occupancy: the encoder batch grows from 15 to M*15 rows, which is where a
launch-bound workload actually wins.

  python bench/batch_images.py --num-images 5 --num-seeds 3
  python bench/batch_images.py --num-images 16 --num-seeds 3 --encoder-chunk 240
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import common
from common import ROOT, env_fingerprint, make_args, sync
from batch_seeds import cache_clip_load


def chunked_encode(cl, x, chunk):
    """Encoder forward, optionally split to bound activation memory.

    Splitting changes nothing mathematically (rows are independent with BatchNorm in
    eval mode) but trades some launch amortisation for a lower peak footprint.
    """
    if not chunk or x.shape[0] <= chunk:
        return cl.forward_inspection_clip_resnet(x)
    fcs, convs = [], None
    for i in range(0, x.shape[0], chunk):
        fc, cv = cl.forward_inspection_clip_resnet(x[i:i + chunk])
        fcs.append(fc)
        convs = [[c] for c in cv] if convs is None else [a + [b] for a, b in zip(convs, cv)]
    return torch.cat(fcs, 0), [torch.cat(c, 0) for c in convs]


def batched_loss_multi(cl, sketches, targets, mode="train", chunk=None):
    """Per-item CLIPConvLoss over many (image, seed) pairs in one encoder call."""
    device = cl.device
    xs_all, ys_all, counts = [], [], []
    for x, y in zip(sketches, targets):
        x, y = x.to(device), y.to(device)
        sk, im = [cl.normalize_transform(x)], [cl.normalize_transform(y)]
        if mode == "train":
            for _ in range(cl.num_augs):
                pair = cl.augment_trans(torch.cat([x, y]))
                sk.append(pair[0].unsqueeze(0))
                im.append(pair[1].unsqueeze(0))
        xs_all.append(torch.cat(sk, 0))
        ys_all.append(torch.cat(im, 0))
        counts.append(len(sk))

    xs = torch.cat(xs_all, 0).to(device)
    ys = torch.cat(ys_all, 0).to(device)
    xs_fc, xs_conv = chunked_encode(cl, xs.contiguous(), chunk)
    ys_fc, ys_conv = chunked_encode(cl, ys.detach(), chunk)

    out, off = [], 0
    for n in counts:
        sl = slice(off, off + n)
        d = {}
        for layer, w in enumerate(cl.args.clip_conv_layer_weights):
            if w:
                d[f"clip_conv_loss_layer{layer}"] = torch.square(
                    xs_conv[layer][sl] - ys_conv[layer][sl]).mean() * w
        if cl.clip_fc_loss_weight:
            d["fc"] = (1 - torch.cosine_similarity(
                xs_fc[sl], ys_fc[sl], dim=1)).mean() * cl.clip_fc_loss_weight
        out.append(d)
        off += n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-images", type=int, default=5)
    ap.add_argument("--num-seeds", type=int, default=3)
    ap.add_argument("--num-paths", type=int, default=16)
    ap.add_argument("--num-iter", type=int, default=2001)
    ap.add_argument("--eval-interval", type=int, default=10)
    ap.add_argument("--save-interval", type=int, default=10 ** 9)
    ap.add_argument("--encoder-chunk", type=int, default=0,
                    help="max rows per encoder call (0 = one call); bounds peak memory")
    ap.add_argument("--manifest", default=None,
                    help="image list (default: data/eval_set.json, else data/manifest.json)")
    ap.add_argument("--out", default=str(ROOT / "bench" / "results" / "batched_images"))
    a = ap.parse_args()

    src = Path(a.manifest) if a.manifest else (ROOT / "data" / "eval_set.json")
    items = json.loads(src.read_text())
    if len(items) < a.num_images:
        items = json.loads((ROOT / "data" / "manifest.json").read_text())
    items = items[:a.num_images]
    seeds = list(range(0, a.num_seeds * 1000, 1000))
    outroot = Path(a.out)
    outroot.mkdir(parents=True, exist_ok=True)

    import config
    import painterly_rendering as pr
    from models.loss import Loss
    from models.painter_params import Painter

    cache_clip_load()
    t_boot = time.perf_counter()

    first = make_args(items[0]["path"], outroot, "probe", num_paths=a.num_paths, seed=0,
                      num_iter=a.num_iter, save_interval=10 ** 9, eval_interval=10 ** 9)
    with common.quiet():
        loss_func = Loss(first)
    cl = loss_func.loss_mapper["clip_conv_loss"]

    pairs = []          # one entry per (image, seed)
    with common.quiet():
        for it in items:
            cls, stem = it["class"], Path(it["rel"]).stem
            ta = make_args(it["path"], outroot, f"{cls}_{stem}_probe",
                           num_paths=a.num_paths, seed=0, num_iter=a.num_iter,
                           save_interval=10 ** 9, eval_interval=10 ** 9)
            inputs, mask = pr.get_target(ta)
            for s in seeds:
                sa = make_args(it["path"], outroot,
                               f"{cls}_{stem}_{a.num_paths}strokes_seed{s}",
                               num_paths=a.num_paths, seed=s, num_iter=a.num_iter,
                               save_interval=a.save_interval,
                               eval_interval=a.eval_interval)
                config.set_seed(s)
                p = Painter(num_strokes=sa.num_paths, args=sa, num_segments=sa.num_segments,
                            imsize=sa.image_scale, device=sa.device,
                            target_im=inputs, mask=mask).to(sa.device)
                p.set_random_noise(0)
                p.init_image(stage=0)
                pairs.append({"painter": p, "args": sa, "target": inputs,
                              "class": cls, "seed": s,
                              "loss_eval": [], "best": 1e9, "best_iter": 0, "cfg": {}})

    params = [q for pr_ in pairs for q in pr_["painter"].parameters()]
    optim = torch.optim.Adam(params, lr=first.lr)
    sync()
    boot = time.perf_counter() - t_boot
    rows = len(pairs) * (1 + cl.num_augs)

    print(f"images      : {len(items)}   seeds: {len(seeds)}   pairs: {len(pairs)}")
    print(f"boot        : {boot:.2f}s  (shared)")
    print(f"CLIP batch  : {rows} sketch + {rows} target rows per iteration"
          f"{f' (chunked at {a.encoder_chunk})' if a.encoder_chunk else ''}")
    print(f"free params : {sum(q.numel() for q in params)}\n")

    torch.cuda.reset_peak_memory_stats()
    min_delta = 1e-5
    t0 = time.perf_counter()
    for epoch in range(a.num_iter):
        for pr_ in pairs:
            pr_["painter"].set_random_noise(epoch)
        optim.zero_grad()
        imgs = [pr_["painter"].get_image() for pr_ in pairs]
        tgts = [pr_["target"] for pr_ in pairs]
        per = batched_loss_multi(cl, imgs, tgts, "train", a.encoder_chunk)
        total = sum(sum(d.values()) for d in per)
        total.backward()
        optim.step()

        if epoch % a.eval_interval == 0:
            with torch.no_grad():
                ev = batched_loss_multi(cl, imgs, tgts, "eval", a.encoder_chunk)
            for pr_, d in zip(pairs, ev):
                v = float(sum(d.values()).item())
                pr_["loss_eval"].append(v)
                for k, t in d.items():
                    pr_["cfg"].setdefault(k, []).append(float(t.item()))
                if v - pr_["best"] < -min_delta:
                    pr_["best"], pr_["best_iter"] = v, epoch
                    pr_["painter"].save_svg(pr_["args"].output_dir, "best_iter")
            del ev
        del imgs, tgts, per, total      # §4.2

    sync()
    loop = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    for pr_ in pairs:
        pr_["painter"].save_svg(pr_["args"].output_dir, "final_svg")
        cfg = vars(pr_["args"]).copy()
        cfg["loss_eval"] = pr_["loss_eval"]
        cfg.update(pr_["cfg"])
        np.save(f"{pr_['args'].output_dir}/config.npy", cfg)

    total_s = boot + loop
    n_sk = len(pairs)
    print(f"loop        : {loop:.1f}s  ({1e3*loop/a.num_iter:.1f} ms/iter)")
    print(f"peak alloc  : {peak:.2f} GB")
    print(f"TOTAL       : {total_s:.1f}s for {n_sk} sketches "
          f"({len(items)} images x {len(seeds)} seeds)")
    print(f"  per sketch: {total_s/n_sk:.2f}s      per image: {total_s/len(items):.1f}s")

    res = {"n_images": len(items), "n_seeds": len(seeds), "pairs": n_sk,
           "num_paths": a.num_paths, "num_iter": a.num_iter,
           "encoder_chunk": a.encoder_chunk, "boot_s": boot, "loop_s": loop,
           "total_s": total_s, "s_per_sketch": total_s / n_sk,
           "s_per_image": total_s / len(items), "ms_per_iter": 1e3 * loop / a.num_iter,
           "peak_alloc_gb": peak,
           "best_loss_eval": [pr_["best"] for pr_ in pairs],
           "env": env_fingerprint()}
    fp = outroot / f"batched_images_M{len(items)}_S{len(seeds)}_n{a.num_paths}.json"
    fp.write_text(json.dumps(res, indent=1))
    print(f"wrote {fp}")


if __name__ == "__main__":
    main()
