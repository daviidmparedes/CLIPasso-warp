#!/usr/bin/env python3
"""
Tier 1.1 -- batch the seeds.

run_object_sketching.py spawns one PROCESS per seed over the same target image. Each
reloads RN101, ViT-B/32 (twice, inside Painter) and U2Net, recomputes the saliency map,
and then runs a 2001-iteration loop that launches ~3800 kernels averaging 5.5 us. The
profile says 41.6% of wall-clock is GPU-idle launch overhead, so running the seeds
concurrently in ONE process should convert three small launch-bound loops into one
larger one.

What is shared: model loading, the target tensor, the U2Net mask, and -- crucially --
a single CLIP encoder call over all seeds' augmented views at once.

What is NOT shared (deliberately, to keep the objective identical to the baseline):
  * each seed keeps its own augmentation draws,
  * each seed's loss is reduced over ITS OWN views only, then summed.
Because the seeds have disjoint parameters, d(sum_i L_i)/d(theta_j) = d(L_j)/d(theta_j),
so one backward over the summed loss gives every seed exactly the gradient it would have
received alone. One Adam over the concatenated parameters is likewise equivalent to three
separate Adams, since Adam's state is per-parameter.

diffvg is NOT batched: separate scenes cannot share a canvas, so rendering stays one call
per seed. Only the CLIP half (57% of per-iteration cost) and startup are amortised.

  python bench/batch_seeds.py --num-paths 16 --num-seeds 3
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import common
from common import ROOT, env_fingerprint, make_args, sync


def cache_clip_load():
    """Painter loads ViT-B/32 twice per instance; Loss loads it again. Cache it.

    clip_attn() does `del model` afterwards, which only drops the local reference --
    the cache keeps the module alive, so this is safe.
    """
    import CLIP_.clip as clip
    orig = clip.load
    cache = {}

    def cached(name, device=None, jit=True, **kw):
        key = (name, str(device), jit)
        if key not in cache:
            cache[key] = orig(name, device=device, jit=jit, **kw)
        return cache[key]

    clip.load = cached
    return orig


def batched_conv_loss(cl, sketches, target, mode="train"):
    """CLIPConvLoss over N sketches in ONE encoder call; returns a per-seed loss list.

    Mirrors CLIPConvLoss.forward exactly, except that the encoder sees all seeds'
    views concatenated and each seed's reduction is taken over its own slice.
    """
    device = cl.device
    y = target.to(device)
    xs_all, ys_all, counts = [], [], []
    for x in sketches:
        x = x.to(device)
        sk, im = [cl.normalize_transform(x)], [cl.normalize_transform(y)]
        if mode == "train":
            for _ in range(cl.num_augs):
                pair = cl.augment_trans(torch.cat([x, y]))
                sk.append(pair[0].unsqueeze(0))
                im.append(pair[1].unsqueeze(0))
        xs_all.append(torch.cat(sk, dim=0))
        ys_all.append(torch.cat(im, dim=0))
        counts.append(len(sk))

    xs = torch.cat(xs_all, dim=0).to(device)
    ys = torch.cat(ys_all, dim=0).to(device)
    xs_fc, xs_conv = cl.forward_inspection_clip_resnet(xs.contiguous())
    ys_fc, ys_conv = cl.forward_inspection_clip_resnet(ys.detach())

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
    ap.add_argument("--target", default=None)
    ap.add_argument("--num-paths", type=int, default=16)
    ap.add_argument("--num-seeds", type=int, default=3)
    ap.add_argument("--num-iter", type=int, default=2001)
    ap.add_argument("--eval-interval", type=int, default=10)
    ap.add_argument("--save-interval", type=int, default=10)
    ap.add_argument("--out", default=str(ROOT / "bench" / "results" / "batched"))
    ap.add_argument("--name", default=None)
    a = ap.parse_args()

    if a.target is None:
        a.target = json.loads((ROOT / "data" / "eval_set.json").read_text())[0]["path"]
    seeds = list(range(0, a.num_seeds * 1000, 1000))
    stem = Path(a.target).stem
    cls = Path(a.target).parent.name
    outroot = Path(a.out)
    outroot.mkdir(parents=True, exist_ok=True)

    import config
    import painterly_rendering as pr
    import sketch_utils as utils
    from models.loss import Loss
    from models.painter_params import Painter

    cache_clip_load()
    t_boot = time.perf_counter()

    # args for seed 0 drive the shared pieces (target, mask, loss)
    base_args = make_args(a.target, outroot, f"{cls}_{stem}_{a.num_paths}strokes_batched",
                          num_paths=a.num_paths, seed=seeds[0], num_iter=a.num_iter,
                          save_interval=10 ** 9, eval_interval=10 ** 9)
    with common.quiet():
        loss_func = Loss(base_args)
        inputs, mask = pr.get_target(base_args)

        # one Painter per seed; reseed before each so stroke init matches the
        # per-process baseline (get_path uses python random, saliency uses numpy)
        painters, per_args = [], []
        for s in seeds:
            sa = make_args(a.target, outroot, f"{cls}_{stem}_{a.num_paths}strokes_seed{s}",
                           num_paths=a.num_paths, seed=s, num_iter=a.num_iter,
                           save_interval=a.save_interval, eval_interval=a.eval_interval)
            config.set_seed(s)
            p = Painter(num_strokes=sa.num_paths, args=sa, num_segments=sa.num_segments,
                        imsize=sa.image_scale, device=sa.device, target_im=inputs, mask=mask)
            p = p.to(sa.device)
            p.set_random_noise(0)
            p.init_image(stage=0)
            painters.append(p)
            per_args.append(sa)

    params = [pt for p in painters for pt in p.parameters()]
    optim = torch.optim.Adam(params, lr=base_args.lr)   # == 3 separate Adams
    sync()
    boot = time.perf_counter() - t_boot
    cl = loss_func.loss_mapper["clip_conv_loss"]

    print(f"target      : {a.target}")
    print(f"seeds       : {seeds}   strokes: {a.num_paths}")
    print(f"boot        : {boot:.2f}s  (shared across all {len(seeds)} seeds)")
    print(f"CLIP batch  : {len(seeds)*(1+cl.num_augs)} sketch + "
          f"{len(seeds)*(1+cl.num_augs)} target images per iteration\n")

    state = [{"loss_eval": [], "best": 1e9, "best_iter": 0, "terminate": False,
              "cfg": {}} for _ in seeds]
    min_delta = 1e-5
    t0 = time.perf_counter()

    for epoch in range(a.num_iter):
        for p in painters:
            p.set_random_noise(epoch)
        optim.zero_grad()
        imgs = [p.get_image() for p in painters]
        per_seed = batched_conv_loss(cl, imgs, inputs.detach(), mode="train")
        total = sum(sum(d.values()) for d in per_seed)
        total.backward()
        optim.step()

        if epoch % a.eval_interval == 0:
            with torch.no_grad():
                ev = batched_conv_loss(cl, imgs, inputs, mode="eval")
            for i, d in enumerate(ev):
                v = float(sum(d.values()).item())
                st = state[i]
                st["loss_eval"].append(v)
                for k, t in d.items():
                    st["cfg"].setdefault(k, []).append(float(t.item()))
                delta = v - st["best"]
                if abs(delta) > min_delta and delta < 0:
                    st["best"], st["best_iter"] = v, epoch
                    painters[i].save_svg(per_args[i].output_dir, "best_iter")
            del ev
        if epoch % a.save_interval == 0:
            for i, p in enumerate(painters):
                p.save_svg(f"{per_args[i].output_dir}/svg_logs", f"svg_iter{epoch}")

        # §4.2: release the graph before the next forward (diffvg's context otherwise
        # stays pinned and its backward falls back to raw cudaMalloc/cudaFree)
        del imgs, per_seed, total

    sync()
    loop = time.perf_counter() - t0
    for i, p in enumerate(painters):
        p.save_svg(per_args[i].output_dir, "final_svg")
        cfg = vars(per_args[i]).copy()
        cfg["loss_eval"] = state[i]["loss_eval"]
        cfg.update(state[i]["cfg"])
        np.save(f"{per_args[i].output_dir}/config.npy", cfg)

    total_s = boot + loop
    print(f"loop        : {loop:.2f}s  ({1e3*loop/a.num_iter:.2f} ms/iter)")
    print(f"TOTAL       : {total_s:.2f}s for {len(seeds)} seeds "
          f"= {total_s/len(seeds):.2f}s per seed")
    for i, s in enumerate(seeds):
        print(f"  seed {s:<5} best loss_eval {state[i]['best']:.5f} @iter {state[i]['best_iter']}")

    res = {"target": a.target, "seeds": seeds, "num_paths": a.num_paths,
           "num_iter": a.num_iter, "boot_s": boot, "loop_s": loop, "total_s": total_s,
           "s_per_seed": total_s / len(seeds), "ms_per_iter": 1e3 * loop / a.num_iter,
           "best_loss_eval": [st["best"] for st in state],
           "env": env_fingerprint()}
    fp = outroot / f"batched_{cls}_{stem}_n{a.num_paths}_x{len(seeds)}.json"
    fp.write_text(json.dumps(res, indent=1))
    print(f"wrote {fp}")


if __name__ == "__main__":
    main()
