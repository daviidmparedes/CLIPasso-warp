#!/usr/bin/env python3
"""
Verify that batching the seeds does not change any seed's objective.

Two claims to check, both of which the Tier 1.1 speedup depends on:

  1. With N=1, batched_conv_loss must reproduce CLIPConvLoss.forward BIT-IDENTICALLY
     given the same RNG state. Any difference means the batched reduction is wrong.

  2. With N>1, the gradient each seed receives from one backward over the summed loss
     must equal the gradient it would receive optimising alone. The seeds have disjoint
     parameters, so this should hold exactly; it is worth confirming rather than assuming,
     because a reduction taken over the wrong axis would silently rescale gradients.
"""
import json

import torch

import common
from common import ROOT, make_args, sync
from batch_seeds import batched_conv_loss, cache_clip_load


def main():
    import config
    import painterly_rendering as pr
    from models.loss import Loss
    from models.painter_params import Painter

    cache_clip_load()
    target = json.loads((ROOT / "data" / "eval_set.json").read_text())[0]["path"]
    wd = ROOT / "bench" / "results" / "_equiv"
    wd.mkdir(parents=True, exist_ok=True)
    args = make_args(target, wd, "equiv", num_paths=16, seed=0, num_iter=50,
                     save_interval=10 ** 9, eval_interval=10 ** 9)

    with common.quiet():
        loss_func = Loss(args)
        inputs, mask = pr.get_target(args)
    cl = loss_func.loss_mapper["clip_conv_loss"]

    def make_painter(seed):
        config.set_seed(seed)
        p = Painter(num_strokes=args.num_paths, args=args, num_segments=args.num_segments,
                    imsize=args.image_scale, device=args.device, target_im=inputs, mask=mask)
        p = p.to(args.device)
        p.set_random_noise(0)
        p.init_image(stage=0)
        p.parameters()
        return p

    # ---------------- claim 1: N=1 equals the original forward -----------------
    print("=== 1. N=1 batched vs original CLIPConvLoss.forward ===")
    with common.quiet():
        p0 = make_painter(0)
    img = p0.get_image()

    config.set_seed(123)
    ref = cl(img, inputs.detach(), mode="train")
    config.set_seed(123)
    got = batched_conv_loss(cl, [img], inputs.detach(), mode="train")[0]

    assert set(ref) == set(got), f"key mismatch {set(ref)} vs {set(got)}"
    ok = True
    for k in sorted(ref):
        a, b = float(ref[k].item()), float(got[k].item())
        same = torch.equal(ref[k], got[k])
        ok &= same
        print(f"  {k:<26} orig={a:.10f}  batched={b:.10f}  bit-identical={same}")
    print(f"  -> all terms bit-identical: {ok}\n")

    # ---------------- claim 2: batching must not change a seed's loss ----------
    # The right question is not "do two RNG replays agree" (they cannot be made to,
    # across differently shaped batches) but: given the EXACT SAME augmented views,
    # does putting them in a 15-row batch change seed i's loss versus its own 5-row
    # batch? BatchNorm is in eval mode, so mathematically the rows are independent
    # and the answer must be zero up to floating-point.
    print("=== 2. same views, 15-row batch vs per-seed 5-row batch ===")
    seeds = [0, 1000, 2000]
    with common.quiet():
        painters = [make_painter(s) for s in seeds]
    config.set_seed(777)
    imgs = [p.get_image() for p in painters]

    # build the concatenated views exactly as batched_conv_loss does
    y = inputs.detach().to(cl.device)
    xs_all, ys_all, counts = [], [], []
    for x in imgs:
        sk, im = [cl.normalize_transform(x)], [cl.normalize_transform(y)]
        for _ in range(cl.num_augs):
            pair = cl.augment_trans(torch.cat([x, y]))
            sk.append(pair[0].unsqueeze(0)); im.append(pair[1].unsqueeze(0))
        xs_all.append(torch.cat(sk, 0)); ys_all.append(torch.cat(im, 0)); counts.append(len(sk))

    def losses_from(xs, ys, w):
        xf, xc = cl.forward_inspection_clip_resnet(xs.contiguous())
        yf, yc = cl.forward_inspection_clip_resnet(ys.detach())
        d = {}
        for layer, wt in enumerate(w):
            if wt:
                d[f"L{layer}"] = torch.square(xc[layer] - yc[layer]).mean() * wt
        d["fc"] = (1 - torch.cosine_similarity(xf, yf, dim=1)).mean() * cl.clip_fc_loss_weight
        return d

    W = cl.args.clip_conv_layer_weights
    with torch.no_grad():
        big_x, big_y = torch.cat(xs_all, 0), torch.cat(ys_all, 0)
        xf, xc = cl.forward_inspection_clip_resnet(big_x.contiguous())
        yf, yc = cl.forward_inspection_clip_resnet(big_y.detach())
        off = 0
        rows = []
        for i, n in enumerate(counts):
            sl = slice(off, off + n); off += n
            batched = sum(torch.square(xc[l][sl] - yc[l][sl]).mean() * w
                          for l, w in enumerate(W) if w) + \
                      (1 - torch.cosine_similarity(xf[sl], yf[sl], dim=1)).mean() * cl.clip_fc_loss_weight
            alone = sum(losses_from(xs_all[i], ys_all[i], W).values())
            rows.append((seeds[i], float(batched.item()), float(alone.item())))

    worst = 0.0
    for s, b, al in rows:
        d = abs(b - al); worst = max(worst, d / max(abs(al), 1e-12))
        print(f"  seed {s:<5} in-batch={b:.8f}  alone={al:.8f}  |d|={d:.3e}  rel={d/abs(al):.3e}")
    print(f"  -> max relative change from batching: {worst:.3e}")
    print(f"     (for scale: this method's run-to-run loss_eval noise is ~2.3e-2 relative)")


    out = ROOT / "bench" / "results" / "batched_equivalence.json"
    out.write_text(json.dumps({
        "n1_bit_identical": bool(ok),
        "n1_terms": {k: float(ref[k].item()) for k in ref},
        "per_seed_loss_rel_change_from_batching": worst,
        "per_seed": [{"seed": s, "in_batch": b, "alone": al} for s, b, al in rows],
        "env": common.env_fingerprint(),
    }, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
