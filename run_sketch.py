#!/usr/bin/env python3
"""
Sketch an image with CLIPasso.

    python run_sketch.py --target my_photo.png
    python run_sketch.py --target my_photo.png --num-strokes 8 --num-sketches 3

CLIPasso draws several sketches of the same image from different random
initialisations and keeps the best one. Upstream runs each of those as a separate
process, which reloads CLIP, the saliency model and the target image every time
and gives the GPU one small sketch to work on at a time. This runs them together
in one process: one CLIP encoder, one target, and every sketch's strokes stepped
by a single optimiser.

That is exact, not an approximation. The sketches have disjoint parameters, so the
gradient of the summed loss with respect to each sketch's control points is that
sketch's own gradient, and one Adam over the concatenated parameters is identical
to N separate Adams because Adam's state is per-parameter. The encoder runs in
eval mode, so its batch normalisation does not mix rows either.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402


def build_args(target, output_dir, name, num_paths, seed, num_iter, save_interval,
               eval_interval, use_gpu, mask_object, fix_scale):
    """Build the args namespace through the repo's own parser.

    Going through config.parse_arguments rather than hand-rolling a namespace means
    this sees exactly the defaults painterly_rendering.py sees, including the device
    setup and the clip_conv_layer_weights parsing that happen inside it.
    """
    argv = ["painterly_rendering.py", str(target),
            "--output_dir", str(output_dir), "--wandb_name", str(name),
            "--num_paths", str(num_paths), "--seed", str(seed),
            "--num_iter", str(num_iter), "--save_interval", str(save_interval),
            "--eval_interval", str(eval_interval), "--use_gpu", str(int(use_gpu)),
            "--mask_object", str(int(mask_object)),
            "--mask_object_attention", str(int(mask_object)),
            "--fix_scale", str(int(fix_scale)),
            "--display_logs", "0", "--display", "0"]
    old, sys.argv = sys.argv, argv
    try:
        return config.parse_arguments()
    finally:
        sys.argv = old


def batched_loss(cl, sketches, target, mode="train"):
    """CLIPConvLoss over N sketches in one encoder call; returns a per-sketch loss dict.

    Mirrors CLIPConvLoss.forward, except the encoder sees every sketch's views
    concatenated and each sketch's reduction is taken over its own slice.
    """
    device = cl.device
    xs_all, ys_all, counts = [], [], []
    for x in sketches:
        x, y = x.to(device), target.to(device)
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


def ensure_u2net():
    p = ROOT / "U2Net_" / "saved_models" / "u2net.pth"
    if p.is_file():
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    print("downloading the U2Net saliency model (one time, ~170 MB) ...")
    # Invoked as a module rather than as a bare `gdown` command: the console script
    # is only on PATH when the virtualenv is activated, and this has to work when
    # the interpreter is called by full path too.
    r = subprocess.run([sys.executable, "-m", "gdown",
                        "https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ",
                        "-O", str(p)])
    if r.returncode != 0 or not p.is_file():
        sys.exit(
            "could not download the U2Net model.\n"
            "Google Drive rate-limits large files; try again shortly, or fetch it "
            "manually to\n    " + str(p) + "\nfrom "
            "https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ")


def main():
    ap = argparse.ArgumentParser(
        description="Sketch an image with CLIPasso.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--target", required=True,
                    help="path to the image to sketch")
    ap.add_argument("--num-strokes", type=int, default=16,
                    help="number of strokes; fewer means more abstract")
    ap.add_argument("--num-sketches", type=int, default=3,
                    help="how many sketches to draw; the best is kept")
    ap.add_argument("--num-iter", type=int, default=2001,
                    help="optimisation steps per sketch")
    ap.add_argument("--output-dir", default=None,
                    help="where to write results (default: output_sketches/<image name>)")
    ap.add_argument("--mask-object", action="store_true",
                    help="mask out the background first; use when the subject is not isolated")
    ap.add_argument("--fix-scale", action="store_true",
                    help="use when the image is not square")
    ap.add_argument("--save-interval", type=int, default=0,
                    help="also save an SVG every N steps (0 = only the final sketches)")
    ap.add_argument("--cpu", action="store_true", help="run on CPU (very slow)")
    a = ap.parse_args()

    target = Path(a.target).expanduser().resolve()
    if not target.is_file():
        sys.exit(f"no such image: {target}")
    use_gpu = not a.cpu and torch.cuda.is_available()
    if not a.cpu and not use_gpu:
        print("CUDA is not available; running on CPU, which is very slow.")

    out = Path(a.output_dir) if a.output_dir else ROOT / "output_sketches" / target.stem
    out.mkdir(parents=True, exist_ok=True)
    ensure_u2net()

    import painterly_rendering as pr
    from models.loss import Loss
    from models.painter_params import Painter

    seeds = list(range(0, a.num_sketches * 1000, 1000))
    save_int = a.save_interval if a.save_interval > 0 else 10 ** 9

    # config.parse_arguments() appends the run name to output_dir and creates it with
    # os.mkdir, not makedirs, so the parent has to exist first and each run's name is
    # what becomes its directory.
    base = build_args(target, out, f"sketch_seed{seeds[0]}", a.num_strokes, seeds[0],
                      a.num_iter, save_int, 10, use_gpu, a.mask_object, a.fix_scale)
    device = base.device
    print(f"sketching {target.name}: {a.num_strokes} strokes, "
          f"{a.num_sketches} sketch(es), {a.num_iter} steps, device {device}")

    t0 = time.perf_counter()
    loss_func = Loss(base)
    inputs, mask = pr.get_target(base)

    painters, per_args = [], []
    for s in seeds:
        sa = build_args(target, out, f"sketch_seed{s}", a.num_strokes, s,
                        a.num_iter, save_int, 10, use_gpu, a.mask_object, a.fix_scale)
        config.set_seed(s)
        p = Painter(num_strokes=sa.num_paths, args=sa, num_segments=sa.num_segments,
                    imsize=sa.image_scale, device=device,
                    target_im=inputs, mask=mask).to(device)
        p.set_random_noise(0)
        p.init_image(stage=0)
        painters.append(p)
        per_args.append(sa)

    params = [pt for p in painters for pt in p.parameters()]
    optim = torch.optim.Adam(params, lr=base.lr)
    cl = loss_func.loss_mapper["clip_conv_loss"]
    print(f"setup took {time.perf_counter() - t0:.1f}s; optimising ...")

    best = [{"loss": float("inf"), "iter": 0} for _ in seeds]
    t1 = time.perf_counter()
    for epoch in range(a.num_iter):
        for p in painters:
            p.set_random_noise(epoch)
        optim.zero_grad()
        imgs = [p.get_image() for p in painters]
        per = batched_loss(cl, imgs, inputs.detach(), mode="train")
        total = sum(sum(d.values()) for d in per)
        total.backward()
        optim.step()

        if epoch % 10 == 0:
            with torch.no_grad():
                ev = batched_loss(cl, imgs, inputs, mode="eval")
            for i, d in enumerate(ev):
                v = float(sum(d.values()).item())
                if v < best[i]["loss"] - 1e-5:
                    best[i] = {"loss": v, "iter": epoch}
                    painters[i].save_svg(per_args[i].output_dir, "best")
            del ev
            if epoch % 200 == 0:
                el = time.perf_counter() - t1
                rate = el / max(epoch, 1)
                print(f"  step {epoch:5d}/{a.num_iter}  "
                      f"best loss {min(b['loss'] for b in best):.4f}  "
                      f"eta {(a.num_iter - epoch) * rate / 60:4.1f} min")
        if a.save_interval > 0 and epoch % a.save_interval == 0:
            for i, p in enumerate(painters):
                p.save_svg(f"{per_args[i].output_dir}/svg_logs", f"iter{epoch}")

        # diffvg allocates outside PyTorch's caching allocator; leaving this
        # iteration's graph alive makes the next backward fall back to raw
        # cudaMalloc/cudaFree, which synchronises the device.
        del imgs, per, total

    if use_gpu:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t1
    for i, p in enumerate(painters):
        p.save_svg(per_args[i].output_dir, "final")

    win = int(np.argmin([b["loss"] for b in best]))
    src = Path(per_args[win].output_dir) / "best.svg"
    dst = out / "best_sketch.svg"
    if src.is_file():
        dst.write_bytes(src.read_bytes())

    print(f"\ndone in {elapsed:.1f}s ({elapsed / a.num_iter * 1e3:.1f} ms/step "
          f"for {a.num_sketches} sketch(es))")
    for i, b in enumerate(best):
        mark = "  <- best" if i == win else ""
        print(f"  sketch {i} (seed {seeds[i]:4d}): loss {b['loss']:.4f} "
              f"at step {b['iter']}{mark}")
    print(f"\nbest sketch: {dst}")
    print(f"all output:  {out}")


if __name__ == "__main__":
    main()
