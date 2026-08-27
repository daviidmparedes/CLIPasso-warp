#!/usr/bin/env python3
"""
Quality guardrails (project brief section 5).

Every speedup is reported as a (speedup, quality delta) pair. This computes the
quality half for a directory of CLIPasso runs:

  1. loss_eval            the repo's own un-augmented eval loss (read from config.npy)
  2. zero-shot top-1      CLIP ViT-B/32, prompt "A sketch of a(n) {class}"
  3. sketch->photo R@k    label-free retrieval against a photo gallery
  4. trajectory divergence  control-point drift vs a reference run, same image+seed
  5. contact sheet        target vs sketch, rendered side by side

ViT-B/32 is deliberately NOT the RN101 that drives the optimisation loss, so metrics
2 and 3 are semi-independent of the objective being optimised.

  python bench/guardrails.py --runs bench/results/baseline/shipped --tag shipped
  python bench/guardrails.py --runs bench/results/baseline/fixed_graphfree --tag fixed \
                             --compare-to bench/results/baseline/shipped
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import common
from common import ROOT, env_fingerprint

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


# ----------------------------------------------------------------- rendering
def render_svg(svg_path, device):
    """Rasterise a saved sketch SVG to a 1x3xHxW tensor in [0,1] on white.

    Mirrors Painter.get_image(): alpha-composite the RGBA render over white, then
    NHWC -> NCHW. Using the SVG (not the saved jpg) keeps this independent of the
    repo's matplotlib logging, which writes a 2-panel figure rather than the sketch.
    """
    import pydiffvg
    w, h, shapes, groups = pydiffvg.svg_to_scene(str(svg_path))
    scene = pydiffvg.RenderFunction.serialize_scene(w, h, shapes, groups)
    img = pydiffvg.RenderFunction.apply(w, h, 2, 2, 0, None, *scene)
    alpha = img[:, :, 3:4]
    img = alpha * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3,
                                             device=img.device) * (1 - alpha)
    return img.permute(2, 0, 1).unsqueeze(0).to(device).clamp(0, 1)


def svg_control_points(svg_path):
    """Flat (N,2) array of every control point in a saved sketch."""
    import pydiffvg
    _, _, shapes, _ = pydiffvg.svg_to_scene(str(svg_path))
    pts = [s.points.detach().cpu().numpy() for s in shapes if hasattr(s, "points")]
    return np.concatenate(pts, axis=0) if pts else np.zeros((0, 2))


# ----------------------------------------------------------------- CLIP side
class Evaluator:
    def __init__(self, device):
        import CLIP_.clip as clip
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device, jit=False)
        self.model.eval()
        self.clip = clip
        self.norm = torch.nn.functional.normalize
        self._mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def encode_tensor(self, img):
        """img: 1x3xHxW in [0,1] already at 224."""
        x = torch.nn.functional.interpolate(img, size=224, mode="bicubic", align_corners=False)
        x = ((x - self._mean) / self._std).to(next(self.model.parameters()).dtype)
        return self.norm(self.model.encode_image(x).float(), dim=-1)

    @torch.no_grad()
    def encode_pil(self, paths, bs=64):
        from PIL import Image
        feats = []
        for i in range(0, len(paths), bs):
            batch = torch.stack([self.preprocess(Image.open(p).convert("RGB"))
                                 for p in paths[i:i + bs]]).to(self.device)
            batch = batch.to(next(self.model.parameters()).dtype)
            feats.append(self.norm(self.model.encode_image(batch).float(), dim=-1))
        return torch.cat(feats)

    @torch.no_grad()
    def encode_prompts(self, class_names):
        # The paper's prompt. "a(n)" is kept verbatim rather than resolved to a/an,
        # so the numbers stay comparable to the published figures.
        toks = self.clip.tokenize([f"A sketch of a(n) {c}" for c in class_names]).to(self.device)
        return self.norm(self.model.encode_text(toks).float(), dim=-1)


# ----------------------------------------------------------------- run loading
def load_runs(run_dir):
    runs = []
    for cfg_path in sorted(Path(run_dir).glob("*/config.npy")):
        cfg = np.load(cfg_path, allow_pickle=True)[()]
        d = cfg_path.parent
        svg = d / "best_iter.svg"
        if not svg.exists():
            svg = d / "final_svg.svg"
        if not svg.exists():
            continue
        le = np.asarray(cfg["loss_eval"], dtype=float)
        runs.append({
            "name": d.name, "dir": d, "svg": svg,
            "target": cfg["target"], "num_paths": int(cfg["num_paths"]),
            "seed": int(cfg["seed"]),
            "loss_eval": le, "best_loss_eval": float(le.min()),
        })
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="directory of CLIPasso run subdirs")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--compare-to", default=None, help="reference run dir for trajectory divergence")
    ap.add_argument("--strokes", type=int, default=None,
                    help="restrict to one stroke count (needed for like-for-like deltas)")
    ap.add_argument("--gallery", type=int, default=2000, help="distractor pool size for retrieval")
    ap.add_argument("--out", default=str(ROOT / "bench" / "results" / "guardrails"))
    a = ap.parse_args()

    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import pydiffvg
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    pydiffvg.set_device(device)

    runs = load_runs(a.runs)
    if a.strokes is not None:
        runs = [r for r in runs if r["num_paths"] == a.strokes]
    if not runs:
        raise SystemExit(f"no runs with a saved SVG under {a.runs}")
    print(f"loaded {len(runs)} runs from {a.runs}")

    manifest = json.loads((ROOT / "data" / "manifest.json").read_text())
    by_path = {m["path"]: m for m in manifest}
    all_classes = sorted({m["class"] for m in manifest})
    prompt_of = {m["class"]: m["prompt_name"] for m in manifest}

    ev = Evaluator(device)

    # ---- render every sketch and encode it
    sk_feats, metas = [], []
    for r in runs:
        img = render_svg(r["svg"], device)
        sk_feats.append(ev.encode_tensor(img))
        m = by_path.get(r["target"])
        if m is None:                      # target outside the sampled corpus
            cls = Path(r["target"]).parent.name
            m = {"class": cls, "prompt_name": cls.replace("_", " "), "path": r["target"]}
        metas.append(m)
        r["class"] = m["class"]
    sk_feats = torch.cat(sk_feats)

    # ---- metric 2: zero-shot classification
    txt_all = ev.encode_prompts([prompt_of.get(c, c.replace("_", " ")) for c in all_classes])
    pred_all = (sk_feats @ txt_all.T).argmax(dim=1).cpu().numpy()
    true_all = np.array([all_classes.index(m["class"]) for m in metas])
    top1_125 = float((pred_all == true_all).mean())

    sub_classes = sorted({m["class"] for m in metas})
    txt_sub = ev.encode_prompts([prompt_of.get(c, c.replace("_", " ")) for c in sub_classes])
    pred_sub = (sk_feats @ txt_sub.T).argmax(dim=1).cpu().numpy()
    true_sub = np.array([sub_classes.index(m["class"]) for m in metas])
    top1_sub = float((pred_sub == true_sub).mean())

    # ---- metric 3: sketch -> photo retrieval
    gallery = manifest[:a.gallery]
    gal_paths = [g["path"] for g in gallery]
    for m in metas:                        # guarantee each true photo is in the gallery
        if m["path"] not in gal_paths:
            gal_paths.append(m["path"])
    print(f"encoding retrieval gallery ({len(gal_paths)} photos) ...")
    gal_feats = ev.encode_pil(gal_paths)
    sims = sk_feats @ gal_feats.T
    ranks = []
    for i, m in enumerate(metas):
        j = gal_paths.index(m["path"])
        ranks.append(int((sims[i] > sims[i, j]).sum().item()) + 1)
    ranks = np.array(ranks)
    r1, r5, r10 = [float((ranks <= k).mean()) for k in (1, 5, 10)]

    # ---- metric 1
    losses = np.array([r["best_loss_eval"] for r in runs])

    res = {
        "tag": a.tag, "n_runs": len(runs), "runs_dir": str(a.runs), "strokes_filter": a.strokes,
        "loss_eval_mean": float(losses.mean()), "loss_eval_std": float(losses.std()),
        "zeroshot_top1_125way": top1_125,
        "zeroshot_top1_subset": top1_sub, "subset_size": len(sub_classes),
        "retrieval_R1": r1, "retrieval_R5": r5, "retrieval_R10": r10,
        "retrieval_median_rank": float(np.median(ranks)),
        "gallery_size": len(gal_paths),
        "per_run": [{"name": r["name"], "class": r["class"], "num_paths": r["num_paths"],
                     "seed": r["seed"], "best_loss_eval": r["best_loss_eval"],
                     "retrieval_rank": int(rk),
                     "zeroshot_correct_125": bool(p == t)}
                    for r, rk, p, t in zip(runs, ranks, pred_all, true_all)],
        "env": env_fingerprint(),
    }

    # ---- metric 4: trajectory divergence vs a reference run
    if a.compare_to:
        ref = {r["name"]: r for r in load_runs(a.compare_to)
               if a.strokes is None or r["num_paths"] == a.strokes}
        divs, finals = [], []
        for r in runs:
            if r["name"] not in ref:
                continue
            rr = ref[r["name"]]
            # Compare final_svg, NOT best_iter: "best" can land on a different
            # iteration in each run, which would compare different points in the
            # trajectory and report divergence that is really just an index mismatch.
            fa, fb = r["dir"] / "final_svg.svg", rr["dir"] / "final_svg.svg"
            if fa.exists() and fb.exists():
                pa, pb = svg_control_points(fa), svg_control_points(fb)
                if pa.shape == pb.shape and pa.size:
                    finals.append(float(np.linalg.norm(pa - pb, axis=1).mean()))
            la, lb = r["loss_eval"], rr["loss_eval"]
            n = min(len(la), len(lb))
            divs.append(float(np.abs(la[:n] - lb[:n]).max()))
        res["trajectory"] = {
            "reference": str(a.compare_to),
            "n_matched": len(divs),
            "max_abs_loss_eval_diff": float(np.max(divs)) if divs else None,
            "mean_final_ctrlpoint_L2_px": float(np.mean(finals)) if finals else None,
            "loss_eval_curves_identical": bool(divs and np.max(divs) == 0.0),
        }

    # ---- metric 5: contact sheet
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
        sel = sorted(runs, key=lambda r: (r["class"], r["num_paths"], r["seed"]))[:24]
        cols = 6
        rows_n = int(np.ceil(len(sel) / cols)) * 2
        fig, axes = plt.subplots(rows_n, cols, figsize=(2.0 * cols, 2.1 * rows_n))
        axes = np.atleast_2d(axes)
        for i, r in enumerate(sel):
            cr, cc = (i // cols) * 2, i % cols
            axes[cr, cc].imshow(Image.open(r["target"]).convert("RGB")); axes[cr, cc].axis("off")
            axes[cr, cc].set_title(f"{r['class']}", fontsize=7)
            sk = render_svg(r["svg"], device)[0].permute(1, 2, 0).cpu().numpy()
            axes[cr + 1, cc].imshow(sk); axes[cr + 1, cc].axis("off")
            axes[cr + 1, cc].set_title(f"n={r['num_paths']} s{r['seed']}", fontsize=7)
        for ax in axes.ravel():
            if not ax.images:
                ax.axis("off")
        plt.tight_layout()
        sheet = outdir / f"contact_sheet_{a.tag}.png"
        plt.savefig(sheet, dpi=110); plt.close()
        res["contact_sheet"] = str(sheet)
        print(f"contact sheet -> {sheet}")
    except Exception as e:
        print(f"contact sheet failed: {type(e).__name__}: {e}")

    print("\n" + "=" * 66)
    print(f"GUARDRAILS  [{a.tag}]   {len(runs)} runs")
    print("=" * 66)
    print(f"  loss_eval (mean+/-sd)      {res['loss_eval_mean']:.5f} +/- {res['loss_eval_std']:.5f}")
    print(f"  zero-shot top-1 (125-way)  {100*top1_125:5.1f}%")
    print(f"  zero-shot top-1 ({len(sub_classes)}-way)  {100*top1_sub:5.1f}%")
    print(f"  retrieval R@1 / R@5 / R@10 {100*r1:5.1f}% / {100*r5:5.1f}% / {100*r10:5.1f}%   "
          f"(gallery {len(gal_paths)})")
    print(f"  retrieval median rank      {np.median(ranks):.0f}")
    if "trajectory" in res:
        t = res["trajectory"]
        print(f"  vs {Path(a.compare_to).name}:")
        print(f"    loss_eval curves identical  {t['loss_eval_curves_identical']}")
        print(f"    max |d loss_eval|           {t['max_abs_loss_eval_diff']:.3e}")
        if t["mean_final_ctrlpoint_L2_px"] is not None:
            print(f"    mean final ctrl-pt L2       {t['mean_final_ctrlpoint_L2_px']:.4f} px")
    print("=" * 66)

    fp = outdir / f"guardrails_{a.tag}.json"
    fp.write_text(json.dumps(res, indent=1))
    print(f"wrote {fp}")


if __name__ == "__main__":
    main()
