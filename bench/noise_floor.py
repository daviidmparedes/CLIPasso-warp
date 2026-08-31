#!/usr/bin/env python3
"""
Run-to-run noise floor for the quality metrics.

CLIPasso is not deterministic: diffvg's backward accumulates with atomicAdd, and
over 2001 Adam steps that chaos-amplifies to ~48 px of mean control-point drift
between two runs with identical config and seed. Every quality delta we report is
therefore only meaningful relative to how far two *identical* runs land from each
other. This measures that.

Given two directories of runs produced by the same configuration, it matches them
by name (image + strokes + seed) and reports, per metric, the paired |A - B|.
The most useful output is `cross_sim`: the CLIP cosine similarity between the two
replicates' final sketches. That is the ceiling on agreement the method itself can
deliver, so any truncated sketch that reaches it is, in CLIP's view, as good as a
rerun -- which is the threshold the early-stopping decision turns on.

  python bench/noise_floor.py --a bench/results/baseline/shipped \
                              --b bench/results/baseline/shipped_rerun \
                              --tag shipped_vs_rerun --strokes 16
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import common  # noqa: F401  (path setup)
from common import ROOT, env_fingerprint
from guardrails import (Evaluator, load_runs, render_svg, retrieval_stats,
                        svg_control_points, zeroshot_stats)


def _final_svg(run):
    """final_svg, not best_iter: 'best' lands on a different iteration in each
    replicate, so comparing best_iter would report an index mismatch as divergence."""
    p = run["dir"] / "final_svg.svg"
    return p if p.exists() else run["svg"]


@torch.no_grad()
def _encode(ev, runs, device, bs=32):
    feats, buf = [], []
    for i, r in enumerate(runs):
        buf.append(render_svg(_final_svg(r), device)[0])
        if len(buf) == bs or i == len(runs) - 1:
            feats.append(ev.encode_tensor(torch.stack(buf)))
            buf = []
    return torch.cat(feats)


def _summary(x):
    x = np.asarray(x, float)
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            "p95": float(np.percentile(x, 95)), "max": float(x.max()), "n": int(len(x))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--strokes", type=int, default=None)
    ap.add_argument("--gallery", type=int, default=2000)
    ap.add_argument("--out", default=str(ROOT / "bench" / "results" / "quality_curve"))
    a = ap.parse_args()

    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    common.require_free_gpu_memory()
    import pydiffvg
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    pydiffvg.set_device(device)

    ra = {r["name"]: r for r in load_runs(a.a)}
    rb = {r["name"]: r for r in load_runs(a.b)}
    names = sorted(set(ra) & set(rb))
    if a.strokes is not None:
        names = [n for n in names if ra[n]["num_paths"] == a.strokes]
    if not names:
        raise SystemExit(f"no run names common to {a.a} and {a.b}")
    A = [ra[n] for n in names]
    B = [rb[n] for n in names]
    print(f"{len(names)} matched replicate pairs")

    manifest = json.loads((ROOT / "data" / "manifest.json").read_text())
    by_path = {m["path"]: m for m in manifest}
    all_classes = sorted({m["class"] for m in manifest})
    prompt_of = {m["class"]: m["prompt_name"] for m in manifest}
    metas = []
    for r in A:
        m = by_path.get(r["target"])
        if m is None:
            cls = Path(r["target"]).parent.name
            m = {"class": cls, "prompt_name": cls.replace("_", " "), "path": r["target"]}
        metas.append(m)

    ev = Evaluator(device)
    txt = ev.encode_prompts([prompt_of.get(c, c.replace("_", " ")) for c in all_classes])
    true_idx = torch.as_tensor([all_classes.index(m["class"]) for m in metas], device=device)
    # A subset classifier needs at least two classes to have a competing prompt;
    # with one class the margin is -inf and every downstream statistic is nan.
    sub_classes = sorted({m["class"] for m in metas})
    has_sub = len(sub_classes) >= 2
    if has_sub:
        txt_sub = ev.encode_prompts(
            [prompt_of.get(c, c.replace("_", " ")) for c in sub_classes])
        true_sub = torch.as_tensor(
            [sub_classes.index(m["class"]) for m in metas], device=device)

    gal_paths = [g["path"] for g in manifest[:a.gallery]]
    for m in metas:
        if m["path"] not in gal_paths:
            gal_paths.append(m["path"])
    print(f"encoding retrieval gallery ({len(gal_paths)} photos) ...")
    gal_feats = ev.encode_pil(gal_paths)
    true_j = torch.as_tensor([gal_paths.index(m["path"]) for m in metas], device=device)

    fa, fb = _encode(ev, A, device), _encode(ev, B, device)
    cross_sim = (fa * fb).sum(-1).cpu().numpy()

    okA, mgA = zeroshot_stats(fa, txt, true_idx)
    okB, mgB = zeroshot_stats(fb, txt, true_idx)
    if has_sub:
        okAs, mgAs = zeroshot_stats(fa, txt_sub, true_sub)
        okBs, mgBs = zeroshot_stats(fb, txt_sub, true_sub)
    rkA, stA = retrieval_stats(fa, gal_feats, true_j)
    rkB, stB = retrieval_stats(fb, gal_feats, true_j)

    leA = np.array([r["best_loss_eval"] for r in A])
    leB = np.array([r["best_loss_eval"] for r in B])

    drift = []
    for x, y in zip(A, B):
        pa, pb = svg_control_points(_final_svg(x)), svg_control_points(_final_svg(y))
        if pa.shape == pb.shape and pa.size:
            drift.append(float(np.linalg.norm(pa - pb, axis=1).mean()))

    res = {
        "tag": a.tag, "a": a.a, "b": a.b, "strokes_filter": a.strokes,
        "n_pairs": len(names), "names": names,
        # the reproducibility ceiling -- the headline number
        "cross_sim": _summary(cross_sim),
        "ctrlpoint_drift_px": _summary(drift) if drift else None,
        "abs_delta": {
            "best_loss_eval": _summary(np.abs(leA - leB)),
            "zeroshot_margin_125way": _summary(np.abs(mgA - mgB)),
            **({"zeroshot_margin_subset": _summary(np.abs(mgAs - mgBs))} if has_sub else {}),
            "sim_true_photo": _summary(np.abs(stA - stB)),
            "log_retrieval_rank": _summary(np.abs(np.log(rkA) - np.log(rkB))),
        },
        "decision_flips": {
            "zeroshot_125way": float((okA != okB).mean()),
            **({"zeroshot_subset": float((okAs != okBs).mean())} if has_sub else {}),
        },
        "aggregate_delta": {   # what a *reported* mean would move by, by luck alone
            "top1_125way": float(okA.mean() - okB.mean()),
            **({"top1_subset": float(okAs.mean() - okBs.mean())} if has_sub else {}),
            "median_rank": float(np.median(rkA) - np.median(rkB)),
        },
        "env": env_fingerprint(),
    }

    print("\n" + "=" * 70)
    print(f"NOISE FLOOR  [{a.tag}]  {len(names)} replicate pairs, n={a.strokes} strokes")
    print("=" * 70)
    c = res["cross_sim"]
    print(f"  CLIP sim between replicates  {c['mean']:.4f} +/- {c['sd']:.4f}  "
          f"(min {min(cross_sim):.4f})")
    print("     ^ agreement ceiling: no truncation can beat this without luck")
    if res["ctrlpoint_drift_px"]:
        d = res["ctrlpoint_drift_px"]
        print(f"  control-point drift          {d['mean']:.2f} +/- {d['sd']:.2f} px")
    print("  paired |A - B| per metric:")
    for k, v in res["abs_delta"].items():
        print(f"    {k:<26} mean {v['mean']:.5f}  p95 {v['p95']:.5f}  max {v['max']:.5f}")
    flips = res["decision_flips"]
    sub_txt = (f", subset {100*flips['zeroshot_subset']:.1f}%" if has_sub
               else "  (subset skipped: only one class present)")
    print(f"  zero-shot decision flips between replicates: "
          f"125-way {100*flips['zeroshot_125way']:.1f}%{sub_txt}")
    print("=" * 70)

    fp = outdir / f"noise_floor_{a.tag}.json"
    fp.write_text(json.dumps(res, indent=1))
    print(f"wrote {fp}")


if __name__ == "__main__":
    main()
