#!/usr/bin/env python3
"""
Quality as a function of truncation iteration (project plan P1).

CLIPasso saves svg_logs/svg_iter{k}.svg every save_interval steps, so a completed
run already contains its own entire optimisation trajectory. This walks those
snapshots and asks, for every k: *if we had stopped here, how good would the
sketch be?*  That answers whether Tier 2.1 (early stopping) is worth implementing
without running a single new optimisation.

The distinction that matters is between the two curves:

  loss_eval        the objective CLIPasso is minimising -- keeps creeping down
  zero-shot / R@k  whether a human-independent model still recognises the object

If recognisability plateaus well before the loss does, truncation is nearly free
and 2.1 is worth building. If they plateau together, 2.1 buys only what the loss
curve already said it would.

  python bench/quality_vs_iter.py --runs bench/results/baseline/shipped --tag shipped
  python bench/quality_vs_iter.py --runs bench/results/baseline/shipped --tag shipped \
                                  --strokes 16 --stride 2
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

import common  # noqa: F401  (path setup)
from common import ROOT, env_fingerprint
from guardrails import Evaluator, load_runs, render_svg, retrieval_stats, zeroshot_stats


def snapshot_iters(run_dir, stride=1):
    """Sorted iteration numbers with a saved SVG, subsampled by `stride`."""
    its = []
    for p in (Path(run_dir) / "svg_logs").glob("svg_iter*.svg"):
        m = re.fullmatch(r"svg_iter(\d+)\.svg", p.name)
        if m:
            its.append(int(m.group(1)))
    its.sort()
    return its[::stride]


@torch.no_grad()
def encode_snapshots(ev, paths, device, bs=32, progress=None):
    """Render each SVG and encode it. Rendering is one diffvg call per scene --
    diffvg has no batched entry point -- but the CLIP side batches, which is where
    the time would otherwise go."""
    feats, buf = [], []
    for i, sp in enumerate(paths):
        buf.append(render_svg(sp, device)[0])
        if len(buf) == bs or i == len(paths) - 1:
            feats.append(ev.encode_tensor(torch.stack(buf)))
            buf = []
            if progress and (len(feats) % 20 == 0 or i == len(paths) - 1):
                progress(i + 1, len(paths))
    return torch.cat(feats)


def _settle(ok, iters):
    """Start of the final unbroken run of True in `ok`, as an iteration number.

    Walking back from the end rather than forward from the start means a curve
    that crosses the threshold early, falls back out, and only later returns is
    credited with the later iteration -- which is what a truncation decision
    actually depends on.
    """
    idx = len(ok)
    while idx > 0 and ok[idx - 1]:
        idx -= 1
    return int(iters[idx]) if idx < len(ok) else None


def frac_of_gain(curve, iters, tol, plateau_frac=0.1, min_span=1e-6):
    """Earliest iteration from which the curve holds >= (1 - tol) of its total gain.

    "Gain" is measured from the value at iteration 0 to the plateau (mean of the
    last `plateau_frac` of the trajectory), so this is scale-free and direction-free:
    it works for loss_eval (decreasing), the zero-shot margin (increasing through
    zero, where a relative tolerance would be meaningless because the final value
    is ~0.004), and retrieval rank (decreasing) with one definition.

    Returns (iteration, plateau, span). span ~ 0 means the metric never moved, so
    the saturation point carries no information and the iteration is returned as None.
    """
    curve = np.asarray(curve, dtype=float)
    k = max(1, int(len(curve) * plateau_frac))
    plateau = float(curve[-k:].mean())
    start = float(curve[0])
    span = plateau - start
    if abs(span) < min_span:
        return None, plateau, span
    progress = (curve - start) / span          # 0 at iter 0, ~1 at the plateau
    return _settle(progress >= (1.0 - tol), iters), plateau, span


def first_within(curve, iters, tol, higher_is_better, plateau_frac=0.1):
    """Earliest iteration from which `curve` stays within `tol` *relative to its
    plateau value*. Kept alongside frac_of_gain because this is the definition the
    earlier loss_eval analysis used ("within 5% of final best"), and loss_eval is
    the one metric where a relative tolerance is meaningful -- it is bounded away
    from zero and its scale is interpretable.
    """
    curve = np.asarray(curve, dtype=float)
    k = max(1, int(len(curve) * plateau_frac))
    plateau = float(curve[-k:].mean())
    if higher_is_better:
        ok = curve >= plateau - abs(plateau) * tol
    else:
        ok = curve <= plateau + abs(plateau) * tol
    return _settle(ok, iters), plateau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "bench" / "results" / "baseline" / "shipped"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--strokes", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1,
                    help="keep every Nth saved snapshot (1 = all of them)")
    ap.add_argument("--gallery", type=int, default=2000)
    ap.add_argument("--out", default=str(ROOT / "bench" / "results" / "quality_curve"))
    a = ap.parse_args()

    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    common.require_free_gpu_memory()
    import pydiffvg
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    pydiffvg.set_device(device)

    runs = load_runs(a.runs)
    if a.strokes is not None:
        runs = [r for r in runs if r["num_paths"] == a.strokes]
    runs = [r for r in runs if (r["dir"] / "svg_logs").is_dir()]
    if not runs:
        raise SystemExit(f"no runs with svg_logs under {a.runs}")

    iters = snapshot_iters(runs[0]["dir"], a.stride)
    runs = [r for r in runs if snapshot_iters(r["dir"], a.stride) == iters]
    print(f"{len(runs)} runs x {len(iters)} snapshots = {len(runs) * len(iters)} sketches "
          f"(iters {iters[0]}..{iters[-1]} step {iters[1] - iters[0]})")

    manifest = json.loads((ROOT / "data" / "manifest.json").read_text())
    by_path = {m["path"]: m for m in manifest}
    all_classes = sorted({m["class"] for m in manifest})
    prompt_of = {m["class"]: m["prompt_name"] for m in manifest}

    metas = []
    for r in runs:
        m = by_path.get(r["target"])
        if m is None:
            cls = Path(r["target"]).parent.name
            m = {"class": cls, "prompt_name": cls.replace("_", " "), "path": r["target"]}
        metas.append(m)
        r["class"] = m["class"]

    ev = Evaluator(device)
    txt = ev.encode_prompts([prompt_of.get(c, c.replace("_", " ")) for c in all_classes])
    true_idx = torch.as_tensor([all_classes.index(m["class"]) for m in metas], device=device)

    # The 125-way prompt set puts n=16 sketches near the floor (top-1 ~10%, R@1 0%),
    # which cannot resolve a saturation point. The subset classifier -- only the
    # classes actually present in this run set -- is the same measurement with a
    # usable dynamic range, so both are reported.
    sub_classes = sorted({m["class"] for m in metas})
    txt_sub = ev.encode_prompts([prompt_of.get(c, c.replace("_", " ")) for c in sub_classes])
    true_sub = torch.as_tensor([sub_classes.index(m["class"]) for m in metas], device=device)

    gal_paths = [g["path"] for g in manifest[:a.gallery]]
    for m in metas:
        if m["path"] not in gal_paths:
            gal_paths.append(m["path"])
    print(f"encoding retrieval gallery ({len(gal_paths)} photos) ...")
    gal_feats = ev.encode_pil(gal_paths)
    true_j = torch.as_tensor([gal_paths.index(m["path"]) for m in metas], device=device)

    # ---- sweep the trajectory. One iteration at a time across all runs, so every
    # metric at iteration k is computed over the same population as at 2000.
    # Features are kept (n_it x n_run x 512, ~19 MB) so the self-similarity metric
    # below can compare each snapshot against its own run's final sketch.
    n_it, n_run = len(iters), len(runs)
    correct = np.zeros((n_it, n_run), bool)
    margin = np.zeros((n_it, n_run))
    correct_sub = np.zeros((n_it, n_run), bool)
    margin_sub = np.zeros((n_it, n_run))
    rank = np.zeros((n_it, n_run), int)
    sim_true = np.zeros((n_it, n_run))
    all_feats = []

    for ti, it in enumerate(iters):
        paths = [r["dir"] / "svg_logs" / f"svg_iter{it}.svg" for r in runs]
        feats = encode_snapshots(ev, paths, device)
        all_feats.append(feats)
        correct[ti], margin[ti] = zeroshot_stats(feats, txt, true_idx)
        correct_sub[ti], margin_sub[ti] = zeroshot_stats(feats, txt_sub, true_sub)
        rank[ti], sim_true[ti] = retrieval_stats(feats, gal_feats, true_j)
        if ti % 10 == 0 or ti == n_it - 1:
            print(f"  iter {it:5d}  top1 {100*correct[ti].mean():5.1f}%  "
                  f"top1_sub {100*correct_sub[ti].mean():5.1f}%  "
                  f"medrank {np.median(rank[ti]):6.0f}  "
                  f"margin {margin[ti].mean():+.4f}", flush=True)

    # ---- perceptual convergence: cosine similarity between each snapshot and the
    # SAME run's final sketch. Unlike the accuracy metrics this has no floor
    # problem and no dependence on class labels -- it answers directly "has this
    # sketch stopped changing in the space CLIP actually sees?", which is the
    # question early stopping turns on.
    feats_t = torch.stack(all_feats)                       # n_it x n_run x D
    self_sim = (feats_t * feats_t[-1:]).sum(-1).cpu().numpy()

    # ---- loss_eval, aligned to the same iteration grid
    loss = np.full((n_it, n_run), np.nan)
    for ri, r in enumerate(runs):
        le = np.asarray(r["loss_eval"], float)
        ev_int = 10  # eval_interval; snapshots and evals share the same cadence
        for ti, it in enumerate(iters):
            j = it // ev_int
            if j < len(le):
                loss[ti, ri] = le[j]

    curves = {
        "zeroshot_top1_125way": correct.mean(axis=1),
        "zeroshot_margin_125way": margin.mean(axis=1),
        "zeroshot_top1_subset": correct_sub.mean(axis=1),
        "zeroshot_margin_subset": margin_sub.mean(axis=1),
        "retrieval_R1": (rank <= 1).mean(axis=1),
        "retrieval_R10": (rank <= 10).mean(axis=1),
        "retrieval_median_rank": np.median(rank, axis=1),
        "log_retrieval_rank": np.log(rank).mean(axis=1),
        "sim_true_photo": sim_true.mean(axis=1),
        "clip_self_sim": self_sim.mean(axis=1),
        "loss_eval": np.nanmean(loss, axis=1),
    }
    higher = {"zeroshot_top1_125way": True, "zeroshot_margin_125way": True,
              "zeroshot_top1_subset": True, "zeroshot_margin_subset": True,
              "retrieval_R1": True, "retrieval_R10": True,
              "retrieval_median_rank": False, "log_retrieval_rank": False,
              "sim_true_photo": True, "clip_self_sim": True, "loss_eval": False}

    # ---- saturation: earliest iteration each aggregate curve settles
    last = iters[-1]
    sat = {}
    for name, c in curves.items():
        sat[name] = {}
        for tol in (0.01, 0.02, 0.05, 0.10):
            it, plateau, span = frac_of_gain(c, iters, tol)
            sat[name][f"gain{int(tol*100)}pct"] = {
                "iter": it, "plateau": plateau, "span": span,
                "speedup": (last / it) if it else None,
            }
        for tol in (0.01, 0.02, 0.05):
            it, plateau = first_within(c, iters, tol, higher[name])
            sat[name][f"rel{int(tol*100)}pct"] = {
                "iter": it, "plateau": plateau,
                "speedup": (last / it) if it else None,
            }

    # ---- per-run saturation for the continuous metrics (a distribution, not a point)
    per_run_sat = {}
    for name, arr in (("clip_self_sim", self_sim),
                      ("sim_true_photo", sim_true),
                      ("zeroshot_margin_subset", margin_sub),
                      ("loss_eval", loss)):
        for tol in (0.02, 0.05, 0.10):
            its_ = [frac_of_gain(arr[:, ri], iters, tol)[0] for ri in range(n_run)]
            good = [x for x in its_ if x is not None]
            per_run_sat[f"{name}_gain{int(tol*100)}pct"] = {
                "mean": float(np.mean(good)) if good else None,
                "median": float(np.median(good)) if good else None,
                "p90": float(np.percentile(good, 90)) if good else None,
                "max": int(np.max(good)) if good else None,
                "n_settled": len(good), "n": n_run,
            }

    res = {
        "tag": a.tag, "runs_dir": a.runs, "strokes_filter": a.strokes,
        "n_runs": n_run, "iters": iters,
        "curves": {k: [float(x) for x in v] for k, v in curves.items()},
        "saturation": sat,
        "per_run_saturation": per_run_sat,
        "per_run": [{"name": r["name"], "class": r["class"], "seed": r["seed"],
                     "num_paths": r["num_paths"]} for r in runs],
        "raw": {"correct": correct.astype(int).tolist(), "rank": rank.tolist(),
                "margin": margin.tolist(), "sim_true_photo": sim_true.tolist(),
                "correct_sub": correct_sub.astype(int).tolist(),
                "margin_sub": margin_sub.tolist(),
                "clip_self_sim": self_sim.tolist()},
        "env": env_fingerprint(),
    }
    fp = outdir / f"quality_vs_iter_{a.tag}.json"
    fp.write_text(json.dumps(res, indent=1))

    print("\n" + "=" * 78)
    print(f"QUALITY vs ITERATION  [{a.tag}]  {n_run} runs, {n_it} snapshots each")
    print("=" * 78)
    print("iteration at which the aggregate curve holds >= (1-tol) of its total")
    print("gain from iteration 0, and never falls back out again:\n")
    print(f"{'metric':<26}{'iter 0':>9}{'final':>9}{'90%':>7}{'95%':>7}"
          f"{'98%':>7}{'99%':>7}{'x@95%':>8}")
    for name in curves:
        sn = sat[name]
        f = lambda t: (str(sn[t]['iter']) if sn[t]['iter'] is not None else "-")  # noqa: E731
        sp = sn["gain5pct"]["speedup"]
        print(f"{name:<26}{curves[name][0]:>9.4f}{curves[name][-1]:>9.4f}"
              f"{f('gain10pct'):>7}{f('gain5pct'):>7}{f('gain2pct'):>7}{f('gain1pct'):>7}"
              f"{(f'{sp:.2f}x' if sp else '-'):>8}")
    print(f"\n  loss_eval, 'within x% of final value' (the earlier definition): "
          f"5% -> {sat['loss_eval']['rel5pct']['iter']}, "
          f"2% -> {sat['loss_eval']['rel2pct']['iter']}, "
          f"1% -> {sat['loss_eval']['rel1pct']['iter']}")
    print("-" * 78)
    print("per-run saturation iteration (distribution over runs):")
    for k, v in per_run_sat.items():
        if v["mean"] is None:
            continue
        print(f"  {k:<34} mean {v['mean']:6.0f}  median {v['median']:6.0f}  "
              f"p90 {v['p90']:6.0f}  max {v['max']:5d}  ({v['n_settled']}/{v['n']})")
    print("=" * 78)
    print(f"wrote {fp}")

    _plot(outdir / f"quality_vs_iter_{a.tag}.png", iters, curves, sat, a.tag, n_run)


def _plot(path, iters, curves, sat, tag, n_run):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"plot skipped: {type(e).__name__}: {e}")
        return
    panels = [("loss_eval", "loss_eval (the objective)", False),
              ("clip_self_sim", "CLIP sim to own final sketch", True),
              ("zeroshot_top1_subset", "zero-shot top-1, subset", True),
              ("zeroshot_margin_subset", "zero-shot margin, subset", True),
              ("sim_true_photo", "cos sim to true photo", True),
              ("log_retrieval_rank", "mean log retrieval rank", False)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    for ax, (key, title, higher) in zip(axes.ravel(), panels):
        c = np.asarray(curves[key])
        ax.plot(iters, c, lw=1.4)
        for tol, col in ((0.05, "tab:orange"), (0.02, "tab:green")):
            sk = sat[key][f"gain{int(tol*100)}pct"]
            if sk["iter"] is not None:
                ax.axvline(sk["iter"], color=col, ls="--", lw=1,
                           label=f"{100-int(tol*100)}% of gain: it {sk['iter']}")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(f"CLIPasso quality vs truncation iteration  [{tag}, {n_run} runs]",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"plot -> {path}")


if __name__ == "__main__":
    main()
