#!/usr/bin/env python3
"""
Is there a stopping rule that beats a fixed iteration count?

analyze_p1.py shows the iteration at which a sketch stops changing varies ~2x
across runs (median ~990, p90 ~1770). A fixed truncation therefore has to be set
for the slowest run, which throws away most of the available speedup. This tests
whether a cheap adaptive signal does better.

The candidate signal is control-point velocity: how far the Bezier points move per
iteration, read straight from the saved SVGs. It costs nothing to compute inside
the training loop (the points are already in memory, no CLIP forward, no render)
and it measures "the sketch has stopped changing" directly rather than through the
objective, which analyze_p1 shows flattens ~1.7x too early.

Every rule is scored the same way: apply it per run, then check whether the sketch
at the iteration it picked has reached the replicate agreement floor -- the point
at which the truncated sketch is as close to the full-length one as a rerun is.

  python bench/stopping_rule.py --curve bench/results/quality_curve/quality_vs_iter_shipped.json \
                                --floor bench/results/quality_curve/noise_floor_shipped_vs_rerun.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from common import ROOT
from guardrails import svg_control_points


def load_velocities(runs, iters, cache):
    """Mean per-control-point displacement between consecutive snapshots, px/iter."""
    iters_l = [int(x) for x in iters]
    if cache.exists():
        d = json.loads(cache.read_text())
        if d["iters"] == iters_l and d["names"] == [r["name"] for r in runs]:
            print(f"velocities from cache {cache}")
            return np.asarray(d["vel"])
    step = iters[1] - iters[0]
    vel = np.zeros((len(iters), len(runs)))
    for ri, r in enumerate(runs):
        d = Path(r["dir"])
        prev = svg_control_points(d / "svg_logs" / f"svg_iter{iters[0]}.svg")
        for ti, it in enumerate(iters[1:], start=1):
            cp = svg_control_points(d / "svg_logs" / f"svg_iter{it}.svg")
            if cp.shape == prev.shape and cp.size:
                vel[ti, ri] = float(np.linalg.norm(cp - prev, axis=1).mean()) / step
            prev = cp
        vel[0, ri] = vel[1, ri]
        print(f"  [{ri+1}/{len(runs)}] {r['name']}", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"iters": iters_l,
                                 "names": [r["name"] for r in runs],
                                 "vel": vel.tolist()}))
    return vel


def smooth_cols(a, w):
    if w <= 1:
        return a
    pad = w // 2
    k = np.ones(w) / w
    return np.stack([np.convolve(np.pad(a[:, i], pad, mode="edge"), k, mode="valid")[:len(a)]
                     for i in range(a.shape[1])], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", required=True)
    ap.add_argument("--floor", required=True)
    ap.add_argument("--window", type=int, default=11)
    ap.add_argument("--warmup", type=int, default=200,
                    help="never stop before this iteration")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cur = json.loads(Path(a.curve).read_text())
    flo = json.loads(Path(a.floor).read_text())
    iters = np.asarray(cur["iters"])
    last = int(iters[-1])
    runs = cur["per_run"]
    n_run = cur["n_runs"]
    self_sim = np.asarray(cur["raw"]["clip_self_sim"])
    floor = flo["cross_sim"]["mean"]

    # per_run in the curve JSON has names but not dirs; rebuild dirs from runs_dir
    base = Path(cur["runs_dir"])
    for r in runs:
        r["dir"] = base / r["name"]

    vel = load_velocities(runs, iters, Path(cur["runs_dir"]).parent.parent /
                          "quality_curve" / f"velocity_{cur['tag']}.json")
    vel_s = smooth_cols(vel, a.window)
    warm = int(np.searchsorted(iters, a.warmup))

    def score(stop_idx):
        """stop_idx: per-run index into `iters`. Returns cost and quality."""
        stop_idx = np.asarray(stop_idx, dtype=int)
        stop_it = iters[stop_idx]
        q = np.array([self_sim[stop_idx[i], i] for i in range(n_run)])
        return {
            "mean_iter": float(stop_it.mean()),
            "speedup": float(last / stop_it.mean()),
            "frac_at_floor": float((q >= floor).mean()),
            "mean_self_sim": float(q.mean()),
            "p10_self_sim": float(np.percentile(q, 10)),
            "min_self_sim": float(q.min()),
        }

    print("=" * 88)
    print(f"STOPPING RULES   {n_run} runs   replicate floor = {floor:.4f} "
          f"(n={flo['n_pairs']} pairs)")
    print("=" * 88)
    print("A rule is good if it reaches a high speedup while keeping frac>=floor near 1.0:")
    print("that is the fraction of sketches whose truncation cost is smaller than the")
    print("cost of simply rerunning the optimisation.\n")
    print(f"{'rule':<34}{'mean it':>9}{'speedup':>9}{'frac>=floor':>13}"
          f"{'mean sim':>10}{'p10 sim':>9}{'min sim':>9}")

    rows = {}

    # --- baseline family: a single fixed iteration for every run
    for it in (400, 600, 800, 1000, 1200, 1400, 1600):
        ti = int(np.searchsorted(iters, it))
        rows[f"fixed @ {it}"] = score(np.full(n_run, ti))

    # --- adaptive family: stop once smoothed velocity falls below tau px/iter
    for tau in (0.02, 0.01, 0.005, 0.002, 0.001):
        idx = []
        for i in range(n_run):
            below = np.where(vel_s[warm:, i] < tau)[0]
            idx.append(warm + int(below[0]) if len(below) else len(iters) - 1)
        rows[f"velocity < {tau} px/iter"] = score(np.array(idx))

    # --- adaptive family: velocity relative to each run's own early velocity,
    # which removes the dependence on stroke count and image scale
    ref = vel_s[warm:warm + 10].mean(axis=0)
    for frac in (0.5, 0.3, 0.2, 0.1):
        idx = []
        for i in range(n_run):
            below = np.where(vel_s[warm:, i] < frac * ref[i])[0]
            idx.append(warm + int(below[0]) if len(below) else len(iters) - 1)
        rows[f"velocity < {frac:g}x own early"] = score(np.array(idx))

    for name, r in rows.items():
        print(f"{name:<34}{r['mean_iter']:>9.0f}{r['speedup']:>8.2f}x"
              f"{100*r['frac_at_floor']:>12.0f}%{r['mean_self_sim']:>10.4f}"
              f"{r['p10_self_sim']:>9.4f}{r['min_self_sim']:>9.4f}")

    # --- the honest comparison: at matched mean cost, does adaptive beat fixed?
    print("\nmatched-cost comparison (adaptive rule vs the fixed cut with the same mean cost):")
    best = []
    for name, r in rows.items():
        if not name.startswith("velocity"):
            continue
        ti = int(np.searchsorted(iters, r["mean_iter"]))
        f = score(np.full(n_run, min(ti, len(iters) - 1)))
        delta = r["frac_at_floor"] - f["frac_at_floor"]
        best.append((delta, name, r, f))
        print(f"  {name:<32} it {r['mean_iter']:6.0f}  frac {100*r['frac_at_floor']:5.0f}%"
              f"   vs fixed@{iters[min(ti, len(iters)-1)]:<5d} frac {100*f['frac_at_floor']:5.0f}%"
              f"   {'+' if delta >= 0 else ''}{100*delta:.0f} pts")
    best.sort(reverse=True)
    if best:
        d, name, r, f = best[0]
        print(f"\n  best adaptive margin: {name} ({'+' if d >= 0 else ''}{100*d:.0f} pts "
              f"at {r['speedup']:.2f}x)")
        if d <= 0.02:
            print("  -> adaptivity does not pay here: at matched cost the fixed cut is")
            print("     as good, so 2.1 should ship as a fixed iteration budget.")

    out = Path(a.out) if a.out else Path(a.curve).parent / "stopping_rules.json"
    out.write_text(json.dumps({"floor": float(floor), "n_runs": int(n_run),
                               "rules": rows, "warmup": a.warmup,
                               "window": a.window}, indent=1))
    print("=" * 88)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
