#!/usr/bin/env python3
"""
P1 verdict: does sketch quality saturate early enough for Tier 2.1 to be worth it?

Combines two measurements:

  bench/quality_vs_iter.py  -> quality at every truncation point, per run
  bench/noise_floor.py      -> how far two identical runs land from each other

and asks the only question that matters for early stopping: *at what iteration is
a truncated sketch already as close to the full-length sketch as a rerun would be?*
Below that iteration, truncation costs something measurable. Above it, the cost is
smaller than the noise the method already has, so stopping there is free.

It also reports signal-to-noise per metric, because several of the guardrails turn
out to have a noise floor larger than their entire trajectory-wide signal, which
makes them incapable of detecting the effect regardless of the answer.

  python bench/analyze_p1.py --curve bench/results/quality_curve/quality_vs_iter_shipped.json \
                             --floor bench/results/quality_curve/noise_floor_shipped_vs_rerun.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from common import ROOT

# metric -> (key in curve["raw"], key in floor["abs_delta"], higher is better)
METRICS = [
    ("clip_self_sim",          "clip_self_sim",  None,                     True),
    ("loss_eval",              None,             "best_loss_eval",         False),
    ("zeroshot_margin_125way", "margin",         "zeroshot_margin_125way", True),
    ("zeroshot_margin_subset", "margin_sub",     "zeroshot_margin_subset", True),
    ("sim_true_photo",         "sim_true_photo", "sim_true_photo",         True),
    ("log_retrieval_rank",     None,             "log_retrieval_rank",     False),
]


def smooth(y, w):
    """Centred moving average. The raw curves wobble by more than the effect we
    are looking for, and a 'first iteration that never falls back out' rule on an
    unsmoothed noisy curve always returns something near the end regardless of
    where the real plateau is."""
    y = np.asarray(y, float)
    if w <= 1:
        return y
    pad = w // 2
    yp = np.pad(y, pad, mode="edge")
    k = np.ones(w) / w
    return np.convolve(yp, k, mode="valid")[:len(y)]


def first_cross(curve, iters, thresh, higher_is_better):
    """Earliest iteration from which the curve is on the good side of `thresh`
    and stays there."""
    ok = (curve >= thresh) if higher_is_better else (curve <= thresh)
    i = len(ok)
    while i > 0 and ok[i - 1]:
        i -= 1
    return int(iters[i]) if i < len(ok) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", required=True)
    ap.add_argument("--floor", required=True)
    ap.add_argument("--window", type=int, default=11,
                    help="moving-average window in snapshots (11 = 110 iterations)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cur = json.loads(Path(a.curve).read_text())
    flo = json.loads(Path(a.floor).read_text())
    iters = np.asarray(cur["iters"])
    last = int(iters[-1])
    raw = cur["raw"]
    runs = cur["per_run"]
    n_run = cur["n_runs"]

    # loss_eval is not in raw (it comes from config.npy), so rebuild it per run
    # from the aggregate curve when a per-run breakdown is not needed.
    arrays = {
        "clip_self_sim": np.asarray(raw["clip_self_sim"]),
        "zeroshot_margin_125way": np.asarray(raw["margin"]),
        "zeroshot_margin_subset": np.asarray(raw["margin_sub"]),
        "sim_true_photo": np.asarray(raw["sim_true_photo"]),
        "log_retrieval_rank": np.log(np.asarray(raw["rank"], float)),
        "loss_eval": np.tile(np.asarray(cur["curves"]["loss_eval"])[:, None], (1, n_run)),
    }

    print("=" * 84)
    print(f"P1 VERDICT   curve={Path(a.curve).name}   floor={Path(a.floor).name}")
    print(f"             {n_run} runs, {len(iters)} snapshots, smoothing window "
          f"{a.window} snapshots ({a.window * (iters[1]-iters[0])} iterations)")
    print("=" * 84)

    # ---------------------------------------------------------------- part 1
    # Can each metric even see the effect? Signal = how far the metric moves over
    # the whole trajectory. Noise = how far two identical runs land apart.
    print("\n1. SIGNAL-TO-NOISE  (can this guardrail detect anything?)\n")
    print(f"{'metric':<24}{'signal':>11}{'noise floor':>13}{'SNR':>8}   verdict")
    snr_rows = {}
    for name, _, fkey, _ in METRICS:
        c = np.asarray(cur["curves"][name], float)
        signal = abs(float(c[-1] - c[0]))
        if fkey is None or fkey not in flo["abs_delta"]:
            print(f"{name:<24}{signal:>11.4f}{'-':>13}{'-':>8}   no paired floor measured")
            continue
        noise = flo["abs_delta"][fkey]["mean"]
        snr = signal / noise if noise else float("inf")
        verdict = ("usable" if snr >= 3 else
                   "marginal" if snr >= 1 else "UNUSABLE: noise exceeds signal")
        snr_rows[name] = {"signal": signal, "noise": noise, "snr": snr, "verdict": verdict}
        print(f"{name:<24}{signal:>11.4f}{noise:>13.5f}{snr:>8.2f}   {verdict}")

    # ---------------------------------------------------------------- part 2
    # The headline. cross_sim is the CLIP agreement two identical runs reach; a
    # snapshot that reaches it is as close to "the answer" as a rerun is.
    print("\n2. TRUNCATION POINT  (where a truncated sketch matches a rerun)\n")
    cs_mean = flo["cross_sim"]["mean"]
    cs_sd = flo["cross_sim"]["sd"]
    n_pairs = flo["n_pairs"]
    print(f"   replicate agreement ceiling: {cs_mean:.4f} +/- {cs_sd:.4f} "
          f"(n={n_pairs} pairs)")

    self_sim = arrays["clip_self_sim"]
    agg = smooth(self_sim.mean(axis=1), a.window)
    trunc = {}
    for label, thresh in (("mean floor", cs_mean),
                          ("conservative (floor + 1 sd)", cs_mean + cs_sd),
                          ("lenient (floor - 1 sd)", cs_mean - cs_sd)):
        it = first_cross(agg, iters, thresh, True)
        sp = (last / it) if it else None
        trunc[label] = {"threshold": thresh, "iter": it, "speedup": sp}
        print(f"   {label:<28} sim >= {thresh:.4f}  ->  iteration "
              f"{(str(it) if it else 'never'):>5}   {(f'{sp:.2f}x' if sp else '-'):>7}")

    per_run_it = []
    for ri in range(n_run):
        it = first_cross(smooth(self_sim[:, ri], a.window), iters, cs_mean, True)
        if it is not None:
            per_run_it.append(it)
    if per_run_it:
        print(f"   per-run: mean {np.mean(per_run_it):.0f}  median "
              f"{np.median(per_run_it):.0f}  p90 {np.percentile(per_run_it, 90):.0f}  "
              f"max {max(per_run_it)}  ({len(per_run_it)}/{n_run} reach the floor)")
        print(f"   -> a p90-safe truncation at iteration "
              f"{np.percentile(per_run_it, 90):.0f} is "
              f"{last/np.percentile(per_run_it, 90):.2f}x")

    # ---------------------------------------------------------------- part 3
    # For every metric: earliest k whose paired gap to the final value is smaller
    # than the replicate floor. Same question, asked of each guardrail separately.
    print("\n3. PER-METRIC: earliest iteration whose gap to iteration "
          f"{last} is below the noise floor\n")
    print(f"{'metric':<24}{'|gap| floor':>13}{'iteration':>11}{'speedup':>10}")
    permetric = {}
    for name, _, fkey, higher in METRICS:
        if fkey is None or fkey not in flo["abs_delta"]:
            continue
        noise = flo["abs_delta"][fkey]["mean"]
        arr = arrays[name]
        gap = np.abs(arr - arr[-1:]).mean(axis=1)      # paired, then averaged
        it = first_cross(smooth(gap, a.window), iters, noise, False)
        sp = (last / it) if it else None
        permetric[name] = {"floor": noise, "iter": it, "speedup": sp}
        print(f"{name:<24}{noise:>13.5f}{(str(it) if it else 'never'):>11}"
              f"{(f'{sp:.2f}x' if sp else '-'):>10}")

    # ---------------------------------------------------------------- part 4
    print("\n4. BY STROKE COUNT  (truncation iteration at the mean replicate floor)\n")
    by_n = {}
    npaths = np.array([r["num_paths"] for r in runs])
    for n in sorted(set(npaths.tolist())):
        sel = npaths == n
        it = first_cross(smooth(self_sim[:, sel].mean(axis=1), a.window), iters, cs_mean, True)
        by_n[int(n)] = {"n_runs": int(sel.sum()), "iter": it,
                        "speedup": (last / it) if it else None,
                        "final_self_sim_at_iter0": float(self_sim[0, sel].mean())}
        print(f"   n={n:<3} strokes  ({sel.sum():2d} runs)  iteration "
              f"{(str(it) if it else 'never'):>5}   "
              f"{(f'{last/it:.2f}x' if it else '-'):>7}")

    # ---------------------------------------------------------------- part 5
    loss_curve = np.asarray(cur["curves"]["loss_eval"], float)
    loss_gain95 = cur["saturation"]["loss_eval"]["gain5pct"]["iter"]
    best_trunc = trunc["mean floor"]["iter"]
    print("\n5. THE LOSS CURVE OVERSTATES THE HEADROOM\n")
    print(f"   loss_eval reaches 95% of its total gain at iteration {loss_gain95} "
          f"({last/loss_gain95:.2f}x)")
    if best_trunc:
        print(f"   but CLIP-space agreement with the final sketch only reaches the")
        print(f"   replicate floor at iteration {best_trunc} ({last/best_trunc:.2f}x)")
        print(f"   -> the objective flattens {best_trunc/loss_gain95:.1f}x earlier than the")
        print(f"      sketch stops changing. Stopping on loss_eval alone would cut")
        print(f"      {best_trunc - loss_gain95} iterations of real, visible refinement.")
    print(f"   loss_eval at iter {loss_gain95}: {loss_curve[np.searchsorted(iters, loss_gain95)]:.4f}"
          f"   at iter {last}: {loss_curve[-1]:.4f}")

    # ---------------------------------------------------------------- part 6
    # A metric with SNR < 1 per run is not permanently useless -- averaging over
    # more runs shrinks the standard error of the mean as 1/sqrt(N). This says how
    # many runs each guardrail would need to resolve a delta of a given size,
    # which is the concrete form of "widen the eval set".
    print("\n6. HOW MANY RUNS TO DETECT A DELTA  (paired, alpha=0.05, power=0.8)\n")
    print("   delta is expressed as a fraction of the metric's own trajectory-wide")
    print("   signal (iteration 0 -> 2000), so the columns are comparable.\n")
    print(f"{'metric':<24}{'50% of signal':>15}{'25%':>8}{'10%':>8}{'5%':>8}")
    power = {}
    for name, _, fkey, _ in METRICS:
        if fkey is None or fkey not in flo["abs_delta"]:
            continue
        # paired |A-B| has mean ~ sigma_d * sqrt(2/pi) for a zero-mean normal
        # difference, so sigma_d = mean|A-B| * sqrt(pi/2)
        sigma_d = flo["abs_delta"][fkey]["mean"] * np.sqrt(np.pi / 2)
        c = np.asarray(cur["curves"][name], float)
        signal = abs(float(c[-1] - c[0]))
        row, cells = {}, []
        for frac in (0.5, 0.25, 0.10, 0.05):
            d = frac * signal
            n = int(np.ceil((2.8 * sigma_d / d) ** 2)) if d > 0 else None
            row[f"{int(frac*100)}pct"] = n
            cells.append(f"{n:>8}" if n and n < 100000 else f"{'>1e5':>8}")
        power[name] = row
        print(f"{name:<24}{cells[0]:>15}{cells[1]}{cells[2]}{cells[3]}")
    print(f"\n   current eval set: {n_run} runs total, {n_run//3} per stroke count.")

    res = {"curve": a.curve, "floor": a.floor, "window": a.window,
           "power_n_runs_needed": power,
           "n_runs": n_run, "n_floor_pairs": n_pairs,
           "cross_sim_mean": cs_mean, "cross_sim_sd": cs_sd,
           "signal_to_noise": snr_rows, "truncation": trunc,
           "per_run_truncation": {
               "mean": float(np.mean(per_run_it)) if per_run_it else None,
               "median": float(np.median(per_run_it)) if per_run_it else None,
               "p90": float(np.percentile(per_run_it, 90)) if per_run_it else None,
               "n": len(per_run_it)},
           "per_metric_gap": permetric, "by_stroke_count": by_n,
           "loss_eval_gain95_iter": loss_gain95}
    out = Path(a.out) if a.out else Path(a.curve).parent / "p1_verdict.json"
    out.write_text(json.dumps(res, indent=1))
    print("\n" + "=" * 84)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
