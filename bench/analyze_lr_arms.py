#!/usr/bin/env python3
"""
N1 verdict: does a working learning-rate schedule pay?

Tabulates the arms produced by bench/run_lr_arms.sh against the const-lr control
(bench/results/batched_freeze) and judges every delta against the n=15
reproducibility floor from bench/results/quality_curve/noise_floor_rep15.json.

The comparisons that decide N1 are the same-budget pairs:

    C vs C0   1200 iterations, cosine decay vs plain truncation
    D vs D0    800 iterations, same

If decay does not beat truncation at matched cost, the schedule is not buying
anything that section 6's fixed budget does not already buy. Arm B (full 2001
iterations with decay) asks the separate question of whether settling improves the
final sketch at no saving at all.

Speedups are arithmetic -- 2001/1200 -- not measured: this GPU is shared, so only
the quality half of each pair needs running.

  bash bench/run_lr_eval.sh          # runs guardrails per arm, then this
"""
import argparse
import json
from pathlib import Path

import numpy as np

from common import ROOT

ARMS = [
    ("control", "batched_freeze", 2001, "const lr, full run (the shipped baseline)"),
    ("B",       "lr_B",           2001, "cosine decay over 2001"),
    ("C0",      "lr_C0",          1200, "const lr, truncated at 1200"),
    ("C",       "lr_C",           1200, "cosine decay over 1200"),
    ("D0",      "lr_D0",           800, "const lr, truncated at 800"),
    ("D",       "lr_D",            800, "cosine decay over 800"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--guardrails", default=str(ROOT / "bench" / "results" / "guardrails"))
    ap.add_argument("--floor", default=str(ROOT / "bench" / "results" / "quality_curve"
                                           / "noise_floor_rep15.json"))
    a = ap.parse_args()

    flo = json.loads(Path(a.floor).read_text())
    fl_loss = flo["abs_delta"]["best_loss_eval"]["mean"]
    fl_marg = flo["abs_delta"]["zeroshot_margin_subset"]["mean"]
    fl_sim = flo["abs_delta"]["sim_true_photo"]["mean"]
    fl_rank = flo["abs_delta"]["log_retrieval_rank"]["mean"]
    flip5 = flo["decision_flips"].get("zeroshot_subset")

    rows = {}
    for name, d, iters, desc in ARMS:
        fp = Path(a.guardrails) / f"guardrails_lr_{name}.json"
        if not fp.exists():
            print(f"  (missing {fp.name} -- arm {name} not evaluated yet)")
            continue
        g = json.loads(fp.read_text())
        rows[name] = {"iters": iters, "desc": desc, "n": g["n_runs"],
                      "loss": g["loss_eval_mean"],
                      "top1_sub": g["zeroshot_top1_subset"],
                      "margin_sub": g["zeroshot_margin_subset_mean"],
                      "sim": g["sim_true_photo_mean"],
                      "logrank": g["log_retrieval_rank_mean"],
                      "medrank": g["retrieval_median_rank"]}
    if "control" not in rows:
        raise SystemExit("no control arm -- run guardrails on bench/results/batched_freeze")
    c = rows["control"]

    print("=" * 92)
    print("N1: LEARNING-RATE SCHEDULE")
    print(f"  floor (n={flo['n_pairs']} replicate pairs): loss {fl_loss:.5f}  "
          f"margin {fl_marg:.5f}  sim {fl_sim:.5f}  log-rank {fl_rank:.4f}")
    if flip5 is not None:
        print(f"  {100*flip5:.0f}% of 5-way zero-shot decisions flip between identical runs")
    print("=" * 92)
    print(f"{'arm':<9}{'iters':>6}{'x':>6}{'loss_eval':>11}{'Δ':>9}{'':>3}"
          f"{'5-way':>7}{'margin':>9}{'Δ':>9}{'':>3}{'medrank':>9}")
    for name, _, iters, desc in ARMS:
        if name not in rows:
            continue
        r = rows[name]
        sp = 2001 / iters
        dl = r["loss"] - c["loss"]
        dm = r["margin_sub"] - c["margin_sub"]
        fl = "  " if name == "control" else ("!!" if abs(dl) > fl_loss else "ok")
        fm = "  " if name == "control" else ("!!" if abs(dm) > fl_marg else "ok")
        print(f"{name:<9}{iters:>6}{sp:>5.2f}x{r['loss']:>11.5f}{dl:>+9.5f} {fl:<2}"
              f"{100*r['top1_sub']:>6.1f}%{r['margin_sub']:>9.5f}{dm:>+9.5f} {fm:<2}"
              f"{r['medrank']:>9.0f}")
    print("-" * 92)
    print("  Δ is vs the const-lr control; 'ok' = inside the replicate floor,")
    print("  '!!' = larger than the floor, i.e. a difference the method itself would not produce.")

    def cmp(a_, b_, label):
        if a_ not in rows or b_ not in rows:
            return
        x, y = rows[a_], rows[b_]
        dl, dm = x["loss"] - y["loss"], x["margin_sub"] - y["margin_sub"]
        verdict = ("decay is better" if dl < -fl_loss else
                   "decay is worse" if dl > fl_loss else
                   "indistinguishable at this sample size")
        print(f"\n  {label}: Δloss {dl:+.5f} (floor {fl_loss:.5f}), "
              f"Δmargin {dm:+.5f}  ->  {verdict}")

    print("\nTHE DECIDING COMPARISONS (same budget, decay vs plain truncation):")
    cmp("C", "C0", "C vs C0 @1200")
    cmp("D", "D0", "D vs D0 @800")
    cmp("B", "control", "B vs control @2001 (does settling help at no saving?)")

    out = Path(a.guardrails).parent / "quality_curve" / "n1_lr_verdict.json"
    out.write_text(json.dumps({"arms": rows, "floor": {
        "loss": fl_loss, "margin_subset": fl_marg, "sim": fl_sim,
        "log_rank": fl_rank, "flip_subset": flip5}}, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
