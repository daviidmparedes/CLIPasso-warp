#!/usr/bin/env python3
"""
Does the per-iteration cost change as optimisation proceeds?

A 50-iteration profile taken from initialisation measures the CHEAPEST regime:
strokes start as ~0.05-radius squiggles and lengthen as they fit the target, so
diffvg's rasteriser and its backward touch progressively more pixels. Extrapolating
an early-iteration profile across a 2001-iteration run therefore understates diffvg
and overstates CLIP (whose cost is fixed -- it always sees 10 224x224 images).

Runs ONE full-length optimisation and stops at checkpoints to measure the phase
split, alongside geometry stats that test the coverage explanation directly.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import common
from common import ROOT, build_pipeline, env_fingerprint, make_args, sync
from profile_iter import PhaseTimer, instrument_diffvg, instrument_loss, one_iteration


def geometry_stats(renderer):
    """Total control-polygon length and rendered ink coverage."""
    tot_len, bbox_diag = 0.0, []
    for path in renderer.shapes:
        p = path.points.detach()
        tot_len += (p[1:] - p[:-1]).norm(dim=1).sum().item()
        bbox_diag.append((p.max(0).values - p.min(0).values).norm().item())
    with torch.no_grad():
        img = renderer.get_image()          # NCHW, white background
        ink = (img < 0.99).any(dim=1).float().mean().item()
    return {"ctrl_polygon_len_px": tot_len,
            "mean_stroke_bbox_diag_px": float(np.mean(bbox_diag)),
            "ink_coverage_frac": ink}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=None)
    ap.add_argument("--num-paths", type=int, default=16)
    ap.add_argument("--total-iters", type=int, default=2001)
    ap.add_argument("--window", type=int, default=30, help="iterations measured per checkpoint")
    ap.add_argument("--checkpoints", type=int, nargs="+",
                    default=[25, 250, 500, 1000, 1500, 1950])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if a.target is None:
        a.target = json.loads((ROOT / "data" / "eval_set.json").read_text())[0]["path"]

    wd = ROOT / "bench" / "results" / "_overtime"
    wd.mkdir(parents=True, exist_ok=True)
    args = make_args(a.target, wd, f"ot_n{a.num_paths}", num_paths=a.num_paths, seed=a.seed,
                     num_iter=a.total_iters + 10, save_interval=10**9, eval_interval=10**9)

    print(f"target={a.target}  n={a.num_paths}  total_iters={a.total_iters}")
    with common.quiet():
        renderer, loss_func, optimizer, inputs, _ = build_pipeline(args)

    pt = PhaseTimer()
    instrument_loss(loss_func, pt)
    instrument_diffvg(pt)

    rows, epoch = [], 0
    for cp in a.checkpoints:
        # advance (unmeasured) to the checkpoint
        while epoch < cp:
            one_iteration(renderer, loss_func, optimizer, inputs, epoch); epoch += 1
        geo = geometry_stats(renderer)

        # fused window: ground-truth wall clock at this stage
        sync()
        import time
        t0 = time.perf_counter()
        for _ in range(a.window):
            one_iteration(renderer, loss_func, optimizer, inputs, epoch); epoch += 1
        sync()
        fused = (time.perf_counter() - t0) * 1e3 / a.window

        # split window: attribution at this stage
        pt.rows.clear()
        for _ in range(a.window):
            one_iteration(renderer, loss_func, optimizer, inputs, epoch, pt, split=True)
            pt.flush(); epoch += 1
        s = pt.stats()

        def g(k):
            return s.get(k, {}).get("mean_ms", 0.0)

        row = {"checkpoint": cp, "fused_ms": fused,
               "diffvg_fwd": g("diffvg_fwd.total"),
               "diffvg_bwd": g("diffvg_bwd"),
               "clip_fwd": g("clip_fwd.total"),
               "clip_bwd": g("clip_bwd"),
               "serialize": g("diffvg_fwd.serialize_scene"), **geo}
        row["diffvg_total"] = row["diffvg_fwd"] + row["diffvg_bwd"]
        row["clip_total"] = row["clip_fwd"] + row["clip_bwd"]
        row["clip_over_diffvg"] = row["clip_total"] / max(row["diffvg_total"], 1e-9)
        rows.append(row)
        print(f"  iter {cp:>5}: fused {fused:6.2f} ms | diffvg {row['diffvg_total']:6.2f} "
              f"(f {row['diffvg_fwd']:5.2f} / b {row['diffvg_bwd']:5.2f}) | "
              f"CLIP {row['clip_total']:6.2f} | ratio {row['clip_over_diffvg']:.2f}x | "
              f"ink {geo['ink_coverage_frac']*100:4.1f}% | len {geo['ctrl_polygon_len_px']:6.0f}px")

    print("\n" + "=" * 96)
    print(f"{'iter':>6}{'fused ms':>10}{'diffvg':>9}{'  fwd':>8}{'  bwd':>8}{'CLIP':>9}"
          f"{'CLIP/dv':>9}{'ink %':>8}{'ctrl len px':>13}")
    print("-" * 96)
    for r in rows:
        print(f"{r['checkpoint']:>6}{r['fused_ms']:>10.2f}{r['diffvg_total']:>9.2f}"
              f"{r['diffvg_fwd']:>8.2f}{r['diffvg_bwd']:>8.2f}{r['clip_total']:>9.2f}"
              f"{r['clip_over_diffvg']:>9.2f}{100*r['ink_coverage_frac']:>8.1f}"
              f"{r['ctrl_polygon_len_px']:>13.0f}")
    print("=" * 96)
    f, l = rows[0], rows[-1]
    print(f"\nfused iteration cost      {f['fused_ms']:.2f} -> {l['fused_ms']:.2f} ms "
          f"({l['fused_ms']/f['fused_ms']:.2f}x over the run)")
    print(f"  diffvg                  {f['diffvg_total']:.2f} -> {l['diffvg_total']:.2f} ms "
          f"({l['diffvg_total']/f['diffvg_total']:.2f}x)")
    print(f"  CLIP                    {f['clip_total']:.2f} -> {l['clip_total']:.2f} ms "
          f"({l['clip_total']/f['clip_total']:.2f}x)")
    print(f"  ink coverage            {100*f['ink_coverage_frac']:.1f}% -> {100*l['ink_coverage_frac']:.1f}%")
    print(f"  control-polygon length  {f['ctrl_polygon_len_px']:.0f} -> {l['ctrl_polygon_len_px']:.0f} px")
    mean_fused = float(np.mean([r["fused_ms"] for r in rows]))
    print(f"\nmean over checkpoints: {mean_fused:.2f} ms/iter "
          f"-> a 2001-iteration seed = {mean_fused*2001/1e3:.1f} s")

    out = ROOT / "bench" / "results" / "profile" / f"over_time_n{a.num_paths}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": vars(a), "env": env_fingerprint(), "rows": rows}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
