#!/usr/bin/env python3
"""
Clean baseline: unmodified CLIPasso over a fixed eval set.

Runs painterly_rendering.py as a SUBPROCESS, exactly the way run_object_sketching.py
does, so the numbers include real process startup and model loading rather than an
in-process approximation. Per-seed wall clock, iterations actually executed (the
min_delta=1e-5 rule stops early), and final loss_eval all get recorded.

  python bench/run_baseline.py                       # 5 imgs x {8,16,32} x 3 seeds
  python bench/run_baseline.py --strokes 16 --seeds 1 --tag quick
  python bench/run_baseline.py --save-interval 1000000 --tag nolog   # isolate logging cost
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import common
from common import ROOT, env_fingerprint


def run_one(target, out_dir, name, num_paths, seed, num_iter, save_interval, timeout=3600):
    """One painterly_rendering.py process; returns timing + loss telemetry."""
    cmd = [sys.executable, "painterly_rendering.py", str(target),
           "--num_paths", str(num_paths),
           "--output_dir", str(out_dir),
           "--wandb_name", name,
           "--num_iter", str(num_iter),
           "--save_interval", str(save_interval),
           "--eval_interval", "10",
           "--seed", str(seed),
           "--use_gpu", "1",
           "--fix_scale", "0",
           "--mask_object", "0",
           "--mask_object_attention", "0",
           "--display_logs", "0",
           "--display", "0"]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    wall = time.perf_counter() - t0
    if p.returncode != 0:
        return {"ok": False, "wall_s": wall,
                "stderr": p.stderr[-1500:], "stdout": p.stdout[-800:]}

    cfg_path = Path(out_dir) / name / "config.npy"
    cfg = np.load(cfg_path, allow_pickle=True)[()]
    loss_eval = np.asarray(cfg["loss_eval"], dtype=float)
    # eval runs every eval_interval, so this recovers how far the loop actually got
    n_evals = len(loss_eval)
    iters_run = n_evals * 10
    return {
        "ok": True,
        "wall_s": wall,
        "best_loss_eval": float(loss_eval.min()),
        "best_iter": int(loss_eval.argmin() * 10),
        "final_loss_eval": float(loss_eval[-1]),
        "n_evals": n_evals,
        "iters_run": iters_run,
        "early_stopped": bool(iters_run < num_iter),
        "ms_per_iter": wall * 1e3 / max(iters_run, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strokes", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--seeds", type=int, default=3, help="number of seeds (0,1000,2000,...)")
    ap.add_argument("--num-iter", type=int, default=2001)
    ap.add_argument("--save-interval", type=int, default=10,
                    help="10 = as shipped. Set huge to measure without SVG/JPEG logging.")
    ap.add_argument("--images", type=int, default=None, help="limit number of eval images")
    ap.add_argument("--tag", default="shipped")
    ap.add_argument("--out", default=str(ROOT / "bench" / "results" / "baseline"))
    a = ap.parse_args()

    eval_set = json.loads((ROOT / "data" / "eval_set.json").read_text())
    if a.images:
        eval_set = eval_set[:a.images]
    seeds = list(range(0, a.seeds * 1000, 1000))

    outdir = Path(a.out) / a.tag
    outdir.mkdir(parents=True, exist_ok=True)
    total = len(eval_set) * len(a.strokes) * len(seeds)
    print(f"baseline '{a.tag}': {len(eval_set)} images x {a.strokes} strokes x {len(seeds)} seeds "
          f"= {total} runs")
    print(f"  num_iter={a.num_iter}  save_interval={a.save_interval}  -> {outdir}\n")

    runs, done = [], 0
    t_start = time.perf_counter()
    for item in eval_set:
        for n in a.strokes:
            per_seed = []
            for seed in seeds:
                name = f"{item['class']}_{Path(item['rel']).stem}_{n}strokes_seed{seed}"
                r = run_one(item["path"], outdir, name, n, seed, a.num_iter, a.save_interval)
                r.update({"image": item["rel"], "class": item["class"],
                          "num_paths": n, "seed": seed, "name": name})
                runs.append(r); per_seed.append(r); done += 1
                if r["ok"]:
                    print(f"  [{done:>3}/{total}] {item['class']:<10} n={n:<3} seed={seed:<5} "
                          f"{r['wall_s']:6.1f}s  {r['ms_per_iter']:5.1f} ms/it  "
                          f"iters={r['iters_run']:<5}{'(early)' if r['early_stopped'] else ''} "
                          f"loss_eval={r['best_loss_eval']:.5f}")
                else:
                    print(f"  [{done:>3}/{total}] {item['class']:<10} n={n:<3} seed={seed:<5} FAILED")
                    print("        " + r["stderr"].strip().splitlines()[-1][:160])
            ok = [x for x in per_seed if x["ok"]]
            if ok:
                best = min(ok, key=lambda x: x["best_loss_eval"])
                # This mirrors run_object_sketching.py: 3 seeds run, best-of-3 kept.
                print(f"        -> image total {sum(x['wall_s'] for x in ok):6.1f}s  "
                      f"best seed={best['seed']} loss_eval={best['best_loss_eval']:.5f}\n")

    elapsed = time.perf_counter() - t_start
    ok = [r for r in runs if r["ok"]]
    summary = {"tag": a.tag, "total_wall_s": elapsed, "n_runs": len(runs), "n_ok": len(ok),
               "save_interval": a.save_interval, "num_iter": a.num_iter,
               "env": env_fingerprint(), "runs": runs}
    if ok:
        by_n = {}
        for n in a.strokes:
            sel = [r for r in ok if r["num_paths"] == n]
            if not sel:
                continue
            by_n[n] = {
                "mean_wall_s_per_seed": float(np.mean([r["wall_s"] for r in sel])),
                "mean_ms_per_iter": float(np.mean([r["ms_per_iter"] for r in sel])),
                "mean_iters_run": float(np.mean([r["iters_run"] for r in sel])),
                "frac_early_stopped": float(np.mean([r["early_stopped"] for r in sel])),
                "mean_best_loss_eval": float(np.mean([r["best_loss_eval"] for r in sel])),
                "std_best_loss_eval": float(np.std([r["best_loss_eval"] for r in sel])),
                "mean_wall_s_per_image_3seeds": float(np.mean([r["wall_s"] for r in sel])) * len(seeds),
            }
        summary["by_strokes"] = by_n
        print("=" * 78)
        print(f"{'n':>4} {'s/seed':>9} {'ms/iter':>9} {'iters':>8} {'early%':>8} "
              f"{'loss_eval':>11} {'+/-':>8} {'s/image(3seed)':>15}")
        print("-" * 78)
        for n, v in by_n.items():
            print(f"{n:>4} {v['mean_wall_s_per_seed']:>9.1f} {v['mean_ms_per_iter']:>9.1f} "
                  f"{v['mean_iters_run']:>8.0f} {100*v['frac_early_stopped']:>7.0f}% "
                  f"{v['mean_best_loss_eval']:>11.5f} {v['std_best_loss_eval']:>8.5f} "
                  f"{v['mean_wall_s_per_image_3seeds']:>15.1f}")
        print("=" * 78)
    print(f"\ntotal {elapsed/60:.1f} min, {len(ok)}/{len(runs)} succeeded")
    fp = Path(a.out) / f"baseline_{a.tag}.json"
    fp.write_text(json.dumps(summary, indent=1))
    print(f"wrote {fp}")


if __name__ == "__main__":
    main()
