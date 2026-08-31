#!/usr/bin/env bash
# N1: does a working learning-rate schedule pay?
#
# Arms, all 5 images x 3 seeds at n=16, all on the batched+frozen pipeline so they
# are directly comparable to bench/results/batched_freeze (the const-lr control,
# n=15, already on disk):
#
#   B    cosine over 2001, run 2001   same cost -- pure quality test: does settling help?
#   C0   const  lr,        run 1200   plain truncation at 1.67x (RESULTS.md section 6's recommendation)
#   C    cosine over 1200, run 1200   same budget WITH decay -- does decay beat truncation?
#   D0   const  lr,        run  800   plain truncation at 2.5x
#   D    cosine over  800, run  800
#
# C vs C0 and D vs D0 are the comparisons that decide N1. B decides whether the
# schedule is worth turning on even at full length.
#
# Speedups here are arithmetic (1200/2001), not measured -- the GPU is shared, so
# only the quality half of each (speedup, quality delta) pair needs running.
set -uo pipefail
source /home/dmiranda/CLIPasso/.venv/bin/activate
cd /home/dmiranda/CLIPasso

run_arm () {  # name  num_iter  lr_scheduler  decay_iters
  local name=$1 iters=$2 sched=$3 decay=$4
  echo "################ arm $name: ${iters} iters, scheduler=$sched, decay=$decay"
  python - "$name" "$iters" "$sched" "$decay" <<'PY'
import json, subprocess, sys, time
from pathlib import Path
name, iters, sched, decay = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
ROOT = Path("/home/dmiranda/CLIPasso")
OUT = ROOT / "bench" / "results" / f"lr_{name}"
ev = json.loads((ROOT / "data" / "eval_set.json").read_text())
tot = 0.0
for it in ev:
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, "bench/batch_seeds.py", "--target", it["path"],
        "--num-paths", "16", "--num-seeds", "3", "--num-iter", iters,
        "--eval-interval", "10", "--save-interval", "1000000",
        "--lr-scheduler", sched, "--lr-schedule", "cosine",
        "--lr-decay-iters", decay, "--out", str(OUT)],
        cwd=ROOT, capture_output=True, text=True)
    if r.returncode:
        print(f"  {it['class']:<10} FAILED\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}", flush=True)
        continue
    d = time.perf_counter() - t0; tot += d
    print(f"  {it['class']:<10} {d:7.1f}s for 3 seeds", flush=True)
print(f"arm {name}: {tot:.1f}s total")
PY
}

run_arm B  2001 1 2001
run_arm C0 1200 0 0
run_arm C  1200 1 1200
run_arm D0  800 0 0
run_arm D   800 1 800
echo "ALL ARMS DONE"
