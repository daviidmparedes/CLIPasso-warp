#!/usr/bin/env bash
# Does a decaying learning rate shrink CLIPasso's run-to-run divergence?
#
# This is the one question N1 opened that is worth GPU time even though the
# schedule turned out not to be a speedup. RESULTS.md section 4.3/6 measures 57.5 px
# of mean control-point drift between two identical const-lr runs, and that number
# is the noise floor every quality claim in this document is judged against. If
# cosine decay halves it, every future comparison becomes correspondingly more
# sensitive -- worth more to the programme than a 1.1x speedup.
#
# Produces a second replicate of arm B (cosine over 2001, full length) so it can be
# paired against bench/results/lr_B, exactly as batched_freeze_rep2 pairs against
# batched_freeze. Quality-only: no timing is read from this, so it is safe to run
# on a contended GPU.
#
#   nohup bash bench/run_lr_replicate.sh > bench/logs/lr_replicate.log 2>&1 &
# then
#   python bench/noise_floor.py --a bench/results/lr_B --b bench/results/lr_B_rep2 \
#          --tag lrB_rep --strokes 16
#   # compare its ctrlpoint_drift_px against noise_floor_rep15.json (57.5 px)
set -uo pipefail
cd /home/dmiranda/CLIPasso
source .venv/bin/activate
python - <<'PY'
import json, subprocess, sys, time
from pathlib import Path
ROOT = Path("/home/dmiranda/CLIPasso")
OUT = ROOT / "bench" / "results" / "lr_B_rep2"
ev = json.loads((ROOT / "data" / "eval_set.json").read_text())
for it in ev:
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, "bench/batch_seeds.py", "--target", it["path"],
        "--num-paths", "16", "--num-seeds", "3", "--num-iter", "2001",
        "--eval-interval", "10", "--save-interval", "1000000",
        "--lr-scheduler", "1", "--lr-schedule", "cosine", "--lr-decay-iters", "2001",
        "--out", str(OUT)], cwd=ROOT, capture_output=True, text=True)
    if r.returncode:
        print(f"  {it['class']:<10} FAILED\n{r.stderr[-800:]}", flush=True); continue
    print(f"  {it['class']:<10} {time.perf_counter()-t0:7.1f}s for 3 seeds", flush=True)
print("LR REPLICATE DONE")
PY
