#!/usr/bin/env bash
# Apples-to-apples: both sides at eval_interval=10, periodic logging off.
set -uo pipefail
source /home/dmiranda/CLIPasso/.venv/bin/activate
cd /home/dmiranda/CLIPasso
echo "=== [$(date +%T)] B: patched per-process baseline, n=16 x 3 seeds x 5 images ==="
python bench/run_baseline.py --strokes 16 --seeds 3 --save-interval 1000000 --tag patched_nolog
echo "=== [$(date +%T)] C: batched seeds, same 5 images ==="
python - <<'PY'
import json, subprocess, sys, time
from pathlib import Path
ROOT = Path("/home/dmiranda/CLIPasso")
ev = json.loads((ROOT/"data"/"eval_set.json").read_text())
tot = 0.0
for it in ev:
    t0 = time.perf_counter()
    subprocess.run([sys.executable, "bench/batch_seeds.py", "--target", it["path"],
                    "--num-paths", "16", "--num-seeds", "3", "--num-iter", "2001",
                    "--eval-interval", "10", "--save-interval", "1000000"],
                   cwd=ROOT, check=True, capture_output=True, text=True)
    d = time.perf_counter()-t0
    tot += d
    print(f"  {it['class']:<10} {d:7.1f}s for 3 seeds = {d/3:6.1f}s/seed", flush=True)
print(f"BATCHED TOTAL {tot:.1f}s over {len(ev)} images = {tot/len(ev)/3:.1f}s per seed")
PY
echo "=== [$(date +%T)] TIER 1.1 RUNS DONE ==="
