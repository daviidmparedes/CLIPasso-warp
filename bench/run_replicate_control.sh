#!/usr/bin/env bash
# Second replicate of the current best pipeline (batched seeds + frozen CLIP).
#
# The existing nondeterminism control was n=3 (one image, three seeds), which is
# too thin to set a noise floor that every quality delta is then judged against.
# This produces a paired replicate of all 15 n=16 runs, so the floor is measured
# over the same population the deltas are.
#
# Writes to a NEW directory; bench/results/batched_freeze is the first replicate
# and must not be overwritten.
set -uo pipefail
source /home/dmiranda/CLIPasso/.venv/bin/activate
cd /home/dmiranda/CLIPasso
python - <<'PY'
import json, subprocess, sys, time
from pathlib import Path
ROOT = Path("/home/dmiranda/CLIPasso")
OUT = ROOT / "bench" / "results" / "batched_freeze_rep2"
ev = json.loads((ROOT / "data" / "eval_set.json").read_text())
tot = 0.0
for it in ev:
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, "bench/batch_seeds.py", "--target", it["path"],
        "--num-paths", "16", "--num-seeds", "3", "--num-iter", "2001",
        "--eval-interval", "10", "--save-interval", "1000000", "--out", str(OUT)],
        cwd=ROOT, check=True, capture_output=True, text=True)
    d = time.perf_counter() - t0; tot += d
    print(f"  {it['class']:<10} {d:7.1f}s for 3 seeds = {d/3:6.1f}s/seed", flush=True)
print(f"REPLICATE 2 TOTAL {tot:.1f}s over {len(ev)} images = {tot/len(ev)/3:.1f}s per seed")
print("NOTE: the GPU was shared with another job, so these timings are not a")
print("      speedup measurement -- only the sketches matter here.")
PY
