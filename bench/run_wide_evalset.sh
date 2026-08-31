#!/usr/bin/env bash
# Widen the quality eval set from 15 runs to 200 (methodology debt item 4).
#
# Section 6's power table is the argument: with 15 runs per stroke count the perceptual
# guardrails resolve about 50% of their trajectory-wide signal, which is why every
# speedup so far lands "inside the noise floor" without that being very informative.
# data/paper_protocol.json (200 images, 10 categories) takes them to roughly 20%.
#
# Runs at 1200 iterations, not 2001, for two reasons: it is the budget section 6
# recommends and N2 will ship, so this validates that recommendation at 40x the
# sample size instead of assuming it; and it halves a job that is ~4 hours on a
# shared card either way.
#
# batch_images.py holds every image of a batch in memory at once (Tier 1.2 measured
# 18.95 GB at M=16), so 200 at once is impossible -- this chunks into groups of 5.
# It also has no offset argument, so each chunk gets its own manifest file.
#
# Quality-only: no timing is read from this, so a contended GPU is fine.
#   nohup bash bench/run_wide_evalset.sh > bench/logs/wide_evalset.log 2>&1 &
set -uo pipefail
cd /home/dmiranda/CLIPasso
source .venv/bin/activate

CHUNK=${CHUNK:-5}
ITERS=${ITERS:-1200}
NIMG=${NIMG:-200}

python - "$CHUNK" "$ITERS" "$NIMG" <<'PY'
import json, subprocess, sys, time
from pathlib import Path
chunk, iters, nimg = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
ROOT = Path("/home/dmiranda/CLIPasso")
OUT = ROOT / "bench" / "results" / "wide1200"
CH = ROOT / "data" / "_chunks"; CH.mkdir(parents=True, exist_ok=True)

items = json.loads((ROOT / "data" / "paper_protocol.json").read_text())[:nimg]
print(f"{len(items)} images, chunks of {chunk}, {iters} iterations, 1 seed each")
t_all = time.perf_counter()
done = 0
for i in range(0, len(items), chunk):
    grp = items[i:i + chunk]
    man = CH / f"chunk_{i//chunk:03d}.json"
    man.write_text(json.dumps(grp))
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, "bench/batch_images.py",
        "--manifest", str(man), "--num-images", str(len(grp)), "--num-seeds", "1",
        "--num-paths", "16", "--num-iter", iters,
        "--eval-interval", "10", "--save-interval", "1000000",
        "--out", str(OUT)], cwd=ROOT, capture_output=True, text=True)
    if r.returncode:
        print(f"  chunk {i//chunk:3d} FAILED\n{r.stdout[-700:]}\n{r.stderr[-700:]}", flush=True)
        continue
    d = time.perf_counter() - t0
    done += len(grp)
    el = time.perf_counter() - t_all
    eta = el / max(done, 1) * (len(items) - done)
    print(f"  chunk {i//chunk:3d}  {len(grp)} imgs  {d:6.1f}s  "
          f"({d/len(grp):5.1f}s/sketch)  {done}/{len(items)} done  "
          f"ETA {eta/60:5.1f} min", flush=True)
print(f"WIDE EVALSET DONE  {done}/{len(items)} in {(time.perf_counter()-t_all)/60:.1f} min")
PY
