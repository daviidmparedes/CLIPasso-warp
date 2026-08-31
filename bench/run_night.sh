#!/usr/bin/env bash
# Everything that needs an UNCONTENDED GPU, in priority order.
#
# Every absolute millisecond in RESULTS.md was measured while another researcher's
# job held 83-95 GB of the 97 GB card. Paired/interleaved measurement makes the
# *ratios* trustworthy, but the absolute numbers are upper bounds and the ladder's
# s/seed figures need one clean pass before they go anywhere public.
#
# Usage:  nohup bash bench/run_night.sh > bench/logs/night.log 2>&1 &
#
# The guard below aborts rather than producing numbers that look clean and are not.
# Raise SKIP_GUARD=1 only if you know the card is yours.
set -uo pipefail
cd /home/dmiranda/CLIPasso
source .venv/bin/activate
mkdir -p bench/logs

MIN_FREE_GB=${MIN_FREE_GB:-80}
if [ "${SKIP_GUARD:-0}" != "1" ]; then
  python - "$MIN_FREE_GB" <<'PY' || exit 1
import sys, torch
need = float(sys.argv[1])
free, total = torch.cuda.mem_get_info()
free, total = free / 2**30, total / 2**30
print(f"GPU: {free:.1f} GB free of {total:.1f} GB (need >= {need})")
if free < need:
    print("ABORT: the card is not free. Timings taken now would be unusable, and")
    print("       diffvg renders silently wrong output near the memory ceiling.")
    sys.exit(1)
PY
fi
echo "=== starting $(date -Is) ==="

# ---------------------------------------------------------------- phase 1
# The ladder, re-measured clean. This is the headline deliverable of the night:
# it converts every s/seed figure in the summary table from "upper bound under
# contention" into a measurement. Quality is unchanged and is not re-run.
echo "### phase 1: clean ladder re-measurement (n=16, 5 images x 3 seeds)"
python bench/run_baseline.py --strokes 16 --seeds 0,1000,2000 --tag night_shipped \
  || echo "phase 1a FAILED"
bash bench/run_tier11.sh 2>&1 | tail -5 || echo "phase 1b FAILED"
bash bench/run_freeze_batched.sh 2>&1 | tail -5 || echo "phase 1c FAILED"

# ---------------------------------------------------------------- phase 2
# 0.4 end-to-end on the real batched pipeline. Currently only measured on a
# synthetic loop (1.03x idle / 1.13x shared); this is the number that would let it
# join the ladder. Requires the --fast-serialize flag in batch_seeds.py.
echo "### phase 2: 0.4 (assert-free serialize) end-to-end"
if grep -q "fast-serialize" bench/batch_seeds.py; then
  bash bench/run_freeze_batched.sh 2>&1 | tail -3
else
  echo "SKIPPED: bench/batch_seeds.py has no --fast-serialize flag yet (see RESULTS.md N3)"
fi
python bench/fast_serialize.py --strokes 16 --iters 400 --verify --train-loop \
  || echo "phase 2b FAILED"

# ---------------------------------------------------------------- phase 3
# Tiling, clean. The M=1..16 sweep showed a monotone trend to 1.5-1.8x on diffvg
# fwd+bwd, but every point was contended; the paired medians and min-ratios
# disagreed by up to 30%.
echo "### phase 3: tiling sweep, clean"
bash bench/run_tiled_sweep.sh 2>&1 | grep -E "^###|forward|kernel reduction|correctness"

# ---------------------------------------------------------------- phase 4
# num_samples 2 -> 1: timing half. The quality half is already provisionally
# negative (the renderer's own gradient cosine falls 0.947 -> 0.730), so this only
# needs enough numbers to close the question.
echo "### phase 4: num_samples 2 -> 1"
for M in 1 5; do
  python bench/tiled_render.py --num-scenes $M --strokes 16 --samples 1 \
      --reps 40 --warmup 5 --profile 2>&1 | grep -E "forward|kernel"
done

echo "=== finished $(date -Is) ==="
