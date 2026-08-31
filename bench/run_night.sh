#!/usr/bin/env bash
# Everything that needs an UNCONTENDED GPU, in priority order.
#
# Every absolute millisecond in RESULTS.md was measured while another researcher's
# job held 83-95 GB of the 97 GB card. Paired/interleaved measurement makes the
# *ratios* trustworthy, but the absolute s/seed figures are upper bounds and need
# one clean pass before they go anywhere public.
#
#   nohup bash bench/run_night.sh > bench/logs/night.log 2>&1 &
#
# SAFETY: every phase writes under bench/results/night/. Nothing here reuses
# bench/run_tier11.sh or bench/run_freeze_batched.sh, because those have hardcoded
# --out paths that would overwrite bench/results/{batched,batched_freeze} and
# baseline/patched_nolog -- all of which are live comparison arms, including the
# N1 control. Re-running a timing measurement must never destroy a quality result.
set -uo pipefail
cd /home/dmiranda/CLIPasso
source .venv/bin/activate
mkdir -p bench/logs bench/results/night

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

# Run the batched pipeline over the 5-image eval set, into a named directory.
# $1 = output subdirectory, remaining args are passed to batch_seeds.py.
batched_over_eval_set () {
  local sub=$1; shift
  python - "$sub" "$@" <<'PY'
import json, subprocess, sys, time
from pathlib import Path
ROOT = Path("/home/dmiranda/CLIPasso")
sub, extra = sys.argv[1], sys.argv[2:]
out = ROOT / "bench" / "results" / "night" / sub
ev = json.loads((ROOT / "data" / "eval_set.json").read_text())
tot = 0.0
for it in ev:
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, "bench/batch_seeds.py", "--target", it["path"],
        "--num-paths", "16", "--num-seeds", "3", "--num-iter", "2001",
        "--eval-interval", "10", "--save-interval", "1000000",
        "--out", str(out)] + extra, cwd=ROOT, capture_output=True, text=True)
    if r.returncode:
        print(f"  {it['class']:<10} FAILED\n{r.stdout[-800:]}\n{r.stderr[-800:]}", flush=True)
        continue
    d = time.perf_counter() - t0; tot += d
    print(f"  {it['class']:<10} {d:7.1f}s for 3 seeds = {d/3:6.1f}s/seed", flush=True)
print(f"  -> {sub}: {tot:.1f}s over {len(ev)} images = {tot/len(ev)/3:.1f}s per seed")
PY
}

# ---------------------------------------------------------------- phase 1
# The ladder's live arms, re-timed clean. This is the night's headline deliverable:
# it does not discover anything, it converts the summary table's s/seed figures
# from contended upper bounds into measurements.
#
# The 126.3 s/seed "as shipped" number cannot be reproduced from this working tree
# (0.1/0.2/0.3 are applied here). Re-deriving it needs `git checkout main` first,
# which is deliberately NOT automated overnight -- do it by hand if the ladder's
# base point matters more than its ratios.
echo "### phase 1a: per-process arm, n=16 x 3 seeds x 5 images"
python bench/run_baseline.py --strokes 16 --seeds 3 --save-interval 1000000 \
  --tag night_perproc --out bench/results/night/baseline || echo "phase 1a FAILED"

echo "### phase 1b: batched seeds (Tier 1.1 + freeze), clean timing"
batched_over_eval_set batched

# ---------------------------------------------------------------- phase 2
# 0.4 end-to-end on the real pipeline. So far it is only measured on a synthetic
# loop (1.03x idle / 1.13x shared); this is the number that would let it join the
# ladder. Needs a --fast-serialize flag in batch_seeds.py, which does not exist yet.
echo "### phase 2: 0.4 (assert-free serialize) end-to-end"
if grep -q "fast-serialize" bench/batch_seeds.py; then
  batched_over_eval_set batched_fastser --fast-serialize 1
else
  echo "  SKIPPED: bench/batch_seeds.py has no --fast-serialize flag yet (RESULTS.md N3)"
fi
python bench/fast_serialize.py --strokes 16 --iters 400 --verify --train-loop \
  || echo "phase 2b FAILED"

# ---------------------------------------------------------------- phase 3
# Tiling, clean. The M=1..16 sweep trends monotonically to 1.5-1.8x on diffvg
# fwd+bwd, but every point was contended and the paired median and min-ratio
# disagreed by up to 30%.
echo "### phase 3: tiling sweep, clean"
bash bench/run_tiled_sweep.sh 2>&1 | grep -E "^###|forward|kernel reduction|correctness"

# ---------------------------------------------------------------- phase 4
# num_samples 2 -> 1: timing half. The quality half already looks negative (the
# renderer's own gradient cosine falls 0.947 -> 0.730), so this just closes it.
echo "### phase 4: num_samples 2 -> 1"
for M in 1 5; do
  python bench/tiled_render.py --num-scenes $M --strokes 16 --samples 1 \
      --reps 40 --warmup 5 --profile 2>&1 | grep -E "forward|kernel"
done

echo "=== finished $(date -Is) ==="
echo "Quality for phases 1-2 is unchanged and was NOT re-run; score any new arm with"
echo "  python bench/guardrails.py --runs bench/results/night/<arm> --tag night_<arm> --strokes 16"
