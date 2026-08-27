#!/usr/bin/env bash
# Sequential (never concurrent -- GPU contention would corrupt every timing).
set -uo pipefail
source /home/dmiranda/CLIPasso/.venv/bin/activate
cd /home/dmiranda/CLIPasso
echo "=== [$(date +%T)] A: shipped baseline, 5 imgs x {8,16,32} x 3 seeds ==="
python bench/run_baseline.py --strokes 8 16 32 --seeds 3 --tag shipped
echo "=== [$(date +%T)] B: no-logging variant at n=16 (isolates SVG/JPEG write cost) ==="
python bench/run_baseline.py --strokes 16 --seeds 3 --save-interval 1000000 --tag nolog
echo "=== [$(date +%T)] ALL BASELINES DONE ==="
