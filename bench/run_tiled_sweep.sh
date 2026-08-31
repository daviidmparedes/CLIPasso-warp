#!/usr/bin/env bash
# Tiling sweep: does merging M scenes into one raster pay, and how does it scale?
# Uses many reps because the GPU is shared -- see the min/median note in
# tiled_render.py's bench().
set -uo pipefail
cd /home/dmiranda/CLIPasso
for M in 1 2 4 8 16; do
  echo "################ M=$M samples=2"
  .venv/bin/python bench/tiled_render.py --num-scenes $M --strokes 16 \
      --reps 40 --warmup 5 --profile 2>&1 | grep -vi warning
done
echo "################ M=5 samples=1 (num_samples_x/y 2->1)"
.venv/bin/python bench/tiled_render.py --num-scenes 5 --strokes 16 --samples 1 \
    --reps 40 --warmup 5 --profile 2>&1 | grep -vi warning
