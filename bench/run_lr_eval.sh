#!/usr/bin/env bash
# Score every N1 arm with the guardrails, then tabulate against the n=15 floor.
set -uo pipefail
cd /home/dmiranda/CLIPasso
source .venv/bin/activate
for pair in "control:batched_freeze" "B:lr_B" "C0:lr_C0" "C:lr_C" "D0:lr_D0" "D:lr_D"; do
  name="${pair%%:*}"; dir="bench/results/${pair##*:}"
  [ -d "$dir" ] || { echo "skip $name (no $dir)"; continue; }
  echo "### guardrails: arm $name"
  python bench/guardrails.py --runs "$dir" --tag "lr_$name" --strokes 16 2>&1 \
    | grep -E "WARNING|loss_eval|top-1|median rank|margin|sim to true"
done
python bench/analyze_lr_arms.py
