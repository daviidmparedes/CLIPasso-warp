#!/usr/bin/env bash
# Re-run the P1 analysis chain against the widened (n=15) replicate control.
# The noise floor feeds both the truncation verdict and the stopping-rule scoring,
# so all three have to be regenerated together whenever the control changes.
set -uo pipefail
cd /home/dmiranda/CLIPasso
Q=bench/results/quality_curve
.venv/bin/python bench/noise_floor.py \
    --a bench/results/batched_freeze --b bench/results/batched_freeze_rep2 \
    --tag rep15 --strokes 16 2>&1 | grep -vi warning
.venv/bin/python bench/analyze_p1.py --curve $Q/quality_vs_iter_shipped.json \
    --floor $Q/noise_floor_rep15.json --out $Q/p1_verdict_rep15.json 2>&1 | grep -vi warning
.venv/bin/python bench/stopping_rule.py --curve $Q/quality_vs_iter_shipped.json \
    --floor $Q/noise_floor_rep15.json --out $Q/stopping_rules_rep15.json 2>&1 | grep -vi warning
