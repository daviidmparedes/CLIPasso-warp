#!/usr/bin/env bash
set -uo pipefail
source /home/dmiranda/CLIPasso/.venv/bin/activate
cd /home/dmiranda/CLIPasso
python - <<'PY'
import json, subprocess, sys, time
from pathlib import Path
ROOT=Path("/home/dmiranda/CLIPasso")
ev=json.loads((ROOT/"data"/"eval_set.json").read_text())
tot=0.0
for it in ev:
    t0=time.perf_counter()
    r=subprocess.run([sys.executable,"bench/batch_seeds.py","--target",it["path"],
        "--num-paths","16","--num-seeds","3","--num-iter","2001",
        "--eval-interval","10","--save-interval","1000000",
        "--out",str(ROOT/"bench"/"results"/"batched_freeze")],
        cwd=ROOT,check=True,capture_output=True,text=True)
    d=time.perf_counter()-t0; tot+=d
    boot=[l for l in r.stdout.splitlines() if l.startswith("boot")]
    print(f"  {it['class']:<10} {d:7.1f}s for 3 seeds = {d/3:6.1f}s/seed   {boot[0] if boot else ''}",flush=True)
print(f"BATCHED+FREEZE TOTAL {tot:.1f}s over {len(ev)} images = {tot/len(ev)/3:.1f}s per seed")
print(f"  reference (1.1 batched, same 5 images): 766.5s total = 51.1s per seed")
print(f"  marginal speedup from freeze+lazy: {766.5/tot:.3f}x")
PY
