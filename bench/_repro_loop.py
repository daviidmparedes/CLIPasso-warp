"""Bisect helper: my in-process loop, run as a standalone subprocess."""
import json, sys, time
import common
from common import ROOT, build_pipeline, make_args, sync


N = int(sys.argv[1]) if len(sys.argv) > 1 else 2001
tgt = json.loads((ROOT/"data"/"eval_set.json").read_text())[0]["path"]
wd = ROOT/"bench"/"results"/"_repro"; wd.mkdir(parents=True, exist_ok=True)
args = make_args(tgt, wd, "repro", num_paths=16, seed=0, num_iter=N+10,
                 save_interval=10**9, eval_interval=10**9)
t_boot = time.perf_counter()
with common.quiet():
    r, lf, o, inp, _ = build_pipeline(args)
sync(); boot = time.perf_counter() - t_boot
t0 = time.perf_counter(); marks = {}
SYNC = len(sys.argv) > 2 and sys.argv[2] == "sync"
for e in range(N):
    r.set_random_noise(e)
    o.zero_grad_()
    img = r.get_image()
    loss = sum(lf(img, inp.detach(), r.get_color_parameters(), r, e, o).values())
    loss.backward()
    o.step_()
    if SYNC:
        del img, loss        # release the previous iteration's autograd graph
    if e+1 in (100, 500, 1000, 2000):
        sync(); marks[e+1] = time.perf_counter() - t0
sync(); tot = time.perf_counter() - t0
print(f"boot={boot:.2f}s  loop={tot:.2f}s  total={boot+tot:.2f}s  mean={1000*tot/N:.2f} ms/iter")
prev_k = prev_v = 0
for k, v in marks.items():
    print(f"  iters {prev_k:>5}->{k:<5} marginal {1000*(v-prev_v)/(k-prev_k):6.2f} ms/iter")
    prev_k, prev_v = k, v
