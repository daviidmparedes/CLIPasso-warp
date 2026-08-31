#!/usr/bin/env python3
"""
P2, the finding that actually matters: most of diffvg's forward launches are asserts.

Profiling diffvg per kernel (bench/profile_diffvg.py) shows that at n=16 strokes the
forward issues 124 CUDA kernels but only ONE of them is diffvg's rasteriser. The rest
come from pydiffvg.RenderFunction.serialize_scene, which for every shape runs

    assert(torch.isfinite(shape.points).all())

-- a full GPU reduction plus a device-to-host synchronisation, per stroke, per
iteration -- and then copies each shape's points to the host separately.

Stripping just the asserts takes the forward from 124 kernels to 21 (5.9x fewer,
10.7 -> 3.8 launches per stroke). In a workload measured at 41.6% GPU-idle, launches
are the currency, so this is worth measuring end to end.

Rather than hand-rewriting serialize_scene (and risking a behavioural difference),
this rewrites diffvg's own source: parse it, delete the Assert nodes, recompile. The
result is by construction identical to upstream except that the assertions are gone.

  python bench/fast_serialize.py --verify          # identical args + identical pixels
  python bench/fast_serialize.py --bench --iters 400
"""
import argparse
import ast
import inspect
import json
import textwrap
import time
from pathlib import Path

import numpy as np
import torch

import common
from common import ROOT, env_fingerprint, make_args


class _StripAsserts(ast.NodeTransformer):
    def visit_Assert(self, node):
        return None


def build_assert_free_serialize():
    """Recompile pydiffvg's serialize_scene with every `assert` removed."""
    import pydiffvg
    from pydiffvg import render_pytorch

    fn = pydiffvg.RenderFunction.serialize_scene
    src = textwrap.dedent(inspect.getsource(fn))
    tree = ast.parse(src)
    fdef = tree.body[0]
    fdef.decorator_list = []                  # drop @staticmethod; we want a plain fn
    fdef.name = "serialize_scene_no_asserts"
    n_before = sum(isinstance(n, ast.Assert) for n in ast.walk(tree))
    tree = _StripAsserts().visit(tree)
    ast.fix_missing_locations(tree)
    ns = dict(vars(render_pytorch))           # same globals as the original
    exec(compile(tree, "<serialize_scene_no_asserts>", "exec"), ns)
    return ns["serialize_scene_no_asserts"], n_before


def enable(verbose=True):
    """Install the assert-free serialize_scene process-wide. Opt-in.

    Safe by construction: --verify checks that the recompiled function returns an
    argument list equal entry-for-entry to upstream's and renders bit-identical
    pixels, and it is the same source with `assert` nodes deleted. What is given up
    is diffvg's guard against NaN control points -- upstream raises AssertionError,
    this renders whatever NaNs produce. CLIPasso's loss goes NaN in that case
    anyway, so the failure is still loud, just one step later.

        import fast_serialize; fast_serialize.enable()
    """
    import pydiffvg
    fn, n = build_assert_free_serialize()
    pydiffvg.RenderFunction.serialize_scene = staticmethod(fn)
    if verbose:
        print(f"fast_serialize: {n} asserts removed from serialize_scene")
    return fn


def build_painter(target, outdir, strokes, seed=0):
    import painterly_rendering as pr
    args = make_args(target, outdir, "fs", num_paths=strokes, seed=seed,
                     num_iter=1, save_interval=10 ** 9, eval_interval=10 ** 9)
    inputs, mask = pr.get_target(args)
    r = pr.load_renderer(args, inputs, mask)
    r.set_random_noise(0)
    r.init_image(stage=0)
    r.parameters()
    return r, inputs, args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strokes", type=int, default=16)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--full-loop", action="store_true",
                    help="time the render forward+backward, not just the forward")
    ap.add_argument("--train-loop", action="store_true",
                    help="time a REAL CLIPasso iteration: render + CLIP loss + step")
    ap.add_argument("--out", default=str(ROOT / "bench" / "results" / "diffvg"))
    a = ap.parse_args()
    if not (a.verify or a.bench or a.train_loop):
        a.verify = a.bench = True

    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    common.require_free_gpu_memory()
    import pydiffvg

    fast, n_asserts = build_assert_free_serialize()
    print(f"recompiled serialize_scene with {n_asserts} assert statements removed")

    ev = json.loads((ROOT / "data" / "eval_set.json").read_text())
    work = outdir / "_fs"; work.mkdir(parents=True, exist_ok=True)
    r, inputs, args = build_painter(ev[0]["path"], work, a.strokes)
    canvas = r.canvas_width
    stock = pydiffvg.RenderFunction.serialize_scene
    res = {"strokes": a.strokes, "asserts_removed": n_asserts, "env": env_fingerprint()}

    if a.verify:
        sa = stock(canvas, canvas, r.shapes, r.shape_groups)
        sb = fast(canvas, canvas, r.shapes, r.shape_groups)
        same_len = len(sa) == len(sb)
        mismatch = []
        for i, (x, y) in enumerate(zip(sa, sb)):
            if torch.is_tensor(x):
                if not (torch.is_tensor(y) and x.shape == y.shape
                        and torch.equal(x.cpu(), y.cpu())):
                    mismatch.append(i)
            elif x != y:
                mismatch.append(i)
        ia = pydiffvg.RenderFunction.apply(canvas, canvas, 2, 2, 0, None, *sa)
        ib = pydiffvg.RenderFunction.apply(canvas, canvas, 2, 2, 0, None, *sb)
        pix = float((ia - ib).abs().max())
        print(f"  arg lists same length: {same_len}, mismatched entries: {len(mismatch)}")
        print(f"  max |pixel difference|: {pix:.3e}  "
              f"({'bit-identical' if pix == 0 else 'DIFFERENT'})")
        res.update({"args_equal": same_len and not mismatch, "pixel_diff": pix})
        if not (same_len and not mismatch and pix == 0):
            raise SystemExit("ABORT: the assert-free version is not equivalent")

    if a.bench:
        # Paired within one process: alternate the two implementations iteration by
        # iteration. Both produce identical args, so alternating cannot change any
        # result -- but it does force both arms through the same contention, which
        # a shared GPU makes essential.
        def render_with(ser):
            sc = ser(canvas, canvas, r.shapes, r.shape_groups)
            img = pydiffvg.RenderFunction.apply(canvas, canvas, 2, 2, 0, None, *sc)
            return img

        def full_with(ser):
            for s in r.shapes:
                s.points.grad = None
            img = render_with(ser)
            opacity = img[:, :, 3:4]
            rgb = opacity * img[:, :, :3] + torch.ones(
                img.shape[0], img.shape[1], 3, device=device) * (1 - opacity)
            (rgb ** 2).sum().backward()

        fn = full_with if a.full_loop else render_with
        label = "render fwd+bwd" if a.full_loop else "render forward"
        for _ in range(10):
            fn(stock); fn(fast)
        torch.cuda.synchronize()
        ts, tf = [], []
        for _ in range(a.iters):
            t0 = time.perf_counter(); fn(stock); torch.cuda.synchronize()
            t1 = time.perf_counter(); fn(fast); torch.cuda.synchronize()
            t2 = time.perf_counter()
            ts.append((t1 - t0) * 1e3); tf.append((t2 - t1) * 1e3)
        ts, tf = np.array(ts), np.array(tf)
        ratio = ts / tf
        res["bench"] = {
            "what": label, "iters": a.iters,
            "stock_ms": {"min": float(ts.min()), "median": float(np.median(ts))},
            "fast_ms": {"min": float(tf.min()), "median": float(np.median(tf))},
            "min_ratio": float(ts.min() / tf.min()),
            "median_paired_ratio": float(np.median(ratio)),
            "p25": float(np.percentile(ratio, 25)), "p75": float(np.percentile(ratio, 75)),
        }
        b = res["bench"]
        print(f"\n  {label}, {a.iters} paired reps, n={a.strokes} strokes")
        print(f"    stock  min {b['stock_ms']['min']:7.3f} ms   "
              f"median {b['stock_ms']['median']:7.3f} ms")
        print(f"    fast   min {b['fast_ms']['min']:7.3f} ms   "
              f"median {b['fast_ms']['median']:7.3f} ms")
        print(f"    speedup: min-ratio {b['min_ratio']:.2f}x   "
              f"paired median {b['median_paired_ratio']:.2f}x "
              f"[{b['p25']:.2f}-{b['p75']:.2f}]")
        print(f"    saved per render: "
              f"{b['stock_ms']['min'] - b['fast_ms']['min']:.3f} ms (min-based)")

    if a.train_loop:
        # The end-to-end number. Alternating the serialiser inside a live training
        # loop is safe precisely because --verify proved the two produce identical
        # args: the trajectory is whatever it would have been, and each arm sees
        # the same contention, the same iteration index, and the same optimiser
        # state distribution.
        renderer, loss_func, optimizer, target, _ = common.build_pipeline(args)
        optimizer.init_optimizers()
        stock_ser = pydiffvg.RenderFunction.serialize_scene

        def one_iter(ser):
            pydiffvg.RenderFunction.serialize_scene = staticmethod(ser)
            try:
                optimizer.zero_grad_()
                renderer.set_random_noise(0)
                sketches = renderer.get_image().to(args.device)
                losses = loss_func(sketches, target.detach(),
                                   renderer.get_color_parameters(), renderer, 0, optimizer)
                loss = sum(list(losses.values()))
                loss.backward()
                optimizer.step_()
                del sketches, losses, loss
            finally:
                pydiffvg.RenderFunction.serialize_scene = staticmethod(stock_ser)

        for _ in range(10):
            one_iter(stock_ser); one_iter(fast)
        torch.cuda.synchronize()
        ts, tf = [], []
        for _ in range(a.iters):
            t0 = time.perf_counter(); one_iter(stock_ser); torch.cuda.synchronize()
            t1 = time.perf_counter(); one_iter(fast); torch.cuda.synchronize()
            t2 = time.perf_counter()
            ts.append((t1 - t0) * 1e3); tf.append((t2 - t1) * 1e3)
        ts, tf = np.array(ts), np.array(tf)
        ratio = ts / tf
        res["train_loop"] = {
            "iters": a.iters,
            "stock_ms": {"min": float(ts.min()), "median": float(np.median(ts))},
            "fast_ms": {"min": float(tf.min()), "median": float(np.median(tf))},
            "min_ratio": float(ts.min() / tf.min()),
            "median_paired_ratio": float(np.median(ratio)),
            "p25": float(np.percentile(ratio, 25)), "p75": float(np.percentile(ratio, 75)),
        }
        t = res["train_loop"]
        print(f"\n  FULL CLIPasso iteration, {a.iters} paired reps, n={a.strokes}")
        print(f"    stock  min {t['stock_ms']['min']:7.3f} ms   "
              f"median {t['stock_ms']['median']:7.3f} ms")
        print(f"    fast   min {t['fast_ms']['min']:7.3f} ms   "
              f"median {t['fast_ms']['median']:7.3f} ms")
        print(f"    end-to-end speedup: min-ratio {t['min_ratio']:.3f}x   "
              f"paired median {t['median_paired_ratio']:.3f}x "
              f"[{t['p25']:.2f}-{t['p75']:.2f}]")

    suffix = "_full" if a.full_loop else ("_train" if a.train_loop else "")
    fp = outdir / f"fast_serialize_n{a.strokes}{suffix}.json"
    fp.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {fp}")


if __name__ == "__main__":
    main()
