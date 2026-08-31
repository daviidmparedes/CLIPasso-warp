#!/usr/bin/env python3
"""Shared setup for the CLIPasso benchmark harnesses.

Everything here goes through the repo's own config.parse_arguments() rather than
hand-rolling an args namespace, so the benchmarks are guaranteed to see exactly the
defaults the real entry point sees (including the pydiffvg device setup and the
clip_conv_layer_weights string parsing, both of which happen inside that function).
"""
import contextlib
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402


# run_object_sketching.py's exact invocation of painterly_rendering.py.
# Reproduced here so the benchmark measures the shipped configuration, not a guess.
RUNNER_DEFAULTS = dict(
    num_iter=2001,
    save_interval=10,
    use_gpu=1,
    fix_scale=0,
    mask_object=0,
    mask_object_attention=0,
    display_logs=0,
    display=0,
)


def make_args(target, output_dir, wandb_name, num_paths=16, seed=0, **overrides):
    """Build an args namespace by driving the repo's own argument parser."""
    import config

    opts = dict(RUNNER_DEFAULTS)
    opts.update(overrides)
    argv = ["painterly_rendering.py", str(target),
            "--output_dir", str(output_dir),
            "--wandb_name", str(wandb_name),
            "--num_paths", str(num_paths),
            "--seed", str(seed)]
    for k, v in opts.items():
        argv += [f"--{k}", str(v)]

    old_argv = sys.argv
    sys.argv = argv
    try:
        args = config.parse_arguments()
    finally:
        sys.argv = old_argv
    return args


def build_pipeline(args):
    """Instantiate the exact objects painterly_rendering.main() would.

    Returns (renderer, loss_func, optimizer, target_tensor, init_img).
    """
    import painterly_rendering as pr
    from models.loss import Loss
    from models.painter_params import PainterOptimizer

    loss_func = Loss(args)
    inputs, mask = pr.get_target(args)
    renderer = pr.load_renderer(args, inputs, mask)
    optimizer = PainterOptimizer(args, renderer)
    renderer.set_random_noise(0)
    init_img = renderer.init_image(stage=0)
    optimizer.init_optimizers()
    return renderer, loss_func, optimizer, inputs, init_img


class GpuSampler:
    """Polls nvidia-smi in a background thread for utilisation + memory.

    torch.cuda.memory_allocated only sees the caching allocator; diffvg allocates
    outside it, so the only honest number for total footprint comes from nvidia-smi.
    """

    def __init__(self, interval=0.05, device=0):
        self.interval, self.device = interval, device
        self.util, self.mem = [], []
        self._stop = threading.Event()
        self._t = None

    def _run(self):
        cmd = ["nvidia-smi", f"--id={self.device}",
               "--query-gpu=utilization.gpu,memory.used",
               "--format=csv,noheader,nounits"]
        while not self._stop.is_set():
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout.strip()
                u, m = out.split(",")
                self.util.append(float(u))
                self.mem.append(float(m))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._t:
            self._t.join(timeout=2)

    def summary(self):
        if not self.util:
            return {"gpu_util_mean": None, "gpu_util_max": None,
                    "gpu_mem_used_mb_max": None, "n_samples": 0}
        return {
            "gpu_util_mean": round(float(np.mean(self.util)), 1),
            "gpu_util_max": round(float(np.max(self.util)), 1),
            "gpu_mem_used_mb_max": round(float(np.max(self.mem)), 1),
            "n_samples": len(self.util),
        }


@contextlib.contextmanager
def quiet():
    """Silence the repo's tqdm/print noise without hiding real errors."""
    devnull = open(os.devnull, "w")
    old = sys.stdout
    sys.stdout = devnull
    try:
        yield
    finally:
        sys.stdout = old
        devnull.close()


def env_fingerprint():
    """Record everything that could plausibly move a timing number."""
    import torchvision
    fp = {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "python": sys.version.split()[0],
    }
    if torch.cuda.is_available():
        fp.update({
            "gpu": torch.cuda.get_device_name(0),
            "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "arch_list": torch.cuda.get_arch_list(),
        })
    try:
        fp["driver"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        pass
    try:
        fp["git_commit"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            capture_output=True, text=True, timeout=5).stdout.strip()
        fp["git_branch"] = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT,
            capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        pass
    return fp


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Stopwatch:
    """Accumulates named, GPU-synchronised durations in milliseconds."""

    def __init__(self):
        self.acc = {}

    @contextlib.contextmanager
    def __call__(self, name):
        sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            sync()
            self.acc.setdefault(name, []).append((time.perf_counter() - t0) * 1e3)

    def stats(self, drop_warmup=0):
        out = {}
        for k, v in self.acc.items():
            v = v[drop_warmup:]
            if not v:
                continue
            a = np.asarray(v)
            out[k] = {"mean_ms": float(a.mean()), "median_ms": float(np.median(a)),
                      "std_ms": float(a.std()), "min_ms": float(a.min()),
                      "max_ms": float(a.max()), "n": int(a.size)}
        return out


def require_free_gpu_memory(min_free_gb=4.0, hard=False):
    """Warn (or abort) when the GPU is nearly full.

    diffvg allocates with raw cudaMalloc/cudaFree, outside PyTorch's caching
    allocator and without checking the result. Under exhaustion it does not raise
    -- it renders into a buffer it did not get, and the output is silently wrong.
    Observed on this machine while sharing the GPU: every sketch in a run encoded
    to the same feature vector, which turned zero-shot top-1 from 13.3% into 0.0%
    with no error anywhere. Any measurement taken near the memory ceiling is
    suspect, so check before starting rather than debugging the numbers afterwards.
    """
    if not torch.cuda.is_available():
        return None
    free_b, total_b = torch.cuda.mem_get_info()
    free_gb, total_gb = free_b / 2**30, total_b / 2**30
    if free_gb < min_free_gb:
        msg = (f"only {free_gb:.1f} GB free of {total_gb:.1f} GB. diffvg fails "
               f"silently under memory pressure and will produce wrong output.")
        if hard:
            raise SystemExit(f"ABORT: {msg}")
        print(f"WARNING: {msg}\n         Results from this run should not be trusted.",
              flush=True)
    return free_gb
