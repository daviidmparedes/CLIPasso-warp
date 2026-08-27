#!/usr/bin/env python3
"""
Idempotent source patches to make the 2021 CLIPasso tree run on
Python 3.12 / PyTorch 2.9 / NumPy 2.x / SciPy 1.18.

Every patch is a (file, old, new, why) tuple. Re-running is a no-op.
Run:  python bench/apply_compat_patches.py [--check]
"""
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

PATCHES = [
    (
        "CLIP_/clip/auxilary.py",
        "pad = F._pad",
        "pad = F.pad  # [compat] F._pad was removed after torch 1.9; F.pad is the same op",
        "torch.nn.functional._pad no longer exists (torch>=1.10). It was only ever an "
        "alias introduced to dodge a __torch_function__ recursion. Blocks ALL imports.",
    ),
    (
        "models/painter_params.py",
        "from scipy.ndimage.filters import gaussian_filter",
        "from scipy.ndimage import gaussian_filter  # [compat] scipy.ndimage.filters namespace removed in SciPy 2.0",
        "scipy.ndimage.filters is deprecated (warns on 1.18) and is removed in SciPy 2.0.",
    ),
    (
        "models/painter_params.py",
        "prob_sum = sum_attn[self.inds[:,0].astype(np.int), self.inds[:,1].astype(np.int)]",
        "prob_sum = sum_attn[self.inds[:,0].astype(int), self.inds[:,1].astype(int)]  # [compat] np.int removed in NumPy 1.24",
        "np.int was removed in NumPy 1.24. Only reachable via --saliency_model dino "
        "(default is clip), so this is latent rather than fatal -- patched anyway.",
    ),
    (
        "sketch_utils.py",
        "net.load_state_dict(torch.load(model_dir))",
        "net.load_state_dict(torch.load(model_dir, weights_only=True))  # [compat] torch>=2.6 flipped weights_only default",
        "torch 2.6 flipped the weights_only default to True. u2net.pth is a plain "
        "state_dict so this is safe, and being explicit stops a future default change biting us.",
    ),
]


def main():
    check_only = "--check" in sys.argv
    applied, already, missing = 0, 0, 0
    for rel, old, new, why in PATCHES:
        p = ROOT / rel
        if not p.exists():
            print(f"  MISSING FILE  {rel}")
            missing += 1
            continue
        src = p.read_text()
        if new in src:
            print(f"  already       {rel}: {old[:58]}")
            already += 1
        elif old in src:
            if check_only:
                print(f"  NEEDS PATCH   {rel}: {old[:58]}")
            else:
                p.write_text(src.replace(old, new))
                print(f"  patched       {rel}: {old[:58]}")
                print(f"                why: {why}")
            applied += 1
        else:
            print(f"  NOT FOUND     {rel}: {old[:58]}  (upstream changed?)")
            missing += 1
    print(f"\n{applied} to apply / {already} already applied / {missing} problems")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
