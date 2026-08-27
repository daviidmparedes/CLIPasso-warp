#!/usr/bin/env python3
"""
Build the CLIPasso benchmark corpus from the Sketchy photo set.

Why Sketchy and not the original image_test/:
  - image_test/{10,15,20,25} carries no usable class label, and quality guardrail #2
    (CLIP zero-shot "A sketch of a(n) {class}") needs one.
  - Sketchy is 125 classes x exactly 100 photos, all 256x256, object-centric --
    which is the distribution CLIPasso was designed for. SketchyCOCO's Object.tar
    on this server is 0 bytes, so it was never an option.

Outputs (all deterministic given --seed):
  data/sketchy2000/<class>/<file>.jpg   2000 photos, 16 per class x 125 classes
  data/manifest.json                    every sampled image with its class label
  data/eval_set.json                    the FIXED 5-image guardrail set (quality guardrails, sec 5)
  data/paper_protocol.json              200 images / 10 categories, mirroring the paper's eval
"""
import argparse
import json
import random
import shutil
from pathlib import Path

SRC = Path("/home/shared_data/sketches/sketchy/photo")
DST_ROOT = Path(__file__).resolve().parent.parent / "data"

# The 10 categories used for the paper-comparable protocol. Chosen to overlap the
# animal/vehicle categories CLIPasso reports on, and to span rigid vs deformable
# shape, which is where stroke-count sensitivity shows up most.
PAPER_CATEGORIES = [
    "horse", "camel", "elephant", "giraffe", "zebra",
    "cat", "dog", "bicycle", "airplane", "sailboat",
]

# The 5 guardrail images. One per class, spanning: deformable quadruped, rigid
# man-made with thin structure, compact blob, tall/thin, and a wing/limb outline.
# Held fixed forever so every (speedup, quality delta) row is comparable.
EVAL_CLASSES = ["horse", "bicycle", "teapot", "giraffe", "butterfly"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-total", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dst", type=Path, default=DST_ROOT / "sketchy2000")
    ap.add_argument("--force", action="store_true", help="re-copy even if dst exists")
    args = ap.parse_args()

    classes = sorted(p.name for p in SRC.iterdir() if p.is_dir())
    per_class = args.n_total // len(classes)
    print(f"{len(classes)} classes, sampling {per_class}/class -> {per_class*len(classes)} images")

    rng = random.Random(args.seed)
    manifest = []
    if args.dst.exists() and args.force:
        shutil.rmtree(args.dst)
    args.dst.mkdir(parents=True, exist_ok=True)

    for ci, cls in enumerate(classes):
        photos = sorted(p.name for p in (SRC / cls).iterdir() if p.suffix.lower() == ".jpg")
        picked = rng.sample(photos, min(per_class, len(photos)))
        (args.dst / cls).mkdir(exist_ok=True)
        for fn in picked:
            dst = args.dst / cls / fn
            if not dst.exists():
                shutil.copy2(SRC / cls / fn, dst)
            manifest.append({
                "path": str(dst),
                "rel": f"{cls}/{fn}",
                "class": cls,
                "class_idx": ci,
                # class name -> natural-language prompt form for zero-shot eval.
                # Sketchy uses "car_(sedan)" / "hot-air_balloon" style names.
                "prompt_name": cls.replace("_", " ").replace("(", "").replace(")", "").strip(),
            })

    DST_ROOT.mkdir(parents=True, exist_ok=True)
    (DST_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote manifest.json ({len(manifest)} images)")

    by_class = {}
    for m in manifest:
        by_class.setdefault(m["class"], []).append(m)

    # Guardrail eval set: first image (sorted) of each eval class, so it never drifts.
    eval_set = []
    for cls in EVAL_CLASSES:
        if cls not in by_class:
            raise SystemExit(f"eval class {cls!r} missing from sample -- raise --n-total")
        eval_set.append(sorted(by_class[cls], key=lambda m: m["rel"])[0])
    (DST_ROOT / "eval_set.json").write_text(json.dumps(eval_set, indent=1))
    print(f"wrote eval_set.json: {[m['rel'] for m in eval_set]}")

    # Paper-protocol subset: 20 per category x 10 categories = 200.
    # Drawn from the FULL Sketchy source, not the 2000-image sample, because
    # 2000/125 = 16 per class < the 20 the protocol needs.
    rng2 = random.Random(args.seed + 1)
    paper = []
    for ci, cls in enumerate(PAPER_CATEGORIES):
        pool = sorted(q.name for q in (SRC / cls).iterdir() if q.suffix.lower() == ".jpg")
        for fn in rng2.sample(pool, min(20, len(pool))):
            dst = args.dst / cls / fn
            dst.parent.mkdir(exist_ok=True)
            if not dst.exists():
                shutil.copy2(SRC / cls / fn, dst)
            paper.append({
                "path": str(dst), "rel": f"{cls}/{fn}", "class": cls, "class_idx": ci,
                "prompt_name": cls.replace("_", " ").replace("(", "").replace(")", "").strip(),
            })
    (DST_ROOT / "paper_protocol.json").write_text(json.dumps(paper, indent=1))
    print(f"wrote paper_protocol.json ({len(paper)} images, {len(PAPER_CATEGORIES)} categories)")


if __name__ == "__main__":
    main()
