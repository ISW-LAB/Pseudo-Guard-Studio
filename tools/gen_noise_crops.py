#!/usr/bin/env python3
"""Fabricate the validator's training crops, and a manifest for the human review screen.

The validator learns "good vs noise" from crops a RULE manufactures out of the human's labels:

    good      the labeled boxes, lightly jittered                 → clf_train_yes/
    empty     background boxes the rule says hold no object       → clf_train_no/
    deviated  labeled boxes displaced off their object            → clf_train_no/

Running this as its own step is what makes the rule reviewable: the app pauses training here,
shows these crops, and lets the human change the rule and regenerate before the validator
learns anything. The output folders are exactly where ``train_and_predict.py --stage validator``
loads approved crops from, so confirming the review costs no extra work.

Writes a JSON manifest — counts, the effective rule, and a thumbnail sample per type — which is
what the review screen renders. Seeded, so regenerating with the same rule gives the same crops.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root
from tools import common                                          # noqa: E402
from tools.common import log                                      # noqa: E402

common.enable_utf8_output()   # a redirected stdout must not be code-page limited


def jitter_boxes(boxes, jitter: float, width: int, height: int):
    """Perturb each box's centre and size by ±jitter (a fraction of the box).

    The positives must not be pixel-perfect ground truth: a validator trained only on exact
    boxes learns "is this box perfectly placed", not "is there an object here", and then rejects
    every slightly-off detection the detector actually produces.
    """
    import random
    import torch
    if jitter <= 0:
        return boxes
    out = boxes.clone()
    for i in range(len(out)):
        x1, y1, x2, y2 = out[i].tolist()
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        cx += random.uniform(-jitter, jitter) * w
        cy += random.uniform(-jitter, jitter) * h
        w *= 1 + random.uniform(-jitter, jitter)
        h *= 1 + random.uniform(-jitter, jitter)
        nx1, ny1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
        nx2, ny2 = min(float(width), cx + w / 2), min(float(height), cy + h / 2)
        if nx2 - nx1 >= 2 and ny2 - ny1 >= 2:
            out[i] = torch.tensor([nx1, ny1, nx2, ny2])
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--work-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="clf_train_yes/ and clf_train_no/ are written here")
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--train-scope", default="human", choices=["human", "all"])
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda:0")
    ap.add_argument("--noise-config", type=Path, default=None,
                    help="JSON of NoiseGenerationConfig overrides (the human-tuned rule)")
    ap.add_argument("--preprocess", default="none", choices=common.PREPROCESS_MODES)
    ap.add_argument("--max-crops", type=int, default=8000)
    ap.add_argument("--sample-per-type", type=int, default=24,
                    help="how many thumbnails per type to name in the manifest")
    ap.add_argument("--seed", type=int, default=42)
    return ap


def generate(args, manifest_out: dict) -> dict:
    import random
    import numpy as np
    import torch
    from PIL import Image

    common.bootstrap_path()
    from pseudoguard.data.noise_generator import NoiseGenerator

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.device = common.resolve_device(args.device)

    cfg = common.build_noise_config(args.noise_config)      # raises ValueError on bad ratios
    raw = common.load_overrides(args.noise_config)
    # UI-only knob: a FIXED number of deviated cases per image, so the review screen shows a
    # comparable sample per image instead of a count that swings with object density.
    dev_per_image = int(raw.get("deviation_per_image", 0) or 0)
    generator = NoiseGenerator(cfg, device=args.device)

    seed, _scope = common.seed_images(args.images, args.labels, args.train_scope)
    if not seed:
        raise SystemExit("no labeled images with boxes in scope")
    seed_names = [name for name, _ in seed]
    src_dir = common.ensure_preprocessed(args.images, seed_names,
                                         args.work_dir / "prep", args.preprocess)

    yes_dir, no_dir = args.out_dir / "clf_train_yes", args.out_dir / "clf_train_no"
    for d in (yes_dir, no_dir):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)

    good_ratio = cfg.good_crop_ratio
    negative_ratio = cfg.empty_crop_ratio + cfg.deviation_ratio
    n_good = n_empty_total = n_dev_total = 0
    samples = {"good": [], "empty": [], "deviated": []}

    def save(crop, folder, prefix, idx, bucket):
        fname = f"{prefix}_{idx:07d}.jpg"
        crop.convert("RGB").save(folder / fname, quality=85)
        if len(samples[bucket]) < args.sample_per_type:
            samples[bucket].append(fname)

    for name, boxes in seed:
        if (n_good + n_empty_total + n_dev_total) >= args.max_crops:
            break
        with Image.open(src_dir / name) as im:
            img = im.convert("RGB")
            W, H = img.size
            box_tensor = torch.tensor(
                [[(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H]
                 for _c, cx, cy, w, h in boxes], dtype=torch.float32)
            if box_tensor.numel() == 0:
                continue
            goods = generator.good_crops(img, jitter_boxes(box_tensor, cfg.good_crop_jitter, W, H))
            # Negatives are sized RELATIVE to the positives actually produced, so the ratios hold
            # per image rather than only in aggregate.
            n_noise = int(len(goods) * negative_ratio / good_ratio) if good_ratio > 0 else 0
            n_empty = int(n_noise * cfg.empty_crop_ratio / (negative_ratio or 1.0))
            n_dev = dev_per_image if dev_per_image > 0 else (n_noise - n_empty)
            empties = generator.empty_crops(img, box_tensor, n_empty) if n_empty > 0 else []
            deviated = generator.deviated_crops(img, box_tensor, n_dev) if n_dev > 0 else []
        for c in goods:
            save(c, yes_dir, "good", n_good, "good")
            n_good += 1
        for c in empties:
            save(c, no_dir, "empty", n_empty_total, "empty")
            n_empty_total += 1
        for c in deviated:
            save(c, no_dir, "dev", n_dev_total, "deviated")
            n_dev_total += 1

    if n_good == 0 or (n_empty_total + n_dev_total) == 0:
        raise SystemExit(f"need both good and noise crops "
                         f"(good={n_good} noise={n_empty_total + n_dev_total}); seed too small")

    effective = cfg.describe()
    effective["deviation_per_image"] = dev_per_image or 10
    manifest_out = {"ok": True, "seed_images": len(seed_names), "preprocess": args.preprocess,
                    "counts": {"good": n_good, "empty": n_empty_total, "deviated": n_dev_total,
                               "noise": n_empty_total + n_dev_total,
                               "total": n_good + n_empty_total + n_dev_total},
                    "config": effective, "samples": samples}
    log(f"[crops] good={n_good} empty={n_empty_total} deviated={n_dev_total} "
        f"(rule={cfg.negative_rule}, shift={cfg.deviation_shift}, preprocess={args.preprocess})")
    return manifest_out


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    for a in ("images", "labels", "work_dir", "out_dir", "manifest"):
        setattr(args, a, getattr(args, a).resolve())

    manifest = {"ok": False}
    try:
        manifest = generate(args, manifest)
    except (SystemExit, ValueError) as e:
        manifest = {"ok": False, "error": str(e)}
        log(f"[crops:fail] {e}")
    except Exception as e:
        manifest = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        log("[crops:fail] " + traceback.format_exc())

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if manifest.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
