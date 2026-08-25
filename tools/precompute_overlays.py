#!/usr/bin/env python3
"""Freeze the candidate overlay for a folder of images.

An overlay is the app's AI in a file: every candidate box with both scores, computed once.
Serving one means the annotator needs no GPU, no torch and no model on disk, and every session
sees byte-identical AI output — which is what makes a user study or a demo reproducible.

    # real: run the trained detector + validator over the images
    python tools/precompute_overlays.py --images DIR --detector det.pt --validator val.pt \\
        --out artifacts/overlay.json

    # synthetic: no models at all — exercises the whole selection path for tests and demos
    python tools/precompute_overlays.py --synthetic --out artifacts/overlay.json

Schema: { image_id: [ {box_id, box_xyxy, label, det_conf, p_good}, … ], … }
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root
from tools.common import (IMG_EXTS, bootstrap_path, enable_utf8_output,  # noqa: E402
                          resolve_device)

enable_utf8_output()   # a redirected stdout must not be code-page limited


def synthetic_overlay(n_images: int = 20, gold_density: int = 3, seed: int = 42) -> dict:
    """A fake overlay whose shape matches a real one.

    Each image gets ``gold_density`` high-P(good) "true" candidates plus a variable number of
    low-scoring ones — the noise a permissive collection threshold really does admit. That is
    enough structure for the operating point, the adapters and the metrics to be exercised
    end to end without a model.
    """
    rng = random.Random(seed)
    overlay: dict[str, list[dict]] = {}
    for i in range(n_images):
        img = f"img_{i:04d}.jpg"
        cands: list[dict] = []
        for _ in range(gold_density):                       # true objects: tight, high p_good
            x, y = rng.uniform(0, 500), rng.uniform(0, 500)
            cands.append({"box_id": f"{img}:{len(cands)}", "box_xyxy": [x, y, x + 60, y + 60],
                          "label": rng.randint(0, 2), "det_conf": rng.uniform(0.4, 0.9),
                          "p_good": rng.uniform(0.7, 0.98)})
        for _ in range(rng.randint(4, 12)):                 # noise: what a flat low cut admits
            x, y = rng.uniform(0, 560), rng.uniform(0, 560)
            cands.append({"box_id": f"{img}:{len(cands)}",
                          "box_xyxy": [x, y, x + rng.uniform(20, 80), y + rng.uniform(20, 80)],
                          "label": rng.randint(0, 2), "det_conf": rng.uniform(0.05, 0.5),
                          "p_good": rng.uniform(0.05, 0.55)})
        overlay[img] = cands
    return overlay


def real_overlay(args) -> dict:
    """Run the trained detector + validator over every image in ``--images``."""
    bootstrap_path()
    from pgcount.backend import FrozenBackend

    backend = FrozenBackend(Path(args.detector), Path(args.validator),
                            pseudo_conf_threshold=args.pseudo_conf,
                            det_model_type=args.det_model_type, det_size=args.det_size,
                            det_img_size=args.det_img_size,
                            device=resolve_device(args.device)).load()
    img_paths = sorted(p for p in Path(args.images).iterdir()
                       if p.is_file() and p.suffix in IMG_EXTS)
    overlay: dict[str, list[dict]] = {}
    for img_path in img_paths:
        img_id = img_path.name
        overlay[img_id] = [{"box_id": c.box_id, "box_xyxy": list(c.box_xyxy), "label": c.label,
                            "det_conf": c.det_conf, "p_good": c.p_good}
                           for c in backend.candidates(img_id, img_path)]
    return overlay


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--synthetic", action="store_true", help="generate a fake overlay (no models)")
    ap.add_argument("--images", type=Path, help="folder of images (real mode)")
    ap.add_argument("--detector", type=Path, help="detector checkpoint (real mode)")
    ap.add_argument("--validator", type=Path, help="validator checkpoint (real mode)")
    ap.add_argument("--det-model-type", default="yolov8", help="yolov8|yolov11|yolo26|rtdetr")
    ap.add_argument("--det-size", default="n", help="n|s|m|l|x")
    ap.add_argument("--det-img-size", type=int, default=640)
    ap.add_argument("--pseudo-conf", type=float, default=0.05,
                    help="detector collection threshold (deliberately permissive)")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda:0")
    ap.add_argument("--n-images", type=int, default=20, help="synthetic mode only")
    ap.add_argument("--gold-density", type=int, default=3, help="synthetic mode only")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not args.synthetic and not (args.images and args.detector and args.validator):
        build_parser().error("real mode needs --images, --detector and --validator "
                             "(or pass --synthetic)")
    overlay = (synthetic_overlay(args.n_images, args.gold_density) if args.synthetic
               else real_overlay(args))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(overlay, ensure_ascii=False, indent=2), encoding="utf-8")
    n_boxes = sum(len(v) for v in overlay.values())
    print(f"[overlay] {len(overlay)} images, {n_boxes} candidates → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
