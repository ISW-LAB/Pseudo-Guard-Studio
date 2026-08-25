#!/usr/bin/env python3
"""Exporting the finished annotations.

YOLO needs no export — the app writes that format as it goes, so "export YOLO" is just the path
to the labels folder. COCO is a real conversion: normalised centre boxes become absolute
top-left boxes, and class ids become 1-based category ids.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import state
from .labelio import image_size, list_images, load_yolo


def export_coco() -> dict:
    """Write ``instances_export.json`` next to the labels, in COCO detection format.

    Boxes are clamped to the image and degenerate ones are dropped: a box the user dragged off
    the canvas is a valid thing to have on screen but not a valid annotation to publish.
    """
    images_dir, labels_dir, classes = state.CFG["images"], state.CFG["labels"], state.CFG["classes"]
    images, annotations, ann_id = [], [], 1
    for i, name in enumerate(list_images(images_dir), 1):
        W, H = image_size(images_dir / name)
        images.append({"id": i, "file_name": name, "width": W, "height": H})
        for b in load_yolo(labels_dir, Path(name).stem):
            x1 = max(0.0, (b["cx"] - b["w"] / 2) * W)
            y1 = max(0.0, (b["cy"] - b["h"] / 2) * H)
            x2 = min(float(W), (b["cx"] + b["w"] / 2) * W)
            y2 = min(float(H), (b["cy"] + b["h"] / 2) * H)
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                continue
            annotations.append({"id": ann_id, "image_id": i, "category_id": b["cls"] + 1,
                                "bbox": [x1, y1, bw, bh], "area": bw * bh, "iscrowd": 0})
            ann_id += 1
    coco = {"images": images, "annotations": annotations,
            "categories": [{"id": j + 1, "name": c} for j, c in enumerate(classes)]}
    out = labels_dir / "instances_export.json"
    out.write_text(json.dumps(coco), encoding="utf-8")
    return {"format": "coco", "path": str(out), "images": len(images),
            "annotations": len(annotations)}


def export_yolo() -> dict:
    """YOLO is the native on-disk format — nothing to convert, just say where it is."""
    return {"format": "yolo", "labels_dir": str(state.CFG["labels"])}
