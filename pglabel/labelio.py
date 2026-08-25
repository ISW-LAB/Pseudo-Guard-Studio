#!/usr/bin/env python3
"""Reading and writing annotations on disk — YOLO ``.txt``, one file per image.

Pure functions over explicit directories: nothing here reads application state, so the label
format is testable on its own and the same helpers serve the app, the training tools and the
tests. A box is always the normalised dict ``{"cls", "cx", "cy", "w", "h"}`` in [0, 1].
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG")


def list_images(images_dir: Path) -> List[str]:
    """File names (not paths) of every image in ``images_dir``, sorted for a stable UI order."""
    return sorted(p.name for p in Path(images_dir).iterdir()
                  if p.is_file() and p.suffix in IMG_EXTS)


def load_yolo(labels_dir: Path, stem: str) -> List[dict]:
    """Read ``<labels_dir>/<stem>.txt``; an absent or empty file is simply no boxes.

    ``utf-8-sig`` because a label file edited in a Windows editor can carry a BOM, which would
    otherwise turn the first class id into an unparseable token.
    """
    f = Path(labels_dir) / f"{stem}.txt"
    out: List[dict] = []
    if f.exists():
        for line in f.read_text(encoding="utf-8-sig").splitlines():
            p = line.split()
            if len(p) >= 5:
                out.append({"cls": int(float(p[0])), "cx": float(p[1]), "cy": float(p[2]),
                            "w": float(p[3]), "h": float(p[4])})
    return out


def clip_norm_box(b: dict) -> Optional[dict]:
    """Clamp a normalised box to the image and drop it if that leaves it degenerate.

    Keeps every saved label a spec-valid YOLO row even when the user dragged a box partly or
    fully off the canvas before saving: the box is cropped to the image edge instead.
    """
    x1, y1 = b["cx"] - b["w"] / 2, b["cy"] - b["h"] / 2
    x2, y2 = b["cx"] + b["w"] / 2, b["cy"] + b["h"] / 2
    x1, y1 = min(max(x1, 0.0), 1.0), min(max(y1, 0.0), 1.0)
    x2, y2 = min(max(x2, 0.0), 1.0), min(max(y2, 0.0), 1.0)
    w, h = x2 - x1, y2 - y1
    if w <= 1e-6 or h <= 1e-6:
        return None
    return {"cls": int(b["cls"]), "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2, "w": w, "h": h}


def save_yolo(labels_dir: Path, stem: str, boxes: List[dict]) -> None:
    """Write ``<labels_dir>/<stem>.txt``, skipping boxes that clip away to nothing.

    ``newline="\\n"`` is deliberate: Windows text mode would write CRLF, so the same annotation
    session would produce byte-different label files depending on which OS it ran on.
    """
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for b in boxes:
        c = clip_norm_box(b)
        if c is not None:
            lines.append(f"{c['cls']} {c['cx']:.6f} {c['cy']:.6f} {c['w']:.6f} {c['h']:.6f}")
    (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""),
                                            encoding="utf-8", newline="\n")


def image_size(image_path: Path) -> tuple[int, int]:
    """(width, height) of an image, read through Pillow and closed immediately."""
    from PIL import Image
    with Image.open(image_path) as im:
        return im.size
