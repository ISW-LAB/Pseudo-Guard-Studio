"""Crop extraction utility.

Mirrors the exact cropping logic used by the reused validator
(the validator's own crop path in ``pseudoguard``) so the crops
the app scores are identical to the crops the DenseNet validator was trained/eval'd
on: same clamping, same invalid-box skipping, same PIL ``.crop((x1,y1,x2,y2))``.

Pure PIL — no torch — so it is unit-testable standalone.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple, Union

try:
    from PIL import Image
except Exception:  # keep importable without Pillow (torch env has it)
    Image = None  # type: ignore


def clamp_box(box: Sequence[float], img_w: int, img_h: int) -> Tuple[int, int, int, int] | None:
    """Clamp an xyxy box to image bounds; return int coords or None if degenerate.

    Matches quality_filter.py:99-106 exactly (note the asymmetric clamp: x1/y1 to
    ``w-1``/``h-1``, x2/y2 to ``w``/``h``), so crop geometry is identical.
    """
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), img_w - 1))
    y1 = max(0, min(int(y1), img_h - 1))
    x2 = max(0, min(int(x2), img_w))
    y2 = max(0, min(int(y2), img_h))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def extract_crops(
    image: Union["Image.Image", str, Path],
    boxes_xyxy: Sequence[Sequence[float]],
) -> Tuple[List["Image.Image"], List[int]]:
    """Extract RGB crops for boxes in ORIGINAL-image pixel coords.

    Parameters
    ----------
    image      : a PIL.Image (RGB) or a path to load.
    boxes_xyxy : list of (x1, y1, x2, y2) in the image's pixel coordinate system
                 (exactly what ``detector.predict`` returns for a full-size image).

    Returns
    -------
    (crops, keep_indices)
        ``crops[k]`` is the PIL crop for original box ``keep_indices[k]``.
        Invalid/degenerate boxes are skipped, so ``len(crops) == len(keep_indices)``
        may be < ``len(boxes_xyxy)``. The caller must use ``keep_indices`` to keep
        boxes/scores/labels aligned with the crop scores.
    """
    if Image is None:
        raise RuntimeError("Pillow not available; install pillow to extract crops")

    img = image if hasattr(image, "crop") else Image.open(image)
    img = img.convert("RGB")
    w, h = img.size

    crops: List["Image.Image"] = []
    keep: List[int] = []
    for i, box in enumerate(boxes_xyxy):
        clamped = clamp_box(box, w, h)
        if clamped is None:
            continue
        crops.append(img.crop(clamped))
        keep.append(i)
    return crops, keep
