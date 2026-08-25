#!/usr/bin/env python3
"""The negative-crop rule as the UI exposes it, plus the preview the review screen draws.

Training pauses between the detector and the validator so the human can SEE the crops the rule
fabricates and adjust it. That review screen needs two things this module provides:

    validate_noise_config  clamp whatever the browser posted into a rule the generator accepts
    preview_deviated       the deviated-box geometry for one image, in pure Python

The preview matters more than it looks: it is what makes the deviation-shift slider legible.
Moving the slider redraws the boxes instantly, in the app's own process, with no torch and no
crop generation — so the human sees what "0.4" versus "1.2" means before committing to a run
that takes minutes. The geometry here is deliberately the same as the generator's refined rule
(``pseudoguard.data.noise_generator``); if one changes, the other must.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

from . import state
from .geometry import iou_xyxy
from .labelio import image_size, load_yolo

# UI-exposed rule knobs and their defaults. These mirror ``pseudoguard.config``; the app ships
# the REFINED rule with a 0.80 centre shift, which is the setting the paper reports.
NOISE_DEFAULTS = {
    "negative_rule": "refined",
    "good_crop_ratio": 0.4,
    "empty_crop_ratio": 0.3,
    "deviation_ratio": 0.3,
    "good_crop_jitter": 0.05,
    "deviation_max_iou": 0.10,          # baseline rule only
    "deviation_min_area_ratio": 0.7,
    "negative_iou_reject": 0.75,        # refined rule: reject a negative at/above this IoU
    "deviation_shift": 0.80,            # centre displacement as a fraction of the box size
    "deviation_per_image": 10,          # how many deviated cases to draw per image
}

SHIFT_MIN, SHIFT_MAX = 0.1, 2.0
SHIFT_BAND = 0.12                       # scalar shift → [shift-band, shift+band]
PER_IMAGE_MIN, PER_IMAGE_MAX = 1, 30


def defaults() -> dict:
    return dict(NOISE_DEFAULTS)


def validate_noise_config(cfg: dict) -> dict:
    """Merge a posted config over the defaults and make it structurally valid.

    The three crop ratios are renormalised to sum to 1.0 rather than rejected, so the
    generator's ratio check can never trip on a half-typed number from a slider, and unknown
    keys are dropped rather than forwarded.
    """
    d = defaults()
    for k, v in (cfg or {}).items():
        if k in d:
            d[k] = v
    try:
        g = float(d["good_crop_ratio"])
        e = float(d["empty_crop_ratio"])
        dv = float(d["deviation_ratio"])
    except (TypeError, ValueError):
        g = e = dv = 0.0
    total = (g + e + dv) or 1.0
    d["good_crop_ratio"], d["empty_crop_ratio"], d["deviation_ratio"] = g / total, e / total, dv / total
    if d.get("negative_rule") not in ("baseline", "refined"):
        d["negative_rule"] = "baseline"
    try:
        d["deviation_shift"] = min(SHIFT_MAX, max(SHIFT_MIN, float(d.get("deviation_shift", 0.8))))
    except (TypeError, ValueError):
        d["deviation_shift"] = NOISE_DEFAULTS["deviation_shift"]
    try:
        n = int(round(float(d.get("deviation_per_image", PER_IMAGE_MIN))))
        d["deviation_per_image"] = min(PER_IMAGE_MAX, max(PER_IMAGE_MIN, n))
    except (TypeError, ValueError):
        d["deviation_per_image"] = NOISE_DEFAULTS["deviation_per_image"]
    return d


def shift_band(shift: float) -> tuple[float, float]:
    """The sampling band a scalar centre shift expands into (matches pseudoguard.config)."""
    return max(0.05, float(shift) - SHIFT_BAND), float(shift) + SHIFT_BAND


def preview_deviated(image_name: str, cfg: dict, n: int = None) -> dict:
    """Ground-truth and deviated boxes for one seed image, normalised, at the chosen shift.

    Pure Python and pure PIL: no torch, so the review screen stays responsive and works in the
    label-only install where torch is not present at all.
    """
    if n is None:
        n = int(cfg.get("deviation_per_image", NOISE_DEFAULTS["deviation_per_image"]))
    W, H = image_size(state.CFG["images"] / image_name)
    boxes = load_yolo(state.CFG["labels"], Path(image_name).stem)
    gt_px = [((b["cx"] - b["w"] / 2) * W, (b["cy"] - b["h"] / 2) * H,
              (b["cx"] + b["w"] / 2) * W, (b["cy"] + b["h"] / 2) * H) for b in boxes]

    shift = float(cfg.get("deviation_shift", NOISE_DEFAULTS["deviation_shift"]))
    lo, hi = shift_band(shift)
    reject = (float(cfg.get("negative_iou_reject", 0.75)) if cfg.get("negative_rule") == "refined"
              else float(cfg.get("deviation_max_iou", 0.25)))
    min_area_ratio = float(cfg.get("deviation_min_area_ratio", 0.7))

    deviated = []
    for _ in range(n):
        if not gt_px:
            break
        g = random.choice(gt_px)
        w, h = g[2] - g[0], g[3] - g[1]
        cx, cy = (g[0] + g[2]) / 2, (g[1] + g[3]) / 2
        original_area = w * h
        for _attempt in range(100):
            frac = random.uniform(lo, hi)
            theta = random.uniform(0, 2 * math.pi)
            ncx, ncy = cx + frac * w * math.cos(theta), cy + frac * h * math.sin(theta)
            nw, nh = w * random.uniform(0.7, 1.3), h * random.uniform(0.7, 1.3)
            if nw * nh < min_area_ratio * original_area:
                continue
            x1, y1 = max(0.0, ncx - nw / 2), max(0.0, ncy - nh / 2)
            x2, y2 = min(float(W), ncx + nw / 2), min(float(H), ncy + nh / 2)
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue
            if max((iou_xyxy((x1, y1, x2, y2), gg) for gg in gt_px), default=0.0) < reject:
                deviated.append([x1, y1, x2, y2])
                break

    def normalise(b):
        return {"cx": (b[0] + b[2]) / 2 / W, "cy": (b[1] + b[3]) / 2 / H,
                "w": (b[2] - b[0]) / W, "h": (b[3] - b[1]) / H}

    return {"image": image_name, "width": W, "height": H, "shift": round(shift, 2),
            "gt": [normalise(g) for g in gt_px],
            "deviated": [normalise(b) for b in deviated]}
