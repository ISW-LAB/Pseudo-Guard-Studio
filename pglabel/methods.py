#!/usr/bin/env python3
"""Acceptance methodologies, and the two per-dataset decisions derived from the human seed.

A "methodology" is a policy for turning the model's scored candidates into accepted boxes.
The proposed one is Pseudo-Guard with COUNT-GUIDED acceptance: the number of boxes accepted per
image comes from the box counts in the images the human has already labeled, instead of from a
flat threshold picked once. The comparison baselines are plain detector-confidence cuts over the
same candidate pool, which is what makes them a fair comparison — only the acceptance rule
differs.

Two things are recomputed from the seed as the human labels more images (``compute_adaptive``):

    containment  whether nested boxes are duplicates to remove or real objects to keep
    max_topk     an upper bound on count-guided K, taken from the observed density

Both move DURING a session, so callers re-read them rather than caching them at startup.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from . import state
from .geometry import CONTAINMENT_THRESHOLD, containment
from .labelio import load_yolo

GROUP_PROPOSED = "Proposed · Pseudo-Guard"
GROUP_COMPARISON = "Comparison baselines"

# ``seed: True`` marks a methodology that NEEDS the human count prior; with no usable seed it
# falls back to auto-adaptive (Otsu) and the UI says so rather than silently changing behaviour.
METHODS = [
    # -- Proposed: Pseudo-Guard + count-guided acceptance (adaptive per-image K from the seed) --
    {"id": "pseudoguard", "group": GROUP_PROPOSED, "label": "Pseudo-Guard (proposed)",
     "backend": "validator", "op": "adaptive", "thr": 0.5, "seed": True},
    {"id": "manual", "group": GROUP_PROPOSED, "label": "🎚 Manual threshold (confidence slider)",
     "backend": "validator", "op": "manual", "seed": False},
    # -- Comparison baselines: detector confidence only -----------------------------------
    {"id": "conf25", "group": GROUP_COMPARISON, "label": "Confidence (≥ 0.25)",
     "backend": "confidence", "op": "fixed", "thr": 0.25, "seed": False},
    {"id": "conf50", "group": GROUP_COMPARISON, "label": "Confidence (≥ 0.50)",
     "backend": "confidence", "op": "fixed", "thr": 0.50, "seed": False},
    {"id": "conf75", "group": GROUP_COMPARISON, "label": "Confidence (≥ 0.75)",
     "backend": "confidence", "op": "fixed", "thr": 0.75, "seed": False},
    {"id": "conf90", "group": GROUP_COMPARISON, "label": "Confidence (≥ 0.90)",
     "backend": "confidence", "op": "fixed", "thr": 0.90, "seed": False},
]
METHOD_MAP = {m["id"]: m for m in METHODS}
DEFAULT_METHOD = "pseudoguard"

# Regimes in which removing nested boxes destroys real objects rather than duplicates:
#   • the seed shows genuine nesting (a helmet inside a person), or
#   • the seed is sparse, where nested detections are multi-scale duplicates of ONE object.
NEST_FRACTION_OFF = 0.03
SPARSE_MEAN_OFF = 2.5
# Count-guided K is capped at the observed density rather than a loose fixed 100.
TOPK_DENSITY_MULTIPLIER = 1.5


def public_methods() -> list:
    """The registry as the UI needs it (no internal fields)."""
    return [{"id": m["id"], "label": m["label"], "group": m.get("group", ""),
             "seed": m.get("seed", False)}
            for m in METHODS if not m.get("hidden")]


def seed_stats() -> Tuple[dict, object]:
    """Count prior from the HUMAN seed only.

    AI-written images are excluded on purpose: re-applying Auto-label ALL must never let the
    machine's own output drift the prior it is then measured against.
    """
    from pgcount.seed_density import estimate_seed_density
    seed = {}
    for img in state.HUMAN_SET:
        boxes = load_yolo(state.CFG["labels"], Path(img).stem)
        if boxes:
            seed[img] = [None] * len(boxes)
    return seed, estimate_seed_density(seed, min_images=1, min_total_boxes=1)


def compute_adaptive() -> dict:
    """Recompute the per-dataset decisions from the current human seed labels.

    Cheap (it only reads the seed's label files), so it is called before candidate use and on
    every status poll instead of being cached — the decision has to move as the human labels.
    """
    counts, nested_pairs, total_pairs = [], 0, 0
    for img in state.HUMAN_SET:
        boxes = load_yolo(state.CFG["labels"], Path(img).stem)
        if not boxes:
            continue
        counts.append(len(boxes))
        px = [[b["cx"] - b["w"] / 2, b["cy"] - b["h"] / 2,
               b["cx"] + b["w"] / 2, b["cy"] + b["h"] / 2] for b in boxes]
        for i in range(len(px)):
            for j in range(len(px)):
                if i != j:
                    total_pairs += 1
                    if containment(px[i], px[j]) >= CONTAINMENT_THRESHOLD:
                        nested_pairs += 1
    if not counts:
        state.ADAPT.update(containment=True, max_topk=None, nest_frac=None, seed_mean=None)
        return state.ADAPT
    mean = sum(counts) / len(counts)
    p90 = sorted(counts)[min(len(counts) - 1, int(0.9 * len(counts)))]
    nest_frac = nested_pairs / total_pairs if total_pairs else 0.0
    off = (nest_frac >= NEST_FRACTION_OFF) or (mean < SPARSE_MEAN_OFF)
    state.ADAPT.update(containment=(not off),
                       max_topk=max(2, round(p90 * TOPK_DENSITY_MULTIPLIER)),
                       nest_frac=round(nest_frac, 4), seed_mean=round(mean, 2))
    return state.ADAPT


def containment_on() -> bool:
    """The live containment decision (defaults to on until a seed says otherwise)."""
    return state.ADAPT["containment"] is not False


def make_operating_point(method_id: str, raw_candidates: list, stats):
    """Build the acceptance policy for a methodology.

    Returns ``(operating_point, candidates, label, fell_back)``. ``fell_back`` is True when a
    seed-dependent methodology had no usable seed and degraded to auto-adaptive — the UI shows
    that, because silently changing the acceptance rule would misreport what the method did.
    """
    from pgcount.config import CountGuidedConfig, Backend, OperatingPointStrategy
    from pgcount.count_guided_labeler import Candidate
    from pgcount.operating_point import (AdaptivePerImageK, PerImageTopK, GlobalTau,
                                         AutoAdaptiveTau, ThresholdSelect, QuantileRate)
    m = METHOD_MAP.get(method_id, METHOD_MAP[DEFAULT_METHOD])
    backend = Backend.CONFIDENCE if m["backend"] == "confidence" else Backend.VALIDATOR
    cands = [Candidate.from_raw(r, backend) for r in raw_candidates]
    cfg = CountGuidedConfig(
        backend=backend, fixed_filter_threshold=float(m.get("thr", 0.5)),
        strategy=(OperatingPointStrategy.GLOBAL_TAU if m["op"] == "global_tau"
                  else OperatingPointStrategy.PER_IMAGE_TOPK))
    if state.ADAPT["max_topk"]:                    # adaptive density cap on count-guided K
        cfg.max_topk_per_image = int(state.ADAPT["max_topk"])

    label, op_kind, fell_back = m["label"], m["op"], False
    if m.get("seed") and not stats.reliable:
        op_kind = "otsu"
        label = m["label"] + " → auto-adaptive (no human seed yet)"
        fell_back = True

    if op_kind == "adaptive":
        op = AdaptivePerImageK(cfg).fit(stats, cands)
    elif op_kind == "topk":
        op = PerImageTopK(cfg).fit(stats, cands)
    elif op_kind == "global_tau":
        op = GlobalTau(cfg).fit(stats, cands)
    elif op_kind == "otsu":
        op = AutoAdaptiveTau(cfg).fit(None, cands)
    elif op_kind == "rate":
        op = QuantileRate(cfg, rate=float(m.get("rate", 0.5))).fit(None, cands)
    else:                                          # fixed threshold (validator or confidence)
        op = ThresholdSelect(float(m.get("thr", 0.5))).fit()
    return op, cands, label, fell_back
