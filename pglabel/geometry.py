#!/usr/bin/env python3
"""Box overlap arithmetic and the two duplicate-removal passes built on it.

Two different notions of "these boxes are the same object" are needed, and conflating them is
what produces either stacked boxes or vanished objects:

    IoU overlap    two boxes fire on the same object at roughly the same extent.
    CONTAINMENT    one box sits almost entirely inside a strictly larger one — a part of an
                   object detected separately from the whole (a wheel inside a car), or a
                   genuinely nested object (a helmet on a person).

Which of those two is a duplicate depends on the dataset, so containment removal is a decision
the caller passes in (``use_containment``) rather than a constant here. It is derived from the
human seed in ``pglabel.methods``; this module stays pure and has no opinion.

Everything takes and returns xyxy pixel tuples, and no function here reads application state.
"""

from __future__ import annotations

from typing import List, Sequence

# A smaller box whose area is at least this fraction inside a larger box counts as contained.
CONTAINMENT_THRESHOLD = 0.80

# Class-agnostic NMS at the source: fold a detector's many overlapping firings on one object
# into a single box BEFORE any acceptance policy runs. 0.45 removes all top-3 overlaps on the
# benchmark datasets without merging genuinely adjacent objects.
NMS_IOU = 0.45

# Second, stricter pass applied to what actually gets SAVED (see dedup).
DEDUP_IOU = 0.55


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union of two xyxy boxes; 0.0 when they do not overlap."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = area_xyxy(a) + area_xyxy(b) - inter
    return inter / union if union > 0 else 0.0


def area_xyxy(a: Sequence[float]) -> float:
    return max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])


def containment(inner: Sequence[float], outer: Sequence[float]) -> float:
    """Fraction of ``inner``'s area that lies inside ``outer`` (0..1)."""
    ix = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    ai = area_xyxy(inner)
    return (ix * iy / ai) if ai > 0 else 0.0


def nested(a: Sequence[float], b: Sequence[float]) -> bool:
    """True when either box is largely contained in the other."""
    return (containment(a, b) >= CONTAINMENT_THRESHOLD
            or containment(b, a) >= CONTAINMENT_THRESHOLD)


def nms_candidates(cands: List, use_containment: bool) -> List:
    """Greedy class-agnostic NMS over scored candidates, applied at the SOURCE.

    Ranked by ``p_good`` — the validator score the pipeline actually SELECTS on. p_good and
    det_conf are decoupled by design, so ranking by the wrong one would keep the box that is
    least likely to survive acceptance and would feed the wrong score into the count prior.

    Overlap handling matches ``dedup``: an IoU duplicate collapses to the higher-p_good box,
    while a nested pair collapses to the LARGER box (the whole object), never the part-box.
    """
    kept: List = []
    for c in sorted(cands, key=lambda c: -float(c.p_good)):
        ca = area_xyxy(c.box_xyxy)
        drop = False
        for k in kept:
            if iou_xyxy(c.box_xyxy, k.box_xyxy) > NMS_IOU:
                drop = True                      # duplicate → keep the earlier, higher-p_good box
                break
            if use_containment and nested(c.box_xyxy, k.box_xyxy) and ca <= area_xyxy(k.box_xyxy):
                drop = True                      # c is the smaller nested box → drop it
                break
        if drop:
            continue
        if use_containment:                      # c survives: evict smaller boxes nested with it
            kept = [k for k in kept
                    if not (nested(k.box_xyxy, c.box_xyxy) and area_xyxy(k.box_xyxy) < ca)]
        kept.append(c)
    return kept


def dedup(cands: List, use_containment: bool) -> List:
    """Final clean-up so what gets SAVED equals what the UI showed.

    1. drop near-duplicates (IoU > DEDUP_IOU), keeping the earlier box;
    2. when containment removal is on, drop a smaller box largely contained in a strictly
       larger kept box.

    Step 2 follows the SAME decision as ``nms_candidates`` on purpose. On a dataset whose seed
    shows genuine nesting the caller turns containment off, and the saved labels have to honour
    that: filtering unconditionally here would let inner objects survive NMS only to be deleted
    on the way to disk, silently cancelling the adaptive decision.
    """
    kept: List = []
    for c in cands:
        if not any(iou_xyxy(c.box_xyxy, k.box_xyxy) > DEDUP_IOU for k in kept):
            kept.append(c)
    if not use_containment:                      # nested objects are real on this dataset
        return kept
    out: List = []
    for c in kept:
        ca = area_xyxy(c.box_xyxy)
        if any(area_xyxy(o.box_xyxy) > ca
               and containment(c.box_xyxy, o.box_xyxy) >= CONTAINMENT_THRESHOLD
               for o in kept if o is not c):
            continue                             # smaller box mostly inside a bigger one
        out.append(c)
    return out
