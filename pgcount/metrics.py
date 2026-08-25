"""Artifact-level metrics: count fidelity, defect-escape, gate operating curve.

These are the SE-headline outcomes (design §6.1, RQ3/RQ4). Implemented in pure
python so the scaffold runs standalone; for the final paper, swap the matcher for
``pseudoguard/utils/box_ops.py`` (compute_batch_pseudo_accuracy)
so human/detector/gate outputs sit on one identical scale (design §6.1 note).

Box format everywhere: (x1, y1, x2, y2) absolute pixels.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Mapping, Sequence


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def greedy_match(preds: Sequence[Sequence[float]], golds: Sequence[Sequence[float]],
                 iou_thr: float = 0.5) -> list[tuple[int, int, float]]:
    """Greedy IoU matching. Returns list of (pred_idx, gold_idx, iou)."""
    pairs = []
    for pi, p in enumerate(preds):
        for gi, g in enumerate(golds):
            v = iou(p, g)
            if v >= iou_thr:
                pairs.append((v, pi, gi))
    pairs.sort(reverse=True)
    used_p, used_g, matches = set(), set(), []
    for v, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi); used_g.add(gi)
        matches.append((pi, gi, v))
    return matches


# --------------------------------------------------------------------- RQ4
@dataclass
class CountFidelity:
    mean_signed_dev: float     # E[accepted_count - gold_count] per image (bias)
    mean_abs_dev: float        # E[|accepted - gold|]
    over_rate: float           # fraction of images with accepted > gold
    under_rate: float


def count_fidelity(accepted_per_image: Mapping[str, int],
                   gold_per_image: Mapping[str, int]) -> CountFidelity:
    """Does the count-guided output match the true per-image object count? (RQ4)

    ⚠ CAUTION (review, fatal finding): count-fidelity is IDENTITY-BLIND — it only
    compares COUNTS. Perfect count-fidelity (abs_dev=0) can coexist with keeping the
    WRONG boxes. It is a density-matching/calibration metric, NOT a quality metric.
    ALWAYS report it paired with ``correct_count_fidelity`` and ``true_object_suppression``.
    """
    imgs = set(accepted_per_image) | set(gold_per_image)
    devs = [accepted_per_image.get(i, 0) - gold_per_image.get(i, 0) for i in imgs]
    return CountFidelity(
        mean_signed_dev=statistics.fmean(devs) if devs else 0.0,
        mean_abs_dev=statistics.fmean([abs(d) for d in devs]) if devs else 0.0,
        over_rate=sum(d > 0 for d in devs) / len(devs) if devs else 0.0,
        under_rate=sum(d < 0 for d in devs) / len(devs) if devs else 0.0,
    )


@dataclass
class CorrectCountFidelity:
    """Count of accepted-AND-CORRECT boxes vs gold — the identity-aware companion (RQ4)."""
    mean_correct_dev: float     # E[correct_accepted - gold_count] (≤0; how far below true)
    mean_recall: float          # E[correct_accepted / gold_count] — true-object coverage
    mean_precision: float       # E[correct_accepted / accepted]


def correct_count_fidelity(accepted_by_image: Mapping[str, Sequence[dict]],
                           golds_by_image: Mapping[str, Sequence[dict]],
                           iou_thr: float = 0.5) -> CorrectCountFidelity:
    """Identity-aware count fidelity (design v4). ``accepted``/``golds`` items are
    dicts with ``box`` (xyxy) and ``label``; a match must have IoU≥thr AND same class.
    Answers the fatal critique that count_fidelity can be perfect while boxes are wrong."""
    imgs = set(accepted_by_image) | set(golds_by_image)
    devs, recs, precs = [], [], []
    for im in imgs:
        acc = list(accepted_by_image.get(im, []))
        gold = list(golds_by_image.get(im, []))
        m = greedy_match([a["box"] for a in acc], [g["box"] for g in gold], iou_thr)
        correct = sum(1 for pi, gi, _ in m if acc[pi].get("label") == gold[gi].get("label"))
        devs.append(correct - len(gold))
        recs.append(correct / len(gold) if gold else 1.0)
        precs.append(correct / len(acc) if acc else 1.0)
    n = len(imgs) or 1
    return CorrectCountFidelity(sum(devs) / n, sum(recs) / n, sum(precs) / n)


def true_object_suppression(accepted_by_image: Mapping[str, Sequence[dict]],
                            golds_by_image: Mapping[str, Sequence[dict]],
                            seed_median: float, iou_thr: float = 0.5) -> dict:
    """Does count-guided DISCARD true objects on dense images? (design v4 primary harm).

    Stratifies recall by whether an image's gold count exceeds the seed median (the
    regime where top-K = round(seed_median) provably caps below the true count).
    Returns recall on 'dense' (gold>seed_median) vs 'sparse' images — the harm must
    show up precisely where count-guiding is supposed to help.
    """
    dense_rec, sparse_rec = [], []
    for im, gold in golds_by_image.items():
        acc = list(accepted_by_image.get(im, []))
        m = greedy_match([a["box"] for a in acc], [g["box"] for g in gold], iou_thr)
        correct = sum(1 for pi, gi, _ in m if acc[pi].get("label") == gold[gi].get("label"))
        recall = correct / len(gold) if gold else 1.0
        (dense_rec if len(gold) > seed_median else sparse_rec).append(recall)
    f = lambda xs: (sum(xs) / len(xs)) if xs else None
    return {"recall_dense_images": f(dense_rec), "n_dense": len(dense_rec),
            "recall_sparse_images": f(sparse_rec), "n_sparse": len(sparse_rec),
            "seed_median": seed_median}


# --------------------------------------------------------------------- RQ3
@dataclass
class DefectEscape:
    n_admitted: int            # boxes that entered the dataset (accepted by gate/human)
    n_defect: int              # admitted boxes that are wrong vs gold
    escape_rate: float         # n_defect / n_admitted
    taxonomy: dict[str, int]   # {spurious, mislocalized, misclassified, missed}


def defect_escape(admitted: Sequence[dict], golds: Sequence[dict],
                  iou_thr: float = 0.5, loc_lo: float = 0.1) -> DefectEscape:
    """Classify defects that escaped into the dataset (design §4.3, RQ3).

    ``admitted``/``golds`` items are dicts with keys ``box`` (xyxy) and ``label``.
    Taxonomy (TIDE-adapted):
        spurious      : admitted box with IoU < loc_lo to any gold  (false object)
        mislocalized  : loc_lo ≤ IoU < iou_thr                       (poor box)
        misclassified : IoU ≥ iou_thr but wrong class
        missed        : gold object with no admitted match (gate/human FN)
    """
    pboxes = [a["box"] for a in admitted]
    gboxes = [g["box"] for g in golds]
    matches = greedy_match(pboxes, gboxes, iou_thr=loc_lo)  # match loosely, then grade
    matched_p = {pi: (gi, v) for pi, gi, v in matches}
    matched_g = {gi for _pi, gi, _v in matches}

    tax = {"spurious": 0, "mislocalized": 0, "misclassified": 0, "missed": 0}
    for pi, a in enumerate(admitted):
        if pi not in matched_p:
            tax["spurious"] += 1
            continue
        gi, v = matched_p[pi]
        if v < iou_thr:
            tax["mislocalized"] += 1
        elif a.get("label") != golds[gi].get("label"):
            tax["misclassified"] += 1
        # else: correct, not a defect
    tax["missed"] = sum(1 for gi in range(len(golds)) if gi not in matched_g)

    n_admitted = len(admitted)
    n_defect = tax["spurious"] + tax["mislocalized"] + tax["misclassified"]
    return DefectEscape(
        n_admitted=n_admitted,
        n_defect=n_defect,
        escape_rate=(n_defect / n_admitted) if n_admitted else 0.0,
        taxonomy=tax,
    )


def operating_curve(scored_candidates: Sequence[dict], golds_by_image: Mapping[str, Sequence[dict]],
                    thresholds: Sequence[float], iou_thr: float = 0.5) -> list[dict]:
    """Gate precision/recall as a function of operating point τ (design §4.3.3, RQ3).

    ``scored_candidates`` items: {image_id, box, label, score}. Returns one row per τ:
    {tau, precision, recall, n_accepted} — the curve used to show a count-guided /
    validator operating point dominates a fixed / confidence one.

    TODO(paper): compute against the reused box_ops matcher and add mIoU per τ.
    """
    by_img: dict[str, list[dict]] = {}
    for c in scored_candidates:
        by_img.setdefault(c["image_id"], []).append(c)

    total_gold = sum(len(g) for g in golds_by_image.values())
    curve = []
    for tau in thresholds:
        tp = fp = 0
        for img, cands in by_img.items():
            acc = [c for c in cands if c["score"] >= tau]
            golds = golds_by_image.get(img, [])
            m = greedy_match([a["box"] for a in acc], [g["box"] for g in golds], iou_thr)
            correct = sum(1 for pi, gi, _ in m if acc[pi].get("label") == golds[gi].get("label"))
            tp += correct
            fp += len(acc) - correct
        n_acc = tp + fp
        curve.append({
            "tau": tau,
            "precision": tp / n_acc if n_acc else 0.0,
            "recall": tp / total_gold if total_gold else 0.0,
            "n_accepted": n_acc,
        })
    return curve
