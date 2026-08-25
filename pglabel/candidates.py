#!/usr/bin/env python3
"""The candidate cache, and the two Automate-Label paths built on it.

One rule governs this module: the operating point is fit on the FULL candidate pool, so that
"Automate this image" and "Auto-label ALL" always agree. Both apply ONE global operating point
to the same predictions; fitting per image would make the single-image button disagree with the
batch run on the same data, which is exactly the kind of inconsistency a human notices and
cannot explain.

The cache is split by COST, because its two layers go stale for different reasons:

    _RAW_CACHE   the model's own output per image. Expensive — a detector plus validator
                 forward pass per image on a live backend — so it is rebuilt only when the
                 model, the dataset, or the set of image files actually changes.
    _VIEW_CACHE  _RAW_CACHE after source-level NMS. Cheap post-processing, but it depends on
                 the containment decision, which is derived from the human seed and therefore
                 FLIPS while the human labels. Keeping it separate means a label edit re-runs
                 NMS, never the model.
"""

from __future__ import annotations

import threading
from pathlib import Path

from . import state
from .geometry import dedup, nms_candidates
from .labelio import image_size, list_images, load_yolo, save_yolo
from .methods import compute_adaptive, containment_on, make_operating_point, seed_stats

# Serialises rebuilds: the HTTP server answers status polls concurrently with an in-flight
# Automate, and an unlocked rebuild had every one of those requests re-run the whole model.
_CAND_LOCK = threading.RLock()
_CAND_KEY = None        # (backend, images dir, image-set signature) behind _RAW_CACHE
_CAND_CONT = None       # containment decision baked into _VIEW_CACHE
_RAW_CACHE: dict = {}
_VIEW_CACHE: dict = {}
_WARMING = threading.Event()


def _cache_key():
    """Identity of the predictions currently cached: which backend, which folder, which files.

    The image-set signature is what makes a file dropped into the images folder actually get
    predicted — keying on the folder path alone served it an empty candidate list forever.
    """
    imgs = state.CFG["images"]
    if imgs is None:
        return None
    # The backend object itself (identity comparison; keeping it in the tuple also keeps it
    # alive, so a new backend can never reuse a freed object's id).
    return (state.CFG["ai"], str(imgs), tuple(list_images(imgs)))


def invalidate() -> None:
    """Drop every cached prediction.

    Called when the model is hot-swapped or the dataset changes. The key check would catch both,
    but an explicit drop frees the memory immediately instead of at the next rebuild.
    """
    global _CAND_KEY, _CAND_CONT
    with _CAND_LOCK:
        _CAND_KEY = _CAND_CONT = None
        _RAW_CACHE.clear()
        _VIEW_CACHE.clear()


def all_candidates() -> dict:
    """Post-NMS candidates for every image, cached.

    May run the model on a cold cache. Callers that must not block on inference use
    ``candidates_if_warm()`` instead.
    """
    global _CAND_KEY, _CAND_CONT, _RAW_CACHE, _VIEW_CACHE
    with _CAND_LOCK:                     # one rebuild total, not one per concurrent request
        compute_adaptive()               # refresh the adaptive decision from the current seed
        cont_now = containment_on()
        key = _cache_key()
        if _CAND_KEY != key:             # model / dataset / image set changed → re-run the model
            ai, imgs = state.CFG["ai"], state.CFG["images"]
            _RAW_CACHE = {img: ai.candidates(img, imgs / img) for img in list_images(imgs)}
            _CAND_KEY, _CAND_CONT = key, None        # raw output changed → the NMS view is stale
        if _CAND_CONT != cont_now:       # containment flipped → redo the CHEAP pass only
            _VIEW_CACHE = {img: nms_candidates(cs, cont_now) for img, cs in _RAW_CACHE.items()}
            _CAND_CONT = cont_now
        return _VIEW_CACHE


def candidates_if_warm():
    """The cached candidates, but only when already up to date — never triggers inference.

    Status polling uses this. /api/status is refetched after every save and every image change,
    and letting a routine UI refresh kick off a full-dataset forward pass froze the tool (and
    inflated the per-image timings a study measures) at unpredictable moments. The lock is taken
    NON-blocking for the same reason: while another request is mid-rebuild, the poll reports no
    AI classes rather than queueing behind minutes of inference.
    """
    if not _CAND_LOCK.acquire(blocking=False):
        return None
    try:
        compute_adaptive()
        if _CAND_KEY is not None and _CAND_KEY == _cache_key() and _CAND_CONT == containment_on():
            return _VIEW_CACHE
    finally:
        _CAND_LOCK.release()
    return None


def warm_async() -> None:
    """Build the cache in the background (at most one worker at a time).

    A status poll must not block on inference, but it should not leave the AI class counts empty
    forever either: a cold poll kicks the build off and a later poll picks up the result.
    """
    if _WARMING.is_set():
        return
    _WARMING.set()

    def work():
        try:
            all_candidates()
        except Exception:
            pass
        finally:
            _WARMING.clear()

    threading.Thread(target=work, daemon=True).start()


# ------------------------------------------------------------------ read-only views
def image_candidates(name: str) -> dict:
    """Every raw candidate for one image (normalised box + both scores).

    Both scores are sent so the UI can move the confidence slider without a round trip per drag.
    """
    W, H = image_size(state.CFG["images"] / name)
    out = []
    for c in all_candidates().get(name, []):
        x1, y1, x2, y2 = c.box_xyxy
        out.append({"cls": int(c.label), "cx": (x1 + x2) / 2 / W, "cy": (y1 + y2) / 2 / H,
                    "w": (x2 - x1) / W, "h": (y2 - y1) / H,
                    "p_good": round(float(c.p_good), 4),
                    "det_conf": round(float(c.det_conf), 4)})
    return {"candidates": out, "width": W, "height": H}


def score_summary(bins: int = 100) -> dict:
    """Histograms of candidate scores over the auto-target images.

    The UI draws the score distribution from this and computes "objects accepted at threshold X"
    for the whole dataset live, without asking the server per slider position.
    """
    cache = all_candidates()
    targets = [img for img in list_images(state.CFG["images"]) if img not in state.HUMAN_SET]
    hist_p, hist_d, n = [0] * bins, [0] * bins, 0
    for img in targets:
        for c in cache.get(img, []):
            hist_p[min(bins - 1, max(0, int(float(c.p_good) * bins)))] += 1
            hist_d[min(bins - 1, max(0, int(float(c.det_conf) * bins)))] += 1
            n += 1
    return {"bins": bins, "p_good_hist": hist_p, "det_conf_hist": hist_d,
            "n_candidates": n, "n_images": len(targets)}


def label_status() -> dict:
    """Per-image progress for the UI.

    ``classes``    class ids present in each image's SAVED labels.
    ``ai_classes`` class ids the current model INFERRED per image, so a class-chip filter can
                   surface images the AI predicts a class in even before Auto-label ALL saves
                   them. Read from the cache ONLY when it is already warm: this endpoint is
                   polled after every save, so it must never be what triggers inference.
    ``adaptive``   the live per-dataset decision, re-read on every poll because it moves as the
                   human labels rather than being fixed at startup.
    """
    images_dir, labels_dir = state.CFG["images"], state.CFG["labels"]
    counts, classes = {}, {}
    for img in list_images(images_dir):
        boxes = load_yolo(labels_dir, Path(img).stem)
        counts[img] = len(boxes)
        classes[img] = sorted({int(b.get("cls", 0)) for b in boxes})
    # Recomputed here, not only on the AI path: the UI shows this decision in manual mode too,
    # and it moves as the human labels. Cheap — it reads the seed's label files, nothing else.
    compute_adaptive()
    ai_classes = {}
    if state.ai_available():
        try:
            cache = candidates_if_warm()             # warm-only → never blocks the poll
            if cache is None:
                warm_async()                         # …but do start building it off-thread
            for img in (counts if cache else {}):
                cs = sorted({int(c.label) for c in cache.get(img, [])})
                if cs:
                    ai_classes[img] = cs
        except Exception:
            ai_classes = {}
    owner = {img: ("human" if img in state.HUMAN_SET else ("auto" if counts[img] > 0 else "none"))
             for img in counts}
    return {"counts": counts, "classes": classes, "ai_classes": ai_classes, "owner": owner,
            "labeled": sum(1 for n in counts.values() if n > 0),
            "human": sum(1 for img in counts if img in state.HUMAN_SET),
            "total": len(counts), "adaptive": dict(state.ADAPT)}


# --------------------------------------------------------------------- automate
def _band(score: float) -> str:
    """Traffic-light band the UI colours a proposed box by."""
    return "green" if score >= 0.65 else ("amber" if score >= 0.35 else "red")


def automate_image(name: str, method: str = "pseudoguard") -> dict:
    """Pre-labels for ONE image, without saving.

    The operating point is fit on the SAME full target pool that Auto-label ALL uses, then only
    this image's selection is kept — which is what guarantees the two buttons agree.
    """
    images_dir = state.CFG["images"]
    W, H = image_size(images_dir / name)
    seed, stats = seed_stats()
    cache = all_candidates()
    targets = [img for img in list_images(images_dir) if img not in state.HUMAN_SET]
    pool = targets if name in targets else targets + [name]   # this image must be in the fit pool
    raw_all = [c for img in pool for c in cache.get(img, [])]
    if not raw_all:
        return {"boxes": [], "op": "no-candidates", "method": method, "fell_back": False,
                "seed_images": len(seed), "seed_median": stats.median_count}
    op, cands, label, fell_back = make_operating_point(method, raw_all, stats)
    selected = [c for c in op.select(cands) if c.image_id == name]
    boxes = []
    for c in dedup(selected, containment_on()):
        x1, y1, x2, y2 = c.box_xyxy
        boxes.append({"cls": int(c.label), "cx": (x1 + x2) / 2 / W, "cy": (y1 + y2) / 2 / H,
                      "w": (x2 - x1) / W, "h": (y2 - y1) / H,
                      "score": round(c.score, 3), "band": _band(c.score), "ai": True})
    return {"boxes": boxes, "op": label, "method": method, "fell_back": fell_back,
            "seed_images": len(seed), "seed_median": stats.median_count}


def automate_all(method: str = "pseudoguard", thr=None, score: str = "p_good",
                 fold_gate=None) -> dict:
    """Auto-label every non-seed image with the chosen methodology and SAVE the result.

    Targets are every image not in ``HUMAN_SET``. Human-labeled images are the seed: they set
    the count prior and are never overwritten, so pressing this again after switching
    methodology re-labels the whole auto set while the human's own labels stay put.

    ``method="manual"`` with ``thr`` accepts every candidate whose chosen ``score``
    ("p_good" or "det_conf") reaches the threshold — that is the confidence slider. Otherwise
    the operating point is fit ONCE on the full target pool, so per-image adaptive K can see
    cross-image density.

    ``fold_gate`` is used by the cycle only: it keeps the accepted density but drops the lowest
    ``fold_gate`` fraction by p_good from what gets re-used as TRAINING data next round. A
    cleaner training set is what stops sparse-data self-training from drifting.
    """
    images_dir, labels_dir = state.CFG["images"], state.CFG["labels"]
    seed, stats = seed_stats()                                  # human seed only
    cache = all_candidates()                                    # same cache as automate_image
    all_imgs = list_images(images_dir)
    targets = [img for img in all_imgs if img not in state.HUMAN_SET]
    if not targets:
        return {"detail": "every image is human-labeled — nothing to auto-label", "auto_labeled": 0}

    by_img, op_label, fell_back = {}, "no-candidates", False
    if method == "manual" and thr is not None:
        t = float(thr)
        for img in targets:
            for c in cache.get(img, []):
                s = float(c.p_good) if score == "p_good" else float(c.det_conf)
                if s >= t:
                    by_img.setdefault(img, []).append(c)
        op_label = f"Manual {('P(good)' if score == 'p_good' else 'conf')} ≥ {t:.2f}"
    else:
        raw_all = [c for img in targets for c in cache.get(img, [])]
        if raw_all:
            op, cands, op_label, fell_back = make_operating_point(method, raw_all, stats)
            for c in op.select(cands):
                by_img.setdefault(c.image_id, []).append(c)

    if fold_gate is not None and 0.0 < float(fold_gate) < 1.0:
        scores = sorted(float(c.p_good) for cs in by_img.values() for c in cs)
        if scores:
            cut = scores[min(int(float(fold_gate) * len(scores)), len(scores) - 1)]
            by_img = {img: [c for c in cs if float(c.p_good) >= cut] for img, cs in by_img.items()}
            op_label += f" · fold-gate(top {int((1 - float(fold_gate)) * 100)}% p_good)"

    use_cont = containment_on()
    per_image = {}
    for img in targets:
        W, H = image_size(images_dir / img)
        boxes = [{"cls": int(c.label),
                  "cx": (c.box_xyxy[0] + c.box_xyxy[2]) / 2 / W,
                  "cy": (c.box_xyxy[1] + c.box_xyxy[3]) / 2 / H,
                  "w": (c.box_xyxy[2] - c.box_xyxy[0]) / W,
                  "h": (c.box_xyxy[3] - c.box_xyxy[1]) / H}
                 for c in dedup(by_img.get(img, []), use_cont)]
        save_yolo(labels_dir, Path(img).stem, boxes)            # overwrite (re-apply)
        per_image[img] = len(boxes)
    return {"op": op_label, "method": method, "seed_images": len(seed), "fell_back": fell_back,
            "seed_median": stats.median_count, "auto_labeled": len(targets),
            "total_boxes": sum(per_image.values()), "per_image": per_image}
