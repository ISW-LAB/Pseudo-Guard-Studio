"""Estimate the seed object-density statistics ``D_seed``.

This is step (1) of the count-guided mechanism (design §4.3): from the handful
of images the human labels first, compute the per-image box-count distribution.
That distribution is what drives the auto-labeler's acceptance operating point.

Pure-python (stdlib only) so it is unit-testable without torch/numpy.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class SeedDensityStats:
    """Per-image box-count statistics over the seed set.

    Attributes
    ----------
    n_images        : number of seed images
    total_boxes     : total boxes across seed images
    mean_count      : mean boxes per image  (drives GLOBAL_TAU total target)
    median_count    : median boxes per image (drives PER_IMAGE_TOPK's K*)
    p90_count       : 90th-percentile boxes per image (headroom for crowded images)
    per_image_counts: raw counts, one per seed image
    reliable        : False if too few seed images / boxes to trust the estimate
    """

    n_images: int
    total_boxes: int
    mean_count: float
    median_count: float
    p90_count: float
    per_image_counts: tuple[int, ...]
    reliable: bool

    def k_star(self, lo: int = 1, hi: int = 100) -> int:
        """Per-image top-K target derived from the seed median, clamped to [lo, hi]."""
        return max(lo, min(hi, round(self.median_count)))

    def target_total(self, n_unlabeled_images: int) -> int:
        """Expected total accepted boxes if unlabeled set matches seed density."""
        return max(0, round(self.mean_count * n_unlabeled_images))


def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    """Nearest-rank percentile on an already-sorted ascending sequence (q in [0,1])."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    idx = min(len(sorted_vals) - 1, max(0, round(q * (len(sorted_vals) - 1))))
    return float(sorted_vals[idx])


def estimate_seed_density(
    seed_labels: Mapping[str, Sequence],
    min_images: int = 3,
    min_total_boxes: int = 3,
) -> SeedDensityStats:
    """Compute ``D_seed`` from hand-labeled seed images.

    Parameters
    ----------
    seed_labels : mapping ``image_id -> list_of_boxes``. Only ``len(boxes)`` is
                  used, so the boxes may be any object (dicts, tuples, arrays).
    min_images, min_total_boxes : reliability floor (design §12.3, "density representativeness").

    Returns
    -------
    SeedDensityStats. ``reliable=False`` means the caller should fall back to the
    fixed-confidence policy rather than count-guiding on a noisy estimate.
    """
    counts = [len(boxes) for boxes in seed_labels.values()]
    n_images = len(counts)
    total_boxes = sum(counts)

    if n_images == 0:
        return SeedDensityStats(0, 0, 0.0, 0.0, 0.0, tuple(), reliable=False)

    counts_sorted = sorted(counts)
    mean_count = total_boxes / n_images
    median_count = float(statistics.median(counts))
    p90_count = _percentile(counts_sorted, 0.90)
    reliable = (n_images >= min_images) and (total_boxes >= min_total_boxes)

    return SeedDensityStats(
        n_images=n_images,
        total_boxes=total_boxes,
        mean_count=mean_count,
        median_count=median_count,
        p90_count=p90_count,
        per_image_counts=tuple(counts),
        reliable=reliable,
    )
