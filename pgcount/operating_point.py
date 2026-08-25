"""The novel core: map seed density → acceptance operating point, then select.

Design doc §4.3.2. Two interchangeable, pre-registerable strategies (§12.1):

  PerImageTopK : for each image keep the K* highest-scoring candidates, where
                 K* = round(seed median box-count). Uses the (currently static,
                 unused) ``max_pseudo_labels_per_image`` knob as the driver.

  GlobalTau    : pick ONE score threshold τ so the total accepted count ≈
                 (#unlabeled images) × (mean seed box-count). Implemented by
                 taking the (target_total)-th largest pooled score as τ — exact,
                 O(n log n), reproducible (the "score p90 probe" idea in
                 iterative_trainer.py:1236 generalized to an arbitrary quantile).

Both operate on *any* candidate object exposing ``.score`` (float) and
``.image_id`` (str) — duck-typed via the ``Scored`` Protocol — so this module
is testable with trivial stand-ins and never imports the heavy backend.

Fairness note (design §3.1): the SAME frozen candidate set feeds every arm;
only the *selection policy* here differs. The fixed-confidence arm (C2) is just
``ThresholdSelect(fixed_filter_threshold)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol, Sequence, runtime_checkable

from .config import CountGuidedConfig, OperatingPointStrategy
from .seed_density import SeedDensityStats


@runtime_checkable
class Scored(Protocol):
    score: float
    image_id: str


def _group_by_image(cands: Iterable[Scored]) -> dict[str, list[Scored]]:
    groups: dict[str, list[Scored]] = {}
    for c in cands:
        groups.setdefault(c.image_id, []).append(c)
    return groups


@dataclass
class OperatingPointResult:
    """What was decided, for telemetry / the count-fidelity analysis (RQ4)."""

    strategy: str
    tau: float | None = None          # global threshold, if GLOBAL_TAU
    k_star: int | None = None         # per-image top-K, if PER_IMAGE_TOPK
    target_total: int | None = None   # intended total accepted (GLOBAL_TAU)
    seed_median: float | None = None
    seed_mean: float | None = None
    reliable: bool = True             # False ⇒ we fell back to fixed policy


class OperatingPoint:
    """Base class: fit an operating point from D_seed, then select candidates."""

    def fit(self, seed: SeedDensityStats, all_candidates: Sequence[Scored]) -> "OperatingPoint":
        raise NotImplementedError

    def select(self, candidates: Sequence[Scored]) -> list[Scored]:
        raise NotImplementedError

    @property
    def result(self) -> OperatingPointResult:
        raise NotImplementedError


class ThresholdSelect(OperatingPoint):
    """Fixed global confidence threshold — the C2 (auto + fixed-conf) baseline arm."""

    def __init__(self, threshold: float):
        self.threshold = float(threshold)
        self._res = OperatingPointResult(strategy="fixed_threshold", tau=self.threshold)

    def fit(self, seed=None, all_candidates=None) -> "ThresholdSelect":
        return self

    def select(self, candidates: Sequence[Scored]) -> list[Scored]:
        return [c for c in candidates if c.score >= self.threshold]

    @property
    def result(self) -> OperatingPointResult:
        return self._res


class PerImageTopK(OperatingPoint):
    """Count-guided: keep top-K* per image (K* from seed median)."""

    def __init__(self, cfg: CountGuidedConfig):
        self.cfg = cfg
        self.k_star: int = cfg.max_topk_per_image
        self._res = OperatingPointResult(strategy=OperatingPointStrategy.PER_IMAGE_TOPK.value)

    def fit(self, seed: SeedDensityStats, all_candidates=None) -> "PerImageTopK":
        self.k_star = seed.k_star(self.cfg.min_topk_per_image, self.cfg.max_topk_per_image)
        self._res = OperatingPointResult(
            strategy=self._res.strategy,
            k_star=self.k_star,
            seed_median=seed.median_count,
            seed_mean=seed.mean_count,
            reliable=seed.reliable,
        )
        return self

    def select(self, candidates: Sequence[Scored]) -> list[Scored]:
        kept: list[Scored] = []
        for _img, group in _group_by_image(candidates).items():
            group_sorted = sorted(group, key=lambda c: c.score, reverse=True)
            # respect a gentle score floor so top-K never admits pure noise
            group_sorted = [c for c in group_sorted if c.score >= self.cfg.score_floor]
            kept.extend(group_sorted[: self.k_star])
        return kept

    @property
    def result(self) -> OperatingPointResult:
        return self._res


class GlobalTau(OperatingPoint):
    """Count-guided: single τ so total accepted ≈ n_unlabeled × mean seed count."""

    def __init__(self, cfg: CountGuidedConfig):
        self.cfg = cfg
        self.tau: float = cfg.fixed_filter_threshold
        self._res = OperatingPointResult(strategy=OperatingPointStrategy.GLOBAL_TAU.value)

    def fit(self, seed: SeedDensityStats, all_candidates: Sequence[Scored]) -> "GlobalTau":
        n_images = len({c.image_id for c in all_candidates})
        target_total = seed.target_total(n_images)
        scores_desc = sorted((c.score for c in all_candidates), reverse=True)

        if target_total <= 0 or not scores_desc:
            tau = 1.0  # accept nothing
        elif target_total >= len(scores_desc):
            tau = self.cfg.score_floor  # accept (almost) everything above floor
        else:
            # the target_total-th largest score is exactly the threshold that
            # admits ~target_total candidates. Deterministic given the frozen scores.
            tau = float(scores_desc[target_total - 1])

        self.tau = max(tau, self.cfg.score_floor)
        self._res = OperatingPointResult(
            strategy=self._res.strategy,
            tau=self.tau,
            target_total=target_total,
            seed_median=seed.median_count,
            seed_mean=seed.mean_count,
            reliable=seed.reliable,
        )
        return self

    def select(self, candidates: Sequence[Scored]) -> list[Scored]:
        return [c for c in candidates if c.score >= self.tau]

    @property
    def result(self) -> OperatingPointResult:
        return self._res


def _otsu_threshold(scores: Sequence[float], bins: int = 64) -> float:
    """Otsu threshold on a [0,1] score list — the split maximizing between-class
    variance. Fully automatic (NO human count signal). Pure python."""
    xs = [s for s in scores if s == s]  # drop NaN
    if len(xs) < 2:
        return 0.5
    hist = [0] * bins
    for s in xs:
        b = min(bins - 1, max(0, int(s * bins)))
        hist[b] += 1
    total = len(xs)
    sum_all = sum((i + 0.5) / bins * h for i, h in enumerate(hist))
    w_b = 0.0
    sum_b = 0.0
    best_var, best_t = -1.0, 0.5
    for i in range(bins):
        w_b += hist[i]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += (i + 0.5) / bins * hist[i]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > best_var:
            best_var, best_t = var_between, (i + 1) / bins
    return best_t


class AutoAdaptiveTau(OperatingPoint):
    """STRONG non-human baseline (review kill-shot #3): auto-adaptive threshold from
    the candidate SCORE distribution itself (Otsu), using NO seed count / human signal.

    This is the honest comparator for C3: C3-vs-AutoAdaptive isolates the *marginal
    value of the human count-prior* specifically, not merely 'adaptive beats a bad
    fixed constant'. If count-guiding cannot beat this, the human signal adds nothing.
    """

    def __init__(self, cfg: CountGuidedConfig):
        self.cfg = cfg
        self.tau = cfg.fixed_filter_threshold
        self._res = OperatingPointResult(strategy="auto_adaptive_otsu")

    def fit(self, seed=None, all_candidates: Sequence[Scored] = ()) -> "AutoAdaptiveTau":
        self.tau = max(self.cfg.score_floor, _otsu_threshold([c.score for c in all_candidates]))
        self._res = OperatingPointResult(strategy="auto_adaptive_otsu", tau=self.tau, reliable=True)
        return self

    def select(self, candidates: Sequence[Scored]) -> list[Scored]:
        return [c for c in candidates if c.score >= self.tau]

    @property
    def result(self) -> OperatingPointResult:
        return self._res


class AdaptivePerImageK(OperatingPoint):
    """IMPROVED count-guided (design v4, review depth-increment): per-image budget that
    combines the HUMAN seed prior (global scale) with a per-image density signal from the
    image's own candidate scores (heterogeneity) — fixing top-K's dense-image suppression.

    K_i = round( seed_mean_count × r_i / mean_j(r_j) ),  clamped to [min, max],
    where r_i = Σ scores on image i (the validator's *expected* object count on that image).
    → dense images (high Σscore) get proportionally more headroom; sparse images stay trimmed;
      the seed mean anchors the GLOBAL scale (the cheap human contribution). Then keep top-K_i.

    Rationale: the review's fatal harm was applying ONE global K to heterogeneous images.
    Making K per-image-adaptive *from the image's own content, scaled by the human prior*
    is the only version that can beat the strong auto baseline AND avoid suppression.
    """

    def __init__(self, cfg: CountGuidedConfig):
        self.cfg = cfg
        self._res = OperatingPointResult(strategy="adaptive_per_image_k")
        self._k_by_image: dict[str, int] = {}

    def fit(self, seed: SeedDensityStats, all_candidates: Sequence[Scored]) -> "AdaptivePerImageK":
        groups = _group_by_image(all_candidates)
        r = {img: sum(max(0.0, c.score) for c in cs) for img, cs in groups.items()}
        mean_r = (sum(r.values()) / len(r)) if r else 0.0
        scale = seed.mean_count
        lo, hi = self.cfg.min_topk_per_image, self.cfg.max_topk_per_image
        self._k_by_image = {
            img: max(lo, min(hi, round(scale * (r[img] / mean_r)))) if mean_r > 0 else lo
            for img in groups
        }
        self._res = OperatingPointResult(
            strategy="adaptive_per_image_k", seed_mean=seed.mean_count,
            seed_median=seed.median_count, reliable=seed.reliable,
        )
        return self

    def select(self, candidates: Sequence[Scored]) -> list[Scored]:
        kept: list[Scored] = []
        for img, group in _group_by_image(candidates).items():
            k = self._k_by_image.get(img, self.cfg.min_topk_per_image)
            g = sorted((c for c in group if c.score >= self.cfg.score_floor),
                       key=lambda c: c.score, reverse=True)
            kept.extend(g[:k])
        return kept

    @property
    def result(self) -> OperatingPointResult:
        return self._res


class QuantileRate(OperatingPoint):
    """COMPARISON baseline: rate / quantile thresholding (CBST / class-balanced
    self-training lineage). Accept the top ``rate`` FRACTION of all candidates by score
    — a fixed acceptance RATE rather than a fixed score threshold, and NO human count.
    """

    def __init__(self, cfg: CountGuidedConfig, rate: float = 0.5):
        self.cfg = cfg
        self.rate = min(1.0, max(0.0, float(rate)))
        self.tau = cfg.fixed_filter_threshold
        self._res = OperatingPointResult(strategy="quantile_rate")

    def fit(self, seed=None, all_candidates: Sequence[Scored] = ()) -> "QuantileRate":
        scores = sorted((c.score for c in all_candidates), reverse=True)
        if not scores or self.rate <= 0:
            self.tau = 1.0
        else:
            k = max(1, int(round(self.rate * len(scores))))
            self.tau = max(self.cfg.score_floor, float(scores[min(k, len(scores)) - 1]))
        self._res = OperatingPointResult(strategy="quantile_rate", tau=self.tau, reliable=True)
        return self

    def select(self, candidates: Sequence[Scored]) -> list[Scored]:
        return [c for c in candidates if c.score >= self.tau]

    @property
    def result(self) -> OperatingPointResult:
        return self._res


def make_operating_point(cfg: CountGuidedConfig) -> OperatingPoint:
    """Factory: pick the count-guided strategy per config (design §4.3.2)."""
    if cfg.strategy is OperatingPointStrategy.PER_IMAGE_TOPK:
        return PerImageTopK(cfg)
    if cfg.strategy is OperatingPointStrategy.GLOBAL_TAU:
        return GlobalTau(cfg)
    raise ValueError(f"unknown strategy: {cfg.strategy}")
