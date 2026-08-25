"""Orchestrator: seed → density → operating point → count-guided pre-labels.

Implements the "Automate Label" backend for each study condition (design §3.1):

    C1_MANUAL            : no pre-labels (human draws everything)
    C2_AUTO_FIXED        : accept candidates with score ≥ fixed threshold (naive)
    C3_AUTO_COUNT_GUIDED : accept via a seed-count-derived operating point (the app)

All conditions consume the SAME frozen candidate set; only the selection policy
differs → clean box-level identification of the count-guided effect (design §3.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence

from .backend import RawCandidate
from .config import Backend, CountGuidedConfig
from .operating_point import (
    AutoAdaptiveTau,
    OperatingPointResult,
    ThresholdSelect,
    make_operating_point,
)
from .seed_density import SeedDensityStats, estimate_seed_density


class Condition(str, Enum):
    C1_MANUAL = "C1_manual"
    C2_AUTO_FIXED = "C2_auto_fixed_conf"          # weak baseline (fixed global τ)
    C2B_AUTO_ADAPTIVE = "C2b_auto_adaptive"        # STRONG baseline: auto-τ from scores, NO human count
    C3_AUTO_COUNT_GUIDED = "C3_auto_count_guided"  # treatment: human seed-count sets the operating point


@dataclass
class Candidate:
    """A scored candidate exposed to selection (satisfies the ``Scored`` protocol)."""

    image_id: str
    box_id: str
    box_xyxy: tuple[float, float, float, float]
    label: int
    score: float          # the selection score (validator P(good) OR det conf, per backend)
    det_conf: float
    p_good: float

    @classmethod
    def from_raw(cls, raw: RawCandidate, backend: Backend) -> "Candidate":
        score = raw.p_good if backend is Backend.VALIDATOR else raw.det_conf
        return cls(
            image_id=raw.image_id,
            box_id=raw.box_id,
            box_xyxy=raw.box_xyxy,
            label=raw.label,
            score=score,
            det_conf=raw.det_conf,
            p_good=raw.p_good,
        )


@dataclass
class PreLabelResult:
    """Output of one Automate-Label run: what to pre-draw + provenance for telemetry."""

    condition: Condition
    accepted: list[Candidate] = field(default_factory=list)
    n_candidates: int = 0
    operating_point: OperatingPointResult | None = None
    seed_stats: SeedDensityStats | None = None


class CountGuidedLabeler:
    """Ties the frozen backend + count-guided operating point into pre-labels."""

    def __init__(self, backend_provider, cfg: CountGuidedConfig):
        # backend_provider: FrozenBackend or PrecomputedBackend (has .candidates)
        self.backend_provider = backend_provider
        self.cfg = cfg

    def _gather(self, image_ids: Sequence[str]) -> list[Candidate]:
        cands: list[Candidate] = []
        for img in image_ids:
            for raw in self.backend_provider.candidates(img):
                cands.append(Candidate.from_raw(raw, self.cfg.backend))
        return cands

    def automate_label(
        self,
        condition: Condition,
        seed_labels: Mapping[str, Sequence],
        unlabeled_image_ids: Sequence[str],
    ) -> PreLabelResult:
        """Produce pre-labels for ``unlabeled_image_ids`` under ``condition``.

        ``seed_labels`` is used ONLY by C3 (count-guided) to fit the operating
        point; C1/C2 ignore it. Fairness: the seed is hand-labeled in every arm.
        """
        if condition is Condition.C1_MANUAL:
            return PreLabelResult(condition=condition, accepted=[], n_candidates=0)

        candidates = self._gather(unlabeled_image_ids)

        if condition is Condition.C2_AUTO_FIXED:
            op = ThresholdSelect(self.cfg.fixed_filter_threshold)
            accepted = op.select(candidates)
            return PreLabelResult(
                condition=condition,
                accepted=accepted,
                n_candidates=len(candidates),
                operating_point=op.result,
            )

        if condition is Condition.C2B_AUTO_ADAPTIVE:
            # STRONG baseline (review kill-shot #3): auto-τ from the score distribution,
            # NO human count signal. C3-vs-C2B isolates the human count-prior's value.
            op = AutoAdaptiveTau(self.cfg).fit(None, candidates)
            accepted = op.select(candidates)
            return PreLabelResult(condition, accepted, len(candidates), op.result)

        # C3 — count-guided
        seed_stats = estimate_seed_density(
            seed_labels,
            min_images=self.cfg.seed.n_seed_images // 2 or 1,
            min_total_boxes=self.cfg.seed.min_seed_boxes,
        )
        if not seed_stats.reliable:
            # Fallback (design §3.6, §12.3): unreliable density ⇒ use fixed policy,
            # but flag it so the analysis can exclude / annotate these sessions.
            op = ThresholdSelect(self.cfg.fixed_filter_threshold)
            accepted = op.select(candidates)
            res = op.result
            res.reliable = False
            return PreLabelResult(condition, accepted, len(candidates), res, seed_stats)

        op = make_operating_point(self.cfg).fit(seed_stats, candidates)
        accepted = op.select(candidates)
        return PreLabelResult(
            condition=condition,
            accepted=accepted,
            n_candidates=len(candidates),
            operating_point=op.result,
            seed_stats=seed_stats,
        )
