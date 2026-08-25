"""Configuration for count-guided auto-labeling.

Mirrors the relevant knobs of the algorithm library (``pseudoguard/config.py``)
so the scaffold stays consistent with the frozen backend:

    pseudo_conf_threshold = 0.05   # detector collects candidates broadly (high recall)
    filter_threshold      = 0.5    # validator P(good) accept cutoff (DEFAULT / fixed-conf arm)
    max_pseudo_labels_per_image = 100  # per-image cap → we repurpose it as the top-K knob

Design doc: §3.1 (conditions), §3.6 (operating point), §4.3 (mechanism).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class OperatingPointStrategy(str, Enum):
    """How the seed box-count maps to an acceptance operating point (design §4.3.2)."""

    #: Per-image top-K: keep the K* highest-scoring candidates per image, where
    #: K* is derived from the seed's median box-count. Robust on crowded images.
    PER_IMAGE_TOPK = "per_image_topk"

    #: Global-τ: pick a single score threshold τ so the *total* accepted count
    #: ≈ (#unlabeled images) × (mean seed box-count). Matches dataset density.
    GLOBAL_TAU = "global_tau"


class Backend(str, Enum):
    """Which per-candidate score drives selection (design §3.1, Tier-1 fixes to validator)."""

    #: Pseudo-Guard detector-decoupled crop validator, P(good) = probs[:, 1]. (Tier-1 default)
    VALIDATOR = "validator"
    #: Raw detector confidence. (Tier-2 comparison — isolates the value of decoupling.)
    CONFIDENCE = "confidence"


@dataclass
class SeedProtocol:
    """How the human seed is collected (design §3.5, §12.3)."""

    n_seed_images: int = 5           # few-label seed size (open Q §12.3: 3–5)
    seed_manual: bool = True         # seed is always hand-labeled (fairness across conditions)
    min_seed_boxes: int = 3          # refuse to auto-configure below this (density unreliable)


@dataclass
class CountGuidedConfig:
    """Top-level config for a count-guided auto-labeling session."""

    strategy: OperatingPointStrategy = OperatingPointStrategy.PER_IMAGE_TOPK
    backend: Backend = Backend.VALIDATOR

    # --- mirrors of pseudoguard/config.py (keep the two in sync) ---
    pseudo_conf_threshold: float = 0.05   # detector candidate-collection threshold
    fixed_filter_threshold: float = 0.5   # the C2 "auto + fixed-conf" arm operating point
    crop_img_size: int = 256              # validator crop size (classification.img_size)

    # --- count-guided bounds (repurpose max_pseudo_labels_per_image = 100) ---
    min_topk_per_image: int = 1
    max_topk_per_image: int = 100
    # A gentle score floor so global-τ never dips into pure noise even on sparse sets.
    score_floor: float = 0.05

    seed: SeedProtocol = field(default_factory=SeedProtocol)

    def describe(self) -> str:
        return (
            f"CountGuidedConfig(strategy={self.strategy.value}, backend={self.backend.value}, "
            f"seed_n={self.seed.n_seed_images}, fixed_thr={self.fixed_filter_threshold})"
        )
