#!/usr/bin/env python3
"""Configuration dataclasses for the Pseudo-Guard algorithm library.

Three settings groups, one per stage of the pipeline:

    DetectionModelConfig      the high-recall proposal generator (YOLO)
    ClassificationModelConfig the crop-level proposal validator (DenseNet)
    NoiseGenerationConfig     the RULE that fabricates the validator's training crops

Only ``NoiseGenerationConfig`` is consumed inside this package (by
``pseudoguard.data.noise_generator``); the two model configs document the defaults the
training tools pass on the command line, so a reader has one place to look up "what does
the paper actually run with".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Half-width of the band a scalar centre-shift expands into (see NoiseGenerationConfig).
SHIFT_BAND = 0.12


@dataclass
class DetectionModelConfig:
    """Proposal generator. Trained from the human seed, then run at a permissive threshold."""

    model_type: str = "yolov8"      # yolov8 | yolov11 | yolo26 | rtdetr
    model_size: str = "n"           # n | s | m | l | x
    img_size: int = 640
    batch_size: int = 16
    epochs: int = 100
    patience: int = 20              # early stopping
    lr: float = 0.001
    weight_decay: float = 0.0005
    device: str = "auto"            # auto → CUDA when present, else CPU (pseudoguard.device)

    # Two different thresholds on purpose: proposals are collected broadly and filtered by the
    # validator, so the collection threshold is far lower than a detector's own operating point.
    conf_threshold: float = 0.5             # reporting / detector-only evaluation
    pseudo_conf_threshold: float = 0.05     # proposal collection (high recall)
    iou_threshold: float = 0.45             # NMS

    seed: int = 42
    pretrained: bool = True


@dataclass
class ClassificationModelConfig:
    """Proposal validator: a binary good/noise classifier over 256x256 object crops."""

    model_type: str = "densenet121"
    img_size: int = 256             # crop size
    batch_size: int = 64
    epochs: int = 6
    patience: int = 3               # early stopping
    lr: float = 1e-4
    weight_decay: float = 1e-5
    device: str = "auto"

    # The validator's job is to catch bad proposals, so noise errors cost more than good errors.
    use_weighted_loss: bool = True
    good_weight: float = 1.0
    noise_weight: float = 2.0

    filter_threshold: float = 0.5   # accept a proposal when P(good) >= this
    pretrained: bool = True


@dataclass
class NoiseGenerationConfig:
    """The rule that fabricates validator training crops from the labeled seed.

    Three crop types, mixed by the ratios below:

        positive  ground-truth boxes with a small jitter                 -> label 1 (good)
        empty     background boxes that contain no annotated object      -> label 0 (noise)
        deviated  ground-truth boxes displaced off their object          -> label 0 (noise)

    ``negative_rule`` selects how the two NEGATIVE types are drawn:

        "baseline"  the original rule - empty boxes are large random background boxes
                    (rejected on any IoU > 0.1), deviated boxes are shifted 0.5~2.0x the box
                    size (usually fully off the object) and rejected above ``deviation_max_iou``.
        "refined"   the published rule - empty box sizes are a probabilistic mix of
                    GT-similar and large, deviated boxes are displaced by ``deviation_shift``
                    of the box size so partial overlap is deliberately kept, and BOTH types are
                    accepted only when their IoU with EVERY ground-truth box stays below
                    ``negative_iou_reject``.

    Keeping both means the negative-construction rule can be varied as an experiment axis
    while "baseline" behaviour stays byte-identical to the original implementation.
    """

    # -- mixture ---------------------------------------------------------------------
    good_crop_ratio: float = 0.4
    empty_crop_ratio: float = 0.3
    deviation_ratio: float = 0.3

    # -- positive crops --------------------------------------------------------------
    good_crop_jitter: float = 0.05          # +/-5% jitter around the ground-truth box

    # -- empty (background) crops ----------------------------------------------------
    empty_crop_min_size: Tuple[int, int] = (50, 50)
    empty_crop_max_attempts: int = 50       # sampling attempts per requested crop
    empty_similar_size_prob: float = 0.5    # refined: P(size copied from a random GT box)
    empty_size_jitter: Tuple[float, float] = (0.7, 1.3)   # refined: jitter on that copied size

    # -- deviated crops --------------------------------------------------------------
    # ``deviation_shift`` is the centre displacement as a FRACTION of the box size; it expands
    # into ``deviation_shift_range`` (+/- SHIFT_BAND) unless a range is passed explicitly.
    # Passing the scalar used to be silently ignored, which is why it is a real field here.
    deviation_shift: float = 0.80
    deviation_shift_range: Optional[Tuple[float, float]] = None
    deviation_min_area_ratio: float = 0.7   # keep at least 70% of the original box area
    deviation_max_iou: float = 0.25         # baseline only: reject above this IoU with any GT

    # -- shared negative acceptance criterion (refined) -------------------------------
    negative_iou_reject: float = 0.75       # reject a negative whose IoU with ANY GT reaches this

    negative_rule: str = "baseline"         # "baseline" | "refined"

    # -- model-based noise (iterations 1+, unused by the app's rule-based path) --------
    high_conf_threshold: float = 0.7
    low_conf_threshold: float = 0.5

    def __post_init__(self):
        total = self.good_crop_ratio + self.empty_crop_ratio + self.deviation_ratio
        if abs(total - 1.0) >= 0.01:
            raise ValueError(f"crop ratios must sum to 1.0, got {total}")
        if self.negative_rule not in ("baseline", "refined"):
            raise ValueError(f"negative_rule must be 'baseline' or 'refined', got {self.negative_rule!r}")
        if self.deviation_shift_range is None:
            s = float(self.deviation_shift)
            self.deviation_shift_range = (max(0.05, s - SHIFT_BAND), s + SHIFT_BAND)
        else:                                   # explicit range wins; keep the scalar consistent
            lo, hi = self.deviation_shift_range
            self.deviation_shift_range = (float(lo), float(hi))
            self.deviation_shift = round((float(lo) + float(hi)) / 2, 4)
        self.empty_crop_min_size = tuple(self.empty_crop_min_size)
        self.empty_size_jitter = tuple(self.empty_size_jitter)

    def describe(self) -> dict:
        """Flat dict of the rule — recorded next to generated crops for reproducibility."""
        return {
            "negative_rule": self.negative_rule,
            "good_crop_ratio": self.good_crop_ratio,
            "empty_crop_ratio": self.empty_crop_ratio,
            "deviation_ratio": self.deviation_ratio,
            "good_crop_jitter": self.good_crop_jitter,
            "empty_similar_size_prob": self.empty_similar_size_prob,
            "empty_size_jitter": list(self.empty_size_jitter),
            "deviation_shift": self.deviation_shift,
            "deviation_shift_range": list(self.deviation_shift_range),
            "deviation_min_area_ratio": self.deviation_min_area_ratio,
            "deviation_max_iou": self.deviation_max_iou,
            "negative_iou_reject": self.negative_iou_reject,
        }


@dataclass
class PipelineConfig:
    """The three stage configs together — what a training run is fully described by."""

    detection: DetectionModelConfig = field(default_factory=DetectionModelConfig)
    classification: ClassificationModelConfig = field(default_factory=ClassificationModelConfig)
    noise: NoiseGenerationConfig = field(default_factory=NoiseGenerationConfig)
    classes: List[str] = field(default_factory=lambda: ["object"])
