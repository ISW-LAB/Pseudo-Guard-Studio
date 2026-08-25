"""Pseudo-Guard — the frozen algorithm library behind the annotation workflow.

Four pieces, in the order a training run uses them:

    config              settings dataclasses (detector, validator, negative-crop rule)
    data.det_loader     YOLO-layout dataset discovery and loading
    data.noise_generator the RULE that fabricates good/empty/deviated validator crops
    models.detection    YOLO wrapper (proposal generator)
    models.classification DenseNet wrapper (proposal validator)

Importing this package does NOT import torch: the sub-modules that need it import it
themselves, so the label-only application can depend on ``pseudoguard.config`` and
``pseudoguard.device`` without pulling in a 2 GB stack.
"""

from .config import (
    ClassificationModelConfig,
    DetectionModelConfig,
    NoiseGenerationConfig,
    PipelineConfig,
)

__all__ = [
    "ClassificationModelConfig",
    "DetectionModelConfig",
    "NoiseGenerationConfig",
    "PipelineConfig",
]

__version__ = "1.0.0"
