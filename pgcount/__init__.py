"""pgcount — count-guided acceptance for human-AI collaborative annotation.

The idea in one paragraph: a human labels a few *seed* images; the per-image box count of
that seed (the dataset's object density) sets the *operating point* — a global score
threshold or a per-image top-K — that "Automate Label" then applies to the model's
candidates. Acceptance is therefore matched to how crowded this dataset actually is,
instead of to a threshold guessed once and reused everywhere.

The separation of concerns this package exists to hold: ``pseudoguard`` produces candidates
and scores them; ``pgcount`` decides which candidates are ACCEPTED. Nothing here trains or
mutates a model, so the acceptance policy can be changed, compared or ablated without
touching the models — which is what makes "same AI, different collaboration" measurable.

Modules
-------
config               strategy / backend / seed-protocol settings
seed_density         seed box counts -> the density statistic D_seed
operating_point      D_seed -> per-image top-K or global tau (the core mechanism)
backend              candidate sources: live models, or a precomputed overlay
count_guided_labeler candidates -> selection -> pre-labels
metrics              count fidelity, defect escape, operating curves
telemetry            event-sourced JSONL logging of the human-AI session
adapters/            export pre-labels to Label Studio / CVAT
"""

from .config import CountGuidedConfig, OperatingPointStrategy, Backend
from .seed_density import SeedDensityStats, estimate_seed_density
from .operating_point import (
    OperatingPoint,
    PerImageTopK,
    GlobalTau,
    AutoAdaptiveTau,
    make_operating_point,
)
from .count_guided_labeler import Candidate, CountGuidedLabeler, Condition

__all__ = [
    "CountGuidedConfig",
    "OperatingPointStrategy",
    "Backend",
    "SeedDensityStats",
    "estimate_seed_density",
    "OperatingPoint",
    "PerImageTopK",
    "GlobalTau",
    "AutoAdaptiveTau",
    "make_operating_point",
    "Candidate",
    "CountGuidedLabeler",
    "Condition",
]

__version__ = "0.1.0"
