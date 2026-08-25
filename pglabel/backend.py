#!/usr/bin/env python3
"""The app's view of "is there an AI, and what does it propose for this image?".

One small adapter over ``pgcount``'s two candidate sources, with one property the rest of the
app depends on: it is None-safe. ``AIBackend()`` with nothing configured is a perfectly valid
object whose ``available`` is False and whose ``candidates()`` returns nothing — so the manual
annotator works end to end with no model, no overlay and no torch installed.

Two ways to have an AI:

    overlay              a precomputed candidate JSON (what training writes, and what a
                         packaged install ships with) — no torch needed at all.
    detector + validator live checkpoints, scored on demand — needs the training environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional


class AIBackend:
    """Candidate source for the app; ``available`` False means "manual mode"."""

    def __init__(self, overlay: Optional[str] = None, detector: Optional[str] = None,
                 validator: Optional[str] = None, device: str = "auto",
                 det_model_type: str = "yolov8"):
        self._overlay = None
        self._frozen = None
        self.available = False
        self.kind = "none"
        if overlay and Path(overlay).exists():
            from pgcount.backend import PrecomputedBackend
            self._overlay = PrecomputedBackend.from_json(Path(overlay))
            self.available = True
            self.kind = "overlay"
        elif detector and validator:
            from pgcount.backend import FrozenBackend
            self._frozen = FrozenBackend(Path(detector), Path(validator),
                                         det_model_type=det_model_type, device=device).load()
            self.available = True
            self.kind = "live"

    def candidates(self, image_id: str, image_path: Path) -> List:
        if self._overlay is not None:
            return self._overlay.candidates(image_id)
        if self._frozen is not None:
            return self._frozen.candidates(image_id, image_path)
        return []
