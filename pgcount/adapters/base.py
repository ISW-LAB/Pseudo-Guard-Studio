"""Adapter interface: count-guided pre-labels → OSS pre-annotation format."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..count_guided_labeler import Candidate, PreLabelResult


@dataclass
class ImageMeta:
    """Minimal image info the OSS formats need (pixel dims + a reference/URL)."""

    image_id: str
    ref: str          # path or URL the OSS tool will load
    width: int
    height: int


def band_of(score: float, tau_low: float = 0.35, tau_high: float = 0.65) -> str:
    """green/amber/red band for UI coloring (design §4.6). Only the validator/gate
    score is ever surfaced — never detector confidence (decoupling discipline)."""
    if score >= tau_high:
        return "green"
    if score >= tau_low:
        return "amber"
    return "red"


class PreAnnotationAdapter:
    """Base class. Subclasses render accepted candidates into a tool-specific task."""

    model_version = "pgcount-v0.1"

    def __init__(self, class_names: Sequence[str]):
        self.class_names = list(class_names)

    def _class_name(self, label: int) -> str:
        return self.class_names[label] if 0 <= label < len(self.class_names) else str(label)

    def build_task(self, meta: ImageMeta, accepted: Sequence[Candidate]) -> dict:
        """Return ONE OSS task (image + pre-annotations). Implemented per tool."""
        raise NotImplementedError

    def build_tasks(self, metas: dict[str, ImageMeta], result: PreLabelResult) -> list[dict]:
        """Group the result's accepted boxes by image and build one task each.

        Images with no accepted candidates still get an (empty pre-annotation)
        task so the human reviews them (design: don't hide un-proposed images —
        counters miss carry-over / satisficing, RQ5).
        """
        by_img: dict[str, list[Candidate]] = {img: [] for img in metas}
        for c in result.accepted:
            by_img.setdefault(c.image_id, []).append(c)
        return [self.build_task(metas[img], by_img.get(img, [])) for img in metas]

    def write(self, tasks: list[dict], path: Path) -> Path:
        """Serialize tasks to disk in the tool's import format. Implemented per tool."""
        raise NotImplementedError
