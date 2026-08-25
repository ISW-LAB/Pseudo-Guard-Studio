"""Event-sourced telemetry — the source of the study's human-side metrics.

Design doc §4.5. Append-only JSONL, one row per human/tool event, keyed by
``(session_id, image_id, box_id)`` so the offline analysis can join tool decisions
to human actions to gold (box-level matched analysis, design §3.1 / §6.2).

Timestamps: pass server time in from the caller (the harness forbids wall-clock
in some contexts). In the FastAPI serve layer, stamp on receipt.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, TextIO


# Canonical event vocabulary (design §4.5). Keep this list authoritative.
EVENT_TYPES = (
    "session_start",
    "seed_labeled",          # {image_id, n_boxes}
    "automate_clicked",      # {condition, seed_count_stats, chosen_tau_or_K, n_accepted}
    "candidate_rendered",    # {box_id, score, band}
    "box_accept",
    "box_reject",
    "box_adjust",            # {box_id, dxywh}
    "box_add",               # net-new human box (not from AI)
    "box_delete",
    "class_change",          # {box_id, from, to}
    "hover",                 # {box_id, dwell_ms}
    "image_finalized",       # {image_id, n_boxes, dwell_ms}
    "undo",
    "session_end",
)


@dataclass
class Event:
    ts_ms: int
    session_id: str
    event_type: str
    image_id: Optional[str] = None
    box_id: Optional[str] = None
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.event_type not in EVENT_TYPES:
            raise ValueError(f"unknown event_type {self.event_type!r}; add it to EVENT_TYPES")


class TelemetryLogger:
    """Append-only JSONL writer. One file per session."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: Optional[TextIO] = None

    def __enter__(self) -> "TelemetryLogger":
        self._fh = open(self.path, "a", encoding="utf-8")
        return self

    def __exit__(self, *exc) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

    def log(self, event: Event) -> None:
        line = json.dumps(asdict(event), ensure_ascii=False, separators=(",", ":"))
        if self._fh is None:  # allow use without the context manager
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        else:
            self._fh.write(line + "\n")

    def emit(self, ts_ms: int, session_id: str, event_type: str, **kw) -> None:
        """Convenience: build + log in one call. Splits image_id/box_id from payload."""
        image_id = kw.pop("image_id", None)
        box_id = kw.pop("box_id", None)
        self.log(Event(ts_ms, session_id, event_type, image_id, box_id, payload=kw))


def read_events(path: Path) -> list[dict]:
    """Load a session's events for offline metric derivation."""
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
