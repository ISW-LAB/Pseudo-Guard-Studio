"""Label Studio adapter — export count-guided pre-labels as LS *predictions*.

Label Studio ingests pre-annotations via the task ``predictions`` field, with
object-detection boxes as ``rectanglelabels`` whose x/y/width/height are given as
PERCENTAGES of the image dimensions. Importing these makes each accepted candidate
appear as an editable box the human can accept/adjust/reject.

Labeling-config expected on the LS project side (document in the study SOP)::

    <View>
      <Image name="image" value="$image"/>
      <RectangleLabels name="label" toName="image">
        <Label value="class_a"/> ...
      </RectangleLabels>
    </View>

The ``score``/``band`` ride along in each region's ``meta`` so the front-end
plugin (design §4.4 ②) can color boxes and log ``candidate_rendered`` telemetry.
Ref: Label Studio "Import pre-annotations" docs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from ..count_guided_labeler import Candidate
from .base import ImageMeta, PreAnnotationAdapter, band_of


class LabelStudioAdapter(PreAnnotationAdapter):

    def _region(self, c: Candidate, meta: ImageMeta) -> dict:
        x1, y1, x2, y2 = c.box_xyxy
        W, H = max(1, meta.width), max(1, meta.height)
        return {
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels",
            "original_width": meta.width,
            "original_height": meta.height,
            "image_rotation": 0,
            "value": {
                "x": 100.0 * x1 / W,
                "y": 100.0 * y1 / H,
                "width": 100.0 * (x2 - x1) / W,
                "height": 100.0 * (y2 - y1) / H,
                "rotation": 0,
                "rectanglelabels": [self._class_name(c.label)],
            },
            # not part of the LS schema proper — carried for the UI plugin/telemetry
            "meta": {"pg_box_id": c.box_id, "pg_score": c.score, "pg_band": band_of(c.score)},
        }

    def build_task(self, meta: ImageMeta, accepted: Sequence[Candidate]) -> dict:
        return {
            "data": {"image": meta.ref, "pg_image_id": meta.image_id},
            "predictions": [
                {
                    "model_version": self.model_version,
                    "result": [self._region(c, meta) for c in accepted],
                }
            ],
        }

    def write(self, tasks: list[dict], path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(tasks, fh, ensure_ascii=False, indent=2)
        return path
