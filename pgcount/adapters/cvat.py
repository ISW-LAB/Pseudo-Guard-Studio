"""CVAT adapter (SCAFFOLD) — export count-guided pre-labels for CVAT.

Two integration paths (pick one during implementation, design §12.6):

  (A) CVAT REST SDK (recommended): use ``cvat-sdk`` to attach pre-annotations to a
      task programmatically — best for logging exactly what was shown.
          from cvat_sdk import make_client
          with make_client(host, credentials=(u, p)) as client:
              task = client.tasks.retrieve(task_id)
              task.import_annotations(fmt="CVAT 1.1", filename=xml_path)   # or SDK objects

  (B) "CVAT for images 1.1" XML: emit boxes as <box label=.. xtl.. ytl.. xbr.. ybr..>
      inside <image name=..>, import via the CVAT UI.

This file implements a minimal (B) XML writer so the pipeline is end-to-end
runnable; the SDK path (A) is left as a documented TODO because it needs a live
CVAT server + credentials.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
from xml.sax.saxutils import escape

from ..count_guided_labeler import Candidate
from .base import ImageMeta, PreAnnotationAdapter, band_of


class CvatAdapter(PreAnnotationAdapter):

    def build_task(self, meta: ImageMeta, accepted: Sequence[Candidate]) -> dict:
        # Return an intermediate dict; write() assembles the XML across all images.
        boxes = []
        for c in accepted:
            x1, y1, x2, y2 = c.box_xyxy
            boxes.append({
                "label": self._class_name(c.label),
                "xtl": x1, "ytl": y1, "xbr": x2, "ybr": y2,
                "pg_box_id": c.box_id, "pg_score": c.score, "pg_band": band_of(c.score),
            })
        return {"image_id": meta.image_id, "name": meta.ref,
                "width": meta.width, "height": meta.height, "boxes": boxes}

    def write(self, tasks: list[dict], path: Path) -> Path:
        """Emit a 'CVAT for images 1.1' annotations XML (path B)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = ['<?xml version="1.0" encoding="utf-8"?>', "<annotations>",
                 '  <version>1.1</version>']
        for i, t in enumerate(tasks):
            lines.append(
                f'  <image id="{i}" name="{escape(str(t["name"]))}" '
                f'width="{t["width"]}" height="{t["height"]}">'
            )
            for b in t["boxes"]:
                lines.append(
                    f'    <box label="{escape(b["label"])}" occluded="0" source="pgcount" '
                    f'xtl="{b["xtl"]:.2f}" ytl="{b["ytl"]:.2f}" '
                    f'xbr="{b["xbr"]:.2f}" ybr="{b["ybr"]:.2f}" z_order="0">'
                )
                # CVAT attributes carry the gate score/band for the UI plugin + telemetry
                lines.append(f'      <attribute name="pg_score">{b["pg_score"]:.4f}</attribute>')
                lines.append(f'      <attribute name="pg_band">{b["pg_band"]}</attribute>')
                lines.append("    </box>")
            lines.append("  </image>")
        lines.append("</annotations>")
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    # TODO(impl): path (A) — push via cvat-sdk import_annotations against a live task.
