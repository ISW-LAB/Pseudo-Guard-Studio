"""OSS annotation-tool instrumentation adapters (design §4.4).

Decision (design §4.4): we do NOT build a bespoke canvas. Instead we *instrument*
an existing OSS annotator (Label Studio or CVAT) by exporting the count-guided
pre-labels into its pre-annotation import format, plus a telemetry hook. The thin,
reusable adapter layer IS the software-engineering artifact.

Each adapter turns a ``PreLabelResult`` (from ``count_guided_labeler``) into the
tool-specific "pre-annotation" that appears as editable boxes for the human.
"""

from .base import PreAnnotationAdapter, ImageMeta
from .label_studio import LabelStudioAdapter
from .cvat import CvatAdapter

__all__ = ["PreAnnotationAdapter", "ImageMeta", "LabelStudioAdapter", "CvatAdapter"]
