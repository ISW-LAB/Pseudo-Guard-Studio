"""Candidate sources — the only place in ``pgcount`` that touches models or files on disk.

Both classes expose the same one method::

    candidates(image_id, image_path) -> list[RawCandidate]

where each candidate carries BOTH scores: the detector's confidence and the validator's
P(good). Everything else in this package works purely on those numbers, which is what keeps
the acceptance policy independent of how the candidates were produced.

    FrozenBackend        runs the real models (needs torch + ultralytics). Read-only: the
                         checkpoints are never updated here — training lives in ``tools/``.
    PrecomputedBackend   replays a candidate overlay JSON produced earlier. No GPU, no torch,
                         byte-identical candidates on every machine, so a study session or a
                         packaged install can run the whole acceptance path without a model.

The application always talks to one of these two, never to ``pseudoguard`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .crops import extract_crops


@dataclass
class RawCandidate:
    """One detector-proposed box with both score sources (design §4.2)."""

    image_id: str
    box_id: str
    box_xyxy: tuple[float, float, float, float]
    label: int
    det_conf: float          # detector confidence (logged, NOT shown to humans)
    p_good: float            # validator P(good) — the decoupled score


class FrozenBackend:
    """Loads a frozen (detector, validator) pair and yields scored candidates."""

    def __init__(
        self,
        detector_ckpt: Path,
        validator_ckpt: Path,
        pseudo_conf_threshold: float = 0.05,   # config.py: high-recall collection
        det_iou_threshold: float = 0.45,       # NMS IoU (matches PseudoLabeler)
        det_model_type: str = "yolov8",        # must match the detector checkpoint
        det_size: str = "n",
        det_img_size: int = 640,
        val_model_type: str = "densenet121",   # overridden by checkpoint if present
        crop_img_size: int = 256,              # classification.img_size
        num_classes: int = 2,                  # good/noise
        device: str = "auto",      # "auto" -> CUDA when present, else CPU
    ):
        self.detector_ckpt = Path(detector_ckpt)
        self.validator_ckpt = Path(validator_ckpt)
        self.pseudo_conf_threshold = pseudo_conf_threshold
        self.det_iou_threshold = det_iou_threshold
        self.det_model_type = det_model_type
        self.det_size = det_size
        self.det_img_size = det_img_size
        self.val_model_type = val_model_type
        self.crop_img_size = crop_img_size
        self.num_classes = num_classes
        from pseudoguard.device import resolve as _resolve
        self.device = _resolve(device)
        self._detector = None
        self._validator = None

    # ------------------------------------------------------------------ loading
    def load(self) -> "FrozenBackend":
        """Load the frozen detector + validator from checkpoints (read-only).

        Needs the training environment (torch + ultralytics); see requirements-train.txt.
        The checkpoints are the ones ``tools/train_and_predict.py`` writes into the work dir:
        ``detection_model.pt`` and ``classification_model.pt``. The validator's architecture,
        crop size and class count are read back OUT of its checkpoint, so a validator trained
        with different settings still loads correctly.
        """
        import torch
        from pseudoguard.models.detection.yolov8_wrapper import YOLOWrapper
        from pseudoguard.models.classification.densenet_wrapper import TorchvisionClassifierWrapper

        # --- detector (YOLO): construct then load trained weights (wrapper pattern) ---
        det = YOLOWrapper(
            model_type=self.det_model_type,
            size=self.det_size,
            img_size=self.det_img_size,
            device=self.device,
        )
        det.load_checkpoint(str(self.detector_ckpt))  # -> YOLO(path, task='detect')
        self._detector = det

        # --- validator (DenseNet): peek checkpoint for arch, construct, load state ---
        ckpt = torch.load(self.validator_ckpt, map_location="cpu", weights_only=False)
        val_model_type = ckpt.get("model_type", self.val_model_type)
        val_img_size = int(ckpt.get("img_size", self.crop_img_size))
        val_num_classes = int(ckpt.get("num_classes", self.num_classes))
        val = TorchvisionClassifierWrapper(
            model_type=val_model_type,
            img_size=val_img_size,
            num_classes=val_num_classes,
            pretrained=False,          # weights come from the checkpoint
            device=self.device,
        )
        val.load_checkpoint(str(self.validator_ckpt))
        self._validator = val
        self.crop_img_size = val_img_size
        return self

    # -------------------------------------------------------------- candidates
    def candidates(self, image_id: str, image_path: Path) -> List[RawCandidate]:
        """Run detector (high-recall) then validator on one image → scored candidates.

        Steps (design §4.2):
          1. detector.predict @ conf=0.05  → boxes(xyxy, orig px), det_conf, labels
          2. extract_crops(image, boxes)   → PIL crops (+ keep-indices for alignment)
          3. validator.predict(crops)      → probs[:,1] = P(good)
        Detector confidence is carried but NEVER surfaced to humans (decoupling, §4.3).
        """
        if self._detector is None or self._validator is None:
            raise RuntimeError("FrozenBackend.load() must be called before candidates()")

        preds = self._detector.predict(
            [str(image_path)],
            conf_threshold=self.pseudo_conf_threshold,
            iou_threshold=self.det_iou_threshold,
        )[0]
        boxes = preds["boxes"].cpu().numpy()      # [N,4] xyxy in original pixels
        det_conf = preds["scores"].cpu().numpy()  # [N]
        labels = preds["labels"].cpu().numpy()    # [N]
        if len(boxes) == 0:
            return []

        crops, keep = extract_crops(image_path, [tuple(b) for b in boxes])
        if not crops:
            return []

        p_good_col = self._validator.predict(crops)  # Tensor[len(crops), 2]
        try:
            p_good = p_good_col[:, 1].cpu().numpy()
        except Exception:  # ndarray fallback path in some wrappers
            import numpy as np
            arr = np.asarray(p_good_col)
            p_good = arr[:, 1] if arr.ndim == 2 else arr.flatten()

        out: List[RawCandidate] = []
        for k, box_idx in enumerate(keep):
            b = boxes[box_idx]
            out.append(
                RawCandidate(
                    image_id=image_id,
                    box_id=f"{image_id}:{box_idx}",
                    box_xyxy=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                    label=int(labels[box_idx]),
                    det_conf=float(det_conf[box_idx]),
                    p_good=float(p_good[k]),
                )
            )
        return out


class PrecomputedBackend:
    """Serves candidates from a frozen, pre-computed overlay JSON (design §3.6).

    This is what the *study sessions* actually use: every participant sees byte-
    identical candidates/scores per image, sessions need no GPU, and results are
    reproducible. Produce the overlay offline with ``scripts/precompute_overlays.py``.

    Overlay schema (one file per dataset)::

        { image_id: [ {box_id, box_xyxy:[x1,y1,x2,y2], label, det_conf, p_good}, ... ], ... }
    """

    def __init__(self, overlay: dict[str, list[dict]]):
        self._overlay = overlay

    @classmethod
    def from_json(cls, path: Path) -> "PrecomputedBackend":
        import json

        with open(path, "r", encoding="utf-8") as fh:
            return cls(json.load(fh))

    def candidates(self, image_id: str, image_path: Optional[Path] = None) -> List[RawCandidate]:
        out: List[RawCandidate] = []
        for rec in self._overlay.get(image_id, []):
            out.append(
                RawCandidate(
                    image_id=image_id,
                    box_id=rec["box_id"],
                    box_xyxy=tuple(rec["box_xyxy"]),
                    label=int(rec.get("label", 0)),
                    det_conf=float(rec["det_conf"]),
                    p_good=float(rec["p_good"]),
                )
            )
        return out

    def all_image_ids(self) -> list[str]:
        return list(self._overlay.keys())
