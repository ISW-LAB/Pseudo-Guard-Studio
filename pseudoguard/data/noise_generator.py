#!/usr/bin/env python3
"""Fabricates the validator's training crops from a labeled seed.

The proposal validator is not trained on detector output — it is trained on crops the RULE
manufactures from ground truth, so it can be initialised before any detector exists:

    positive   ground-truth boxes, lightly jittered            -> label 1 (good)
    empty      background boxes containing no annotated object -> label 0 (noise)
    deviated   ground-truth boxes displaced off their object   -> label 0 (noise)

``NoiseGenerationConfig.negative_rule`` selects how the two negative types are drawn; see
``pseudoguard.config`` for what "baseline" and "refined" mean geometrically. The public entry
point is ``NoiseGenerator.generate_training_crops``; ``empty_boxes``/``deviated_boxes`` expose
the same geometry without cropping, which is what the app's review screen previews.
"""

import logging
from pathlib import Path
import torch
import numpy as np
from typing import List, Tuple, Dict
from PIL import Image
import random
import math

logger = logging.getLogger(__name__)

from pseudoguard.utils.box_ops import (
    box_iou,
    boxes_have_overlap,
    generate_random_box,
    crop_box_from_image,
    gpu_batch_crop_pil,
    xyxy_to_cxcywh,
    cxcywh_to_xyxy
)
from pseudoguard.data.det_loader import YoloDetDataset
from pseudoguard.config import NoiseGenerationConfig
from pseudoguard.device import resolve as resolve_device


def _pick_crop_device(preferred: str = None) -> torch.device:
    """Device for the GPU-batched crop path, with a CPU fallback that always holds.

    ``torch.device("cuda:0")`` constructs fine on a machine with no GPU and only fails later,
    inside roi_align — so the request is resolved through ``pseudoguard.device`` first.
    """
    return torch.device(resolve_device(preferred))


class NoiseGenerator:
    """
    Generates good and noise label crops for classification training.
    """

    def __init__(self, config: NoiseGenerationConfig, device: str = None):
        """
        Initialize noise generator.

        Args:
            config: Noise generation configuration
            device: CUDA device for batched crop. Defaults to current CUDA device
                or CPU if unavailable.
        """
        self.config = config
        self.device = _pick_crop_device(device)
        self._use_gpu = self.device.type == "cuda"

    def generate_training_crops(
        self,
        dataset: YoloDetDataset,
        mode: str = "rule_based",
        max_samples: int = None,
        pseudo_labels: List = None,
        sink=None,
    ) -> Tuple[List[Image.Image], List[int]]:
        """Generate crops with labels (0=noise, 1=good).

        Args:
            dataset: dataset to generate crops from
            mode: "rule_based" or "model_based"
            max_samples: cap on the number of crops
            pseudo_labels: pseudo-labels from the previous iteration (model_based only)
            sink: optional ``sink(crop, label)`` callable. When given, each crop is handed over
                and dropped immediately, so peak memory stays FLAT instead of growing with
                ``max_samples``. A decoded 256x256 RGB crop costs about 200 KB, so the default
                budget of 8,000 crops means ~1.6 GB held at once — enough to fail on a laptop,
                and unbounded when ``max_samples`` is None. Every caller that only writes the
                crops to disk should pass a sink.

        Returns:
            ``(crops, labels)``. ``labels`` is always complete, so ``sum(labels)`` and
            ``len(labels)`` give the good/noise counts either way; ``crops`` is empty when a
            sink was used, because the images have already been consumed.
        """
        if mode == "rule_based":
            return self._generate_rule_based_crops(dataset, max_samples, sink)
        elif mode == "model_based":
            if pseudo_labels is None:
                raise ValueError("pseudo_labels required for model_based mode")
            return self._generate_model_based_crops(dataset, pseudo_labels, max_samples)
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'rule_based' or 'model_based'")

    def _generate_rule_based_crops(
        self,
        dataset: YoloDetDataset,
        max_samples: int = None,
        sink=None,
    ) -> Tuple[List[Image.Image], List[int]]:
        """
        Generate crops using rule-based noise patterns.

        Rules (ratios are configurable; empty/deviated generation depends on
        config.negative_rule — 'baseline' or 'refined'):
        1. Good crops: GT boxes with small jitter → label=1
        2. Empty space crops → label=0
             baseline: large random background boxes (no GT overlap)
             refined : mixed GT-similar / large sizes, IoU with every GT < 0.75
        3. Deviated crops → label=0
             baseline: GT boxes shifted 0.5~2.0x (often fully off) & scaled
             refined : GT boxes deviated ~0.68~0.92x of box size (shift 0.80 ± 0.12),
                       IoU with every GT < 0.75

        Args:
            dataset: Dataset to generate from
            max_samples: Maximum samples per category

        Returns:
            crops, labels
        """
        crops = []
        labels = []

        # Calculate target numbers for each category
        total_good = 0
        total_empty = 0
        total_deviated = 0

        def collect(batch, label):
            """Keep the crop, or hand it straight to the sink and let it be freed.

            The label list is kept either way: it is 8 bytes per crop against ~200 KB for the
            image, and it is what tells the caller how many good/noise samples were produced.
            """
            if sink is None:
                crops.extend(batch)
            else:
                for crop in batch:
                    sink(crop, label)
            labels.extend([label] * len(batch))

        print("\nGenerating rule-based training crops...")
        print(f"  Good crop ratio: {self.config.good_crop_ratio}")
        print(f"  Empty crop ratio: {self.config.empty_crop_ratio}")
        print(f"  Deviated crop ratio: {self.config.deviation_ratio}")

        for idx in range(len(dataset)):
            img, target = dataset[idx]

            # Convert tensor to PIL if needed
            if isinstance(img, torch.Tensor):
                # Reverse ImageNet normalization if applied
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_denorm = img * std + mean
                img_denorm = img_denorm.clamp(0, 1)
                img = Image.fromarray(
                    (img_denorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                )

            boxes = target['boxes']  # xyxy format
            if boxes.shape[0] == 0:
                continue

            # Rule 1: Good crops from GT boxes
            good_crops = self._crop_good_boxes(img, boxes)
            collect(good_crops, 1)
            total_good += len(good_crops)

            # Calculate noise crop count based on ratio
            n_noise_crops = int(len(good_crops) * (
                self.config.empty_crop_ratio + self.config.deviation_ratio
            ) / self.config.good_crop_ratio)

            n_empty = int(n_noise_crops * self.config.empty_crop_ratio /
                         (self.config.empty_crop_ratio + self.config.deviation_ratio))
            n_deviated = n_noise_crops - n_empty

            # Rule 2: Empty space crops
            empty_crops = self._crop_empty_spaces(img, boxes, n_empty)
            collect(empty_crops, 0)
            total_empty += len(empty_crops)

            # Rule 3: Deviated crops
            deviated_crops = self._crop_deviated_boxes(img, boxes, n_deviated)
            collect(deviated_crops, 0)
            total_deviated += len(deviated_crops)

            if max_samples and len(labels) >= max_samples:
                # Trim only what is still in memory. With a sink the crops are already on disk,
                # so trimming the label list would make len(labels) disagree with what was
                # actually written — the cap is reached at an image boundary either way.
                if sink is None:
                    del labels[max_samples:]
                    del crops[max_samples:]
                break

        n = len(labels)
        print(f"\nGenerated {n} crops:")
        if n > 0:
            print(f"  Good: {total_good} ({total_good / n * 100:.1f}%)")
            print(f"  Empty: {total_empty} ({total_empty / n * 100:.1f}%)")
            print(f"  Deviated: {total_deviated} ({total_deviated / n * 100:.1f}%)")
        else:
            print("  WARNING: No crops generated! Dataset may have 0 images or 0 boxes.")

        return crops, labels

    def _crop_good_boxes(
        self,
        img: Image.Image,
        boxes: torch.Tensor
    ) -> List[Image.Image]:
        """
        Crop GT boxes at original label positions (good labels).

        Args:
            img: PIL Image
            boxes: Tensor[N, 4] in xyxy format

        Returns:
            List of cropped images
        """
        if boxes is None or len(boxes) == 0:
            return []

        # Clip boxes to image bounds & drop degenerate ones in one shot
        img_w, img_h = img.size
        b = boxes.to(dtype=torch.float32).clone()
        b[:, 0::2].clamp_(0, img_w)
        b[:, 1::2].clamp_(0, img_h)
        valid = (b[:, 2] - b[:, 0] >= 1) & (b[:, 3] - b[:, 1] >= 1)
        b = b[valid]
        if len(b) == 0:
            return []

        if self._use_gpu:
            try:
                return gpu_batch_crop_pil(img, b, target_size=(256, 256), device=self.device)
            except Exception as e:
                logger.warning(
                    "GPU batch crop failed in _crop_good_boxes, falling back to CPU: %r", e
                )

        crops = []
        for box in b:
            try:
                crops.append(crop_box_from_image(img, box, target_size=(256, 256)))
            except Exception:
                pass
        return crops

    # ── Negative-rule variant selector ──────────────────────────────────────
    @property
    def _negative_rule(self) -> str:
        """Which negative-generation rule set to use: 'baseline' (original) or
        'refined' (Q2/Q3/Q4). Defaults to 'baseline' for back-compat."""
        return getattr(self.config, "negative_rule", "baseline")

    @staticmethod
    def _as_gt_tensor(boxes) -> torch.Tensor:
        return boxes.to(dtype=torch.float32) if isinstance(boxes, torch.Tensor) \
            else torch.as_tensor(boxes, dtype=torch.float32)

    @staticmethod
    def _max_iou_with_gt(box, gt: torch.Tensor) -> float:
        """Max IoU of a single [x1,y1,x2,y2] box against ALL GT boxes (0.0 if no GT)."""
        if gt is None or len(gt) == 0:
            return 0.0
        b = box if isinstance(box, torch.Tensor) else torch.as_tensor(box, dtype=torch.float32)
        b = b.unsqueeze(0) if b.dim() == 1 else b
        return float(box_iou(b.to(dtype=torch.float32), gt).max().item())

    def _sample_empty_box_refined(self, img_w, img_h, gt, min_large, max_large):
        """Refined empty box (Q2): size is a probabilistic MIX of GT-similar and large.
        Returns a Tensor[4] [x1,y1,x2,y2] or None if degenerate."""
        use_similar = (len(gt) > 0) and (random.random() < self.config.empty_similar_size_prob)
        if use_similar:
            g = gt[random.randint(0, len(gt) - 1)]
            gw = float(g[2] - g[0]); gh = float(g[3] - g[1])
            jlo, jhi = self.config.empty_size_jitter
            w = gw * random.uniform(jlo, jhi)
            h = gh * random.uniform(jlo, jhi)
        else:
            # Large box — same scale envelope the baseline rule uses.
            w = float(random.randint(int(min_large), int(max_large)))
            h = float(random.randint(int(min_large), int(max_large)))
        w = max(2.0, min(w, float(img_w)))
        h = max(2.0, min(h, float(img_h)))
        x1 = random.uniform(0, img_w - w) if img_w - w > 0 else 0.0
        y1 = random.uniform(0, img_h - h) if img_h - h > 0 else 0.0
        x2 = min(x1 + w, float(img_w)); y2 = min(y1 + h, float(img_h))
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None
        return torch.tensor([x1, y1, x2, y2], dtype=torch.float32)

    def _collect_empty_boxes(self, img: Image.Image, boxes: torch.Tensor, n: int) -> List[torch.Tensor]:
        """Collect up to `n` EMPTY (background) noise boxes as List[Tensor[4]].
        Branches on negative_rule; shared by the crop and visualization paths."""
        img_width, img_height = img.size
        max_attempts = self.config.empty_crop_max_attempts * n
        min_dim = min(img_width, img_height)
        max_box_size = max(min_dim // 2, 2)
        min_box_size = self.config.empty_crop_min_size[0]
        if min_box_size >= max_box_size:
            min_box_size = max(1, max_box_size - 1)

        refined = self._negative_rule == "refined"
        gt = self._as_gt_tensor(boxes)
        reject_iou = getattr(self.config, "negative_iou_reject", 0.75)

        accepted = []
        for _ in range(max_attempts):
            if len(accepted) >= n:
                break
            if refined:
                # Q2: mixed GT-similar / large size.  Q4: accept only if IoU with EVERY GT < 0.75.
                rb = self._sample_empty_box_refined(img_width, img_height, gt, min_box_size, max_box_size)
                if rb is None:
                    continue
                if self._max_iou_with_gt(rb, gt) >= reject_iou:
                    continue
                accepted.append(rb)
            else:
                random_box = generate_random_box(
                    img_width, img_height,
                    min_size=min_box_size,
                    max_size=max_box_size
                )
                if not boxes_have_overlap(random_box, boxes, threshold=0.1):
                    accepted.append(random_box)
        return accepted

    def _collect_deviated_boxes(self, img: Image.Image, boxes: torch.Tensor, n: int) -> List[torch.Tensor]:
        """Collect up to `n` DEVIATED (mislocalized GT) noise boxes as List[Tensor[4]].
        Branches on negative_rule; shared by the crop and visualization paths."""
        img_w, img_h = img.size
        refined = self._negative_rule == "refined"
        min_area_ratio = self.config.deviation_min_area_ratio
        gt = self._as_gt_tensor(boxes)
        if len(gt) == 0:
            return []

        if refined:
            # Q3: deviate by the configured shift band (default 0.80 ± 0.12 = 0.68~0.92; center
            #     displacement = frac of box size along a random direction).  Q4: reject if IoU ≥ 0.75 with ANY GT.
            f_lo, f_hi = self.config.deviation_shift_range
            reject_iou = getattr(self.config, "negative_iou_reject", 0.75)
        else:
            # Baseline: aggressive shift 0.5~2.0x, reject if IoU > deviation_max_iou with ANY GT.
            reject_iou = self.config.deviation_max_iou

        max_attempts_per_crop = 100
        batch_size = 10
        accepted = []
        for _ in range(n):
            box = gt[random.randint(0, len(gt) - 1)]
            cx, cy, w, h = xyxy_to_cxcywh(box.unsqueeze(0))[0].tolist()
            orig_area = w * h

            attempts_remaining = max_attempts_per_crop
            found = False
            while attempts_remaining > 0 and not found:
                this_batch = min(batch_size, attempts_remaining)
                attempts_remaining -= this_batch

                candidates = []
                for _ in range(this_batch):
                    if refined:
                        frac = random.uniform(f_lo, f_hi)
                        theta = random.uniform(0, 2 * math.pi)
                        cx_new = cx + frac * w * math.cos(theta)
                        cy_new = cy + frac * h * math.sin(theta)
                    else:
                        cx_new = cx + random.choice([-1, 1]) * random.uniform(0.5, 2.0) * w
                        cy_new = cy + random.choice([-1, 1]) * random.uniform(0.5, 2.0) * h

                    w_new = w * random.uniform(0.7, 1.3)
                    h_new = h * random.uniform(0.7, 1.3)
                    if w_new * h_new < min_area_ratio * orig_area:
                        continue

                    x1 = max(0.0, cx_new - w_new / 2.0)
                    y1 = max(0.0, cy_new - h_new / 2.0)
                    x2 = min(float(img_w), cx_new + w_new / 2.0)
                    y2 = min(float(img_h), cy_new + h_new / 2.0)
                    if x2 - x1 < 10 or y2 - y1 < 10 or (x2 - x1) * (y2 - y1) < min_area_ratio * orig_area:
                        continue
                    candidates.append([x1, y1, x2, y2])

                if not candidates:
                    continue

                cands = torch.as_tensor(candidates, dtype=torch.float32)
                max_iou = box_iou(cands, gt).max(dim=1).values
                # Baseline keeps the original ≤ semantics; refined rejects at ≥ 0.75 (accept < 0.75).
                mask = (max_iou <= reject_iou) if not refined else (max_iou < reject_iou)
                survivors = mask.nonzero(as_tuple=False).flatten().tolist()
                if survivors:
                    accepted.append(torch.as_tensor(candidates[survivors[0]], dtype=torch.float32))
                    found = True
        return accepted

    def _crop_empty_spaces(
        self,
        img: Image.Image,
        boxes: torch.Tensor,
        n: int
    ) -> List[Image.Image]:
        """
        Generate crops from empty regions.

        baseline: random background boxes with no GT overlap (IoU>0.1 rejected).
        refined : GT-similar/large mixed sizes, accepted only if IoU with every GT < 0.75.

        Args:
            img: PIL Image
            boxes: Tensor[N, 4] of GT boxes
            n: Number of crops to generate

        Returns:
            List of cropped images
        """
        accepted = self._collect_empty_boxes(img, boxes, n)
        if not accepted:
            return []

        # accepted is List[Tensor[4]]; stack instead of as_tensor (which mis-interprets
        # a list of multi-element tensors).
        boxes_tensor = torch.stack(accepted).to(dtype=torch.float32)
        if self._use_gpu:
            try:
                return gpu_batch_crop_pil(img, boxes_tensor, target_size=(256, 256), device=self.device)
            except Exception as e:
                logger.warning(
                    "GPU batch crop failed in _crop_empty_spaces, falling back to CPU: %r", e
                )

        crops = []
        for rbox in boxes_tensor:
            try:
                crops.append(crop_box_from_image(img, rbox, target_size=(256, 256)))
            except Exception:
                pass
        return crops

    def _crop_deviated_boxes(
        self,
        img: Image.Image,
        boxes: torch.Tensor,
        n: int
    ) -> List[Image.Image]:
        """
        Generate hard-negative crops by shifting GT boxes.

        baseline: aggressive shift 0.5~2.0x box size, reject IoU > deviation_max_iou (0.25).
        refined : center displacement of deviation_shift_range x box size along a random
                  direction (default 0.80 ± 0.12 = 0.68~0.92, Q3), accepted only if IoU with
                  every GT < negative_iou_reject (0.75, Q4).

        Args:
            img: PIL Image
            boxes: Tensor[N, 4] of GT boxes in xyxy format
            n: Number of crops to generate

        Returns:
            List of cropped images
        """
        accepted = self._collect_deviated_boxes(img, boxes, n)
        if not accepted:
            return []

        boxes_tensor = torch.stack(accepted).to(dtype=torch.float32)
        if self._use_gpu:
            try:
                return gpu_batch_crop_pil(img, boxes_tensor, target_size=(256, 256), device=self.device)
            except Exception:
                pass

        crops = []
        for db in boxes_tensor:
            try:
                crops.append(crop_box_from_image(img, db, target_size=(256, 256)))
            except Exception:
                pass
        return crops

    # ---- public crop API -------------------------------------------------------------
    # generate_training_crops() covers the whole dataset in one call. These three expose the
    # SAME per-image steps, so a caller that needs its own mixture or its own file naming (the
    # review-crop generator does) does not have to reimplement the geometry.
    def good_crops(self, img: Image.Image, boxes: torch.Tensor) -> List[Image.Image]:
        """Positive crops: the ground-truth boxes themselves."""
        return self._crop_good_boxes(img, boxes)

    def empty_crops(self, img: Image.Image, boxes: torch.Tensor, n: int) -> List[Image.Image]:
        """Up to ``n`` background crops containing no annotated object."""
        return self._crop_empty_spaces(img, boxes, n)

    def deviated_crops(self, img: Image.Image, boxes: torch.Tensor, n: int) -> List[Image.Image]:
        """Up to ``n`` crops of ground-truth boxes displaced off their object."""
        return self._crop_deviated_boxes(img, boxes, n)

    # ---- box-coordinate variants (same selection, no cropping) -------------------------
    def empty_boxes(self, img: Image.Image, boxes: torch.Tensor, n: int) -> List[List[float]]:
        """Return up to n EMPTY noise boxes as [x1,y1,x2,y2] pixels (mirrors _crop_empty_spaces)."""
        return [
            [float(v) for v in b.tolist()]
            for b in self._collect_empty_boxes(img, boxes, n)
        ]

    def deviated_boxes(self, img: Image.Image, boxes: torch.Tensor, n: int) -> List[List[float]]:
        """Return up to n DEVIATED noise boxes as [x1,y1,x2,y2] pixels (mirrors _crop_deviated_boxes)."""
        return [
            [float(v) for v in b.tolist()]
            for b in self._collect_deviated_boxes(img, boxes, n)
        ]

    def _generate_model_based_crops(
        self,
        dataset: YoloDetDataset,
        pseudo_labels: List[Dict],
        max_samples: int = None
    ) -> Tuple[List[Image.Image], List[int]]:
        """
        Generate crops using model-based noise patterns.

        Uses pseudo-labels from previous iteration:
        1. Good crops: High confidence (>0.7) detections → label=1
        2. Noise crops: Low confidence (0.3-0.5) detections → label=0
        3. Discard medium confidence (0.5-0.7) as ambiguous

        Args:
            dataset: Dataset to generate from
            pseudo_labels: List of pseudo-label dicts from previous iteration
            max_samples: Maximum samples per category

        Returns:
            crops, labels
        """
        crops = []
        labels = []

        total_good = 0
        total_noise = 0

        print(f"\nGenerating model-based training crops...")
        print(f"  High conf threshold: {self.config.high_conf_threshold}")
        print(f"  Low conf threshold: {self.config.low_conf_threshold}")

        for pl in pseudo_labels:
            img_path = pl['image_path']
            boxes_raw = pl['boxes']
            scores = pl['scores']

            if len(boxes_raw) == 0:
                continue

            img = Image.open(img_path).convert('RGB')

            # Bucket boxes by confidence band into one (boxes, labels) batch per image.
            per_image_boxes = []
            per_image_labels = []
            for box, score in zip(boxes_raw, scores):
                score_val = score.item() if hasattr(score, 'item') else float(score)
                if score_val > self.config.high_conf_threshold:
                    per_image_boxes.append(box.tolist() if hasattr(box, 'tolist') else list(box))
                    per_image_labels.append(1)
                elif score_val < self.config.low_conf_threshold:
                    per_image_boxes.append(box.tolist() if hasattr(box, 'tolist') else list(box))
                    per_image_labels.append(0)
                # medium → discard

            if not per_image_boxes:
                continue

            boxes_tensor = torch.as_tensor(per_image_boxes, dtype=torch.float32)
            try:
                if self._use_gpu:
                    img_crops = gpu_batch_crop_pil(img, boxes_tensor, target_size=(256, 256), device=self.device)
                else:
                    img_crops = [crop_box_from_image(img, b, target_size=(256, 256)) for b in boxes_tensor]
            except Exception:
                img_crops = []
                for b in boxes_tensor:
                    try:
                        img_crops.append(crop_box_from_image(img, b, target_size=(256, 256)))
                    except Exception:
                        img_crops.append(None)

            for c, lab in zip(img_crops, per_image_labels):
                if c is None:
                    continue
                crops.append(c)
                labels.append(lab)
                if lab == 1:
                    total_good += 1
                else:
                    total_noise += 1

            if max_samples and len(crops) >= max_samples:
                crops = crops[:max_samples]
                labels = labels[:max_samples]
                break

        print(f"\nGenerated {len(crops)} crops:")
        if len(crops) > 0:
            print(f"  Good (high conf): {total_good} ({total_good/len(crops)*100:.1f}%)")
            print(f"  Noise (low conf): {total_noise} ({total_noise/len(crops)*100:.1f}%)")
        else:
            print(f"  WARNING: No crops generated from pseudo-labels!")

        # Balance classes if needed
        if total_good == 0 or total_noise == 0:
            print("WARNING: Imbalanced classes! Falling back to rule-based generation.")
            return self._generate_rule_based_crops(dataset, max_samples)

        return crops, labels
