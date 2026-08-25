#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bounding box operations for Auto-Labeling System

Used by: noise_generator.py, iterative_trainer.py
"""

import torch
import numpy as np
from typing import Tuple, List, Union
from PIL import Image
from torchvision.ops import roi_align


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Calculate IoU between two sets of boxes.

    Args:
        boxes1: Tensor[N, 4] in xyxy format
        boxes2: Tensor[M, 4] in xyxy format

    Returns:
        Tensor[N, M] with IoU values
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou


def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from xyxy to cxcywh format (center coordinates).

    Args:
        boxes: Tensor[N, 4] in xyxy format

    Returns:
        Tensor[N, 4] in cxcywh format (cx, cy, w, h)
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from cxcywh to xyxy format.

    Args:
        boxes: Tensor[N, 4] in cxcywh format (cx, cy, w, h)

    Returns:
        Tensor[N, 4] in xyxy format (x1, y1, x2, y2)
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def compute_pseudo_label_accuracy(
    pseudo_boxes_norm: np.ndarray,
    pseudo_classes: np.ndarray,
    gt_boxes_norm: List[List[float]],
    gt_classes: List[int],
    iou_threshold: float = 0.5,
    class_agnostic: bool = False
) -> dict:
    """
    Compute IoU-based accuracy between pseudo-labels and ground truth for a single image.

    Uses greedy matching: pairs are matched in descending IoU order,
    requiring IoU >= threshold and (unless class_agnostic) a class match.

    Args:
        pseudo_boxes_norm: [N, 4] numpy array in normalized cxcywh format
        pseudo_classes: [N] numpy array of class IDs
        gt_boxes_norm: List of [cx, cy, w, h] normalized GT boxes
        gt_classes: List of GT class IDs
        iou_threshold: IoU threshold for matching (default: 0.5)
        class_agnostic: If True, match on IoU only (ignore class IDs). REQUIRED when
            the detector is trained single-class (nc=1) but GT files carry the
            original multi-class IDs — otherwise every match fails on class and
            precision/recall/IoU collapse to 0 even for perfectly localized boxes.

    Returns:
        Dict with 'precision', 'recall', 'f1', 'matched', 'gt_count', 'pseudo_count'
        Returns None if both are empty (skip image).
    """
    n_pseudo = len(pseudo_boxes_norm) if isinstance(pseudo_boxes_norm, np.ndarray) and pseudo_boxes_norm.ndim == 2 else 0
    n_gt = len(gt_boxes_norm)

    if n_pseudo == 0 and n_gt == 0:
        return None  # skip image

    if n_pseudo == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'matched': 0, 'gt_count': n_gt, 'pseudo_count': 0, 'matched_iou_sum': 0.0}

    if n_gt == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'matched': 0, 'gt_count': 0, 'pseudo_count': n_pseudo, 'matched_iou_sum': 0.0}

    # Convert to xyxy tensors for IoU computation
    pseudo_t = cxcywh_to_xyxy(torch.tensor(pseudo_boxes_norm, dtype=torch.float32))
    gt_t = cxcywh_to_xyxy(torch.tensor(gt_boxes_norm, dtype=torch.float32))

    iou_matrix = box_iou(pseudo_t, gt_t)  # [N_pseudo, N_gt]

    # Greedy matching in descending IoU order
    matched = 0
    iou_sum = 0.0
    pseudo_matched = set()
    gt_matched = set()

    # Flatten IoU matrix and sort by descending IoU
    iou_flat = iou_matrix.reshape(-1)
    sorted_indices = torch.argsort(iou_flat, descending=True)

    for idx in sorted_indices:
        idx_val = idx.item()
        pi = idx_val // n_gt
        gi = idx_val % n_gt
        iou_val = iou_flat[idx].item()

        if iou_val < iou_threshold:
            break  # all remaining are below threshold

        if pi in pseudo_matched or gi in gt_matched:
            continue

        # Class match check (skipped for class-agnostic / single-class detection)
        if not class_agnostic:
            p_cls = int(pseudo_classes[pi]) if isinstance(pseudo_classes, np.ndarray) else int(pseudo_classes)
            g_cls = int(gt_classes[gi])
            if p_cls != g_cls:
                continue

        pseudo_matched.add(pi)
        gt_matched.add(gi)
        matched += 1
        iou_sum += iou_val

    precision = matched / n_pseudo if n_pseudo > 0 else 0.0
    recall = matched / n_gt if n_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'matched': matched,
        'gt_count': n_gt,
        'pseudo_count': n_pseudo,
        'matched_iou_sum': float(iou_sum)
    }


def compute_batch_pseudo_accuracy(
    pseudo_labels_list: List[dict],
    unlabeled_imgs: List,
    iou_threshold: float = 0.5,
    class_agnostic: bool = False
) -> dict:
    """
    Compute IoU-based pseudo-label accuracy across all unlabeled images.

    For each unlabeled image, compares pseudo-label boxes against
    the original ground truth labels using IoU-based greedy matching.

    Reads pseudo-label boxes from the on-disk YOLO format label files
    (normalized cxcywh) instead of the in-memory dict to ensure correct
    box format and reflect any in-place filtering updates.

    Args:
        pseudo_labels_list: List of pseudo-label dicts. Each dict must have
                           'image_path', 'label_path', and 'num_boxes'.
        unlabeled_imgs: List of Path objects for unlabeled images (used to find GT labels).
        iou_threshold: IoU threshold for matching (default: 0.5)

    Returns:
        Dict with 'micro_precision', 'micro_recall', 'micro_f1',
              'total_matched', 'total_gt', 'total_pseudo'
    """
    from pathlib import Path
    from pseudoguard.data.det_loader import _image_to_label_path, _read_yolo_label_file

    # Build image_path -> pseudo_label dict for fast lookup
    pseudo_by_image = {}
    for pl in pseudo_labels_list:
        img_key = Path(pl['image_path']).stem
        pseudo_by_image[img_key] = pl

    total_matched = 0
    total_gt = 0
    total_pseudo = 0
    total_matched_iou = 0.0

    for img_path in unlabeled_imgs:
        img_path = Path(img_path)
        img_key = img_path.stem

        # Read ground truth
        gt_label_path = _image_to_label_path(img_path, img_path.parent.parent.parent)
        gt_classes, gt_boxes_norm = _read_yolo_label_file(gt_label_path)

        # Read pseudo labels from on-disk label file (YOLO normalized cxcywh format).
        # NOTE: ``_read_yolo_label_file`` returns python lists; ``compute_pseudo_label_accuracy``
        # only counts pseudo boxes when the input is a 2-D numpy array, so we
        # MUST convert here — otherwise n_pseudo silently drops to 0 and
        # micro_precision/total_pseudo are reported as 0 even when boxes exist.
        pl = pseudo_by_image.get(img_key)
        if pl is not None and pl.get('num_boxes', 0) > 0:
            pseudo_label_path = Path(pl['label_path'])
            if pseudo_label_path.exists():
                _classes_list, _boxes_list = _read_yolo_label_file(pseudo_label_path)
                pseudo_classes = np.array(_classes_list, dtype=np.int64)
                pseudo_boxes = (
                    np.array(_boxes_list, dtype=np.float32).reshape(-1, 4)
                    if len(_boxes_list) > 0
                    else np.zeros((0, 4), dtype=np.float32)
                )
            else:
                pseudo_boxes = np.zeros((0, 4), dtype=np.float32)
                pseudo_classes = np.array([], dtype=np.int64)
        else:
            pseudo_boxes = np.zeros((0, 4), dtype=np.float32)
            pseudo_classes = np.array([], dtype=np.int64)

        result = compute_pseudo_label_accuracy(
            pseudo_boxes, pseudo_classes,
            gt_boxes_norm, gt_classes,
            iou_threshold=iou_threshold,
            class_agnostic=class_agnostic
        )

        if result is not None:
            total_matched += result['matched']
            total_gt += result['gt_count']
            total_pseudo += result['pseudo_count']
            total_matched_iou += result.get('matched_iou_sum', 0.0)

    micro_precision = total_matched / total_pseudo if total_pseudo > 0 else 0.0
    micro_recall = total_matched / total_gt if total_gt > 0 else 0.0
    micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)
                if (micro_precision + micro_recall) > 0 else 0.0)
    mean_matched_iou = total_matched_iou / total_matched if total_matched > 0 else 0.0

    return {
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        'total_matched': total_matched,
        'total_gt': total_gt,
        'total_pseudo': total_pseudo,
        'mean_matched_iou': mean_matched_iou
    }


def gpu_batch_crop_and_resize(
    image: torch.Tensor,
    boxes: torch.Tensor,
    target_size: Tuple[int, int] = (256, 256),
) -> torch.Tensor:
    """
    GPU-accelerated batch crop and resize using roi_align.

    Args:
        image: [C, H, W] uint8 or float tensor (any device)
        boxes: [N, 4] xyxy pixel coordinates (any device)
        target_size: (height, width) for output crops

    Returns:
        [N, C, target_h, target_w] float32 tensor on same device as image, range 0~1
    """
    if boxes.numel() == 0:
        return torch.empty(0, image.shape[0], *target_size,
                           dtype=torch.float32, device=image.device)

    # roi_align requires float input [1, C, H, W]
    img_float = image.unsqueeze(0).float()
    # uint8 input (0~255) needs normalization; avoid GPU sync from .max()
    if image.dtype == torch.uint8:
        img_float = img_float / 255.0

    # roi_align expects [N, 5] with (batch_idx, x1, y1, x2, y2)
    batch_idx = torch.zeros(boxes.shape[0], 1, device=boxes.device, dtype=boxes.dtype)
    rois = torch.cat([batch_idx, boxes.float()], dim=1)

    crops = roi_align(
        img_float, rois,
        output_size=target_size,
        spatial_scale=1.0,
        aligned=True,
    )
    return crops  # [N, C, target_h, target_w] float32, 0~1


def gpu_batch_crop_pil(
    img_pil: Image.Image,
    boxes: Union[List, torch.Tensor],
    target_size: Tuple[int, int] = (256, 256),
    device: Union[str, torch.device] = "cuda",
) -> List[Image.Image]:
    """
    GPU-batched crop+resize for a single PIL image, returning a list of PIL crops.

    Replacement for the per-box ``crop_box_from_image`` loop. Loads the source
    image to GPU once, runs all crops in a single ``roi_align`` kernel, then
    copies the resulting tensor batch back to CPU and converts to PIL.

    Args:
        img_pil: Source PIL image (RGB).
        boxes: ``[N, 4]`` xyxy pixel coordinates (tensor or list-of-lists).
        target_size: ``(h, w)`` for output crops.
        device: CUDA device to use; falls back to CPU if CUDA unavailable.

    Returns:
        List of N PIL Images at ``target_size``. Empty list if ``boxes`` is empty.
    """
    if not isinstance(boxes, torch.Tensor):
        if len(boxes) == 0:
            return []
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
    if boxes.numel() == 0:
        return []

    if isinstance(device, str):
        device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    arr = np.array(img_pil.convert("RGB"), copy=True)  # HWC uint8, writable
    img_tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous().to(device)  # CHW uint8

    crops = gpu_batch_crop_and_resize(img_tensor, boxes.to(device), target_size=target_size)
    # crops: [N, C, h, w] float32 0~1 on device — move to CPU and convert
    crops_np = (crops.clamp(0, 1) * 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    return [Image.fromarray(c) for c in crops_np]


def crop_box_from_image(
    image: Union[Image.Image, np.ndarray, torch.Tensor],
    box: Union[List, Tuple, torch.Tensor],
    target_size: Tuple[int, int] = (256, 256)
) -> Image.Image:
    """
    Crop box region from image and resize.

    Args:
        image: PIL Image, numpy array, or torch Tensor
        box: Box coordinates in xyxy format [x1, y1, x2, y2]
        target_size: Target size for resized crop (width, height)

    Returns:
        PIL Image of cropped and resized region
    """
    # Convert to PIL Image if needed
    if isinstance(image, torch.Tensor):
        # Assume CHW format, convert to HWC
        if image.dim() == 3 and image.shape[0] in [1, 3]:
            image = image.permute(1, 2, 0)
        # Convert to numpy
        image = image.cpu().numpy()

    if isinstance(image, np.ndarray):
        # Ensure uint8
        if image.dtype != np.uint8:
            # Assume [0, 1] range if float
            if image.dtype in [np.float32, np.float64]:
                image = (image * 255).astype(np.uint8)
        # Convert to PIL
        image = Image.fromarray(image)

    # Extract box coordinates
    if isinstance(box, torch.Tensor):
        box = box.cpu().tolist()

    x1, y1, x2, y2 = box

    # Ensure valid box
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    # Clip to image bounds
    img_width, img_height = image.size
    x1 = max(0, min(x1, img_width))
    y1 = max(0, min(y1, img_height))
    x2 = max(0, min(x2, img_width))
    y2 = max(0, min(y2, img_height))

    # Crop
    crop = image.crop((x1, y1, x2, y2))

    # Resize
    if target_size:
        crop = crop.resize(target_size, Image.BILINEAR)

    return crop


def boxes_have_overlap(
    box: Union[List, Tuple, torch.Tensor],
    boxes: torch.Tensor,
    threshold: float = 0.1
) -> bool:
    """
    Check if a box has significant overlap with any box in a set.

    Args:
        box: Single box [x1, y1, x2, y2]
        boxes: Tensor[N, 4] of boxes
        threshold: IoU threshold for overlap

    Returns:
        True if any overlap exceeds threshold
    """
    if boxes.shape[0] == 0:
        return False

    if not isinstance(box, torch.Tensor):
        box = torch.tensor(box, dtype=torch.float32)

    box = box.unsqueeze(0)  # [1, 4]

    iou = box_iou(box, boxes)  # [1, N]

    return (iou > threshold).any().item()


def generate_random_box(
    img_width: int,
    img_height: int,
    min_size: int = 50,
    max_size: int = 200
) -> torch.Tensor:
    """
    Generate a random box within image bounds.

    Args:
        img_width: Image width
        img_height: Image height
        min_size: Minimum box size
        max_size: Maximum box size

    Returns:
        Tensor[4] with box coordinates [x1, y1, x2, y2]
    """
    # Clamp max_size to image dimensions
    effective_max_w = min(max_size, img_width)
    effective_max_h = min(max_size, img_height)

    # Adjust min_size if image is too small (ensure low < high for randint)
    effective_min_w = min(min_size, effective_max_w - 1) if effective_max_w > 1 else 1
    effective_min_h = min(min_size, effective_max_h - 1) if effective_max_h > 1 else 1

    # Final safety: ensure valid range
    effective_min_w = max(1, effective_min_w)
    effective_min_h = max(1, effective_min_h)
    effective_max_w = max(effective_min_w + 1, effective_max_w)
    effective_max_h = max(effective_min_h + 1, effective_max_h)

    # Random width and height
    w = np.random.randint(effective_min_w, effective_max_w)
    h = np.random.randint(effective_min_h, effective_max_h)

    # Random position
    x1 = np.random.randint(0, max(1, img_width - w))
    y1 = np.random.randint(0, max(1, img_height - h))

    x2 = min(x1 + w, img_width)
    y2 = min(y1 + h, img_height)

    return torch.tensor([x1, y1, x2, y2], dtype=torch.float32)
