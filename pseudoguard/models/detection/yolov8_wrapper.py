#!/usr/bin/env python3
"""Detection wrapper — one interface over YOLOv8 / YOLOv11 / YOLO26 / RT-DETR.

The rest of the pipeline only calls ``train``, ``predict``, ``evaluate``, ``save_checkpoint``
and ``load_checkpoint``, so swapping the proposal generator is a one-argument change
(``model_type``) rather than a code change. Ultralytics is imported lazily inside the methods
that need it: constructing a wrapper must stay cheap and import-safe in an environment that
only does inference bookkeeping.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import torch
import yaml
from abc import ABC, abstractmethod

from pseudoguard.data.det_loader import YoloDetDataset
from pseudoguard.device import resolve as resolve_device


class DetectionModelBase(ABC):
    """Abstract base class for detection models"""

    @abstractmethod
    def train(
        self,
        train_dataset: YoloDetDataset,
        val_dataset: YoloDetDataset,
        epochs: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Train the model, return metrics"""
        pass

    @abstractmethod
    def predict(
        self,
        images: List[torch.Tensor],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Run inference.

        Returns:
            List of predictions, each containing:
            - 'boxes': Tensor[N, 4] (xyxy)
            - 'scores': Tensor[N]
            - 'labels': Tensor[N]
        """
        pass

    @abstractmethod
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        pass

    @abstractmethod
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        pass


class YOLOWrapper(DetectionModelBase):
    """Wrapper for YOLOv8/v11/v26/RT-DETR using ultralytics library"""

    def __init__(
        self,
        model_type: str = "yolov8",  # yolov8, yolov11, yolo26, rtdetr
        size: str = "n",  # n, s, m, l, x
        img_size: int = 640,
        device: str = "auto"
    ):
        """
        Initialize YOLO/RT-DETR wrapper.

        Args:
            model_type: Model type (yolov8, yolov11, yolo26, rtdetr)
            size: Model size (n=nano, s=small, m=medium, l=large, x=xlarge)
            img_size: Input image size
            device: Device to use
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics not installed. Please install: pip install ultralytics"
            )

        self.model_type = model_type
        self.size = size
        self.img_size = img_size
        self.device = resolve_device(device)      # "auto"/absent CUDA -> cpu, never a late crash

        # Load pretrained model — RT-DETR uses different naming convention
        if model_type == "rtdetr":
            from ultralytics import RTDETR
            model_name = f"rtdetr-{size}.pt"  # e.g., rtdetr-l.pt
            self.model = RTDETR(model_name)
        else:
            # ultralytics naming: yolov8 -> yolov8n.pt, yolov11 -> yolo11n.pt, yolo26 -> yolo26n.pt
            pt_prefix = "yolo11" if model_type == "yolov11" else model_type
            model_name = f"{pt_prefix}{size}.pt"
            self.model = YOLO(model_name, task='detect')
        self.model.to(self.device)

        print(f"Loaded {model_name} on {self.device}")

    def train(
        self,
        train_dataset: YoloDetDataset,
        val_dataset: YoloDetDataset,
        epochs: int,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train YOLO model.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Number of epochs
            **kwargs: Additional training parameters

        Returns:
            Dictionary with training metrics
        """
        # Create temporary data.yaml
        data_yaml_path = self._create_data_yaml(train_dataset, val_dataset)

        # Training parameters
        project = kwargs.get('project', './runs/train')
        name = kwargs.get('name', 'exp')
        batch_size = kwargs.get('batch_size', 16)
        patience = kwargs.get('patience', 20)
        lr0 = kwargs.get('lr', 0.001)
        num_workers = kwargs.get('num_workers', 8)

        print(f"\nTraining {self.model_type}{self.size} for {epochs} epochs...")
        print(f"  Batch size: {batch_size}")
        print(f"  Device: {self.device}")
        print(f"  Project: {project}")

        # Ensure overrides dict has required keys (fix for multi-iteration training)
        # After first training, ultralytics may reset overrides dict
        if not hasattr(self.model, 'overrides') or 'model' not in self.model.overrides:
            if not hasattr(self.model, 'overrides'):
                self.model.overrides = {}
            if self.model_type == "rtdetr":
                self.model.overrides['model'] = f"rtdetr-{self.size}.pt"
            else:
                self.model.overrides['model'] = f"{self.model_type}{self.size}.pt"
            self.model.overrides['task'] = 'detect'

        # Train
        results = self.model.train(
            data=str(data_yaml_path),
            epochs=epochs,
            imgsz=self.img_size,
            batch=batch_size,
            patience=patience,
            device=self.device,
            project=project,
            name=name,
            lr0=lr0,
            verbose=True,
            plots=False,
            workers=num_workers,
        )

        # Extract metrics
        metrics = self._extract_metrics(results)

        return metrics

    def predict(
        self,
        images: List[torch.Tensor],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Run inference on images.

        Args:
            images: List of image tensors or image paths
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS

        Returns:
            List of predictions
        """
        results = self.model.predict(
            images,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=self.img_size,
            verbose=False,
            device=self.device
        )

        # Convert to standard format
        predictions = []
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                predictions.append({
                    'boxes': r.boxes.xyxy.cpu(),  # [N, 4]
                    'scores': r.boxes.conf.cpu(),  # [N]
                    'labels': r.boxes.cls.long().cpu()  # [N]
                })
            else:
                # No detections
                predictions.append({
                    'boxes': torch.empty(0, 4),
                    'scores': torch.empty(0),
                    'labels': torch.empty(0, dtype=torch.long)
                })

        return predictions

    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # YOLO model is saved automatically during training
        # We save the best weights
        if hasattr(self.model, 'trainer') and self.model.trainer:
            best_path = Path(self.model.trainer.best)
            if best_path.exists():
                import shutil
                shutil.copy(best_path, path)
                print(f"Saved checkpoint to {path}")
            else:
                # Save current model
                self.model.save(path)
        else:
            self.model.save(path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        import gc
        from ultralytics import YOLO

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        # Free previous model before loading new one to prevent GPU memory leak
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            torch.cuda.empty_cache()
            gc.collect()

        if self.model_type == "rtdetr":
            from ultralytics import RTDETR
            self.model = RTDETR(str(path))
        else:
            self.model = YOLO(str(path), task='detect')
        self.model.to(self.device)
        print(f"Loaded checkpoint from {path} on {self.device}")

    def get_state_dict(self):
        """
        Return the underlying PyTorch model's state_dict.

        Fast alternative to save_checkpoint() + load_checkpoint() for in-memory
        snapshot/restore (e.g., iterative training loops). Avoids file I/O and
        the heavy reconstruction performed by load_checkpoint().
        """
        return self.model.model.state_dict()

    def restore_state(self, state_dict):
        """
        Restore the underlying PyTorch model's weights from a state_dict
        produced by get_state_dict().

        Fast alternative to load_checkpoint(): performs an in-place
        load_state_dict() on the existing model, avoiding the
        `del self.model; YOLO(path)` reload cycle (which re-parses the .pt
        file, rebuilds the network, and re-moves it to device).

        Args:
            state_dict: A state_dict previously returned by get_state_dict().
        """
        self.model.model.load_state_dict(state_dict)

    def _create_data_yaml(
        self,
        train_dataset: YoloDetDataset,
        val_dataset: YoloDetDataset
    ) -> Path:
        """
        Create temporary YAML config for YOLO training.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset

        Returns:
            Path to created YAML file
        """
        # Get dataset root and class names
        spec = train_dataset.spec

        # Paths - try multiple candidates for flexible dataset structures
        # Train path candidates
        train_candidates = [
            spec.root / "images" / "train",
            spec.root / "images" / "train2012",
            spec.root / "images" / "train2007",
            spec.root / "images" / "train2017",
            spec.root / "images",
        ]
        train_path = None
        for candidate in train_candidates:
            if candidate.exists() and candidate.is_dir():
                train_path = candidate
                break

        if train_path is None:
            train_path = spec.root / "images" / "train"  # Default fallback

        # Val path candidates
        val_candidates = [
            spec.root / "images" / "val",
            spec.root / "images" / "val2012",
            spec.root / "images" / "val2007",
            spec.root / "images" / "val2017",
            spec.root / "images" / "valid",
            spec.root / "images" / "test",
            train_path,  # Use train as fallback for validation
        ]
        val_path = None
        for candidate in val_candidates:
            if candidate.exists() and candidate.is_dir():
                val_path = candidate
                break

        if val_path is None:
            val_path = train_path  # Ultimate fallback - use train for val

        # Extract class information from labels
        # Try multiple label directory candidates
        classes_found = set()
        labels_candidates = [
            spec.root / "labels" / "train",
            spec.root / "labels" / "train2012",
            spec.root / "labels" / "train2007",
            spec.root / "labels" / "train2017",
            spec.root / "labels",
        ]

        labels_dir = None
        for candidate in labels_candidates:
            if candidate.exists() and candidate.is_dir():
                # For flat structure, try train_*.txt pattern
                if candidate.name == "labels":
                    label_files = list(candidate.glob("train_*.txt"))
                    if not label_files:
                        label_files = list(candidate.glob("*.txt"))
                else:
                    label_files = list(candidate.glob("*.txt"))

                if label_files:
                    labels_dir = candidate
                    # Extract classes from found label files
                    for label_file in label_files:
                        with open(label_file, 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if parts:
                                    classes_found.add(int(parts[0]))
                    break

        # Create class names dict
        nc = max(classes_found) + 1 if classes_found else 1
        class_names = {i: str(i) for i in range(nc)}

        # Create YAML content (use forward slashes for ultralytics compatibility)
        yaml_content = {
            'path': str(spec.root).replace("\\", "/"),
            'train': str(train_path.relative_to(spec.root)).replace("\\", "/"),
            'val': str(val_path.relative_to(spec.root)).replace("\\", "/"),
            'nc': nc,
            'names': class_names
        }

        # Save to temp file (cross-platform)
        import tempfile
        yaml_path = Path(tempfile.gettempdir()) / f"yolo_data_{spec.name}.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)

        print(f"Created data YAML: {yaml_path}")
        print(f"  Train: {train_path}")
        print(f"  Val: {val_path}")
        print(f"  Classes: {nc}")

        return yaml_path

    # ──────────── MC-Dropout & TTA for Comparison Study ────────────

    def predict_with_dropout(
        self,
        images: List,
        n_passes: int = 5,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        dropout_rate: float = 0.1
    ) -> List[List[Dict[str, torch.Tensor]]]:
        """
        Run N stochastic forward passes with dropout enabled for uncertainty estimation.
        (Gal & Ghahramani, ICML 2016)

        Args:
            images: List of image paths or tensors
            n_passes: Number of stochastic forward passes
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            dropout_rate: Dropout probability

        Returns:
            List of per-pass predictions. Each pass is a list of per-image prediction dicts.
        """
        import torch.nn as nn

        # Find or inject dropout layers
        model_backbone = self.model.model
        injected_dropouts = []
        existing_dropouts = []

        for name, module in model_backbone.named_modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d)):
                existing_dropouts.append(module)

        # If no dropout layers exist, inject Dropout2d before detect head
        if not existing_dropouts:
            # Find the last conv/bottleneck layer before the detection head
            # In ultralytics YOLO, model.model[-1] is the Detect head
            detect_head = model_backbone.model[-1] if hasattr(model_backbone, 'model') else None
            if detect_head is not None:
                # Wrap detect head's forward with dropout
                original_forward = detect_head.forward
                dropout_layer = nn.Dropout2d(p=dropout_rate)
                dropout_layer.to(self.device)
                injected_dropouts.append((detect_head, original_forward, dropout_layer))

                def make_dropout_forward(orig_fn, drop):
                    def forward_with_dropout(x):
                        if isinstance(x, (list, tuple)):
                            x = [drop(xi) for xi in x]
                        else:
                            x = drop(x)
                        return orig_fn(x)
                    return forward_with_dropout

                detect_head.forward = make_dropout_forward(original_forward, dropout_layer)

        # Enable dropout for stochastic inference
        dropout_modules = existing_dropouts
        if not dropout_modules and injected_dropouts:
            dropout_modules = [item[2] for item in injected_dropouts]

        # Save training states, then enable dropout
        original_training_states = {}
        for module in dropout_modules:
            original_training_states[id(module)] = module.training
            module.train()

        # Run N forward passes
        all_passes = []
        with torch.no_grad():
            for _ in range(n_passes):
                preds = self.predict(images, conf_threshold, iou_threshold)
                all_passes.append(preds)

        # Restore original states
        for module in dropout_modules:
            module.train(original_training_states.get(id(module), False))

        # Remove injected dropout wrappers
        for detect_head, original_forward, _ in injected_dropouts:
            detect_head.forward = original_forward

        return all_passes

    def predict_with_tta(
        self,
        image_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Run prediction with test-time augmentation (TTA).
        Returns predictions for: original, hflip, scale 0.8, scale 1.2.

        All boxes are mapped back to original image coordinates.

        Args:
            image_path: Path to the image
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS

        Returns:
            List of prediction dicts, one per augmented view (4 total).
        """
        from PIL import Image as PILImage
        import numpy as np
        import tempfile
        import os

        img = PILImage.open(image_path).convert("RGB")
        orig_w, orig_h = img.size

        augmented_results = []

        # 1. Original
        preds_orig = self.predict([image_path], conf_threshold, iou_threshold)[0]
        augmented_results.append(preds_orig)

        # 2. Horizontal flip
        img_flip = img.transpose(PILImage.FLIP_LEFT_RIGHT)
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            img_flip.save(tmp.name)
            preds_flip = self.predict([tmp.name], conf_threshold, iou_threshold)[0]
            os.unlink(tmp.name)
        # Un-flip boxes: x1_new = orig_w - x2_old, x2_new = orig_w - x1_old
        if len(preds_flip['boxes']) > 0:
            boxes = preds_flip['boxes'].clone()
            x1_new = orig_w - boxes[:, 2]
            x2_new = orig_w - boxes[:, 0]
            boxes[:, 0] = x1_new
            boxes[:, 2] = x2_new
            preds_flip['boxes'] = boxes
        augmented_results.append(preds_flip)

        # 3. Scale 0.8x
        new_w, new_h = int(orig_w * 0.8), int(orig_h * 0.8)
        img_small = img.resize((new_w, new_h), PILImage.BILINEAR)
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            img_small.save(tmp.name)
            preds_small = self.predict([tmp.name], conf_threshold, iou_threshold)[0]
            os.unlink(tmp.name)
        # Scale boxes back to original coordinates
        if len(preds_small['boxes']) > 0:
            scale_x = orig_w / new_w
            scale_y = orig_h / new_h
            boxes = preds_small['boxes'].clone()
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            preds_small['boxes'] = boxes
        augmented_results.append(preds_small)

        # 4. Scale 1.2x
        new_w, new_h = int(orig_w * 1.2), int(orig_h * 1.2)
        img_large = img.resize((new_w, new_h), PILImage.BILINEAR)
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            img_large.save(tmp.name)
            preds_large = self.predict([tmp.name], conf_threshold, iou_threshold)[0]
            os.unlink(tmp.name)
        # Scale boxes back to original coordinates
        if len(preds_large['boxes']) > 0:
            scale_x = orig_w / new_w
            scale_y = orig_h / new_h
            boxes = preds_large['boxes'].clone()
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            preds_large['boxes'] = boxes
        augmented_results.append(preds_large)

        return augmented_results

    def _extract_metrics(self, results) -> Dict[str, Any]:
        """
        Extract metrics from training results.

        Args:
            results: YOLO training results

        Returns:
            Dictionary with metrics
        """
        metrics = {}

        try:
            # Get final metrics from results
            if hasattr(results, 'results_dict'):
                results_dict = results.results_dict
                metrics['map50'] = results_dict.get('metrics/mAP50(B)', 0.0)
                metrics['map50_95'] = results_dict.get('metrics/mAP50-95(B)', 0.0)
                metrics['precision'] = results_dict.get('metrics/precision(B)', 0.0)
                metrics['recall'] = results_dict.get('metrics/recall(B)', 0.0)
            else:
                # Try to get from model validator
                if hasattr(self.model, 'metrics'):
                    metrics['map50'] = float(self.model.metrics.box.map50)
                    metrics['map50_95'] = float(self.model.metrics.box.map)
                    metrics['precision'] = float(self.model.metrics.box.mp)
                    metrics['recall'] = float(self.model.metrics.box.mr)

        except Exception as e:
            print(f"Warning: Could not extract all metrics: {e}")
            metrics = {
                'map50': 0.0,
                'map50_95': 0.0,
                'precision': 0.0,
                'recall': 0.0
            }

        return metrics
