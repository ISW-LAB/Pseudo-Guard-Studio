#!/usr/bin/env python3
"""Crop-classification wrappers — the proposal validator, and the optional species head.

``TorchvisionClassifierWrapper`` (DenseNet-121 by default) is the proposal VALIDATOR: a binary
good/noise head over 256x256 object crops whose P(good) is the score the acceptance policy
selects on. The same class also serves the optional species classifier, which assigns a class
per crop when the detector is run class-agnostically. ``ViTClassifierWrapper`` covers the ViT
backbones, which need their own input-resize shim.

Supported backbones: DenseNet-121, ResNet-50, MobileNetV3-Large, ConvNeXt-Base, ViT-B/16.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Any
from pathlib import Path
from abc import ABC, abstractmethod
from tqdm import tqdm

from pseudoguard.device import resolve as resolve_device


class ClassificationModelBase(ABC):
    """Abstract base class for classification models"""

    @abstractmethod
    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        epochs: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Train the model"""
        pass

    @abstractmethod
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities.

        Returns:
            Tensor[N, 2]: Probabilities for [noise, good]
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


class TorchvisionClassifierWrapper(ClassificationModelBase):
    """Wrapper for torchvision classification models"""

    def __init__(
        self,
        model_type: str = "densenet121",  # densenet121, resnet50, mobilenet_v3_large, convnext_base
        img_size: int = 256,
        num_classes: int = 2,
        pretrained: bool = True,
        device: str = "auto"
    ):
        """
        Initialize classifier wrapper.

        Args:
            model_type: Model type (densenet121, resnet50, mobilenet_v3_large, convnext_base)
            img_size: Input image size
            num_classes: Number of classes (2 for good/noise)
            pretrained: Use pretrained weights
            device: Device to use
        """
        import torchvision.models as models

        self.model_type = model_type
        self.img_size = img_size
        self.num_classes = num_classes
        self.device = resolve_device(device)      # "auto"/absent CUDA -> cpu, never a late crash

        # Load pretrained model
        if model_type == "densenet121":
            if pretrained:
                try:
                    # PyTorch 2.0+ API
                    from torchvision.models import DenseNet121_Weights
                    self.model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
                except:
                    # Older API
                    self.model = models.densenet121(pretrained=True)
            else:
                self.model = models.densenet121(pretrained=False)

            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(in_features, num_classes)

        elif model_type == "resnet50":
            if pretrained:
                try:
                    from torchvision.models import ResNet50_Weights
                    self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
                except:
                    self.model = models.resnet50(pretrained=True)
            else:
                self.model = models.resnet50(pretrained=False)

            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)

        elif model_type == "mobilenet_v3_large":
            if pretrained:
                try:
                    from torchvision.models import MobileNet_V3_Large_Weights
                    self.model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
                except:
                    self.model = models.mobilenet_v3_large(pretrained=True)
            else:
                self.model = models.mobilenet_v3_large(pretrained=False)

            in_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(in_features, num_classes)

        elif model_type == "convnext_base":
            if pretrained:
                try:
                    from torchvision.models import ConvNeXt_Base_Weights
                    self.model = models.convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
                except:
                    self.model = models.convnext_base(pretrained=True)
            else:
                self.model = models.convnext_base(pretrained=False)

            in_features = self.model.classifier[2].in_features
            self.model.classifier[2] = nn.Linear(in_features, num_classes)

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.model = self.model.to(self.device)
        print(f"Loaded {model_type} on {self.device}")

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        epochs: int,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train classification model.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Number of epochs
            **kwargs: Additional parameters

        Returns:
            Dictionary with training metrics
        """
        batch_size = kwargs.get('batch_size', 32)
        lr = kwargs.get('lr', 1e-4)
        weight_decay = kwargs.get('weight_decay', 1e-5)
        patience = kwargs.get('patience', 10)
        use_weighted_loss = kwargs.get('use_weighted_loss', True)
        good_weight = kwargs.get('good_weight', 1.0)
        noise_weight = kwargs.get('noise_weight', 2.0)
        use_amp = kwargs.get('use_amp', True)

        # AMP GradScaler (perf fix rank 5)
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

        # DataLoaders (Windows requires num_workers=0; Linux uses passed value)
        import platform
        _nw_req = kwargs.get('num_workers', 8)
        _nw = 0 if platform.system() == "Windows" else max(0, int(_nw_req))
        _loader_kwargs = dict(num_workers=_nw, pin_memory=True)
        if _nw > 0:
            _loader_kwargs["persistent_workers"] = True
            _loader_kwargs["prefetch_factor"] = 4
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            **_loader_kwargs,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            **_loader_kwargs,
        )

        # Optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )

        # Loss function
        if use_weighted_loss:
            weight = torch.tensor([noise_weight, good_weight]).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=weight)
        else:
            criterion = nn.CrossEntropyLoss()

        # Training loop
        best_val_acc = 0.0
        patience_counter = 0
        history = []

        print(f"\nTraining {self.model_type} for {epochs} epochs...")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {lr}")

        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self._train_epoch(
                train_loader, optimizer, criterion, scaler=scaler, use_amp=use_amp
            )

            # Validate
            val_loss, val_acc = self._validate_epoch(
                val_loader, criterion, use_amp=use_amp
            )

            # Scheduler step
            scheduler.step()

            # Log
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

            history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            })

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)

        # Evaluate best model on val set for detailed metrics
        val_precision, val_recall, val_f1, val_cm = 0.0, 0.0, 0.0, []
        try:
            from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
            self.model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(self.device)
                    with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                        outputs = self.model(inputs)
                    preds = outputs.argmax(dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels.numpy())
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='binary', zero_division=0)
            cm = confusion_matrix(all_labels, all_preds)
            val_precision = float(precision)
            val_recall = float(recall)
            val_f1 = float(f1)
            val_cm = cm.tolist()
            print(f"  Val Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, "
                  f"F1: {val_f1:.4f}")
            print(f"  Confusion Matrix: {val_cm}")
        except Exception as e:
            print(f"  Warning: Could not compute detailed metrics: {e}")

        # Explicitly cleanup DataLoader workers to prevent zombie processes
        del train_loader, val_loader

        return {
            'best_val_acc': best_val_acc,
            'final_train_acc': train_acc,
            'final_val_acc': val_acc,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1,
            'val_confusion_matrix': val_cm,
            'history': history
        }

    def _train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scaler: "torch.amp.GradScaler" = None,
        use_amp: bool = True,
    ):
        """Train for one epoch"""
        self.model.train()

        # Lazily build scaler if caller didn't pass one (back-compat).
        if scaler is None:
            scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(loader, desc="Training", leave=False):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward (AMP autocast)
            optimizer.zero_grad()
            with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                outputs = self.model(images)
                loss = criterion(outputs, labels)

            # Backward (GradScaler pattern)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Metrics
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    def _validate_epoch(self, loader: DataLoader, criterion: nn.Module, use_amp: bool = True):
        """Validate for one epoch"""
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(loader, desc="Validating", leave=False):
                images = images.to(self.device)
                labels = labels.to(self.device)

                with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    def predict(self, images) -> torch.Tensor:
        """
        Run inference.

        Args:
            images: Tensor[N, C, H, W] or List[PIL.Image]

        Returns:
            Tensor[N, 2]: Probabilities for [noise, good]
        """
        from PIL import Image
        from torchvision import transforms

        self.model.eval()

        # Handle List[PIL.Image] input
        if isinstance(images, list) and len(images) > 0 and isinstance(images[0], Image.Image):
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            tensors = [transform(img.convert('RGB')) for img in images]
            images = torch.stack(tensors)

        images = images.to(self.device)

        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.float16):
                logits = self.model(images)
            probs = F.softmax(logits.float(), dim=1)

        return probs.cpu()

    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_type': self.model_type,
            'model_state_dict': self.model.state_dict(),
            'img_size': self.img_size,
            'num_classes': self.num_classes
        }

        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Validate checkpoint structure
        if 'model_state_dict' not in checkpoint:
            raise ValueError(
                f"Checkpoint missing 'model_state_dict' key. "
                f"Available keys: {list(checkpoint.keys())}"
            )

        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {path}")


class _ViTResizeWrapper(nn.Module):
    """Wraps a torchvision ViT so it always receives the size it was built for.

    Pretrained ViTs assert exact input H/W (e.g. 224). Our crop pipeline
    produces 256x256 tensors, so we bilinear-resize on the fly when needed.
    """

    def __init__(self, vit_model: nn.Module, target_size: int):
        super().__init__()
        self.vit = vit_model
        self.target_size = target_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.target_size or x.shape[-2] != self.target_size:
            x = F.interpolate(
                x,
                size=(self.target_size, self.target_size),
                mode="bilinear",
                align_corners=False,
            )
        return self.vit(x)


class ViTClassifierWrapper(ClassificationModelBase):
    """Wrapper for Vision Transformer"""

    def __init__(
        self,
        model_name: str = "vit_b_16",
        img_size: int = 256,
        num_classes: int = 2,
        pretrained: bool = True,
        device: str = "auto"
    ):
        """
        Initialize ViT wrapper.

        Args:
            model_name: Model name (vit_b_16, vit_b_32, vit_l_16)
            img_size: Input image size
            num_classes: Number of classes
            pretrained: Use pretrained weights
            device: Device to use
        """
        import torchvision.models as models

        self.model_name = model_name
        self.img_size = img_size
        self.num_classes = num_classes
        self.device = resolve_device(device)      # "auto"/absent CUDA -> cpu, never a late crash

        # Load ViT
        if model_name == "vit_b_16":
            if pretrained:
                try:
                    from torchvision.models import ViT_B_16_Weights
                    vit = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
                except:
                    vit = models.vit_b_16(pretrained=True)
            else:
                vit = models.vit_b_16(pretrained=False)
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

        # Modify classification head
        in_features = vit.heads.head.in_features
        vit.heads.head = nn.Linear(in_features, num_classes)

        # ViT (vit_b_16 / pretrained) requires exactly 224x224. Our crop
        # pipeline emits img_size (typically 256), so resize on the fly.
        native_size = getattr(vit, "image_size", 224)
        if img_size != native_size:
            self.model = _ViTResizeWrapper(vit, target_size=native_size)
        else:
            self.model = vit

        self.model = self.model.to(self.device)
        print(f"Loaded {model_name} on {self.device} (input {img_size}->{native_size})")

    def train(self, train_dataset, val_dataset, epochs, **kwargs):
        """Training is similar to TorchvisionClassifierWrapper"""
        # Use the same training loop
        wrapper = TorchvisionClassifierWrapper.__new__(TorchvisionClassifierWrapper)
        wrapper.model = self.model
        wrapper.model_type = self.model_name
        wrapper.img_size = self.img_size
        wrapper.num_classes = self.num_classes
        wrapper.device = self.device
        return TorchvisionClassifierWrapper.train(
            wrapper, train_dataset, val_dataset, epochs, **kwargs
        )

    def predict(self, images):
        """Prediction is same as TorchvisionClassifierWrapper"""
        wrapper = TorchvisionClassifierWrapper.__new__(TorchvisionClassifierWrapper)
        wrapper.model = self.model
        wrapper.device = self.device
        return TorchvisionClassifierWrapper.predict(wrapper, images)

    def save_checkpoint(self, path):
        """Save checkpoint"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_name': self.model_name,
            'model_state_dict': self.model.state_dict(),
            'img_size': self.img_size,
            'num_classes': self.num_classes
        }

        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path):
        """Load checkpoint"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Validate checkpoint structure
        if 'model_state_dict' not in checkpoint:
            raise ValueError(
                f"Checkpoint missing 'model_state_dict' key. "
                f"Available keys: {list(checkpoint.keys())}"
            )

        state = checkpoint['model_state_dict']
        # Bridge old (pre-resize-wrapper) checkpoints saved without "vit." prefix.
        if isinstance(self.model, _ViTResizeWrapper):
            sample_key = next(iter(state.keys()), "")
            if not sample_key.startswith("vit."):
                state = {f"vit.{k}": v for k, v in state.items()}
        self.model.load_state_dict(state)
        print(f"Loaded checkpoint from {path}")
