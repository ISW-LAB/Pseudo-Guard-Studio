#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classification dataset for crop-based quality filtering
"""

import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import List, Sequence, Tuple, Union
from PIL import Image


class CropClassificationDataset(Dataset):
    """
    Dataset for classification model training.
    Contains image crops with binary labels (0=noise, 1=good).
    """

    def __init__(
        self,
        crops: List[Image.Image],
        labels: List[int],
        img_size: int = 256,
        augment: bool = True
    ):
        """
        Initialize dataset.

        Args:
            crops: List of PIL Images (cropped boxes)
            labels: List of labels (0=noise, 1=good)
            img_size: Target image size
            augment: Apply data augmentation
        """
        assert len(crops) == len(labels), "Crops and labels must have same length"

        self.crops = crops
        self.labels = labels
        self.img_size = img_size
        self.augment = augment

        # Define transforms
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        """
        Get item.

        Returns:
            image: Tensor[3, H, W]
            label: int (0 or 1)
        """
        crop = self.crops[idx]
        label = self.labels[idx]

        # Ensure RGB
        if crop.mode != 'RGB':
            crop = crop.convert('RGB')

        # Apply transforms
        image = self.transform(crop)

        return image, label

    def get_class_distribution(self):
        """Get class distribution"""
        n_noise = sum(1 for l in self.labels if l == 0)
        n_good = sum(1 for l in self.labels if l == 1)

        return {
            'noise': n_noise,
            'good': n_good,
            'total': len(self.labels)
        }


class CropFolderClassificationDataset(Dataset):
    """
    Disk-backed classification dataset. Reads JPG crops lazily from disk so
    training does not need to keep tens of thousands of PIL images resident
    in RAM. Used by the staged-samples training flow:
        samples_dir/clf_train_yes/*.jpg → label 1
        samples_dir/clf_train_no/*.jpg  → label 0
    """

    def __init__(
        self,
        file_paths: Sequence[Union[str, Path]],
        labels: Sequence[int],
        img_size: int = 256,
        augment: bool = True,
    ):
        assert len(file_paths) == len(labels), "file_paths and labels must have same length"
        self.file_paths = [Path(p) for p in file_paths]
        self.labels = list(labels)
        self.img_size = img_size
        self.augment = augment

        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        with Image.open(self.file_paths[idx]) as crop:
            crop = crop.convert('RGB')
            return self.transform(crop), self.labels[idx]

    def get_class_distribution(self):
        n_noise = sum(1 for l in self.labels if l == 0)
        n_good = sum(1 for l in self.labels if l == 1)
        return {'noise': n_noise, 'good': n_good, 'total': len(self.labels)}


def create_train_val_split_from_folder(
    samples_dir: Union[str, Path],
    val_ratio: float = 0.2,
    img_size: int = 256,
    seed: int = 42,
) -> Tuple[CropFolderClassificationDataset, CropFolderClassificationDataset]:
    """
    Build train/val datasets that read crops directly from disk under
    samples_dir/clf_train_yes (label=1) and samples_dir/clf_train_no (label=0).
    """
    samples_dir = Path(samples_dir)
    yes_dir = samples_dir / "clf_train_yes"
    no_dir = samples_dir / "clf_train_no"

    yes_files = sorted(yes_dir.glob("*.jpg")) if yes_dir.exists() else []
    no_files = sorted(no_dir.glob("*.jpg")) if no_dir.exists() else []

    file_paths = list(yes_files) + list(no_files)
    labels = [1] * len(yes_files) + [0] * len(no_files)

    if not file_paths:
        raise ValueError(f"No staged crops found under {samples_dir}")

    indices = list(range(len(file_paths)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_val = int(len(indices) * val_ratio)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_dataset = CropFolderClassificationDataset(
        [file_paths[i] for i in train_indices],
        [labels[i] for i in train_indices],
        img_size=img_size, augment=True,
    )
    val_dataset = CropFolderClassificationDataset(
        [file_paths[i] for i in val_indices],
        [labels[i] for i in val_indices],
        img_size=img_size, augment=False,
    )
    return train_dataset, val_dataset


def create_train_val_split(
    crops: List[Image.Image],
    labels: List[int],
    val_ratio: float = 0.2,
    img_size: int = 256
):
    """
    Create train/val split from crops.

    Args:
        crops: List of image crops
        labels: List of labels
        val_ratio: Validation set ratio
        img_size: Image size

    Returns:
        train_dataset, val_dataset
    """
    import random

    # Create indices
    indices = list(range(len(crops)))
    random.shuffle(indices)

    # Split
    n_val = int(len(indices) * val_ratio)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    # Create datasets
    train_crops = [crops[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]

    val_crops = [crops[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]

    train_dataset = CropClassificationDataset(
        train_crops, train_labels, img_size=img_size, augment=True
    )

    val_dataset = CropClassificationDataset(
        val_crops, val_labels, img_size=img_size, augment=False
    )

    return train_dataset, val_dataset
