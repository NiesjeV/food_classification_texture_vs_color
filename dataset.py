"""
dataset.py — Food Recognition Dataset
Supports three input streams:
  'color'   : RGB with heavy Gaussian blur (texture destroyed)
  'texture' : Grayscale (color destroyed)
  'full'    : Normal RGB (baseline)
"""

import os
import csv
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


# ---------------------------------------------------------------------------
# Per-stream transforms
# ---------------------------------------------------------------------------

IMG_SIZE = 128

def get_transform(stream: str, augment: bool = False):
    """Return the transform pipeline for a given stream."""
    assert stream in ("color", "texture", "full", "scrambled"), f"Unknown stream: {stream}"

    base_aug = [
        T.RandomHorizontalFlip(),
        T.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.RandomRotation(15),
    ] if augment else [
        T.Resize(IMG_SIZE + 16),
        T.CenterCrop(IMG_SIZE),
    ]
    if stream == "color":
        # Keep hue/saturation, destroy fine texture via blur
        return T.Compose([
            *base_aug,
            T.GaussianBlur(kernel_size=15, sigma=(4.0, 6.0)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    elif stream == "texture":
        # Destroy color, keep edges/texture
        return T.Compose([
            *base_aug,
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])

    else:  # full
        extra = [T.RandomGrayscale(p=0.05)] if augment else []
        return T.Compose([
            *base_aug,
            *extra,
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class FoodDataset(Dataset):
    """
    Expects:
      root/
        train/  (or test/)
          <img_name>.jpg  ...
      train_labels.csv   — columns: img_name, label
    """

    def __init__(self, root: str, split: str = "train",
                 stream: str = "full", augment: bool = False):
        self.root = Path(root)
        self.split = split
        self.stream = stream
        self.transform = get_transform(stream, augment=(augment and split == "train"))

        self.samples = []  # list of (img_path, label)

        if split in ("train", "val"):
            label_file = self.root / "train_labels.csv"
            img_dir    = self.root / "train_set"
            self._load_labeled(label_file, img_dir)
        else:
            img_dir = self.root / "test_set"
            self._load_unlabeled(img_dir)

    def _load_labeled(self, label_file: Path, img_dir: Path):
        with open(label_file, newline="") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames
            # Auto-detect column names (handle img_name / image / filename etc.)
            img_col   = next((c for c in cols if "img"   in c.lower()
                              or "file" in c.lower() or "name" in c.lower()), cols[0])
            label_col = next((c for c in cols if "label" in c.lower()
                              or "class" in c.lower() or "cat"  in c.lower()), cols[1])
            for row in reader:
                img_path = img_dir / row[img_col]
                if img_path.exists():
                    self.samples.append((img_path, int(row[label_col])))

    def _load_unlabeled(self, img_dir: Path):
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                self.samples.append((p, -1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        return x, label, str(img_path.name)


# ---------------------------------------------------------------------------
# Multi-stream dataset (returns all three inputs simultaneously)
# ---------------------------------------------------------------------------

class MultiStreamFoodDataset(Dataset):
    """
    Returns (x_color, x_texture, x_full, label, img_name) per sample.
    Used for the fused model training.
    """

    def __init__(self, root: str, split: str = "train", augment: bool = False):
        self.datasets = {
            s: FoodDataset(root, split, stream=s, augment=augment)
            for s in ("color", "texture", "full")
        }
        # All three share the same file list
        assert len(set(len(d) for d in self.datasets.values())) == 1, \
            "Stream datasets have different sizes"

    def __len__(self):
        return len(self.datasets["full"])

    def __getitem__(self, idx):
        x_c, label, name = self.datasets["color"][idx]
        x_t, _,     _    = self.datasets["texture"][idx]
        x_f, _,     _    = self.datasets["full"][idx]
        return x_c, x_t, x_f, label, name


# ---------------------------------------------------------------------------
# Convenience: train/val split from labeled data
# ---------------------------------------------------------------------------

def split_dataset(dataset: FoodDataset, val_frac: float = 0.15, seed: int = 42):
    """Return train and val subsets."""
    from torch.utils.data import Subset, random_split
    n = len(dataset)
    n_val = int(n * val_frac)
    n_train = n - n_val
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val], generator=generator)


def make_loaders(root: str, stream: str = "full",
                 batch_size: int = 64, num_workers: int = 4,
                 val_frac: float = 0.15):
    """One-liner to get train/val DataLoaders for a single stream."""
    full_ds = FoodDataset(root, split="train", stream=stream, augment=True)
    train_ds, val_ds = split_dataset(full_ds, val_frac=val_frac)

    # Val set should not augment — patch its transform
    val_ds.dataset.transform = get_transform(stream, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def make_multistream_loaders(root: str, batch_size: int = 64,
                              num_workers: int = 4, val_frac: float = 0.15):
    """Train/val DataLoaders for the fused multi-stream model."""
    full_ds = MultiStreamFoodDataset(root, split="train", augment=True)
    from torch.utils.data import random_split
    n = len(full_ds)
    n_val = int(n * val_frac)
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(full_ds, [n - n_val, n_val], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader