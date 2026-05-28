"""
Dataset utilities for image dehazing.
Supports: RESIDE-Indoor (ITS), RESIDE-Outdoor (OTS), custom paired data.
"""

import os
import random
from pathlib import Path
from typing import Tuple, Optional, Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF


# ──────────────────────────────────────────────
# Synthetic Haze Augmentation  (online synthesis)
# ──────────────────────────────────────────────

def add_haze(
    img: np.ndarray,          # float32 [0,1] BGR
    beta: float = None,       # scattering coefficient
    A: float = None,          # atmospheric light
) -> np.ndarray:
    """
    ASM (Atmospheric Scattering Model):
        I(x) = J(x)*t(x) + A*(1 - t(x))
        t(x) = exp(-beta * d(x))
    Uses a smooth random depth map to create realistic haze gradients.
    """
    h, w = img.shape[:2]
    beta = beta if beta is not None else random.uniform(0.5, 2.5)
    A    = A    if A    is not None else random.uniform(0.7, 1.0)

    # Smooth depth map (gradient + noise)
    y_lin  = np.linspace(0.5, 1.5, h)
    x_lin  = np.linspace(0.8, 1.2, w)
    depth  = np.outer(y_lin, x_lin)                    # [h, w]
    noise  = cv2.GaussianBlur(
        np.random.rand(h, w).astype(np.float32), (99, 99), 0
    )
    depth  = (depth + 0.3 * noise).astype(np.float32)
    depth /= depth.max()

    t = np.exp(-beta * depth)[..., np.newaxis]         # [h, w, 1]
    hazy = img * t + A * (1 - t)
    return np.clip(hazy, 0, 1).astype(np.float32)


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class DehazeDataset(Dataset):
    """
    Expects a directory layout:
        root/
          hazy/   *.png / *.jpg
          clear/  *.png / *.jpg   (same filenames)

    If clear/ doesn't exist or `synthetic_haze=True`,
    images from hazy/ are treated as clear, and haze is
    synthesised on-the-fly.
    """

    IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

    def __init__(
        self,
        root: str,
        patch_size: int = 256,
        augment: bool = True,
        synthetic_haze: bool = False,
        beta_range: Tuple[float, float] = (0.5, 2.5),
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        root = Path(root)
        self.patch_size    = patch_size
        self.augment       = augment
        self.synthetic     = synthetic_haze
        self.beta_range    = beta_range

        hazy_dir  = root / 'hazy'
        clear_dir = root / 'clear'

        if self.synthetic or not clear_dir.exists():
            # Use all images as clear → synthesise haze
            src_dir = hazy_dir if hazy_dir.exists() else root
            self.clear_paths = sorted(
                p for p in src_dir.iterdir()
                if p.suffix.lower() in self.IMG_EXTS
            )
            self.hazy_paths = None
        else:
            self.hazy_paths = sorted(
                p for p in hazy_dir.iterdir()
                if p.suffix.lower() in self.IMG_EXTS
            )
            self.clear_paths = [clear_dir / p.name for p in self.hazy_paths]

        if max_samples:
            self.clear_paths = self.clear_paths[:max_samples]
            if self.hazy_paths:
                self.hazy_paths = self.hazy_paths[:max_samples]

        # 🔍 Debug info
        print(f"[Dataset] Total clear images: {len(self.clear_paths)}")
        if self.hazy_paths is not None:
            print(f"[Dataset] Total hazy images: {len(self.hazy_paths)}")

        # 🚨 Hard check to prevent silent failure
        if len(self.clear_paths) == 0:
            raise RuntimeError(
                f"No training images found in {root}. "
                f"Check folder structure: {root}/hazy and {root}/clear"
            )

    # ── helpers ───────────────────────────────
    def _load(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    def _random_crop(self, *imgs):
        h, w = imgs[0].shape[:2]
        ps = self.patch_size
        if h < ps or w < ps:
            imgs = tuple(
                cv2.resize(im, (max(w, ps), max(h, ps))) for im in imgs
            )
            h, w = imgs[0].shape[:2]
        top  = random.randint(0, h - ps)
        left = random.randint(0, w - ps)
        return tuple(im[top:top+ps, left:left+ps] for im in imgs)

    def _augment(self, hazy, clear):
        # Horizontal flip
        if random.random() > 0.5:
            hazy, clear = hazy[:, ::-1], clear[:, ::-1]
        # Vertical flip
        if random.random() > 0.3:
            hazy, clear = hazy[::-1], clear[::-1]
        # 90° rotation
        k = random.randint(0, 3)
        if k:
            hazy = np.rot90(hazy, k).copy()
            clear = np.rot90(clear, k).copy()
        return hazy, clear

    @staticmethod
    def _to_tensor(img: np.ndarray) -> torch.Tensor:
        # HWC → CHW
        return torch.from_numpy(img.transpose(2, 0, 1)).float()

    # ── Dataset API ───────────────────────────
    def __len__(self):
        return len(self.clear_paths)

    def __getitem__(self, idx: int):
        clear = self._load(self.clear_paths[idx])

        if self.synthetic or self.hazy_paths is None:
            beta  = random.uniform(*self.beta_range)
            hazy  = add_haze(clear, beta=beta)
        else:
            hazy  = self._load(self.hazy_paths[idx])

        # Patch crop
        hazy, clear = self._random_crop(hazy, clear)

        # Augmentation
        if self.augment:
            hazy, clear = self._augment(hazy, clear)

        hazy  = np.ascontiguousarray(hazy)
        clear = np.ascontiguousarray(clear)

        return self._to_tensor(hazy), self._to_tensor(clear)


# ──────────────────────────────────────────────
# DataLoader factories
# ──────────────────────────────────────────────

def get_loaders(
    data_root: str,
    patch_size: int = 256,
    batch_size: int = 8,
    num_workers: int = 4,
    val_split: float = 0.1,
    synthetic_haze: bool = True,
    max_samples: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:

    full_ds = DehazeDataset(
        data_root,
        patch_size=patch_size,
        augment=True,
        synthetic_haze=synthetic_haze,
        max_samples=max_samples,
    )

    n_val   = max(1, int(len(full_ds) * val_split))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    # val augmentation OFF
    val_ds.dataset.augment = False

    loader_kw = dict(
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,  **loader_kw
    )
    val_loader = DataLoader(
        val_ds,   batch_size=batch_size, shuffle=False, **loader_kw
    )

    return train_loader, val_loader


# ──────────────────────────────────────────────
# Demo: generate a synthetic hazy sample
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Create a dummy clear image (gradient)
    clear = np.zeros((256, 256, 3), dtype=np.float32)
    clear[..., 0] = np.linspace(0, 1, 256)
    clear[..., 2] = np.linspace(1, 0, 256)

    hazy = add_haze(clear, beta=1.5, A=0.9)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(clear); axes[0].set_title("Clear"); axes[0].axis('off')
    axes[1].imshow(hazy);  axes[1].set_title("Hazy (synthetic)"); axes[1].axis('off')
    plt.tight_layout()
    plt.savefig("synthetic_haze_demo.png", dpi=120)
    print("Saved synthetic_haze_demo.png")