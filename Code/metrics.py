"""
Image quality metrics for dehazing evaluation.
  - PSNR  (Peak Signal-to-Noise Ratio)
  - SSIM  (Structural Similarity Index)
Both are computed on CPU numpy arrays and as differentiable torch ops.
"""

import math
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────
# NumPy implementations  (for evaluation loops)
# ──────────────────────────────────────────────

def psnr_np(pred: np.ndarray, target: np.ndarray, max_val: float = 1.0) -> float:
    """
    PSNR in dB.  Both arrays should be float32 in [0, max_val].
    Higher is better.  Typical range for dehazing: 20–35 dB.
    """
    mse = np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    return 20.0 * math.log10(max_val / math.sqrt(mse))


def _ssim_np(
    pred: np.ndarray,
    target: np.ndarray,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> float:
    """Single-channel SSIM."""
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    # Gaussian kernel
    coords = np.arange(window_size) - window_size // 2
    g = np.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    kernel = np.outer(g, g)

    def conv(img):
        from scipy.ndimage import convolve
        return convolve(img.astype(np.float64), kernel, mode='reflect')

    mu1 = conv(pred)
    mu2 = conv(target)
    mu1_sq, mu2_sq, mu12 = mu1**2, mu2**2, mu1*mu2
    s1  = conv(pred**2)   - mu1_sq
    s2  = conv(target**2) - mu2_sq
    s12 = conv(pred*target) - mu12

    num = (2*mu12 + C1) * (2*s12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (s1 + s2 + C2)
    return float(np.mean(num / (den + 1e-10)))


def ssim_np(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """
    Multi-channel SSIM (averages over colour channels).
    Both arrays: float32 HxWxC or HxW.
    """
    if pred.ndim == 2:
        return _ssim_np(pred, target, data_range=data_range)
    scores = [
        _ssim_np(pred[..., c], target[..., c], data_range=data_range)
        for c in range(pred.shape[2])
    ]
    return float(np.mean(scores))


# ──────────────────────────────────────────────
# PyTorch implementations  (for training-time monitoring)
# ──────────────────────────────────────────────

def psnr_torch(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """Batch PSNR; returns mean over batch."""
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])  # [B]
    psnr = 20.0 * torch.log10(torch.tensor(max_val, device=pred.device)) - \
           10.0 * torch.log10(mse + 1e-10)
    return psnr.mean()


# ──────────────────────────────────────────────
# Evaluation helper
# ──────────────────────────────────────────────

class MetricTracker:
    """Running mean of PSNR and SSIM over an evaluation set."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._psnr_sum  = 0.0
        self._ssim_sum  = 0.0
        self._count     = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """pred, target: float32 [B,C,H,W] in [0,1]."""
        B = pred.shape[0]
        for b in range(B):
            p = pred[b].permute(1, 2, 0).cpu().numpy()
            t = target[b].permute(1, 2, 0).cpu().numpy()
            self._psnr_sum += psnr_np(p, t)
            self._ssim_sum += ssim_np(p, t)
        self._count += B

    @property
    def psnr(self) -> float:
        return self._psnr_sum / max(1, self._count)

    @property
    def ssim(self) -> float:
        return self._ssim_sum / max(1, self._count)

    def summary(self) -> dict:
        return {'PSNR': round(self.psnr, 4), 'SSIM': round(self.ssim, 4)}


# ──────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np
    a = np.random.rand(256, 256, 3).astype(np.float32)
    b = a + np.random.randn(256, 256, 3).astype(np.float32) * 0.05
    b = np.clip(b, 0, 1)
    print(f"PSNR: {psnr_np(b, a):.2f} dB")
    print(f"SSIM: {ssim_np(b, a):.4f}")