"""
Loss functions for image dehazing:
  - PerceptualLoss  (VGG16 feature matching)
  - SSIMLoss
  - CharbonnierLoss (robust L1)
  - CombinedLoss    (weighted sum used in training)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ──────────────────────────────────────────────
# Charbonnier (smooth L1 variant)
# ──────────────────────────────────────────────

class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps2))


# ──────────────────────────────────────────────
# SSIM Loss
# ──────────────────────────────────────────────

class SSIMLoss(nn.Module):
    """Differentiable SSIM loss (1 - SSIM)."""

    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        kernel = self._gaussian_kernel(window_size, sigma)
        # [1, 1, K, K] → broadcast over channels
        self.register_buffer('kernel', kernel.unsqueeze(0).unsqueeze(0))
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    @staticmethod
    def _gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return torch.outer(g, g)

    def _apply_kernel(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Process each channel independently
        x_flat = x.view(B * C, 1, H, W)
        pad = self.window_size // 2
        out = F.conv2d(x_flat, self.kernel, padding=pad)
        return out.view(B, C, H, W)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mu1 = self._apply_kernel(pred)
        mu2 = self._apply_kernel(target)

        mu1_sq  = mu1 * mu1
        mu2_sq  = mu2 * mu2
        mu12    = mu1 * mu2

        sigma1_sq = self._apply_kernel(pred * pred)   - mu1_sq
        sigma2_sq = self._apply_kernel(target * target) - mu2_sq
        sigma12   = self._apply_kernel(pred * target)  - mu12

        ssim_map = (
            (2 * mu12 + self.C1) * (2 * sigma12 + self.C2)
        ) / (
            (mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2)
        )
        return 1.0 - ssim_map.mean()


# ──────────────────────────────────────────────
# Perceptual (VGG) Loss
# ──────────────────────────────────────────────

class PerceptualLoss(nn.Module):
    """
    Feature matching on VGG16 relu2_2 and relu3_3 layers.
    Frozen — only used for gradient computation towards dehaze model.
    """

    VGG_LAYERS = {
        'relu2_2': 9,
        'relu3_3': 16,
    }
    WEIGHTS = {'relu2_2': 1.0, 'relu3_3': 0.5}

    # ImageNet normalisation
    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.slices = nn.ModuleDict()
        prev = 0
        for name, end in self.VGG_LAYERS.items():
            self.slices[name] = nn.Sequential(*list(vgg.children())[prev:end])
            prev = end
        for p in self.parameters():
            p.requires_grad = False

    def _normalise(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.MEAN.to(x.device)
        std  = self.STD.to(x.device)
        return (x - mean) / std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_n   = self._normalise(pred)
        target_n = self._normalise(target)
        loss = 0.0
        feat_p = pred_n
        feat_t = target_n
        for name, layer in self.slices.items():
            feat_p = layer(feat_p)
            feat_t = layer(feat_t)
            loss  += self.WEIGHTS[name] * F.l1_loss(feat_p, feat_t.detach())
        return loss


# ──────────────────────────────────────────────
# Combined training loss
# ──────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """
    L_total = w_char * L_char  +  w_ssim * L_ssim  +  w_perc * L_perc

    Default weights balance reconstruction fidelity + perceptual quality.
    """

    def __init__(
        self,
        w_char: float = 1.0,
        w_ssim: float = 0.3,
        w_perc: float = 0.1,
        use_perceptual: bool = True,
    ):
        super().__init__()
        self.w_char = w_char
        self.w_ssim = w_ssim
        self.w_perc = w_perc if use_perceptual else 0.0

        self.char_loss = CharbonnierLoss()
        self.ssim_loss = SSIMLoss()
        self.perc_loss = PerceptualLoss() if use_perceptual else None

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        l_char = self.char_loss(pred, target)
        l_ssim = self.ssim_loss(pred, target)

        total = self.w_char * l_char + self.w_ssim * l_ssim

        components = {'char': l_char.item(), 'ssim': l_ssim.item()}

        if self.perc_loss is not None and self.w_perc > 0:
            l_perc = self.perc_loss(pred, target)
            total += self.w_perc * l_perc
            components['perc'] = l_perc.item()

        components['total'] = total.item()
        return total, components


from typing import Tuple  # noqa: E402 (placed at bottom for readability)