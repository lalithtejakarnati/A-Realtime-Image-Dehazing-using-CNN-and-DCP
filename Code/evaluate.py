import argparse
import csv
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from contextlib import nullcontext

from model   import AetherLiftNet
from dataset import DehazeDataset, add_haze
from metrics import MetricTracker, psnr_np, ssim_np


# Inference helpers

@torch.no_grad()
def infer_image(model, img_bgr: np.ndarray, device, tile_size: int = 512) -> np.ndarray:
    """
    Tile-based inference for arbitrary-resolution images.
    Falls back to full-image if smaller than tile_size.
    """
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    if h <= tile_size and w <= tile_size:
        t = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
        with nullcontext():
            out = model(t)
        out_np = out.squeeze(0).float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        return cv2.cvtColor((out_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # Tiled
    result = np.zeros_like(rgb)
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = rgb[y:y+tile_size, x:x+tile_size]
            t = torch.from_numpy(tile.transpose(2, 0, 1)).unsqueeze(0).to(device)
            with nullcontext():
                out = model(t)
            result[y:y+tile_size, x:x+tile_size] = \
                out.squeeze(0).float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()

    return cv2.cvtColor((result * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


# Evaluation on dataset

def evaluate_dataset(model, data_root: str, device, n_samples: int = 200):
    ds = DehazeDataset(data_root, patch_size=256, augment=False, synthetic_haze=False,
                       max_samples=n_samples)
    print(f"[DEBUG] Using dataset path: {data_root}")
    tracker = MetricTracker()
    loader  = torch.utils.data.DataLoader(ds, batch_size=16, num_workers=0)

    model.eval()
    for hazy, clear in loader:
        hazy, clear = hazy.to(device), clear.to(device)
        with torch.inference_mode():
            pred = model(hazy)
        tracker.update(pred, clear)

    # handle different MetricTracker implementations safely
    if hasattr(tracker, "result"):
        return tracker.result()
    elif hasattr(tracker, "summary"):
        return tracker.summary()
    elif hasattr(tracker, "avg"):
        # assume keys exist
        return {
            "PSNR": tracker.avg("PSNR"),
            "SSIM": tracker.avg("SSIM")
        }
    else:
        raise RuntimeError("MetricTracker has no compatible output method")

# Visualisation: comparison grid

def visualise_samples(model, data_root: str, device, save_path: str, n: int = 4):
    ds = DehazeDataset(data_root, patch_size=256, augment=False, synthetic_haze=True,
                       max_samples=n)
    model.eval()

    fig = plt.figure(figsize=(16, 4 * n), facecolor='#0e0e14')
    gs  = gridspec.GridSpec(n, 4, figure=fig, hspace=0.04, wspace=0.03)

    col_titles = ['Hazy Input', 'Network Output', 'Ground Truth', 'Diff ×4']
    for col, title in enumerate(col_titles):
        ax = fig.add_subplot(gs[0, col])
        ax.set_title(title, color='white', fontsize=13, fontweight='bold', pad=8)
        ax.axis('off')

    for row in range(n):
        hazy_t, clear_t = ds[row]
        hazy_t  = hazy_t.unsqueeze(0).to(device)
        clear_t = clear_t.unsqueeze(0).to(device)

        with torch.no_grad(), nullcontext():
            pred_t = model(hazy_t)

        def t2np(t):
            return t.squeeze(0).float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()

        hazy_np  = t2np(hazy_t)
        pred_np  = t2np(pred_t)
        clear_np = t2np(clear_t)
        diff_np  = np.clip(np.abs(pred_np - clear_np) * 4, 0, 1)

        p  = psnr_np(pred_np, clear_np)
        ss = ssim_np(pred_np, clear_np)

        images = [hazy_np, pred_np, clear_np, diff_np]
        for col, img in enumerate(images):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(img)
            if col == 1:
                ax.set_xlabel(f'PSNR {p:.2f} dB  SSIM {ss:.3f}',
                              color='#7fffb2', fontsize=10)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=col == 1)
            for spine in ax.spines.values():
                spine.set_edgecolor('#333')

    plt.savefig(save_path, dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved comparison grid: {save_path}")

# Visualisation: training curves


def plot_training_curves(csv_path: str, save_dir: str):
    epochs, train_loss, val_loss, psnr_vals, ssim_vals = [], [], [], [], []

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            train_loss.append(float(row['train_loss']))
            val_loss.append(float(row['val_loss']))
            psnr_vals.append(float(row['PSNR']))
            ssim_vals.append(float(row['SSIM']))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor='#0e0e14')
    for ax in axes:
        ax.set_facecolor('#1a1a24')
        ax.tick_params(colors='#aaa')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333')

    # Loss
    axes[0].plot(epochs, train_loss, color='#ff6b6b', lw=2, label='Train')
    axes[0].plot(epochs, val_loss,   color='#4ecdc4', lw=2, label='Val')
    axes[0].set_title('Loss', color='white', fontsize=13)
    axes[0].legend(facecolor='#1a1a24', labelcolor='white')
    axes[0].set_xlabel('Epoch', color='#aaa')

    # PSNR
    axes[1].plot(epochs, psnr_vals, color='#7fff7f', lw=2)
    axes[1].set_title('PSNR (dB)', color='white', fontsize=13)
    axes[1].set_xlabel('Epoch', color='#aaa')
    axes[1].set_ylabel('dB', color='#aaa')

    # SSIM
    axes[2].plot(epochs, ssim_vals, color='#ffcd60', lw=2)
    axes[2].set_title('SSIM', color='white', fontsize=13)
    axes[2].set_xlabel('Epoch', color='#aaa')

    plt.tight_layout()
    out = Path(save_dir) / 'training_curves.png'
    plt.savefig(str(out), dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved training curves: {out}")


# Single image evaluation

def evaluate_single(model, img_path: str, device, save_dir: str, beta: float = 1.5):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    hazy_rgb = add_haze(rgb, beta=beta, A=0.85)
    hazy_bgr = cv2.cvtColor((hazy_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    dehazed_bgr = infer_image(model, hazy_bgr, device)

    out_path = Path(save_dir) / (Path(img_path).stem + '_dehazed.png')
    cv2.imwrite(str(out_path), dehazed_bgr)

    p  = psnr_np(
        cv2.cvtColor(dehazed_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255,
        rgb
    )
    ss = ssim_np(
        cv2.cvtColor(dehazed_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255,
        rgb
    )
    print(f"Single image: PSNR={p:.2f} dB  SSIM={ss:.4f}  → {out_path}")
    return p, ss


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate AetherLiftNet')
    p.add_argument('--weights',     default='models/best.pth', type=str)
    p.add_argument('--data_root',   default='data/val',        type=str)
    p.add_argument('--results_dir', default='results',         type=str)
    p.add_argument('--base_ch',     default=48,                type=int)
    p.add_argument('--n_samples',   default=200,               type=int)
    p.add_argument('--n_vis',       default=4,                 type=int)
    p.add_argument('--csv_path',    default='logs/metrics.csv',type=str)
    p.add_argument('--image',       default=None,              type=str,
                   help='Path to a single image for quick eval')
    return p.parse_args()


if __name__ == '__main__':
    cfg    = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # match architecture used during training (base_ch=48)
    model = AetherLiftNet(base_ch=cfg.base_ch).to(device).eval()
    if Path(cfg.weights).exists():
        ckpt = torch.load(cfg.weights, map_location='cpu')
        model.load_state_dict(ckpt.get('model', ckpt), strict=False)
        print(f"Loaded: {cfg.weights}")

    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if cfg.image:
        evaluate_single(model, cfg.image, device, str(results_dir / 'samples'))
    else:
        print("Evaluating on dataset...")
        metrics = evaluate_dataset(model, cfg.data_root, device, cfg.n_samples)
        print(f"\nDataset metrics: {metrics}")

        print("\nGenerating visual comparisons...")
        visualise_samples(model, cfg.data_root, device,
                          str(results_dir / 'plots' / 'comparison_grid.png'),
                          n=cfg.n_vis)

        if Path(cfg.csv_path).exists():
            plot_training_curves(cfg.csv_path, str(results_dir / 'plots'))