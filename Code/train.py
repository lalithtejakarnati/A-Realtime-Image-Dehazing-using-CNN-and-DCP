"""
AetherLiftNet Training Pipeline
- AMP (CUDA only, disabled on MPS/CPU for stability)
- OneCycleLR scheduler
- Gradient clipping
- Best checkpoint saving
- TensorBoard + CSV logging
"""

import csv, time, argparse, logging
from pathlib import Path
from typing import Dict

import warnings
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings('ignore', category=FutureWarning)

from model   import AetherLiftNet
from dataset import get_loaders
from losses  import CombinedLoss
from metrics import MetricTracker


def best_device():
    if torch.backends.mps.is_available(): return torch.device('mps')
    if torch.cuda.is_available():         return torch.device('cuda')
    return torch.device('cpu')


def setup_logger(log_dir):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log = logging.getLogger('AetherLiftNet')
    log.setLevel(logging.INFO)
    if log.handlers: log.handlers.clear()
    fmt = logging.Formatter('%(asctime)s | %(message)s', '%H:%M:%S')
    for h in [logging.StreamHandler(),
              logging.FileHandler(Path(log_dir) / 'train.log')]:
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


def train_epoch(model, loader, criterion, optimizer, scaler, device, log):
    model.train()
    running, n = {}, 0
    use_amp = (device.type == 'cuda')
    for step, (hazy, clear) in enumerate(loader):
        hazy  = hazy.to(device,  non_blocking=True)
        clear = clear.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            loss, comps = criterion(model(hazy), clear)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        for k, v in comps.items():
            running[k] = running.get(k, 0) + v
        n += 1
        if step % 50 == 0:
            log.info(f"  step {step:4d}/{len(loader)}  " +
                     "  ".join(f"{k}={v/n:.4f}" for k, v in running.items()))
    return {k: v/n for k, v in running.items()}


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    tracker, running, n = MetricTracker(), {}, 0
    use_amp = (device.type == 'cuda')
    for hazy, clear in loader:
        hazy  = hazy.to(device,  non_blocking=True)
        clear = clear.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            pred = model(hazy)
            _, comps = criterion(pred, clear)
        tracker.update(pred.float(), clear.float())
        for k, v in comps.items():
            running[k] = running.get(k, 0) + v
        n += 1
    metrics = {k: v/n for k, v in running.items()}
    metrics.update(tracker.summary())
    return metrics


def save_ckpt(state, path):
    torch.save(state, path)


def load_ckpt(path, model, optimizer=None, scaler=None):
    ckpt  = torch.load(path, map_location='cpu', weights_only=False)
    state = ckpt.get('model', ckpt.get('model_state_dict', ckpt)) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=False)
    if optimizer and 'optimizer' in ckpt:
        try: optimizer.load_state_dict(ckpt['optimizer'])
        except: pass
    if scaler and 'scaler' in ckpt:
        try: scaler.load_state_dict(ckpt['scaler'])
        except: pass
    return ckpt.get('epoch', 0), ckpt.get('best_psnr', 0.0)


def train(cfg):
    device = best_device()
    log    = setup_logger(cfg.log_dir)
    log.info(f"Device : {device}")
    log.info(f"Config : {vars(cfg)}")

    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=cfg.log_dir)

    train_loader, val_loader = get_loaders(
        data_root=cfg.data_root, patch_size=cfg.patch_size,
        batch_size=cfg.batch_size, num_workers=cfg.num_workers,
        synthetic_haze=cfg.synthetic_haze, max_samples=cfg.max_samples,
    )
    # 🚨 Safety check: ensure dataset is not empty
    if len(train_loader) == 0:
        raise RuntimeError(
            f"No training batches found for data_root='{cfg.data_root}'. "
            "Expected structure: <root>/hazy and <root>/clear"
        )
    log.info(f"Train:{len(train_loader)} batches  Val:{len(val_loader)} batches")

    model  = AetherLiftNet(base_ch=cfg.base_ch, num_res_blocks=cfg.num_res_blocks).to(device)
    params = sum(p.numel() for p in model.parameters())
    log.info(f"Params : {params/1e6:.2f}M")

    criterion = CombinedLoss(
        w_char=cfg.w_char, w_ssim=cfg.w_ssim,
        w_perc=cfg.w_perc, use_perceptual=cfg.use_perceptual,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr, steps_per_epoch=len(train_loader),
        epochs=cfg.epochs, pct_start=0.1, anneal_strategy='cos',
    )
    use_amp = (device.type == 'cuda')
    scaler  = torch.amp.GradScaler(device.type, enabled=use_amp)

    start_epoch, best_psnr = 0, 0.0
    if cfg.resume and Path(cfg.resume).exists():
        start_epoch, best_psnr = load_ckpt(cfg.resume, model, optimizer, scaler)
        log.info(f"Resumed epoch {start_epoch}  best PSNR={best_psnr:.2f}dB")
    else:
        log.info("Starting from scratch")

    csv_file   = open(Path(cfg.log_dir) / 'metrics.csv', 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'train_loss', 'val_loss', 'PSNR', 'SSIM', 'lr'])

    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()
        lr = scheduler.get_last_lr()[0]
        log.info(f"\n{'='*55}\nEpoch {epoch+1}/{cfg.epochs}  lr={lr:.2e}")

        tr = train_epoch(model, train_loader, criterion, optimizer, scaler, device, log)
        scheduler.step()
        vl = val_epoch(model, val_loader, criterion, device)

        log.info(
            f"  train={tr['total']:.4f}  val={vl['total']:.4f}"
            f"  PSNR={vl['PSNR']:.2f}dB  SSIM={vl['SSIM']:.4f}"
            f"  time={time.time()-t0:.1f}s"
        )

        writer.add_scalars('Loss', {'train': tr['total'], 'val': vl['total']}, epoch)
        writer.add_scalar('PSNR', vl['PSNR'], epoch)
        writer.add_scalar('SSIM', vl['SSIM'], epoch)

        csv_writer.writerow([
            epoch+1, round(tr['total'],6), round(vl['total'],6),
            round(vl['PSNR'],4), round(vl['SSIM'],4), lr
        ])
        csv_file.flush()

        if vl['PSNR'] > best_psnr:
            best_psnr = vl['PSNR']
            save_ckpt({
                'epoch': epoch+1, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'best_psnr': best_psnr, 'cfg': vars(cfg),
            }, str(Path(cfg.save_dir) / 'best.pth'))
            log.info(f"  ✓ Best PSNR {best_psnr:.2f}dB → best.pth")

        if (epoch+1) % cfg.save_every == 0:
            save_ckpt({
                'epoch': epoch+1, 'model': model.state_dict(), 'best_psnr': best_psnr,
            }, str(Path(cfg.save_dir) / f'epoch_{epoch+1:04d}.pth'))

    csv_file.close()
    writer.close()
    log.info(f"\nDone. Best PSNR: {best_psnr:.2f}dB")


def parse_args():
    p = argparse.ArgumentParser(description='Train AetherLiftNet')
    # Default now points directly to training folder to avoid path mistakes
    p.add_argument('--data_root',      default='data/train',    type=str)
    p.add_argument('--save_dir',       default='models',  type=str)
    p.add_argument('--log_dir',        default='logs',    type=str)
    p.add_argument('--epochs',         default=40,       type=int)
    p.add_argument('--batch_size',     default=4,         type=int)
    p.add_argument('--patch_size',     default=160,       type=int)
    p.add_argument('--lr',             default=3e-4,      type=float)
    p.add_argument('--weight_decay',   default=1e-4,      type=float)
    p.add_argument('--base_ch',        default=48,        type=int)
    p.add_argument('--num_res_blocks', default=4,         type=int)
    p.add_argument('--w_char',         default=1.0,       type=float)
    p.add_argument('--w_ssim',         default=0.6,       type=float)
    p.add_argument('--w_perc',         default=0.2,       type=float)
    p.add_argument('--use_perceptual', default=True,     action=argparse.BooleanOptionalAction)
    p.add_argument('--synthetic_haze', default=True,      action=argparse.BooleanOptionalAction)
    p.add_argument('--num_workers',    default=0,         type=int)
    p.add_argument('--save_every',     default=10,        type=int)
    p.add_argument('--max_samples',    default=3000,      type=int)
    p.add_argument('--resume',         default=None,      type=str)
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())