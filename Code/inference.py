"""
Real-Time Dehazing — Simple & Reliable
=======================================
Default: shows enhanced/sharpened output (safe for any scene)
--dcp    : enables Dark Channel Prior (good for actually hazy images)
--use_cnn: uses trained CNN (only after 50+ epochs of training)

Press Q/ESC to quit, S to save screenshot
"""

import argparse
import threading
import time
import warnings
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

warnings.filterwarnings('ignore')
cv2.setUseOptimized(True)
cv2.setNumThreads(0)


# ──────────────────────────────────────────────
# Mode 1: Simple enhance — safe for ANY scene
# ──────────────────────────────────────────────

def enhance(frame: np.ndarray) -> np.ndarray:
    """
    Sharpening + contrast enhancement.
    Never blows out or adds artifacts. Works on any image.
    """
    # Unsharp mask — sharpens edges
    blur  = cv2.GaussianBlur(frame, (0, 0), 2)
    sharp = cv2.addWeighted(frame, 1.3, blur, -0.3, 0)

    # CLAHE on luminance only — boosts local contrast without colour shift
    lab   = cv2.cvtColor(sharp, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ──────────────────────────────────────────────
# Mode 2: DCP — for actually hazy/foggy images
# ──────────────────────────────────────────────

def dcp_dehaze(frame: np.ndarray, omega: float = 0.85) -> np.ndarray:
    """
    Dark Channel Prior dehazing.
    Use ONLY on images that are actually hazy/foggy.
    On clear images this will over-brighten.
    """
    img  = frame.astype(np.float32) / 255.0
    patch = 9

    # Dark channel
    min_ch = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    dark   = cv2.erode(min_ch, kernel)

    # Atmospheric light — capped to avoid bright windows
    n      = max(int(img.shape[0] * img.shape[1] * 0.001), 1)
    flat_d = dark.flatten()
    flat_i = img.reshape(-1, 3)
    idx    = np.argpartition(flat_d, -n)[-n:]
    A      = np.clip(flat_i[idx].max(axis=0), 0, 0.85).astype(np.float32)

    # Transmission
    norm = np.clip(img / (A + 1e-6), 0, 1)
    min_norm = np.min(norm, axis=2)
    t_coarse = 1.0 - omega * cv2.erode(min_norm, kernel)

    # Guided filter refine
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    r, eps = 20, 1e-3
    ksize  = (2*r+1, 2*r+1)
    mean_g  = cv2.blur(gray, ksize)
    mean_p  = cv2.blur(t_coarse, ksize)
    mean_gp = cv2.blur(gray * t_coarse, ksize)
    mean_gg = cv2.blur(gray * gray, ksize)
    a = (mean_gp - mean_g * mean_p) / (mean_gg - mean_g**2 + eps)
    b = mean_p - a * mean_g
    t = np.clip(cv2.blur(a, ksize) * gray + cv2.blur(b, ksize), 0.2, 1.0)

    # Recover
    J = (img - A) / t[:, :, np.newaxis] + A
    J = np.clip(J, 0, 1)

    out = (J * 255).astype(np.uint8)
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ──────────────────────────────────────────────
# Mode 3: CNN — only after proper training
# ──────────────────────────────────────────────

def best_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def load_cnn(weights_path, base_ch, device, h, w):
    # Use the correct trained model class
    from model import AetherLiftNet

    # Load checkpoint FIRST
    ckpt = torch.load(weights_path, map_location=device)

    # Extract config (to match training)
    cfg = ckpt.get("cfg", {})

    # Build model EXACTLY like training
    net = AetherLiftNet(
        base_ch=cfg.get("base_ch", base_ch),
        num_res_blocks=cfg.get("num_res_blocks", 4)
    )

    # Load weights properly
    net.load_state_dict(ckpt["model"], strict=False)

    net.eval().to(device)

    print("Loaded CNN with config:", cfg)

    return net


def cnn_dehaze(model, frame, h, w, device):
    # Resize FIRST (important)
    resized = cv2.resize(frame, (w, h))

    # Convert BGR → RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Normalize to [0,1]
    img = rgb.astype(np.float32) / 255.0

    # HWC → CHW
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)

    # Postprocess
    out = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    out = np.clip(out, 0, 1)
    out = (out * 255).astype(np.uint8)

    # RGB → BGR
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    # 🔥 Post-processing (critical for visual quality)
    out = enhance(out)

    # Resize back
    return cv2.resize(out, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)


# ──────────────────────────────────────────────
# Camera reader thread
# ──────────────────────────────────────────────

class Camera(threading.Thread):
    def __init__(self, cam_id, w, h):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(cam_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {cam_id}")
        self._frame = None
        self._lock  = threading.Lock()
        self._stop  = threading.Event()

    def run(self):
        while not self._stop.is_set():
            ret, f = self.cap.read()
            if ret:
                with self._lock:
                    self._frame = f

    def read(self):
        with self._lock:
            return self._frame

    def stop(self):
        self._stop.set()
        self.cap.release()


# ──────────────────────────────────────────────
# FPS tracker
# ──────────────────────────────────────────────

class FPS:
    def __init__(self):
        self._t = time.perf_counter()
        self._d = deque(maxlen=30)

    def tick(self):
        now = time.perf_counter()
        self._d.append(now - self._t)
        self._t = now
        return 1.0 / (sum(self._d) / len(self._d)) if self._d else 0.0


# ──────────────────────────────────────────────
# OSD
# ──────────────────────────────────────────────

def draw_osd(left, right, fps, inf_ms, mode_label):
    h = left.shape[0]
    for img in (left, right):
        img[:44, :280] = (img[:44, :280] * 0.4).astype(np.uint8)
    f = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(left,  f"FPS {fps:5.1f}",      (8, 18), f, 0.55, (  0, 255, 120), 2)
    cv2.putText(left,  f"Inf {inf_ms:.1f}ms",  (8, 38), f, 0.55, (  0, 200, 255), 2)
    cv2.putText(right, mode_label,              (8, 28), f, 0.60, (255, 220,   0), 2)
    cv2.putText(left,  "INPUT",   (8, h-10), f, 0.7, ( 80,  80, 255), 2)
    cv2.putText(right, "OUTPUT",  (8, h-10), f, 0.7, ( 80, 255,  80), 2)
    div = np.full((h, 4, 3), [0, 200, 255], dtype=np.uint8)
    return np.hstack([left, div, right])


# ──────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────

def run(cfg):
    # Determine mode
    use_cnn = cfg.use_cnn
    use_dcp = cfg.dcp

    if use_cnn:
        if not Path(cfg.weights).exists():
            print(f"ERROR: weights not found at {cfg.weights}")
            print("Make sure best.pth exists or train the model first.")
            return
        device = best_device()
        h, w   = cfg.infer_size
        print(f"Loading CNN on {device}...")
        model  = load_cnn(cfg.weights, cfg.base_ch, device, h, w)
        mode_label = "CNN"
        print("CNN loaded ✓")
    elif use_dcp:
        mode_label = "DCP dehaze"
        print("Mode: DCP (best for hazy/foggy images)")
    else:
        mode_label = "Enhanced"
        print("Mode: Enhance (sharpening + contrast)")

    print("Opening camera...")
    cam = Camera(cfg.camera_id, cfg.cam_width, cfg.cam_height)
    cam.start()
    time.sleep(0.4)

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    shot_n  = 0
    fps     = FPS()
    inf_ms  = 0.0
    panel   = None

    cv2.namedWindow('AetherLiftNet', cv2.WINDOW_NORMAL)
    print("Running — Q/ESC quit | S save\n")

    try:
        while True:
            frame = cam.read()
            if frame is None:
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break
                time.sleep(0.001)
                continue

            t0 = time.perf_counter()

            if use_cnn:
                output = cnn_dehaze(model, frame, h, w, device)
            elif use_dcp:
                output = dcp_dehaze(frame, omega=cfg.omega)
            else:
                output = enhance(frame)

            inf_ms = (time.perf_counter() - t0) * 1000
            panel  = draw_osd(frame.copy(), output, fps.tick(), inf_ms, mode_label)
            cv2.imshow('AetherLiftNet', panel)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('s') and panel is not None:
                p = save_dir / f'shot_{shot_n:04d}.png'
                cv2.imwrite(str(p), panel)
                print(f"Saved: {p}")
                shot_n += 1

    finally:
        cam.stop()
        cv2.destroyAllWindows()
        print("Done.")


# ──────────────────────────────────────────────
# Image mode
# ──────────────────────────────────────────────

def run_image(cfg):
    import os
    print(f"Loading image: {cfg.image}")
    print("Exists:", os.path.exists(cfg.image))

    img = cv2.imread(cfg.image)

    if img is None:
        raise RuntimeError(f"Failed to load image: {cfg.image} (check path or format)")

    if cfg.use_cnn and Path(cfg.weights).exists():
        device  = best_device()
        h, w    = cfg.infer_size
        model   = load_cnn(cfg.weights, cfg.base_ch, device, h, w)
        output  = cnn_dehaze(model, img, h, w, device)
        label   = 'CNN'
    elif cfg.dcp:
        output  = dcp_dehaze(img, omega=cfg.omega)
        label   = 'DCP'
    else:
        output  = enhance(img)
        label   = 'Enhanced'

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / (Path(cfg.image).stem + f'_{label.lower()}.png')
    cv2.imwrite(str(out_path), output)
    print(f"Saved: {out_path}")

    h   = img.shape[0]
    div = np.full((h, 4, 3), [0, 200, 255], dtype=np.uint8)
    panel = np.hstack([img, div, output])
    cv2.putText(panel, 'Input',  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
    cv2.putText(panel, label,    (img.shape[1]+14, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
    if panel.shape[1] > 1400:
        s = 1400 / panel.shape[1]
        panel = cv2.resize(panel, (1400, int(panel.shape[0] * s)))
    cv2.imshow('Result', panel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--image',      default=None,              type=str)
    p.add_argument('--weights',    default='models/best.pth', type=str)
    p.add_argument('--use_cnn',    action='store_true',       help='Use CNN (needs 50+ epochs training)')
    p.add_argument('--dcp',        action='store_true',       help='Use DCP (for actually hazy images)')
    p.add_argument('--camera_id',  default=0,                 type=int)
    p.add_argument('--cam_width',  default=1280,              type=int)
    p.add_argument('--cam_height', default=720,               type=int)
    p.add_argument('--omega',      default=0.85,              type=float)
    p.add_argument('--infer_size', default=[160, 160],        type=int, nargs=2, metavar=('H','W'))
    p.add_argument('--base_ch',    default=48,                type=int)
    p.add_argument('--save_dir',   default='results/samples', type=str)
    return p.parse_args()


if __name__ == '__main__':
    cfg = parse_args()
    if cfg.image:
        run_image(cfg)
    else:
        run(cfg)