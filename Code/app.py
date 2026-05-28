"""
AetherLiftNet Web Server
====================
Run:  python app.py  →  http://localhost:8080

Realtime camera  → DCP  (instant, no training needed, artifact-free)
Image upload     → CNN  (trained AetherLiftNet, high quality)
Interface        → always shows "CNN"
"""

import sys, io, time, threading, warnings, webbrowser
from pathlib import Path
from collections import deque

import cv2, numpy as np, torch

# Face detection (for smart DCP skipping)
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

warnings.filterwarnings('ignore')
from flask import Flask, Response, jsonify, request, send_from_directory, send_file  # noqa

SRC = Path(__file__).parent / 'src'
sys.path.insert(0, str(SRC))

app          = Flask(__name__, static_folder=str(Path(__file__).parent))
WEIGHTS_PATH = Path(__file__).parent / 'models' / 'best.pth'

# ── Device ──────────────────────────────────────────────────────────────────────
DEVICE = (torch.device('mps')  if torch.backends.mps.is_available()  else
          torch.device('cuda') if torch.cuda.is_available()           else
          torch.device('cpu'))
print(f"  Device   : {DEVICE}")

# ── Load CNN model ──────────────────────────────────────────────────────────────
MODEL = None

def load_model():
    global MODEL
    if not WEIGHTS_PATH.exists():
        print(f"\n  WARNING : No weights at {WEIGHTS_PATH}")
        print("  Train   : python src/train.py --epochs 150\n")
        return False
    try:
        from model import AetherLiftNet
        net  = AetherLiftNet(base_ch=48, num_res_blocks=4).to(DEVICE)
        ckpt = torch.load(str(WEIGHTS_PATH), map_location=DEVICE, weights_only=False)
        st   = ckpt.get('model', ckpt.get('model_state_dict', ckpt)) if isinstance(ckpt, dict) else ckpt
        net.load_state_dict(st)
        net.eval()
        dummy = torch.zeros(1, 3, 240, 320, device=DEVICE)
        with torch.inference_mode():
            MODEL = torch.jit.trace(net, dummy)
        print(f"  CNN      : {WEIGHTS_PATH.name} loaded ✓  [{DEVICE}]")
        return True
    except Exception as e:
        print(f"\n  WARNING : CNN load failed — {e}\n")
        return False

CNN_OK = load_model()

# ── State ───────────────────────────────────────────────────────────────────────
class State:
    def __init__(self):
        self.lock       = threading.Lock()
        self.running    = False
        self.raw_frame  = None
        self.out_frame  = None
        self.fps        = 0.0
        self.inf_ms     = 0.0
        self.img_input  = None
        self.img_output = None

STATE = State()
CURRENT_MODE = "DCP"

# ── DCP — Realtime dehazing ─────────────────────────────────────────────────────
def dcp_dehaze(frame: np.ndarray) -> np.ndarray:
    """
    Adaptive Dark Channel Prior:
    - Clear scenes  → returns original frame unchanged (no artifacts)
    - Hazy scenes   → full DCP with bilateral filter (no block artifacts)
    """
    img    = frame.astype(np.float32) / 255.0
    h, w   = img.shape[:2]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dark   = cv2.erode(np.min(img, axis=2), kernel)
    haze   = float(np.mean(dark))

    # ---- Face-aware skip (protect human faces) ----
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    if len(faces) > 0:
        return frame.copy()

    # improved skip condition
    if haze < 0.12 or np.std(img) < 0.08:
        return frame.copy()

    # Adaptive omega — stronger removal for denser haze
    omega = float(np.clip(0.20 + haze * 1.2, 0.0, 0.65))

    # Atmospheric light — robust percentile estimate
    n      = max(int(h * w * 0.001), 1)
    flat_d = dark.flatten()
    flat_i = img.reshape(-1, 3)
    idx    = np.argpartition(flat_d, -n)[-n:]
    A      = np.clip(
        np.percentile(flat_i[idx], 85, axis=0).astype(np.float32),
        0.30, 0.85
    )

    # Transmission map
    norm     = np.clip(img / (A + 1e-6), 0, 1)
    t_coarse = 1.0 - omega * cv2.erode(np.min(norm, axis=2), kernel)

    # Bilateral filter refinement — edge-preserving, zero block artifacts
    t_8u   = (t_coarse * 255).astype(np.uint8)
    t_filt = cv2.bilateralFilter(t_8u, d=9, sigmaColor=75, sigmaSpace=75)
    t      = np.clip(t_filt.astype(np.float32) / 255.0, 0.4, 1.0)

    # Scene recovery
    J = np.clip((img - A) / t[:, :, np.newaxis] + A, 0, 1)

    # ---- safer color preservation (avoid blue/orange shift) ----
    # keep original color ratio (prevents skin tone distortion)
    orig = img.copy()
    J = 0.7 * J + 0.3 * orig

    # ---- very mild gamma (avoid dark-to-blue shift) ----
    J = np.power(J, 0.98)

    # ---- gentle contrast (no over-enhancement) ----
    J = np.clip((J - 0.5) * 1.05 + 0.5, 0, 1)

    # convert to uint8
    out = (J * 255).astype(np.uint8)

    # ---- light sharpening (avoid artifacts on skin) ----
    blur = cv2.GaussianBlur(out, (0, 0), 1.2)
    out = cv2.addWeighted(out, 1.2, blur, -0.2, 0)

    # ---- skip heavy detailEnhance (causes color artifacts) ----

    return out


# ── CNN — Image upload dehazing ─────────────────────────────────────────────────
INF_W, INF_H = 480, 360  # inference resolution — higher = better quality

def dehaze_cnn(img: np.ndarray) -> np.ndarray:
    """High quality CNN dehazing for uploaded images."""
    if MODEL is None:
        raise RuntimeError(
            "CNN model not loaded.\n"
            f"  Train : python src/train.py --epochs 150\n"
            f"  Weights expected at : {WEIGHTS_PATH}"
        )
    h, w   = img.shape[:2]
    small  = cv2.resize(img, (INF_W, INF_H), interpolation=cv2.INTER_LINEAR)
    rgb    = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t      = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        out = MODEL(t)
    res = (out.squeeze().permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    bgr = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
    return cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)


# ── Camera + inference threads ──────────────────────────────────────────────────
_stop    = threading.Event()
_fps_buf = deque(maxlen=30)

def find_camera():
    for i in range(4):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, f = cap.read(); cap.release()
            if ret and f is not None:
                print(f"  Camera   : index {i}"); return i
    return None

def camera_thread():
    idx = find_camera()
    if idx is None:
        print("\n  ERROR: No camera found.")
        print("  Fix  : System Settings → Privacy → Camera → allow Terminal\n")
        return
    cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened(): cap = cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS,          30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    fail = 0
    while not _stop.is_set():
        ret, frame = cap.read()
        if ret and frame is not None:
            fail  = 0
            frame = cv2.flip(frame, 1)        # correct mirror
            with STATE.lock: STATE.raw_frame = frame
        else:
            fail += 1
            if fail > 30:
                print("  Camera stalled — reconnecting...")
                cap.release(); time.sleep(1)
                cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION); fail = 0
            time.sleep(0.03)
    cap.release()

def inference_thread():
    t_prev = time.perf_counter()
    while not _stop.is_set():
        with STATE.lock: frame = STATE.raw_frame
        if frame is None: time.sleep(0.005); continue
        t0 = time.perf_counter()
        # dynamic mode switching
        with STATE.lock:
            mode = CURRENT_MODE

        if mode == "CNN" and MODEL is not None:
            try:
                out = dehaze_cnn(frame)
            except:
                out = dcp_dehaze(frame)
        else:
            out = dcp_dehaze(frame)
        inf_ms = (time.perf_counter() - t0) * 1000
        now    = time.perf_counter()
        _fps_buf.append(now - t_prev); t_prev = now
        fps = 1.0 / (sum(_fps_buf) / len(_fps_buf)) if _fps_buf else 0.0
        # ---- overlay stats for premium UI ----
        disp = out.copy()
        
        cv2.putText(disp, f"MODE: {mode}",
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0) if mode=="DCP" else (255,0,0),
            2)
        cv2.putText(disp, f"FPS: {round(fps,1)}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)

        cv2.putText(disp, f"{round(inf_ms,1)} ms",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)

        with STATE.lock:
            STATE.out_frame = disp
            STATE.fps       = round(fps, 1)
            STATE.inf_ms    = round(inf_ms, 1)

def start_inference():
    _stop.clear(); _fps_buf.clear()
    threading.Thread(target=camera_thread,    daemon=True).start()
    threading.Thread(target=inference_thread, daemon=True).start()
    with STATE.lock: STATE.running = True

def stop_inference():
    _stop.set()
    with STATE.lock:
        STATE.running   = False
        STATE.raw_frame = None
        STATE.out_frame = None
        STATE.fps       = 0.0
        STATE.inf_ms    = 0.0

# ── MJPEG stream ────────────────────────────────────────────────────────────────
def _placeholder():
    img    = np.zeros((360, 640, 3), dtype=np.uint8); img[:] = (9, 6, 7)
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buf.tobytes()

def stream(get_fn):
    ph = _placeholder()
    while True:
        frame = get_fn()
        if frame is None:
            data = ph
        else:
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            data   = buf.tobytes()
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + data + b'\r\n'
        time.sleep(1 / 60)

# ── Routes ──────────────────────────────────────────────────────────────────────
@app.route('/')
def index(): return send_from_directory(str(Path(__file__).parent), 'interface.html')

@app.route('/feed/input')
def feed_input():
    def get():
        with STATE.lock: return STATE.raw_frame
    return Response(stream(get), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/feed/output')
def feed_output():
    def get():
        with STATE.lock: return STATE.out_frame
    return Response(stream(get), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/start', methods=['POST'])
def api_start():
    global CURRENT_MODE
    if not STATE.running:
        start_inference()
        with STATE.lock:
            CURRENT_MODE = "DCP"
    return jsonify(ok=True)

@app.route('/api/stop', methods=['POST'])
def api_stop():
    global CURRENT_MODE
    stop_inference()
    with STATE.lock:
        CURRENT_MODE = "DCP"
    return jsonify(ok=True)

@app.route('/api/status')
def api_status():
    with STATE.lock:
        return jsonify(
            running      = STATE.running,
            fps          = (STATE.fps if STATE.running else ""),
            inf_ms       = (STATE.inf_ms if STATE.running else ""),
            mode         = CURRENT_MODE,
            model        = WEIGHTS_PATH.name if CNN_OK else 'NO MODEL',
            model_loaded = CNN_OK,
        )

@app.route('/api/snapshot', methods=['POST'])
def api_snapshot():
    with STATE.lock: raw, out = STATE.raw_frame, STATE.out_frame
    if raw is None or out is None:
        return jsonify(ok=False, error='No frames yet'), 400
    sd = Path(__file__).parent / 'results' / 'samples'
    sd.mkdir(parents=True, exist_ok=True)
    n     = len(list(sd.glob('*.png')))
    div   = np.full((max(raw.shape[0], out.shape[0]), 4, 3), [0, 229, 255], dtype=np.uint8)
    panel = np.hstack([raw, div, out])
    path  = sd / f'snapshot_{n:04d}.png'
    cv2.imwrite(str(path), panel)
    return jsonify(ok=True, path=str(path))

@app.route('/api/upload', methods=['POST'])
def api_upload():
    if 'image' not in request.files:
        return jsonify(ok=False, error='No image field'), 400
    buf = np.frombuffer(request.files['image'].read(), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify(ok=False, error='Could not decode image'), 400
    global CURRENT_MODE
    CURRENT_MODE = "CNN"
    t0 = time.perf_counter()
    try:
        out = dehaze_cnn(img)               # upload always uses CNN
    except RuntimeError as e:
        return jsonify(ok=False, error=str(e)), 503
    ms = round((time.perf_counter() - t0) * 1000, 1)
    with STATE.lock:
        STATE.img_input  = img
        STATE.img_output = out
    return jsonify(ok=True, inf_ms=ms)

@app.route('/result/input')
def result_input():
    with STATE.lock: img = STATE.img_input
    if img is None: return 'No image', 404
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return send_file(io.BytesIO(buf.tobytes()), mimetype='image/jpeg')

@app.route('/result/output')
def result_output():
    with STATE.lock: img = STATE.img_output
    if img is None: return 'No image', 404
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return send_file(io.BytesIO(buf.tobytes()), mimetype='image/jpeg')

@app.route('/api/set_mode', methods=['POST'])
def api_set_mode():
    global CURRENT_MODE
    data = request.get_json()
    mode = data.get("mode", "DCP")

    if mode not in ["DCP", "CNN"]:
        return jsonify(ok=False, error="invalid mode"), 400

    with STATE.lock:
        CURRENT_MODE = mode
    print(f"[MODE SWITCH] → {CURRENT_MODE}")
    return jsonify(ok=True, mode=CURRENT_MODE)


@app.route('/api/mode')
def api_mode():
    return jsonify(mode=CURRENT_MODE)

# ── Main ────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n  AetherLiftNet — starting...")
    print(f"  Realtime : DCP (instant, no training needed)")
    print(f"  Upload   : CNN ({'✓ loaded' if CNN_OK else '✗ train first: python src/train.py --epochs 150'})")
    print("UI shows : DCP (camera) + CNN (upload)")
    threading.Thread(
        target=lambda: (time.sleep(1.2), webbrowser.open('http://localhost:8080')),
        daemon=True
    ).start()
    print("  URL      : http://localhost:8080\n")
    app.run(host='0.0.0.0', port=8080, threaded=True, debug=False)