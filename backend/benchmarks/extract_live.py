"""Step 1 of the SIGN-X live benchmark (run from backend/; see docs/performance.md).

Simulates the browser webcam path for every validation/test clip:
  25 fps source video -> sampled at CAPTURE_FPS -> resized to 640 px wide ->
  JPEG q=80 (canvas.toBlob 0.8) -> cv2.imdecode (as websocket.py does) ->
  real FeatureExtractor (MediaPipe IMAGE mode, one fresh instance per clip).

Caches per-frame feature vectors + hand flags so live-policy replays
(step 2) are deterministic and fast, and records per-stage timings.

Usage: python <this> <out.npz> [capture_fps] [width]
"""
import glob
import re
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import psutil  # optional: CPU / memory figures
except ImportError:
    psutil = None

sys.path.insert(0, os.getcwd())
from app.vision.feature_extractor import FeatureExtractor, landmarks_to_vector  # noqa: E402
from app.vision.preprocessing import normalize_frame  # noqa: E402

OUT = sys.argv[1]
CAPTURE_FPS = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
WIDTH = int(sys.argv[3]) if len(sys.argv) > 3 else 640
ROOT = Path("datasets/isl_include_subset28")
VIDEO_ROOT = Path("datasets/_include_zips/extracted")


def find_video(stem: str) -> Path:
    # e.g. Greetings_50_Alright_MVI_0037_MOV -> extracted/Greetings/50. Alright/MVI_0037.MOV
    cat, num = stem.split("_")[:2]
    for d in glob.glob(str(VIDEO_ROOT / cat / f"{num}. *")):
        for f in os.listdir(d):
            if stem.endswith(re.sub(r"[^A-Za-z0-9]+", "_", f).strip("_")):
                return Path(d) / f
    raise FileNotFoundError(stem)


def browser_frame(frame):
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (WIDTH, round(h * WIDTH / w)), interpolation=cv2.INTER_AREA)
    t0 = time.perf_counter()
    jpeg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
    t1 = time.perf_counter()
    decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    t2 = time.perf_counter()
    return decoded, len(jpeg), (t1 - t0) * 1e3, (t2 - t1) * 1e3


def main():
    proc = psutil.Process() if psutil else None
    if proc:
        proc.cpu_percent(None)
    clips, timings = [], {"encode": [], "decode": [], "mediapipe": [], "normalize": [], "jpeg_bytes": []}
    t_start = time.perf_counter()
    for split in ("validation", "test"):
        for cls in sorted(os.listdir(ROOT / split)):
            for f in sorted(os.listdir(ROOT / split / cls)):
                stem = f.split("__")[0]
                cap = cv2.VideoCapture(str(find_video(stem)))
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                frames = []
                while True:
                    ok, fr = cap.read()
                    if not ok:
                        break
                    frames.append(fr)
                cap.release()
                n_out = max(1, int(round(len(frames) / fps * CAPTURE_FPS)))
                idx = np.round(np.linspace(0, len(frames) - 1, n_out)).astype(int)
                fe = FeatureExtractor()
                vecs, hands = [], []
                for i in idx:
                    img, nbytes, enc, dec = browser_frame(frames[i])
                    t0 = time.perf_counter()
                    raw = fe._extractor.process(img)
                    t1 = time.perf_counter()
                    vec = landmarks_to_vector(normalize_frame(raw))
                    t2 = time.perf_counter()
                    timings["encode"].append(enc); timings["decode"].append(dec)
                    timings["mediapipe"].append((t1 - t0) * 1e3); timings["normalize"].append((t2 - t1) * 1e3)
                    timings["jpeg_bytes"].append(nbytes)
                    vecs.append(vec); hands.append(raw.has_any_hand)
                fe.close()
                clips.append({"split": split, "label": cls, "stem": stem, "vecs": np.stack(vecs), "hands": np.array(hands)})
                print(f"{split} {cls} {stem} frames={len(frames)} sampled={len(idx)} hand_ratio={np.mean(hands):.2f}", flush=True)
    wall = time.perf_counter() - t_start
    meta = {
        "capture_fps": CAPTURE_FPS, "width": WIDTH, "wall_s": wall,
        "process_cpu_percent_avg": proc.cpu_percent(None) if proc else None,
        "rss_mb": proc.memory_info().rss / 2**20 if proc else None,
        "timings": {k: {"median": float(np.median(v)), "p95": float(np.percentile(v, 95)), "n": len(v)} for k, v in timings.items()},
    }
    np.savez_compressed(OUT, meta=json.dumps(meta), clips=np.array(clips, dtype=object))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
