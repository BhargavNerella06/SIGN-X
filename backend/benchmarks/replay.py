"""Step 2 of the SIGN-X live benchmark (run from backend/; see docs/performance.md).

Replays the cached browser-quality features from extract_live.py through the
REAL RealtimePredictor (buffer -> gate -> model -> smoother -> sentence),
with only the MediaPipe call replaced by the cache. Deterministic.

Scenarios
  offline   : whole clip resampled to 32 frames (training-style) -> top-1 / top-5.
  isolated  : fresh session per clip: LEAD idle frames (first frame repeated),
              the sign, TAIL idle frames (last frame repeated). Success = the
              first commit is the true sign.
  continuous: one session, all clips back-to-back in a fixed shuffled order,
              GAP idle frames between. Per clip: was the true sign committed
              during its segment (sign + following gap)? Wrong commits counted.

With settings.live_ensemble on (default) the other checkpoint is averaged in,
exactly as the server does; pass live_ensemble=0 for a single model.

Usage: python replay.py <cache.npz> <split> [model_type] [key=value settings overrides...]
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
import app.inference.predictor as predictor_module  # noqa: E402
from app.config import settings  # noqa: E402
from app.models.model_loader import load_model, with_live_ensemble  # noqa: E402
from app.vision.landmarks import FrameLandmarks, empty_hand  # noqa: E402
from app.vision.preprocessing import resample_or_pad  # noqa: E402

LEAD, TAIL, GAP = 20, 25, 10

cache, split = sys.argv[1], sys.argv[2]
model_type = sys.argv[3] if len(sys.argv) > 3 else "transformer"
for kv in sys.argv[4:]:
    k, v = kv.split("=")
    current = getattr(settings, k)
    setattr(settings, k, v if isinstance(current, str) else type(current)(float(v)))

data = np.load(cache, allow_pickle=True)
clips = [c for c in data["clips"] if c["split"] == split]
bundle = with_live_ensemble(load_model(model_type))  # what the server runs (settings.live_ensemble)
UNK = settings.unknown_label


class CachedExtractor:
    available = True

    def __init__(self, *a, **k):
        self.queue = []

    def extract(self, frame):
        vec, hand = self.queue.pop(0)
        raw = FrameLandmarks(None, empty_hand() if hand else None, None, None)
        return vec, raw

    def reset(self):
        pass

    def close(self):
        pass


predictor_module.FeatureExtractor = CachedExtractor


def new_predictor():
    return predictor_module.RealtimePredictor(model_bundle=bundle)


def run(pred, frames):
    """frames: list of (vec, hand). Returns list of per-frame message dicts."""
    pred.feature_extractor.queue.extend(frames)
    out = []
    for _ in frames:
        m = pred.process_frame(None)
        out.append({"sign": m.sign, "conf": m.confidence, "committed": m.committed})
    return out


def frames_of(c):
    return list(zip(c["vecs"], c["hands"]))


# --- offline (training-style) on live-quality features ---------------------
correct = top5 = 0
for c in clips:
    seq, mask = resample_or_pad(c["vecs"].astype(np.float32), settings.sequence_length)
    r = bundle.classifier.predict(torch.from_numpy(seq)[None], mask=torch.from_numpy(mask)[None], top_k=5)
    correct += r.top_labels[0] == c["label"]
    top5 += c["label"] in r.top_labels
offline = {"top1": correct / len(clips), "top5": top5 / len(clips)}

# --- isolated ---------------------------------------------------------------
iso = {"first_commit_correct": 0, "any_commit_correct": 0, "no_commit": 0, "wrong_commits": 0,
       "lead_commits": 0, "latency_frames": [], "commit_conf": []}
per_class = {}
t_proc = []
for c in clips:
    f = frames_of(c)
    pred = new_predictor()
    lead = run(pred, [f[0]] * LEAD)
    t0 = time.perf_counter()
    body = run(pred, f + [f[-1]] * TAIL)
    t_proc.append((time.perf_counter() - t0) * 1e3 / len(body))
    iso["lead_commits"] += sum(m["committed"] for m in lead)
    commits = [(i, m) for i, m in enumerate(body) if m["committed"]]
    ok_first = bool(commits) and commits[0][1]["sign"] == c["label"]
    iso["first_commit_correct"] += ok_first
    iso["any_commit_correct"] += any(m["sign"] == c["label"] for _, m in commits)
    iso["no_commit"] += not commits
    iso["wrong_commits"] += sum(m["sign"] != c["label"] for _, m in commits)
    for i, m in commits:
        if m["sign"] == c["label"]:
            iso["latency_frames"].append(i)
            iso["commit_conf"].append(m["conf"])
            break
    pc = per_class.setdefault(c["label"], [0, 0])
    pc[0] += ok_first
    pc[1] += 1
n = len(clips)
iso_summary = {
    "first_commit_correct": iso["first_commit_correct"] / n,
    "any_commit_correct": iso["any_commit_correct"] / n,
    "no_commit": iso["no_commit"] / n,
    "wrong_commits_per_clip": iso["wrong_commits"] / n,
    "lead_in_commits": iso["lead_commits"],
    "median_frames_from_sign_start_to_correct_commit": float(np.median(iso["latency_frames"])) if iso["latency_frames"] else None,
    "median_conf_at_correct_commit": float(np.median(iso["commit_conf"])) if iso["commit_conf"] else None,
}

# --- continuous ---------------------------------------------------------------
order = np.random.default_rng(0).permutation(n)
pred = new_predictor()
hit = wrong = 0
for j in order:
    c = clips[j]
    f = frames_of(c)
    seg = run(pred, f + [f[-1]] * GAP)
    commits = [m["sign"] for m in seg if m["committed"]]
    hit += c["label"] in commits
    wrong += sum(s != c["label"] for s in commits)
cont = {"segment_recall": hit / n, "wrong_commits_per_clip": wrong / n}

print(json.dumps({
    "model": model_type, "ensemble": bundle.ensemble_members, "split": split, "n": n,
    "settings": {k: getattr(settings, k) for k in ("realtime_window_frames", "sequence_stride", "confidence_threshold",
                                                    "debounce_repeats", "smoothing_window", "min_hand_frame_ratio")},
    "offline_live_features": offline, "isolated": iso_summary, "continuous": cont,
    "process_frame_ms_excl_mediapipe_avg": float(np.mean(t_proc)),
    "per_class_first_commit": {k: f"{v[0]}/{v[1]}" for k, v in sorted(per_class.items())},
}, indent=1))
