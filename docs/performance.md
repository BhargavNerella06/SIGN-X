# SIGN-X — Live Performance & Recognition Benchmark

Measured on the development PC (Windows 11, 6-core/12-thread CPU, PyTorch
2.13.0+cpu, MediaPipe 1.0.1, Python 3.11.5) on 2026-10-10. No physical
webcam was used: the live path is simulated from INCLUDE source videos as
described below.

## How it is measured (reproducible)

All scripts live in `backend/benchmarks/` and run from `backend/`.

1. **`extract_live.py <out.npz> <fps> <width>`** — simulates the browser for
   every validation/test clip: 25 fps source video → sampled at the capture
   rate → resized to 640 px wide → JPEG q=80 (as `canvas.toBlob(…, 0.8)`) →
   `cv2.imdecode` (as `websocket.py`) → the real `FeatureExtractor`. Caches
   per-frame features and times each stage.
2. **`replay.py <cache.npz> <split> [model] [setting=value…]`** — feeds the
   cached frames through the real `RealtimePredictor` (buffer → no-hands gate
   → model → smoother → sentence). Scenarios:
   - *offline*: whole clip resampled to 32 frames (training-style top-1/top-5);
   - *isolated*: 2 s idle lead-in, the sign, 2.5 s idle tail — success when the
     first commit is the true sign;
   - *continuous*: all clips back to back (fixed order, 1 s gaps) in one
     session — was the sign committed during its segment, and how many wrong
     commits were made.
3. **`sweep.py <cache.npz> validation`** — grid over window / stride /
   threshold / `debounce_repeats`. Tuning uses the **validation** split only;
   the test split was evaluated once at the end.
4. **`e2e_ws.py <port> <fps> <seconds> <flood|inflight1> [server_pid]`** —
   real WebSocket client against a running server with real JPEG frames;
   per-frame round trip, server CPU and memory.

## Where the time goes (per frame, 640 px, 7,860 frames)

| Stage | Median | p95 |
|---|---|---|
| MediaPipe Holistic (IMAGE mode) | **23.3 ms** | 27.6 ms |
| JPEG encode (client-side proxy) | 0.43 ms | 0.64 ms |
| JPEG decode (server) | 0.50 ms | 0.73 ms |
| Normalize + vectorize | 0.08 ms | 0.11 ms |
| Buffer + model + smoothing (whole `process_frame` minus MediaPipe) | 0.2–0.4 ms | — |

MediaPipe is >90 % of server time per frame. The model itself is ~1–2.5 ms
per inference on CPU. A GPU would save under 2 ms per inference, and this
PyTorch build is CPU-only, so CUDA isn't worth setting up for live inference.

## Root causes found

1. **10 fps capture starved the models.** Training clips are 25 fps (mean
   2.5 s). Offline accuracy on training-quality features, subsampled:

   | fps | 25 | 15 | 12.5 | 10 | 7.5 |
   |---|---|---|---|---|---|
   | Transformer | 72.0 % | 73.3 % | 66.7 % | 61.3 % | 52.7 % |
   | BiLSTM | 72.7 % | 68.0 % | 56.7 % | 28.7 % | 14.0 % |

   On browser-quality frames (640 px JPEG through MediaPipe) the Transformer
   scores 59.3 % at 10 fps and 69.3 % at 15 fps (BiLSTM 24.0 % → 61.3 %).
   Resolution/JPEG costs only ~2 points; frame rate is the dominant factor.
2. **Eager commits.** A commit fired on the first vote whose majority cleared
   0.6, including votes from windows that only partly covered a sign.
   `debounce_repeats` existed in the config but was never used.
3. **No backpressure.** The browser sent every captured frame regardless of
   replies; when the server fell behind, frames queued (40 fps stress:
   673 ms median round trip vs 29 ms with one frame in flight).
4. **PyTorch thread spin.** 6 intra-op threads busy-wait between the tiny
   live inferences: ~400 % server CPU at 15 fps versus ~70 % with 1 thread,
   at equal or better latency.

## Changes

| Setting | Before | After |
|---|---|---|
| Browser capture | 10 fps (100 ms) | 15 fps (67 ms) |
| `realtime_window_frames` | 32 (3.2 s @ 10 fps) | 40 (2.7 s @ 15 fps) |
| `sequence_stride` | 8 | 4 |
| `debounce_repeats` | 1 (unused) | 4 agreeing confident votes |
| `inference_threads` (server only) | 6 (default) | 1 |
| Frames in flight | unlimited | 1 (resumes after 2 s if a reply is lost) |

Unchanged: model input (32 frames × 330 features, same `resample_or_pad`),
`confidence_threshold` 0.6, `min_hand_frame_ratio` 0.25, checkpoints,
dataset, class mapping.

## Results

**End to end** (same session, 30 s runs, real server):

| | Frames/s | Round trip median | p95 | Server CPU | RSS |
|---|---|---|---|---|---|
| Before (10 fps) | 10.0 | 26.8–28.1 ms | 31.7–33.0 ms | 155–158 % | ~440 MB |
| After (15 fps) | 15.0 | 26.3–26.5 ms | 29.6–30.2 ms | 61–72 % | ~440 MB |

CPU % is per core (1200 % = whole machine).

**Live recognition, held-out test split** (150 clips, evaluated once):

| | Transformer before | Transformer after | BiLSTM before | BiLSTM after |
|---|---|---|---|---|
| Isolated: first commit correct | 37.3 % | 36.7 % | 20.7 % | 32.7 % |
| Isolated: wrong commits / sign | 0.22 | **0.09** | 0.47 | **0.11** |
| Isolated: no commit (stays UNKNOWN) | 42 % | 54 % | 39 % | 60 % |
| Continuous: sign committed | 28.7 % | 29.3 % | 21.3 % | 25.3 % |
| Continuous: wrong commits / sign | 0.35 | **0.11** | 0.40 | **0.09** |
| Median time from sign start to correct commit | ~1.1 s | ~2.3 s | ~2.7 s | ~2.6 s |

For the Transformer the gain is **precision**: wrong words in the sentence
drop by about two-thirds and continuous commit precision rises from ~45 % to
~72 %, while correct commits stay about the same. The trade-offs are that
more signs stay UNKNOWN and commits arrive near the end of the sign rather
than early. `debounce_repeats` is the knob for this: on validation, 3 gives
more commits with more errors.

## Accuracy: what limits it next

- Split integrity checked: no video id or identical feature array appears in
  more than one split; class order matches across both models and
  `classes.json`.
- Weak classes have 4 test clips each (one error = 25 points). Frequent
  confusions are visually similar signs: GOOD_EVENING/GOOD_NIGHT/GOOD_MORNING,
  SHE→HE, PINK→RED, YELLOW→GREEN.
- **Ensemble (not implemented):** averaging Transformer and BiLSTM
  probabilities scores 78.3 % vs 71.7 % (Transformer alone) on browser-quality
  validation features and 85.0 % vs 78.3 % on training-quality validation
  features. Worth a live-replay evaluation next; it adds ~2 ms and a second
  model.
- Background/idle class: not added. It would need genuinely recorded idle and
  transition clips with a labelling protocol, not synthesized labels.

---

# 2026-10-11 — Live ensemble and re-tuned commit policy

Same scripts and method as above. Note: on this day the PC was on battery
with the "Silent" power plan, so every stage ran ~2× slower than on
2026-10-10 (MediaPipe 43 ms median instead of 23 ms; JPEG encode/decode
doubled too). Before/after comparisons below were measured on the same day.
Two e2e runs that overlapped Windows Modern Standby (Kernel-Power events
506/507) were discarded.

## Where the time goes now

| Stage | Median | p95 |
|---|---|---|
| MediaPipe Holistic, 640 px (7,860 frames) | 43.3 ms | 49.5 ms |
| JPEG encode / decode | 0.94 / 0.99 ms | 1.2 / 1.3 ms |
| Normalize + vectorize | 0.13 ms | 0.18 ms |
| Transformer / BiLSTM forward (1 thread, once per 4 frames) | ~4.7 / ~3.5 ms | — |

Rejected after measuring:
- **MediaPipe VIDEO (tracking) mode:** 43.1 ms vs 43.3 ms. No gain, and
  training used IMAGE mode.
- **320 px capture:** MediaPipe 40 ms vs 43 ms over the full set, but offline
  accuracy on validation fell from 75.0 % to 43.3 %. Capture stays at 640 px.
- **GPU:** an RTX 4050 is present, but the installed PyTorch is `+cpu`. The
  model is ~1 % of per-frame time, so a CUDA build would not change live
  latency. Not installed.

## Biggest live error source: the commit policy, not the model

On browser-quality validation features, the Transformer scored 71.7 % offline
but committed the right sign first in only 36.7 % of isolated clips. 58 % of
clips never committed. Per-window traces show the correct label at
0.4–0.6 confidence for many windows, and the rule needs 4 consecutive raw
votes each ≥ 0.6. Raw single-model votes also flip between neighbouring
windows.

## Change

- `live_ensemble = true` (new): the server averages Transformer and BiLSTM
  softmax probabilities (`EnsembleClassifier`, applied in the model registry
  only). Training and evaluation scripts still load a single model. If the
  second checkpoint is missing, untrained, or has a different class list or
  input dim, the server falls back to `model_type` alone.
- `confidence_threshold` 0.6 → 0.5 and `debounce_repeats` 4 → 3, chosen on the
  validation split. The ensemble's averaged probabilities run lower than a
  single model's. Set `SIGNX_LIVE_ENSEMBLE=false` together with 0.6 / 4 to get
  the previous behaviour.

Validation sweep (60 clips, live replay; continuous = wrong commits per clip):

| | Isolated first correct | Isolated wrong/clip | Continuous recall | Continuous wrong/clip |
|---|---|---|---|---|
| Transformer 0.6 / 4 (before) | 36.7 % | 0.050 | 35.0 % | 0.117 |
| Transformer 0.6 / 3 | 40.0 % | 0.133 | 38.3 % | 0.183 |
| Transformer 0.5 / 3 | 48.3 % | 0.267 | 48.3 % | 0.300 |
| **Ensemble 0.5 / 3 (after)** | **45.0 %** | 0.100 | **41.7 %** | **0.100** |
| Ensemble 0.4 / 3 | 56.7 % | 0.300 | 50.0 % | 0.250 |

At matched wrong-commit rates the ensemble gives more recall than the
Transformer across the whole sweep.

## Results — held-out test split (150 clips, evaluated once)

| | Before (Transformer 0.6 / 4) | After (Ensemble 0.5 / 3) |
|---|---|---|
| Offline top-1 on live features | 69.3 % | 72.0 % |
| Isolated: first commit correct | 36.7 % | **42.0 %** |
| Isolated: no commit (stays UNKNOWN) | 54.0 % | 48.0 % |
| Isolated: wrong commits / sign | 0.093 | 0.147 (worse) |
| Continuous: sign committed | 29.3 % | **35.3 %** |
| Continuous: wrong commits / sign | 0.113 | 0.133 |
| Continuous commit precision | 72 % | 73 % |
| Median frames from sign start to correct commit | 35 | 35 |

The gain on test is smaller than on validation. More signs are recognized at
about the same continuous precision. Isolated signs get more wrong commits
(22 vs 14 of 150 clips).

**End to end** (15 fps, one frame in flight, 30 s runs, same day, interleaved):

| | Frames/s | Round trip median | p95 | Server CPU | RSS |
|---|---|---|---|---|---|
| Ensemble off | 14.1 / 14.1 | 53.1 / 52.4 ms | 62.0 / 61.9 ms | 122–124 % | ~440 MB |
| Ensemble on | 14.3 / 14.4 | 53.1 / 48.7 ms | 65.1 / 61.5 ms | 120–129 % | ~458 MB |

The ensemble's latency cost is within run-to-run noise. It costs ~15–20 MB of
memory.

## What limits recognition next

1. **Data.** Weak classes have 2–4 test clips. The confusions are
   GOOD_MORNING / EVENING / NIGHT, HE / SHE / YOU_PLURAL and
   GREEN / BLUE / COLOUR. More genuine INCLUDE or recorded ISL clips per class
   is the most valuable next step.
2. **Training on live-like input.** Models trained on 25 fps clips lose
   ~3–8 points at 15 fps. Retraining with frame-rate augmentation (random
   subsampling to 12–25 fps), and selecting on live-replay validation, should
   narrow the offline-vs-live gap. Save the result under new checkpoint names.
3. **Idle/transition class.** A background class needs genuinely recorded
   idle and transition clips with a labelling protocol. Not added.
4. **Physical webcam test.** All live numbers here come from replayed INCLUDE
   videos through the real browser-equivalent JPEG path. No physical webcam
   session was run.
