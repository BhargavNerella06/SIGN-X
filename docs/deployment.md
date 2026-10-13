# Deploying SIGN-X (GitHub + Render)

Status: **prepared, not yet deployed.** The Docker image below has not been
built in this environment (no Docker/WSL on the audit machine), so the first
Render build is the first real test of it.

## Architecture

```
Browser (user's device)                         Render
┌──────────────────────────────┐   HTTPS   ┌─────────────────────────────┐
│ signx-web (static site)      │ ────────▶ │ signx-api (Docker web svc)  │
│  getUserMedia → JPEG ~15 fps │   wss://  │  /ws/predict                │
│  speechSynthesis (TTS)       │ ◀──────── │  MediaPipe → 330-d features │
└──────────────────────────────┘  JSON     │  → 40-frame window → 32     │
                                           │  → Transformer+BiLSTM → UI  │
                                           └─────────────────────────────┘
```

- The **camera and speech stay in the browser.** Render has no webcam or
  speakers; the server's `pyttsx3` engine is disabled in the image
  (`SIGNX_TTS_ENGINE=none`) and the frontend speaks with the Web Speech API.
- `GET /api/health` is the health check; `GET /api/model/info` reports
  `"is_trained": true` and `"ensemble_members": ["transformer", "bilstm"]`
  when the real checkpoints are loaded. If it ever says `false`, the
  checkpoints failed to load and predictions are meaningless.
- `POST /api/model/reload` is disabled in the image
  (`SIGNX_ALLOW_MODEL_RELOAD=false`, returns 403): it is unauthenticated.

## Sizing (measured, not guessed)

Measured on the audit machine (AMD Ryzen 7 7445HS, Windows 11, Python 3.11.5,
PyTorch 2.13.0 **CPU build** — no GPU is used anywhere):

| Measurement | Result |
|---|---|
| MediaPipe landmark extraction, 640×360 frame | 23.2 ms median (24.6 p95) |
| Model inference, Transformer+BiLSTM ensemble | 3.2 ms median (4.2 p95) |
| End-to-end WebSocket round trip, localhost, 15 fps | 25.8 ms median, 31.4 ms p95 |
| Sustained rate (target 15 fps, 60 s) | 14.98 fps |
| Server CPU while streaming one client | ~65 % of one core |
| Server RSS | 308–444 MB |

Implications for Render (prices/specs from render.com/pricing, Oct 2026 —
re-check before buying):

| Plan | CPU / RAM | Verdict |
|---|---|---|
| Free | 0.1 CPU / 512 MB | **Not viable.** ~23 ms of CPU per frame becomes ≥230 ms at 0.1 CPU: ≤4 fps, far below the 15 fps the models were tuned for. Also spins down after 15 min idle (~1 min cold start). |
| Starter ($7) | 0.5 CPU / 512 MB | **Not recommended.** RSS alone (300–450 MB on Windows) leaves almost no headroom in 512 MB; ~50+ ms/frame on half a (likely slower) vCPU makes 15 fps marginal for even one user. |
| **Standard ($25)** | 1 CPU / 2 GB | **Smallest realistic plan** for one, maybe two, simultaneous signers. |
| Pro ($85) | 2 CPU / 4 GB | Several simultaneous users. |

Frame processing runs on the event loop one frame at a time, so concurrent
users share one core (two local streams: medians unchanged, p95 rose from
31 to 36–41 ms). Each connection also creates its own MediaPipe instance
(~0.6 s, blocks other connections briefly).

Network: every frame is one ~20–40 KB JPEG upload + one JSON reply, and the
frontend waits for each reply before sending the next frame, so the
achievable frame rate is capped by round-trip time. Pick the region closest
to your users (`singapore` in `render.yaml` for India).

## Files

| File | Purpose |
|---|---|
| `render.yaml` | Blueprint: `signx-api` (Docker) + `signx-web` (static site) |
| `backend/Dockerfile` | Python 3.11-slim, CPU-only PyTorch, system libs MediaPipe needs (libGL, GLib, PortAudio), downloads + checksum-verifies the MediaPipe asset, binds `0.0.0.0:$PORT` |
| `backend/constraints.txt` | Exact package versions that were tested |
| `backend/.dockerignore` | Keeps datasets, logs and tests out of the image |

Why Docker and not Render's native Python runtime: MediaPipe depends on
`opencv-contrib-python` and `sounddevice`, which need `libGL.so.1`, GLib and
PortAudio; the native runtime does not guarantee them. Render's native
runtime also defaults new services to Python 3.14, which this project has
not been tested on.

## Steps

1. **Push to GitHub** (after review). The repo is ~6.5 MB; the trained
   checkpoints (`backend/models/isl_*.pt`, `classes.json`) are committed
   because the image needs them. Datasets, videos and the MediaPipe asset are
   not.
2. In Render: **New → Blueprint**, select the repo. Render reads `render.yaml`.
3. It asks for the two `sync: false` values. On the first pass the service
   URLs don't exist yet, so:
   - create the blueprint, let `signx-api` deploy, note its URL
     (e.g. `https://signx-api.onrender.com`);
   - set `signx-web` → `VITE_API_BASE_URL=https://signx-api.onrender.com`
     and redeploy it (Vite bakes the value in at **build** time);
   - set `signx-api` → `SIGNX_CORS_ORIGINS=["https://signx-web.onrender.com"]`
     (JSON list, exact origin, no trailing slash) and redeploy it.
4. Verify:
   ```bash
   curl https://signx-api.onrender.com/api/health
   curl https://signx-api.onrender.com/api/model/info   # is_trained must be true
   ```
   Then open the static site, allow the camera, and check the dashboard shows
   **Connected**, "Trained: Yes", and predictions update.

## Environment variables

Backend (`SIGNX_` prefix; full list in `backend/app/config.py`):

| Variable | Image default | Notes |
|---|---|---|
| `PORT` | set by Render | uvicorn binds `0.0.0.0:$PORT` |
| `SIGNX_CORS_ORIGINS` | `["*"]` | set to the frontend origin in production |
| `SIGNX_ENVIRONMENT` | `production` | reported by `/api/health` |
| `SIGNX_TTS_ENGINE` | `none` | speech is browser-side |
| `SIGNX_ALLOW_MODEL_RELOAD` | `false` | `true` re-enables `POST /api/model/reload` |
| `SIGNX_DEVICE` | `cpu` | no GPU on Render web services |

Frontend (build time):

| Variable | Notes |
|---|---|
| `VITE_API_BASE_URL` | backend origin; REST calls and the WebSocket (`http→ws`, `https→wss`) derive from it |
| `VITE_WS_BASE_URL` | optional override for the WebSocket origin only |

## Not verified

- The Docker build itself and MediaPipe/OpenCV imports on Linux.
- Memory and CPU on Render's hardware (figures above are from a Windows laptop).
- Real-network latency and frame rate from a phone/laptop to Render.
- Browser behaviour: camera permission prompts and `speechSynthesis` voices
  were tested with fakes in Node, not in a real browser.
