# SIGN-X inference backend (FastAPI + MediaPipe + PyTorch, CPU only).
#
# Build context: backend/        docker build -t signx-api backend
# Run:                           docker run -p 8000:8000 signx-api
#
# Docker (rather than Render's native Python runtime) because MediaPipe pulls
# in opencv-contrib-python and sounddevice, which need libGL / GLib /
# PortAudio system libraries the native runtime does not guarantee.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libportaudio2 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch first: the default PyPI wheel bundles ~2.5 GB of CUDA libraries.
COPY requirements.txt constraints.txt ./
RUN pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt -c constraints.txt

COPY app ./app
COPY models ./models

# MediaPipe HolisticLandmarker asset (Apache-2.0, not committed). The checksum
# pins the exact file the models were validated with.
ARG MEDIAPIPE_URL=https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task
ARG MEDIAPIPE_SHA256=e2dab61191e2dcd0a15f943d8e3ed1dce13c82dfa597b9dd39f562975a50c3f8
RUN mkdir -p models/mediapipe \
    && if [ ! -f models/mediapipe/holistic_landmarker.task ]; then \
         curl -fsSL -o models/mediapipe/holistic_landmarker.task "$MEDIAPIPE_URL"; \
       fi \
    && echo "$MEDIAPIPE_SHA256  models/mediapipe/holistic_landmarker.task" | sha256sum -c - \
    && test -f models/isl_transformer.pt && test -f models/isl_bilstm.pt && test -f models/classes.json

ENV SIGNX_ENVIRONMENT=production \
    SIGNX_TTS_ENGINE=none \
    SIGNX_ALLOW_MODEL_RELOAD=false \
    SIGNX_DEVICE=cpu

RUN useradd --create-home signx && chown -R signx /app
USER signx

EXPOSE 8000
# Render injects PORT; bind to all interfaces. One worker: each worker would
# load its own copy of the models.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips '*'"]
