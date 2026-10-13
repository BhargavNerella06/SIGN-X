"""SIGN-X FastAPI application entrypoint.

Run with:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

(from the backend/ directory, with the virtual environment activated).
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import routes_dataset, routes_health, routes_model, websocket
from app.config import settings
from app.models.model_loader import get_model_bundle
from app.utils.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    torch.set_num_threads(settings.inference_threads)
    # Load the model once at startup (not per-request/per-frame).
    bundle = get_model_bundle()
    logger.info(
        "SIGN-X backend ready | model=%s trained=%s device=%s classes=%d",
        bundle.model_type, bundle.is_trained, bundle.device, bundle.num_classes,
    )
    yield
    logger.info("SIGN-X backend shutting down")


app = FastAPI(
    title=settings.app_name,
    description="Real-Time Indian Sign Language (ISL) -> Text -> Voice Translator",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_health.router)
app.include_router(routes_model.router)
app.include_router(routes_dataset.router)
app.include_router(websocket.router)


@app.get("/")
def root() -> dict:
    return {
        "app": settings.app_name,
        "docs": "/docs",
        "websocket": "/ws/predict",
    }
