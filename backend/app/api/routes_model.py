"""Model information and management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import settings
from app.models.model_loader import get_model_bundle, reload_model_bundle

router = APIRouter(prefix="/api/model", tags=["model"])


class ReloadRequest(BaseModel):
    model_type: str | None = None


@router.get("/info")
def model_info() -> dict:
    return get_model_bundle().info()


@router.post("/reload")
def reload_model(request: ReloadRequest) -> dict:
    if not settings.allow_model_reload:
        raise HTTPException(status_code=403, detail="Model reload is disabled on this server")
    if request.model_type is not None and request.model_type not in ("transformer", "bilstm"):
        raise HTTPException(status_code=400, detail="model_type must be 'transformer' or 'bilstm'")
    bundle = reload_model_bundle(request.model_type)
    return bundle.info()
