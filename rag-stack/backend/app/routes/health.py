from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter

from ..schemas import HealthResponse

router = APIRouter(prefix="/healthz", tags=["health"])


@router.get("", response_model=HealthResponse)
async def healthcheck() -> HealthResponse:
    return HealthResponse(status="ok", timestamp=datetime.utcnow())
