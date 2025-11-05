from __future__ import annotations

from fastapi import APIRouter

from . import ask, health, ingest

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(ingest.router)
api_router.include_router(ask.router)
