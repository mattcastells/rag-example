from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status

from .config import get_settings


def get_api_key(request: Request) -> str:
    api_key = request.headers.get("x-api-key")
    if not api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing API key")
    return api_key


def require_api_key(api_key: str = Depends(get_api_key)) -> None:
    settings = get_settings()
    if api_key != settings.api_key:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API key")
