from __future__ import annotations

import asyncio
import time
from collections import defaultdict

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import get_settings
from .logging_conf import setup_logging
from .routes import api_router


class RateLimiterMiddleware:
    def __init__(self, app: FastAPI, requests_per_minute: int) -> None:
        self.app = app
        self.requests_per_minute = requests_per_minute
        self._lock = asyncio.Lock()
        self._buckets: dict[str, tuple[int, float]] = defaultdict(lambda: (0, 0.0))

    async def __call__(self, scope, receive, send):  # type: ignore[override]
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request = Request(scope, receive=receive)
        identity = request.headers.get("x-api-key") or request.client.host or "anonymous"
        now = time.time()
        async with self._lock:
            count, timestamp = self._buckets[identity]
            if now - timestamp >= 60:
                count = 0
                timestamp = now
            if count >= self.requests_per_minute:
                response = JSONResponse(
                    {"detail": "Rate limit exceeded"},
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                )
                await response(scope, receive, send)
                return
            self._buckets[identity] = (count + 1, timestamp if timestamp else now)
        await self.app(scope, receive, send)


def create_app() -> FastAPI:
    settings = get_settings()
    setup_logging()
    app = FastAPI(title="rag-stack", version="0.1.0")
    app.include_router(api_router)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"]
    )
    app.add_middleware(RateLimiterMiddleware, requests_per_minute=settings.rate_limit_per_minute)

    return app


app = create_app()
