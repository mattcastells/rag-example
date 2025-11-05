from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    path: Optional[str] = None
    mime: Optional[str] = None
    repo: Optional[str] = None
    tag: Optional[str] = None
    version: Optional[str] = None
    acl: List[str] = Field(default_factory=list)
    meta: dict | None = None


class IngestResult(BaseModel):
    processed: int
    failed: int


class AskRequest(BaseModel):
    q: str
    k: int = 8
    repo: Optional[str] = None
    tag: Optional[str] = None
    acl: Optional[List[str]] = None


class SourceDocument(BaseModel):
    path: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    usage: dict
    timings: dict


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
