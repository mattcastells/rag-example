from __future__ import annotations

import asyncio
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from .config import get_settings
from .db import get_db
from .rag.embeddings import EmbeddingProvider, get_default_embedding_provider
from .rag.retriever import Retriever
from .rag.rerank import get_reranker
from .rag.service import RAGService

_settings = get_settings()
_rag_service: RAGService | None = None
_rag_lock = asyncio.Lock()


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async for session in get_db():
        yield session


async def get_rag_service(session: AsyncSession = None) -> RAGService:
    global _rag_service
    if _rag_service is None:
        async with _rag_lock:
            if _rag_service is None:
                embeddings: EmbeddingProvider = get_default_embedding_provider()
                retriever = Retriever(settings=_settings)
                reranker = get_reranker(_settings) if _settings.enable_rerank else None
                _rag_service = RAGService(
                    settings=_settings,
                    embeddings=embeddings,
                    retriever=retriever,
                    reranker=reranker,
                )
    return _rag_service
