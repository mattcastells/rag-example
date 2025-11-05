from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import Settings
from ..models import RagChunk


@dataclass
class RetrievedChunk:
    chunk: RagChunk
    score: float


class Retriever:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def _postgres_vector_search(
        self,
        session: AsyncSession,
        embedding: Sequence[float],
        k: int,
        repo: str | None,
        tag: str | None,
        acl: list[str] | None,
    ) -> List[RetrievedChunk]:
        params = {"embedding": embedding, "limit": k, "repo": repo, "tag": tag, "acl": acl}
        filters = ["TRUE"]
        if repo:
            filters.append("repo = :repo")
        if tag:
            filters.append("tag = :tag")
        if acl:
            filters.append("acl && :acl")
        where_clause = " AND ".join(filters)
        stmt = text(
            f"""
            SELECT *, 1 - (embedding <=> :embedding) AS similarity
            FROM rag_chunks
            WHERE {where_clause}
            ORDER BY embedding <=> :embedding
            LIMIT :limit
            """
        )
        result = await session.execute(stmt, params)
        rows = result.mappings().all()
        chunks: List[RetrievedChunk] = []
        for row in rows:
            chunk = RagChunk(**{k: row[k] for k in row.keys() if k in RagChunk.__table__.columns.keys()})
            chunks.append(RetrievedChunk(chunk=chunk, score=row["similarity"]))
        return chunks

    @staticmethod
    def _cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
        va = np.array(list(a), dtype=np.float32)
        vb = np.array(list(b), dtype=np.float32)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        if denom == 0:
            return 0.0
        return float(np.dot(va, vb) / denom)

    async def _fallback_search(
        self,
        session: AsyncSession,
        embedding: Sequence[float],
        k: int,
        repo: str | None,
        tag: str | None,
        acl: list[str] | None,
    ) -> List[RetrievedChunk]:
        stmt = select(RagChunk)
        if repo:
            stmt = stmt.where(RagChunk.repo == repo)
        if tag:
            stmt = stmt.where(RagChunk.tag == tag)
        result = await session.execute(stmt)
        chunks = [row[0] for row in result.all()]
        if acl:
            chunks = [c for c in chunks if c.acl is None or any(scope in (c.acl or []) for scope in acl)]
        scored = [
            RetrievedChunk(chunk=c, score=self._cosine_similarity(embedding, c.embedding or []))
            for c in chunks
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:k]

    async def search(
        self,
        session: AsyncSession,
        embedding: Sequence[float],
        k: int,
        repo: str | None = None,
        tag: str | None = None,
        acl: list[str] | None = None,
    ) -> List[RetrievedChunk]:
        dialect = session.bind.dialect if session.bind else None
        if dialect and dialect.name == "postgresql":
            try:
                return await self._postgres_vector_search(session, embedding, k, repo, tag, acl)
            except Exception:
                return await self._fallback_search(session, embedding, k, repo, tag, acl)
        return await self._fallback_search(session, embedding, k, repo, tag, acl)
