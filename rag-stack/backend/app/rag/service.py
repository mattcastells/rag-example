from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from ..config import Settings
from ..models import RagChunk
from .clients import (
    ClaudeClient,
    CompletionMessage,
    OpenAIClient,
    get_claude_client,
    get_openai_client,
)
from .embeddings import EmbeddingProvider
from .prompt import build_prompt
from .retriever import RetrievedChunk, Retriever
from .rerank import Reranker


@dataclass
class AskResult:
    answer: str
    sources: List[dict]
    usage: dict
    timings: dict


class RAGService:
    def __init__(
        self,
        settings: Settings,
        embeddings: EmbeddingProvider,
        retriever: Retriever,
        reranker: Reranker | None = None,
        client: ClaudeClient | OpenAIClient | None = None,
    ) -> None:
        self.settings = settings
        self.embeddings = embeddings
        self.retriever = retriever
        self.reranker = reranker
        self._client_override = client
        self._clients: dict[str, ClaudeClient | OpenAIClient] = {}

    async def ask(
        self,
        session: AsyncSession,
        question: str,
        k: int = 8,
        repo: str | None = None,
        tag: str | None = None,
        acl: Sequence[str] | None = None,
        provider: str | None = None,
    ) -> AskResult:
        timings: dict[str, float] = {}
        start = time.perf_counter()
        query_embedding = await self.embeddings.embed_query(question)
        timings["embedding"] = time.perf_counter() - start

        retrieve_start = time.perf_counter()
        acl_filters = [scope for scope in (acl or []) if scope]
        retrieved = await self.retriever.search(
            session=session,
            embedding=query_embedding,
            k=k,
            repo=repo,
            tag=tag,
            acl=acl_filters or None,
        )
        timings["retrieval"] = time.perf_counter() - retrieve_start

        if not retrieved:
            return AskResult(
                answer="No encontré información suficiente en la base de conocimiento.",
                sources=[],
                usage={},
                timings=timings,
            )

        chunks = [item.chunk for item in retrieved]
        scores = [item.score for item in retrieved]

        if self.reranker:
            rerank_start = time.perf_counter()
            rerank_scores = await self.reranker.rerank(question, [chunk.content for chunk in chunks])
            timings["rerank"] = time.perf_counter() - rerank_start
            combined = list(zip(chunks, rerank_scores))
            combined.sort(key=lambda x: x[1], reverse=True)
            chunks = [c for c, _ in combined[:k]]
            scores = [float(score) for _, score in combined[:k]]

        prompt = build_prompt(self.settings, question, chunks)
        gen_start = time.perf_counter()
        client = self._get_client(provider)
        completion: CompletionMessage = await client.complete(prompt)
        timings["generation"] = time.perf_counter() - gen_start

        usage = completion.usage
        input_tokens = usage.get("input_tokens") or 0
        output_tokens = usage.get("output_tokens") or 0
        estimated_cost = self._estimate_cost(provider, input_tokens, output_tokens)
        if estimated_cost is not None:
            usage["estimated_cost_usd"] = round(estimated_cost, 6)

        sources = []
        for chunk, score in zip(chunks, scores):
            path = chunk.path or "desconocido"
            sources.append({"path": path, "score": float(score)})

        return AskResult(
            answer=completion.text.strip(),
            sources=sources,
            usage=usage,
            timings=timings,
        )

    async def ingest_chunks(
        self,
        session: AsyncSession,
        docs: Sequence[tuple[RagChunk, List[float] | None]],
    ) -> None:
        for chunk, embedding in docs:
            chunk.embedding = embedding
            session.add(chunk)
        await session.commit()

    def _get_client(self, provider: str | None) -> ClaudeClient | OpenAIClient:
        if self._client_override:
            return self._client_override

        selected = (provider or self.settings.default_llm_provider).lower()
        if selected not in self._clients:
            if selected == "claude":
                self._clients[selected] = get_claude_client(self.settings)
            elif selected == "openai":
                self._clients[selected] = get_openai_client(self.settings)
            else:
                raise ValueError(f"Proveedor LLM desconocido: {selected}")
        return self._clients[selected]

    def _estimate_cost(self, provider: str | None, input_tokens: int, output_tokens: int) -> float | None:
        selected = (provider or self.settings.default_llm_provider).lower()
        if selected == "claude":
            return (
                (input_tokens / 1_000) * self.settings.claude_input_cost_per_1k
                + (output_tokens / 1_000) * self.settings.claude_output_cost_per_1k
            )
        if selected == "openai":
            return (
                (input_tokens / 1_000) * self.settings.openai_input_cost_per_1k
                + (output_tokens / 1_000) * self.settings.openai_output_cost_per_1k
            )
        return None
