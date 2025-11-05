from __future__ import annotations

import asyncio
from typing import Iterable, List, Sequence

from sentence_transformers import CrossEncoder

from ..config import Settings


class Reranker:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: CrossEncoder | None = None
        self._lock = asyncio.Lock()

    async def _get_model(self) -> CrossEncoder:
        if self._model is None:
            async with self._lock:
                if self._model is None:
                    loop = asyncio.get_event_loop()
                    self._model = await loop.run_in_executor(None, lambda: CrossEncoder(self.model_name))
        return self._model

    async def rerank(self, query: str, documents: Sequence[str]) -> List[float]:
        model = await self._get_model()
        loop = asyncio.get_event_loop()
        pairs = [(query, doc) for doc in documents]
        scores = await loop.run_in_executor(None, lambda: model.predict(pairs))
        return list(map(float, scores))


def get_reranker(settings: Settings) -> Reranker | None:
    if not settings.enable_rerank:
        return None
    model_name = "BAAI/bge-reranker-large"
    if settings.embeddings_model.endswith("m3"):
        model_name = "BAAI/bge-reranker-v2-m3"
    return Reranker(model_name)
