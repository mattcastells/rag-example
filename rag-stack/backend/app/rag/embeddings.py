from __future__ import annotations

import asyncio
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Protocol

import numpy as np
from sentence_transformers import SentenceTransformer

from ..config import DATA_DIR, get_settings


class EmbeddingProvider(Protocol):
    async def embed_documents(self, texts: List[str]) -> List[List[float]]: ...

    async def embed_query(self, text: str) -> List[float]: ...


class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str, cache_folder: Path | None = None) -> None:
        self.model_name = model_name
        self.cache_folder = cache_folder
        self._model: SentenceTransformer | None = None
        self._lock = asyncio.Lock()

    async def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            async with self._lock:
                if self._model is None:
                    self._model = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: SentenceTransformer(
                            self.model_name,
                            cache_folder=str(self.cache_folder) if self.cache_folder else None,
                        ),
                    )
        return self._model

    @staticmethod
    def _normalize(vectors: Iterable[Iterable[float]]) -> List[List[float]]:
        normalized = []
        for vector in vectors:
            arr = np.array(list(vector), dtype=np.float32)
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
            normalized.append(arr.tolist())
        return normalized

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        model = await self._get_model()
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            ),
        )
        return self._normalize(embeddings)

    async def embed_query(self, text: str) -> List[float]:
        [vector] = await self.embed_documents([text])
        return vector


@lru_cache(maxsize=1)
def get_default_embedding_provider() -> EmbeddingProvider:
    settings = get_settings()
    cache_dir = DATA_DIR / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if settings.embeddings_provider == "sentence-transformers":
        return SentenceTransformerEmbeddings(settings.embeddings_model, cache_dir)
    raise ValueError(f"Unsupported embeddings provider: {settings.embeddings_provider}")
