from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from rank_bm25 import BM25Okapi

from ..models import RagChunk


@dataclass
class HybridResult:
    chunk: RagChunk
    score: float


class HybridRetriever:
    def __init__(self, corpus: Iterable[RagChunk]) -> None:
        self.corpus = list(corpus)
        tokenized = [chunk.content.split() for chunk in self.corpus]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, vector_results: Sequence[tuple[RagChunk, float]], k: int) -> List[HybridResult]:
        bm25_scores = self.bm25.get_scores(query.split())
        fusion = defaultdict(float)
        for chunk, score in vector_results:
            fusion[chunk.id] += score
        for chunk, bm_score in zip(self.corpus, bm25_scores):
            fusion[chunk.id] += bm_score / (bm25_scores.max() + 1e-9)
        scored = [HybridResult(chunk=self._find_chunk(cid), score=score) for cid, score in fusion.items()]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:k]

    def _find_chunk(self, cid: int) -> RagChunk:
        for chunk in self.corpus:
            if chunk.id == cid:
                return chunk
        raise KeyError(cid)
