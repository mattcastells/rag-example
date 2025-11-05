from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import List

from ..config import DATA_DIR, get_settings
from ..db import lifespan_session
from ..models import RagChunk
from ..rag.embeddings import get_default_embedding_provider
from ..rag.service import RAGService
from ..rag.retriever import Retriever
from ..rag.rerank import get_reranker
from .loaders import load_from_directory
from .splitter import ChunkSplitter


async def ingest_directory(path: Path, repo: str, tag: str, version: str | None, acl: List[str]) -> None:
    settings = get_settings()
    embeddings = get_default_embedding_provider()
    retriever = Retriever(settings)
    reranker = get_reranker(settings) if settings.enable_rerank else None
    service = RAGService(settings=settings, embeddings=embeddings, retriever=retriever, reranker=reranker)

    docs = load_from_directory(path)
    splitter = ChunkSplitter()
    async with lifespan_session() as session:
        for doc in docs:
            metadata = {
                "path": str(doc.path),
                "mime": doc.mime,
                "repo": repo,
                "tag": tag,
                "version": version,
                "acl": acl,
            }
            chunks = splitter.split(doc.content, metadata)
            texts = [chunk.content for chunk in chunks]
            embeddings_list = await embeddings.embed_documents(texts)
            rag_chunks = [
                RagChunk(
                    content=chunk.content,
                    path=metadata["path"],
                    mime=metadata["mime"],
                    repo=metadata["repo"],
                    tag=metadata["tag"],
                    version=metadata["version"],
                    acl=acl,
                    meta={**chunk.metadata, "source": metadata["path"]},
                )
                for chunk in chunks
            ]
            await service.ingest_chunks(session, list(zip(rag_chunks, embeddings_list)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest documents into rag-stack")
    parser.add_argument("--path", type=str, default=str(DATA_DIR / "docs"))
    parser.add_argument("--repo", type=str, required=True)
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--acl", type=str, default="public")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.path)
    acl = [scope.strip() for scope in args.acl.split(",") if scope.strip()]
    asyncio.run(ingest_directory(path, args.repo, args.tag, args.version, acl))


if __name__ == "__main__":
    main()
