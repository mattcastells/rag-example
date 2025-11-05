from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from langchain.text_splitter import RecursiveCharacterTextSplitter

DEFAULT_CHUNK_SIZE = 3200  # ~800 tokens
DEFAULT_CHUNK_OVERLAP = 480  # ~120 tokens


@dataclass
class TextChunk:
    content: str
    metadata: dict


class ChunkSplitter:
    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def split(self, text: str, metadata: dict | None = None) -> List[TextChunk]:
        metadata = metadata or {}
        docs = self.splitter.create_documents([text], metadatas=[metadata])
        return [TextChunk(content=doc.page_content, metadata=dict(doc.metadata)) for doc in docs]
