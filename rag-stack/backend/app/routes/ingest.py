from __future__ import annotations

import io
from typing import List

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth import require_api_key
from ..deps import get_rag_service, get_session
from ..ingest.splitter import ChunkSplitter
from ..models import RagChunk
from ..rag.embeddings import get_default_embedding_provider
from ..schemas import IngestResult

router = APIRouter(prefix="/ingest", tags=["ingest"], dependencies=[Depends(require_api_key)])


async def _read_upload(file: UploadFile) -> str:
    data = await file.read()
    suffix = (file.filename or "").lower()
    if suffix.endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix.endswith(('.md', '.markdown', '.txt', '.log')):
        return data.decode("utf-8")
    if suffix.endswith(('.html', '.htm')):
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(data.decode("utf-8"), "html.parser")
        return soup.get_text(separator="\n")
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unsupported file: {file.filename}")


@router.post("", response_model=IngestResult)
async def ingest_endpoint(
    files: List[UploadFile] = File(...),
    repo: str = Form(...),
    tag: str = Form(...),
    version: str | None = Form(None),
    acl: str = Form("public"),
    session: AsyncSession = Depends(get_session),
):
    embeddings = get_default_embedding_provider()
    splitter = ChunkSplitter()
    rag_service = await get_rag_service()

    processed = 0
    failed = 0

    for upload in files:
        try:
            text = await _read_upload(upload)
        except Exception:  # pragma: no cover - validated in tests
            failed += 1
            continue
        metadata = {
            "path": upload.filename or "unknown",
            "mime": upload.content_type or "text/plain",
            "repo": repo,
            "tag": tag,
            "version": version,
            "acl": [scope.strip() for scope in acl.split(",") if scope.strip()],
        }
        chunks = splitter.split(text, metadata)
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
                acl=metadata["acl"],
                meta={**chunk.metadata, "source": metadata["path"]},
            )
            for chunk in chunks
        ]
        await rag_service.ingest_chunks(session, list(zip(rag_chunks, embeddings_list)))
        processed += 1

    return IngestResult(processed=processed, failed=failed)
