from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth import require_api_key
from ..deps import get_rag_service, get_session
from ..schemas import AskRequest, AskResponse, LLMProvider

router = APIRouter(prefix="/ask", tags=["ask"], dependencies=[Depends(require_api_key)])


def _build_response(result) -> AskResponse:
    return AskResponse(answer=result.answer, sources=result.sources, usage=result.usage, timings=result.timings)


@router.get("", response_model=AskResponse)
async def ask_get(
    q: str = Query(..., min_length=3, max_length=2048),
    k: int = Query(8, ge=1, le=20),
    repo: Optional[str] = Query(None),
    tag: Optional[str] = Query(None),
    acl: Optional[str] = Query(None),
    session: AsyncSession = Depends(get_session),
    provider: Optional[LLMProvider] = Query(None),
):
    rag_service = await get_rag_service()
    acl_list = [scope.strip() for scope in acl.split(",") if scope.strip()] if acl else None
    result = await rag_service.ask(
        session,
        question=q,
        k=k,
        repo=repo,
        tag=tag,
        acl=acl_list,
        provider=provider.value if provider else None,
    )
    return _build_response(result)


@router.post("", response_model=AskResponse)
async def ask_post(
    request: AskRequest,
    session: AsyncSession = Depends(get_session),
):
    rag_service = await get_rag_service()
    result = await rag_service.ask(
        session,
        question=request.q,
        k=request.k,
        repo=request.repo,
        tag=request.tag,
        acl=request.acl,
        provider=request.provider.value if request.provider else None,
    )
    return _build_response(result)
