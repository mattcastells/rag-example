import asyncio

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def configure_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("CLAUDE_MODEL", "claude-test")
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///./test.db")
    monkeypatch.setenv("API_KEY", "dev-key")
    monkeypatch.setenv("ENABLE_RERANK", "false")
    monkeypatch.setenv("ENABLE_HYBRID", "false")
    monkeypatch.setenv("EMBEDDINGS_PROVIDER", "sentence-transformers")
    monkeypatch.setenv("EMBEDDINGS_MODEL", "test")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "100")
    yield


@pytest.fixture
async def db_session(monkeypatch):
    from app import config, db, models

    config.get_settings.cache_clear()
    settings = config.get_settings()

    engine = create_async_engine(settings.database_url, echo=False, future=True)
    db.engine = engine
    db.AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(models.Base.metadata.create_all)

    try:
        yield db.AsyncSessionLocal
    finally:
        async with engine.begin() as conn:
            await conn.run_sync(models.Base.metadata.drop_all)
        await engine.dispose()


class StubEmbeddings:
    async def embed_documents(self, texts):
        return [[float(len(text) % 10), 0.0, 0.0] for text in texts]

    async def embed_query(self, text):
        return [0.1, 0.1, 0.1]


class StubClient:
    async def complete(self, prompt):
        from app.rag.clients import CompletionMessage

        return CompletionMessage(text="respuesta", usage={"input_tokens": 10, "output_tokens": 20})


@pytest.fixture
def stubbed_rag(monkeypatch):
    from app import config, deps
    from app.rag.retriever import Retriever
    from app.rag.service import RAGService

    stub_embeddings = StubEmbeddings()
    monkeypatch.setattr("app.rag.embeddings.get_default_embedding_provider", lambda: stub_embeddings)
    config.get_settings.cache_clear()
    settings = config.get_settings()
    service = RAGService(settings=settings, embeddings=stub_embeddings, retriever=Retriever(settings), reranker=None, client=StubClient())

    async def _get_service():
        return service

    monkeypatch.setattr(deps, "get_rag_service", _get_service)
    yield service
    deps._rag_service = None
