import io

from starlette.datastructures import UploadFile

from app.routes.ask import ask_get
from app.routes.ingest import ingest_endpoint


async def test_ask_returns_answer(db_session, stubbed_rag):
    async with db_session() as session:
        upload = UploadFile(filename="setup.txt", file=io.BytesIO(b"Guia de instalacion"), content_type="text/plain")
        await ingest_endpoint(
            files=[upload],
            repo="company",
            tag="v1",
            version="1.0",
            acl="public",
            session=session,
        )

        response = await ask_get(
            q="Como instalar?",
            k=1,
            repo="company",
            tag="v1",
            acl="public",
            session=session,
        )
        assert response.answer
        assert response.sources
        assert response.sources[0].path.endswith("setup.txt")
        assert response.usage["estimated_cost_usd"] >= 0
