import io

from sqlalchemy import select
from starlette.datastructures import UploadFile

from app.models import RagChunk
from app.routes.ingest import ingest_endpoint


async def test_ingest_endpoint(db_session, stubbed_rag):
    async with db_session() as session:
        upload = UploadFile(filename="doc.txt", file=io.BytesIO(b"Contenido de prueba"), content_type="text/plain")
        result = await ingest_endpoint(
            files=[upload],
            repo="company",
            tag="v1",
            version="1.0",
            acl="public",
            session=session,
        )
        assert result.processed == 1
        assert result.failed == 0

        rows = (await session.execute(select(RagChunk))).scalars().all()
        assert len(rows) == 1
        assert rows[0].repo == "company"
        assert rows[0].acl == ["public"]
