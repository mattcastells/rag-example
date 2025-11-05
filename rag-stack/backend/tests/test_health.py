from app.routes.health import healthcheck


async def test_healthz():
    response = await healthcheck()
    assert response.status == "ok"
