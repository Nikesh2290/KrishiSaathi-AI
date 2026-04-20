import pytest
from fastapi import FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient

from models.errors import (
    ErrorCode,
    ErrorEnvelope,
    KrishiHTTPException,
    register_exception_handlers,
)


def test_envelope_serializes_all_fields():
    env = ErrorEnvelope.build(
        code=ErrorCode.UPSTREAM_RATE_LIMIT,
        message="quota exhausted",
        retry_after_seconds=30,
    )
    d = env.model_dump()
    assert d["error"]["code"] == "UPSTREAM_RATE_LIMIT"
    assert d["error"]["retryable"] is True
    assert d["error"]["fallback_hint"] == "USE_ONDEVICE"
    assert d["error"]["retry_after_seconds"] == 30


@pytest.mark.asyncio
async def test_krishi_http_exception_returns_envelope():
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/boom")
    async def boom():
        raise KrishiHTTPException(
            status_code=429,
            code=ErrorCode.UPSTREAM_RATE_LIMIT,
            message="busy",
            retry_after_seconds=10,
        )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/boom")
    assert r.status_code == 429
    body = r.json()
    assert body["error"]["code"] == "UPSTREAM_RATE_LIMIT"
    assert body["error"]["fallback_hint"] == "USE_ONDEVICE"
    assert body["error"]["retryable"] is True


@pytest.mark.asyncio
async def test_fastapi_http_exception_is_wrapped():
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/notfound")
    async def nf():
        raise HTTPException(status_code=404, detail="no such farmer")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/notfound")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "FARMER_NOT_FOUND"
    assert r.json()["error"]["retryable"] is False
