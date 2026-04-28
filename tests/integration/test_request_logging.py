"""Request logging middleware: correlation id and safe failure path."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
async def test_health_returns_x_request_id():
    from api.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
        r = await client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.headers.get("x-request-id")
    assert len(r.headers["x-request-id"]) >= 8


@pytest.mark.asyncio
async def test_x_request_id_header_echoed():
    from api.main import app

    custom = "test-req-id-abc123"
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
        r = await client.get("/api/v1/health", headers={"X-Request-Id": custom})
    assert r.status_code == 200
    assert r.headers.get("x-request-id") == custom
