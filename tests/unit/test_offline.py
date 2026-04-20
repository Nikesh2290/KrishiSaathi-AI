"""Offline data tests."""

from __future__ import annotations

import pytest

from offline.sync import SyncJob


@pytest.mark.asyncio
async def test_sync_job_httpx(monkeypatch):
    import httpx

    class DummyClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return None

        async def get(self, url: str):
            req = httpx.Request("GET", url)
            return httpx.Response(200, json={"ok": True}, request=req)

    monkeypatch.setattr("offline.sync.httpx.AsyncClient", lambda *a, **k: DummyClient())
    j = SyncJob()
    out = await j.run()
    assert out.get("ok") is True


def test_bootstrap_creates_files(tmp_path, monkeypatch):
    from offline import bootstrap_data as bd

    monkeypatch.setattr(bd, "ROOT", tmp_path / "data")
    bd.bootstrap_all()
    assert (tmp_path / "data" / "scheme_index.json").exists()
    assert (tmp_path / "data" / "mandi_prices.csv").exists()
    assert (tmp_path / "data" / "weather_history.json").exists()
