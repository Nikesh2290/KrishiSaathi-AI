"""Pytest fixtures."""

from __future__ import annotations

import asyncio
import os

import pytest

from config.settings import get_settings


@pytest.fixture(autouse=True)
def test_env(tmp_path_factory):
    root = tmp_path_factory.mktemp("ks")
    os.environ["DATABASE_PATH"] = str(root / "test.db")
    os.environ["CHROMA_PATH"] = str(root / "chroma")
    os.environ["GOOGLE_AI_STUDIO_KEY"] = ""
    os.environ["CONNECTIVITY_MODE"] = "local"
    os.environ["CORS_ORIGINS"] = "http://localhost:3000"
    # Avoid writing ./logs during tests when .env sets LOG_FILE
    os.environ["LOG_FILE"] = ""
    get_settings.cache_clear()

    async def _init():
        from db.sqlite_client import init_db

        await init_db()

    asyncio.run(_init())
    yield
    for k in ("DATABASE_PATH", "CHROMA_PATH", "CONNECTIVITY_MODE", "LOG_FILE"):
        os.environ.pop(k, None)
    get_settings.cache_clear()
