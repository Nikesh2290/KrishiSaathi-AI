"""FastAPI entrypoint."""

from __future__ import annotations

from contextlib import asynccontextmanager
import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import auth, cache as cache_routes, conversation, farmer, health, jobs, market, query, sync, voice, weather
from api.middleware.request_logging import install_request_logging
from config.logging import configure_logging
from config.settings import get_settings
from db.sqlite_client import init_db
from models.errors import register_exception_handlers
from modules.scheme import vector_store
from offline.bootstrap_data import bootstrap_all


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    await init_db(settings)
    bootstrap_all()
    try:
        vector_store.build_index()
    except Exception:
        import logging

        logging.getLogger(__name__).exception("vector_store.build_index() failed")
    if settings.supabase_db_configured:
        import logging as _log

        async def _startup_supabase_row_sync() -> None:
            try:
                from offline.supabase_sync import SupabaseSync

                await SupabaseSync(settings).run_without_vectors()
            except Exception:
                _log.getLogger(__name__).exception(
                    "SupabaseSync run_without_vectors on startup failed"
                )

        async def _startup_scheme_vectors() -> None:
            try:
                from offline.supabase_sync import SupabaseSync

                n = await SupabaseSync(settings).sync_scheme_vectors()
                _log.getLogger(__name__).info("startup scheme_vectors chunks=%s", n)
            except Exception:
                _log.getLogger(__name__).exception(
                    "SupabaseSync scheme vectors on startup failed"
                )

        asyncio.create_task(_startup_supabase_row_sync())
        asyncio.create_task(_startup_scheme_vectors())

    if settings.redis_configured and settings.startup_cache_warmup:
        import logging as _log

        async def _startup_warm() -> None:
            try:
                from cache.warmup import run_warmup, scopes_for_cron

                await run_warmup(
                    settings,
                    scopes=scopes_for_cron("frequent"),
                    include_scheme_vectors=False,
                )
            except Exception:
                _log.getLogger(__name__).exception("Startup cache warm-up failed")

        asyncio.create_task(_startup_warm())

    yield

    try:
        from db import supabase_client

        await supabase_client.close_supabase_http()
    except Exception:
        import logging

        logging.getLogger(__name__).exception("close_supabase_http on shutdown failed")


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(
        level=settings.log_level,
        json_logs=bool(settings.log_json),
        log_file=(settings.log_file or None),
    )
    app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    install_request_logging(app)
    register_exception_handlers(app)
    app.include_router(auth.router)
    app.include_router(query.router)
    app.include_router(conversation.router)
    app.include_router(farmer.router)
    app.include_router(health.router)
    app.include_router(cache_routes.router)
    app.include_router(jobs.router)
    app.include_router(sync.router)
    app.include_router(market.router)
    app.include_router(weather.router)
    app.include_router(voice.router)
    return app


app = create_app()
