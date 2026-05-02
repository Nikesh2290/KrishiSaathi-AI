"""FastAPI entrypoint."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import auth, farmer, health, query, sync, weather
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
        try:
            from offline.supabase_sync import SupabaseSync

            await SupabaseSync(settings).run()
        except Exception:
            import logging

            logging.getLogger(__name__).exception("SupabaseSync on startup failed")
    yield


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
    app.include_router(farmer.router)
    app.include_router(health.router)
    app.include_router(sync.router)
    app.include_router(weather.router)
    return app


app = create_app()
