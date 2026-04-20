"""FastAPI entrypoint."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import farmer, health, query
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
        pass
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_exception_handlers(app)
    app.include_router(query.router)
    app.include_router(farmer.router)
    app.include_router(health.router)
    return app


app = create_app()
