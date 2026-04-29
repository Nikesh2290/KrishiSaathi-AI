"""Supabase email/password auth — returns stable farmer_id (auth user UUID)."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException

from config.settings import get_settings
from db import supabase_client
from db.sqlite_client import upsert_auth_state
from models.auth import AuthResponse, LoginRequest, SignupRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


@router.post("/signup", response_model=AuthResponse)
async def signup(body: SignupRequest) -> AuthResponse:
    settings = get_settings()
    if not settings.supabase_auth_configured:
        raise HTTPException(status_code=503, detail="Supabase Auth not configured")
    try:
        sess = await supabase_client.auth_sign_up(body.email, body.password, settings)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.warning("signup failed: %s", e)
        raise HTTPException(status_code=400, detail=str(e)) from e
    expires_at = int(time.time()) + sess.expires_in
    await upsert_auth_state(
        sess.user_id,
        sess.access_token,
        sess.refresh_token,
        expires_at,
        settings,
    )
    return AuthResponse(
        farmer_id=sess.user_id,
        access_token=sess.access_token,
        expires_in=sess.expires_in,
        refresh_token=sess.refresh_token or None,
    )


@router.post("/login", response_model=AuthResponse)
async def login(body: LoginRequest) -> AuthResponse:
    settings = get_settings()
    if not settings.supabase_auth_configured:
        raise HTTPException(status_code=503, detail="Supabase Auth not configured")
    try:
        sess = await supabase_client.auth_sign_in(body.email, body.password, settings)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e)) from e
    except Exception as e:
        logger.warning("login failed: %s", e)
        raise HTTPException(status_code=401, detail=str(e)) from e
    expires_at = int(time.time()) + sess.expires_in
    await upsert_auth_state(
        sess.user_id,
        sess.access_token,
        sess.refresh_token,
        expires_at,
        settings,
    )
    return AuthResponse(
        farmer_id=sess.user_id,
        access_token=sess.access_token,
        expires_in=sess.expires_in,
        refresh_token=sess.refresh_token or None,
    )
