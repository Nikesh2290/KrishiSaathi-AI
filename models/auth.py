"""Auth API payloads."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class SignupRequest(BaseModel):
    email: str = Field(..., min_length=3, description="User email address")
    password: str = Field(..., min_length=6)


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=3)
    password: str = Field(..., min_length=1)


class AuthResponse(BaseModel):
    """farmer_id is the stable Supabase auth user UUID for offline + online."""

    farmer_id: str
    access_token: str
    expires_in: int = 3600
    refresh_token: Optional[str] = None
