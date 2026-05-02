"""Conversation / session metadata CRUD."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from config.settings import get_settings
from db.persistence import persist_conversation_metadata, resolve_conversations_by_farmer
from db.sqlite_client import get_conversation_metadata

router = APIRouter(prefix="/api/v1", tags=["conversation"])


class CreateConversationBody(BaseModel):
    farmer_id: str = Field(..., description="Authenticated user / farmer UUID")
    title: Optional[str] = Field(None, description="Optional session title")


def _serialize_conversation_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize SQLite unix timestamps vs Supabase ISO strings for JSON."""
    out = dict(row)
    for key in ("created_at", "updated_at"):
        v = out.get(key)
        if isinstance(v, int):
            out[key] = datetime.fromtimestamp(v, tz=timezone.utc).isoformat()
    # Never expose sync flag to clients
    out.pop("synced", None)
    return out


@router.post("/conversation")
async def create_conversation(
    body: CreateConversationBody,
    connectivity: str = Query(
        "online",
        description="offline = SQLite only until next sync",
    ),
) -> Dict[str, Any]:
    settings = get_settings()
    conversation_id = str(uuid.uuid4())
    await persist_conversation_metadata(
        conversation_id,
        body.farmer_id,
        body.title,
        connectivity,
        settings=settings,
    )
    meta = await get_conversation_metadata(conversation_id, settings)
    if not meta:
        return {
            "conversation_id": conversation_id,
            "farmer_id": body.farmer_id,
            "title": body.title,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    return _serialize_conversation_row(meta)


@router.get("/farmer/{farmer_id}/conversations")
async def list_conversations(
    farmer_id: str,
    connectivity: str = Query("online"),
) -> List[Dict[str, Any]]:
    rows = await resolve_conversations_by_farmer(farmer_id, connectivity, get_settings())
    return [_serialize_conversation_row(dict(r)) for r in rows]
