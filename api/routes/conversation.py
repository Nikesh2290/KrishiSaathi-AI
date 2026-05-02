"""Conversation / session metadata CRUD."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from config.settings import get_settings
from db.persistence import (
    persist_conversation_metadata,
    resolve_conversation_history,
    resolve_conversations_by_farmer,
)
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


def _serialize_history_message(row: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "query_text": row.get("query_text"),
        "intent": row.get("intent"),
        "response": row.get("response"),
        "data_source": row.get("data_source"),
        "conversation_id": row.get("conversation_id"),
    }
    rid = row.get("id")
    if rid is not None:
        out["id"] = str(rid)
    ts = row.get("timestamp")
    if isinstance(ts, int):
        out["timestamp"] = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    else:
        out["timestamp"] = ts
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


@router.get("/farmer/{farmer_id}/conversations/{conversation_id}/history")
async def get_conversation_history(
    farmer_id: str,
    conversation_id: str,
    connectivity: str = Query(
        "online",
        description="offline = read SQLite only",
    ),
) -> Dict[str, Any]:
    bundle = await resolve_conversation_history(
        farmer_id, conversation_id, connectivity, get_settings()
    )
    if not bundle:
        raise HTTPException(
            status_code=404,
            detail="conversation not found or access denied",
        )
    meta = _serialize_conversation_row(dict(bundle["meta"]))
    messages = [_serialize_history_message(dict(m)) for m in bundle["messages"]]
    return {**meta, "messages": messages}
