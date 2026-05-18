"""Supabase Auth + PostgREST + RPC via httpx (no heavy native deps)."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

from config.settings import Settings, get_settings
from models.farmer import FarmerTwin

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = httpx.Timeout(8.0, connect=5.0)
_SYNC_TIMEOUT = httpx.Timeout(8.0, connect=5.0)

_SUPABASE_LIMITS = httpx.Limits(max_connections=50, max_keepalive_connections=20)
_supabase_http: httpx.AsyncClient | None = None
_supabase_sync_http: httpx.Client | None = None


def get_supabase_http() -> httpx.AsyncClient:
    global _supabase_http
    if _supabase_http is None:
        _supabase_http = httpx.AsyncClient(
            timeout=_DEFAULT_TIMEOUT,
            limits=_SUPABASE_LIMITS,
        )
    return _supabase_http


async def close_supabase_http() -> None:
    global _supabase_http
    if _supabase_http is not None:
        await _supabase_http.aclose()
        _supabase_http = None


def get_supabase_sync_http() -> httpx.Client:
    global _supabase_sync_http
    if _supabase_sync_http is None:
        _supabase_sync_http = httpx.Client(
            timeout=_SYNC_TIMEOUT,
            limits=_SUPABASE_LIMITS,
        )
    return _supabase_sync_http


def _svc_key(settings: Settings) -> str:
    return settings.supabase_service_role_key or settings.supabase_anon_key


def _headers_svc(settings: Settings) -> Dict[str, str]:
    key = _svc_key(settings)
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _headers_anon(settings: Settings) -> Dict[str, str]:
    k = settings.supabase_anon_key
    return {
        "apikey": k,
        "Authorization": f"Bearer {k}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_json_float_list(value: Any) -> List[float]:
    """Normalize embeddings (numpy/list/tuple) into JSON-safe float list."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        value = list(value)
    return [float(x) for x in value]


def _rest_root(settings: Settings) -> str:
    return settings.supabase_url.rstrip("/") + "/rest/v1"


def _join_rest(settings: Settings, path: str) -> str:
    return _rest_root(settings).rstrip("/") + "/" + path.lstrip("/")


class AuthSessionInfo:
    def __init__(
        self,
        user_id: str,
        access_token: str,
        refresh_token: str,
        expires_in: int,
    ) -> None:
        self.user_id = user_id
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.expires_in = expires_in


async def _post_json_expect_ok(url: str, headers: Dict[str, str], body: Dict[str, Any]) -> Dict[str, Any]:
    client = get_supabase_http()
    r = await client.post(url, headers=headers, json=body)
    if r.status_code >= 400:
        try:
            err = r.json()
        except Exception:
            err = r.text[:600]
        raise ValueError(str(err))
    if not r.content:
        return {}
    try:
        return dict(r.json()) if isinstance(r.json(), dict) else {}
    except Exception:
        return {}


def _parse_auth_session(data: Dict[str, Any]) -> AuthSessionInfo:
    """Normalize Supabase signup/token JSON into AuthSessionInfo."""
    tok = str(data.get("access_token") or "")
    rin = str(data.get("refresh_token") or "")
    exp = int(data.get("expires_in") or 3600)

    sess = data.get("session") if isinstance(data.get("session"), dict) else None
    if sess:
        tok = str(sess.get("access_token") or tok)
        rin = str(sess.get("refresh_token") or rin)
        exp = int(sess.get("expires_in") or exp)

    uid = ""
    user = data.get("user") if isinstance(data.get("user"), dict) else None
    if user and user.get("id"):
        uid = str(user["id"])
    elif sess:
        inner = sess.get("user") if isinstance(sess.get("user"), dict) else None
        if inner and inner.get("id"):
            uid = str(inner["id"])
    if not uid:
        uid = str(data.get("user_id") or data.get("id") or "")
    if not uid:
        raise ValueError(
            data.get("error_description")
            or data.get("msg")
            or data.get("message")
            or "Auth response missing user id."
        )
    if not tok:
        raise ValueError(
            "No access token returned (often email confirmation is enabled). "
            "Disable \"Confirm email\" for the anon provider in Supabase Auth for dev/testing."
        )

    return AuthSessionInfo(user_id=uid, access_token=tok, refresh_token=rin, expires_in=exp)


async def auth_sign_up(email: str, password: str, settings: Optional[Settings] = None) -> AuthSessionInfo:
    settings = settings or get_settings()
    if not settings.supabase_auth_configured:
        raise RuntimeError("Supabase Auth not configured")
    base = settings.supabase_url.rstrip("/")
    url = f"{base}/auth/v1/signup"
    data = await _post_json_expect_ok(
        url, _headers_anon(settings), {"email": email, "password": password}
    )
    return _parse_auth_session(data)


async def auth_sign_in(email: str, password: str, settings: Optional[Settings] = None) -> AuthSessionInfo:
    settings = settings or get_settings()
    if not settings.supabase_auth_configured:
        raise RuntimeError("Supabase Auth not configured")
    base = settings.supabase_url.rstrip("/")
    url = f"{base}/auth/v1/token?grant_type=password"
    data = await _post_json_expect_ok(
        url, _headers_anon(settings), {"email": email, "password": password}
    )
    return _parse_auth_session(data)


# --- Farmer twin ---


async def get_farmer_twin_remote(farmer_id: str, settings: Optional[Settings] = None) -> Optional[FarmerTwin]:
    settings = settings or get_settings()
    if not settings.supabase_db_configured:
        return None
    from urllib.parse import urlencode

    q = urlencode({"farmer_id": f"eq.{farmer_id}", "select": "payload"})
    url = _join_rest(settings, "farmer_twin") + f"?{q}"
    headers = dict(_headers_svc(settings))
    try:
        client = get_supabase_http()
        r = await client.get(url, headers=headers)
        if r.status_code == 404 or r.status_code == 406:
            return None
        if r.status_code >= 400:
            logger.warning("get_farmer_twin_remote: %s %s", r.status_code, r.text[:200])
            return None
        rows = r.json()
        if not isinstance(rows, list) or not rows:
            return None
        payload = rows[0].get("payload")
        if isinstance(payload, str):
            payload = json.loads(payload)
        return FarmerTwin.model_validate(payload)
    except Exception as e:
        logger.warning("get_farmer_twin_remote failed: %s", e)
        return None


async def upsert_farmer_twin_remote(twin: FarmerTwin, settings: Optional[Settings] = None) -> None:
    settings = settings or get_settings()
    if not settings.supabase_db_configured:
        return
    payload = twin.model_dump(mode="json")
    row = {"farmer_id": twin.farmer_id, "payload": payload, "updated_at": _utc_now_iso()}
    headers = dict(_headers_svc(settings))
    headers["Prefer"] = "resolution=merge-duplicates"
    url = _join_rest(settings, "farmer_twin")
    client = get_supabase_http()
    r = await client.post(url, headers=headers, json=row)
    if r.status_code >= 400:
        txt = ""
        try:
            txt = r.text[:400]
        except Exception:
            pass
        logger.warning("upsert_farmer_twin_remote: %s %s", r.status_code, txt)
        raise RuntimeError(txt or str(r.status_code))


# --- Conversation metadata ---


async def upsert_conversation_metadata_remote(
    conversation_id: str,
    farmer_id: str,
    title: Optional[str],
    *,
    created_at_unix: Optional[int] = None,
    updated_at_unix: Optional[int] = None,
    settings: Optional[Settings] = None,
) -> None:
    settings = settings or get_settings()
    if not settings.supabase_db_configured:
        return
    created_iso = _utc_now_iso()
    updated_iso = created_iso
    if created_at_unix is not None:
        created_iso = datetime.fromtimestamp(
            created_at_unix, tz=timezone.utc
        ).isoformat()
    if updated_at_unix is not None:
        updated_iso = datetime.fromtimestamp(
            updated_at_unix, tz=timezone.utc
        ).isoformat()
    row: Dict[str, Any] = {
        "conversation_id": conversation_id.strip(),
        "farmer_id": farmer_id,
        "title": title,
        "created_at": created_iso,
        "updated_at": updated_iso,
    }
    headers = dict(_headers_svc(settings))
    headers["Prefer"] = "resolution=merge-duplicates"
    url = _join_rest(settings, "conversation_metadata?on_conflict=conversation_id")
    client = get_supabase_http()
    r = await client.post(url, headers=headers, json=row)
    if r.status_code >= 400:
        raise RuntimeError(r.text[:500])


async def get_conversations_by_farmer_remote(
    farmer_id: str, settings: Optional[Settings] = None
) -> List[Dict[str, Any]]:
    settings = settings or get_settings()
    if not settings.supabase_db_configured:
        return []
    from urllib.parse import urlencode

    q = urlencode(
        {
            "farmer_id": f"eq.{farmer_id}",
            "select": "conversation_id,farmer_id,title,created_at,updated_at",
            "order": "created_at.desc",
        }
    )
    url = _join_rest(settings, "conversation_metadata") + f"?{q}"
    headers = dict(_headers_svc(settings))
    client = get_supabase_http()
    r = await client.get(url, headers=headers)
    if r.status_code >= 400:
        logger.warning("get_conversations_by_farmer_remote: %s %s", r.status_code, r.text[:200])
        return []
    data = r.json()
    if not isinstance(data, list):
        return []
    return data


async def get_conversation_metadata_remote(
    conversation_id: str, settings: Optional[Settings] = None
) -> Optional[Dict[str, Any]]:
    settings = settings or get_settings()
    if not settings.supabase_db_configured:
        return None
    from urllib.parse import urlencode

    cid = (conversation_id or "").strip()
    if not cid:
        return None
    q = urlencode(
        {
            "conversation_id": f"eq.{cid}",
            "select": "conversation_id,farmer_id,title,created_at,updated_at",
        }
    )
    url = _join_rest(settings, "conversation_metadata") + f"?{q}"
    headers = dict(_headers_svc(settings))
    client = get_supabase_http()
    r = await client.get(url, headers=headers)
    if r.status_code >= 400:
        logger.warning("get_conversation_metadata_remote: %s %s", r.status_code, r.text[:200])
        return None
    data = r.json()
    if isinstance(data, list) and data:
        row = data[0]
        return dict(row) if isinstance(row, dict) else None
    return None


# --- Query history ---


async def insert_query_history_remote(
    query_text: str,
    intent: str,
    response: str,
    data_source: str,
    *,
    sqlite_timestamp_unix: Optional[int] = None,
    conversation_id: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> None:
    settings = settings or get_settings()
    if not settings.supabase_db_configured:
        return
    ts_iso = _utc_now_iso()
    if sqlite_timestamp_unix is not None:
        ts_iso = datetime.fromtimestamp(sqlite_timestamp_unix, tz=timezone.utc).isoformat()
    row = {
        "query_text": query_text,
        "intent": intent,
        "response": response,
        "timestamp": ts_iso,
        "data_source": data_source,
    }
    if conversation_id:
        row["conversation_id"] = conversation_id
    headers = dict(_headers_svc(settings))
    url = _join_rest(settings, "query_history")
    client = get_supabase_http()
    r = await client.post(url, headers=headers, json=row)
    if r.status_code >= 400:
        raise RuntimeError(r.text[:500])


def _query_history_ts_sort_key(row: Dict[str, Any]) -> float:
    v = row.get("timestamp")
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str) and v:
        try:
            s = v.replace("Z", "+00:00")
            return datetime.fromisoformat(s).timestamp()
        except Exception:
            return 0.0
    return 0.0


async def get_query_history_by_conversation_remote(
    conversation_id: str, settings: Optional[Settings] = None
) -> List[Dict[str, Any]]:
    settings = settings or get_settings()
    if not settings.supabase_db_configured:
        return []
    from urllib.parse import urlencode

    cid = (conversation_id or "").strip()
    if not cid:
        return []
    # select=* avoids PostgREST quirks with the reserved column name "timestamp"; sort in-process.
    q = urlencode({"conversation_id": f"eq.{cid}", "select": "*"})
    url = _join_rest(settings, "query_history") + f"?{q}"
    headers = dict(_headers_svc(settings))
    client = get_supabase_http()
    r = await client.get(url, headers=headers)
    if r.status_code >= 400:
        logger.warning("get_query_history_by_conversation_remote: %s %s", r.status_code, r.text[:200])
        return []
    data = r.json()
    if not isinstance(data, list):
        return []
    rows = [dict(row) for row in data if isinstance(row, dict)]
    rows.sort(key=_query_history_ts_sort_key)
    return rows


async def delete_conversation_remote(
    conversation_id: str, settings: Optional[Settings] = None
) -> None:
    """Delete query turns then conversation row in Supabase (FK would only NULL query rows)."""
    settings = settings or get_settings()
    if not settings.supabase_db_configured:
        return
    from urllib.parse import urlencode

    cid = (conversation_id or "").strip()
    if not cid:
        return
    filt = urlencode({"conversation_id": f"eq.{cid}"})
    headers = dict(_headers_svc(settings))
    qh_url = _join_rest(settings, "query_history") + f"?{filt}"
    cm_url = _join_rest(settings, "conversation_metadata") + f"?{filt}"
    client = get_supabase_http()
    r1 = await client.delete(qh_url, headers=headers)
    if r1.status_code >= 400:
        raise RuntimeError(
            r1.text[:500] if r1.text else f"delete query_history failed: {r1.status_code}"
        )
    r2 = await client.delete(cm_url, headers=headers)
    if r2.status_code >= 400:
        raise RuntimeError(
            r2.text[:500] if r2.text else f"delete conversation_metadata failed: {r2.status_code}"
        )


def _match_schemes_via_http(settings: Settings, query_embedding: List[float], k: int) -> List[Dict[str, Any]]:
    """Sync httpx wrapper for embedding path (runs in threadpool if needed)."""
    url = _join_rest(settings, "rpc/match_scheme_vectors")
    headers = _headers_svc(settings)
    body = {"query_embedding": _to_json_float_list(query_embedding), "match_count": k}
    client = get_supabase_sync_http()
    r = client.post(url, headers=headers, json=body)
    if r.status_code >= 400:
        logger.warning("match_scheme_vectors RPC: %s %s", r.status_code, r.text[:300])
        return []
    data = r.json()
    if not isinstance(data, list):
        return []
    out: List[Dict[str, Any]] = []
    for row in data:
        sj = row.get("scheme_json")
        if sj is None:
            continue
        if isinstance(sj, dict):
            out.append(sj)
        elif isinstance(sj, str):
            out.append(json.loads(sj))
    return out


def search_schemes_vector_remote_sync(
    query: str, k: int = 5, settings: Optional[Settings] = None
) -> List[Dict[str, Any]]:
    """Sync path using Chroma default embedding + RPC (used from vector_store.search sync API)."""
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    settings = settings or get_settings()
    if not settings.supabase_db_configured:
        return []
    ef = DefaultEmbeddingFunction()
    emb = ef([query])[0]
    return _match_schemes_via_http(settings, emb, k)


async def upsert_scheme_vector_rows(
    rows: List[Dict[str, Any]], settings: Optional[Settings] = None
) -> None:
    settings = settings or get_settings()
    if not settings.supabase_db_configured or not rows:
        return
    headers = dict(_headers_svc(settings))
    headers["Prefer"] = "resolution=merge-duplicates"
    # Explicit conflict target makes upsert behavior deterministic on unique scheme_id.
    url = _join_rest(settings, "scheme_vectors?on_conflict=scheme_id")
    payload_rows: List[Dict[str, Any]] = []
    for row in rows:
        prepared = dict(row)
        if "embedding" in prepared:
            prepared["embedding"] = _to_json_float_list(prepared["embedding"])
        payload_rows.append(prepared)
    client = get_supabase_http()
    r = await client.post(url, headers=headers, json=payload_rows)
    if r.status_code >= 400:
        logger.warning("upsert_scheme_vector_rows: %s %s", r.status_code, r.text[:500])
        raise RuntimeError(r.text[:500])
