"""Request logging + request_id correlation."""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI, Request, Response

from config.logging import request_id_var

logger = logging.getLogger("api.request")

# Avoid huge or sensitive blobs in access logs (query strings can contain tokens).
_MAX_QUERY_LEN = 500
_MAX_USER_AGENT_LEN = 200


def _truncate(s: str | None, max_len: int) -> str | None:
    if s is None:
        return None
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def install_request_logging(app: FastAPI) -> None:
    @app.middleware("http")
    async def _log_requests(request: Request, call_next):  # type: ignore[no-untyped-def]
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        token = request_id_var.set(rid)
        start = time.perf_counter()
        response: Response | None = None
        try:
            response = await call_next(request)
        except BaseException:
            duration_ms = (time.perf_counter() - start) * 1000.0
            logger.info(
                "http",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "query": _truncate(request.url.query or "", _MAX_QUERY_LEN),
                    "status": 500,
                    "duration_ms": round(duration_ms, 2),
                    "client": request.client.host if request.client else None,
                    "user_agent": _truncate(
                        request.headers.get("user-agent"), _MAX_USER_AGENT_LEN
                    ),
                },
            )
            request_id_var.reset(token)
            raise
        else:
            duration_ms = (time.perf_counter() - start) * 1000.0
            logger.info(
                "http",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "query": _truncate(request.url.query or "", _MAX_QUERY_LEN),
                    "status": response.status_code,
                    "duration_ms": round(duration_ms, 2),
                    "client": request.client.host if request.client else None,
                    "user_agent": _truncate(
                        request.headers.get("user-agent"), _MAX_USER_AGENT_LEN
                    ),
                },
            )
            request_id_var.reset(token)
            response.headers["X-Request-Id"] = rid
            return response

