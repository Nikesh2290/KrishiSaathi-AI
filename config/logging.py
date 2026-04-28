"""Centralized production-grade logging.

Goals:
- Single config point for the whole codebase (incl. uvicorn loggers)
- Structured logs (JSON by default) with full stacktraces
- Request correlation via request_id (propagates through async contextvars)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextvars import ContextVar
from logging.config import dictConfig
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional


request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        rid = getattr(record, "request_id", None)
        if rid:
            payload["request_id"] = rid

        # Common debug fields only when meaningful.
        if record.funcName:
            payload["func"] = record.funcName
        if record.lineno:
            payload["line"] = record.lineno

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        # Include any extra structured fields passed via logger.*(extra={...})
        # Avoid overwriting reserved keys.
        reserved = set(payload.keys()) | {
            "name",
            "message",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
        }
        for k, v in record.__dict__.items():
            if k in reserved or k.startswith("_"):
                continue
            try:
                json.dumps(v)
                payload[k] = v
            except TypeError:
                payload[k] = str(v)

        return json.dumps(payload, ensure_ascii=False)


_configured = False


def configure_logging(
    *,
    level: str = "INFO",
    json_logs: bool = True,
    log_file: Optional[str] = None,
) -> None:
    """Configure root + uvicorn loggers exactly once."""
    global _configured
    if _configured:
        return
    _configured = True

    level = (level or "INFO").upper()

    # Inject request_id into all LogRecords via contextvar.
    old_factory = logging.getLogRecordFactory()

    def record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = old_factory(*args, **kwargs)
        record.request_id = request_id_var.get()
        return record

    logging.setLogRecordFactory(record_factory)

    handlers: Dict[str, Dict[str, Any]] = {
        "console": {
            "class": "logging.StreamHandler",
            "level": level,
            "formatter": "json" if json_logs else "plain",
            "stream": "ext://sys.stdout",
        }
    }

    root_handlers = ["console"]

    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": level,
            "formatter": "json" if json_logs else "plain",
            "filename": log_file,
            "maxBytes": 10 * 1024 * 1024,
            "backupCount": 3,
            "encoding": "utf-8",
        }
        root_handlers.append("file")

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "plain": {
                    "format": "%(asctime)s %(levelname)s %(name)s [rid=%(request_id)s] %(message)s",
                },
                "json": {"()": "config.logging.JsonFormatter"},
            },
            "handlers": handlers,
            "root": {"level": level, "handlers": root_handlers},
            "loggers": {
                # Align uvicorn with our handlers and formatting.
                "uvicorn": {"level": level, "handlers": root_handlers, "propagate": False},
                "uvicorn.error": {"level": level, "handlers": root_handlers, "propagate": False},
                "uvicorn.access": {"level": level, "handlers": root_handlers, "propagate": False},
            },
        }
    )

    # Ensure truly unhandled exceptions still get logged.
    _install_global_exception_hooks()


def _install_global_exception_hooks() -> None:
    logger = logging.getLogger("krishisaathi.unhandled")

    def excepthook(exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        logger.critical("Unhandled exception", exc_info=(exc_type, exc, tb))

    sys.excepthook = excepthook

    # asyncio event loop exceptions (background tasks etc.)
    try:
        import asyncio

        def loop_exception_handler(loop, context):  # type: ignore[no-untyped-def]
            err = context.get("exception")
            if err:
                logger.error("Asyncio exception: %s", err, exc_info=err)
            else:
                logger.error("Asyncio exception context: %s", context)

        try:
            loop = asyncio.get_event_loop()
            loop.set_exception_handler(loop_exception_handler)
        except RuntimeError:
            # No running loop at import time (common in uvicorn); FastAPI lifespan will create one.
            pass
    except Exception:
        # Never fail app startup due to logging hooks.
        return

