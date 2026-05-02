"""Unified error envelope and FastAPI handlers."""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ErrorCode(str, Enum):
    VALIDATION_ERROR = "VALIDATION_ERROR"
    LOCATION_MISSING = "LOCATION_MISSING"
    FARMER_NOT_FOUND = "FARMER_NOT_FOUND"
    IMAGE_REF_EXPIRED = "IMAGE_REF_EXPIRED"
    IMAGE_TOO_LARGE = "IMAGE_TOO_LARGE"
    IMAGE_UNSUPPORTED_TYPE = "IMAGE_UNSUPPORTED_TYPE"
    LLM_TIMEOUT = "LLM_TIMEOUT"
    UPSTREAM_RATE_LIMIT = "UPSTREAM_RATE_LIMIT"
    UPSTREAM_UNAVAILABLE = "UPSTREAM_UNAVAILABLE"
    INTERNAL_ERROR = "INTERNAL_ERROR"


_RETRYABLE = {
    ErrorCode.LLM_TIMEOUT,
    ErrorCode.UPSTREAM_RATE_LIMIT,
    ErrorCode.UPSTREAM_UNAVAILABLE,
}

_FALLBACK_USE_ONDEVICE = {
    ErrorCode.LLM_TIMEOUT,
    ErrorCode.UPSTREAM_RATE_LIMIT,
    ErrorCode.UPSTREAM_UNAVAILABLE,
}

_FALLBACK_RETRY_ONLINE = {
    ErrorCode.INTERNAL_ERROR,
}


class ErrorBody(BaseModel):
    code: ErrorCode
    message: str
    retryable: bool
    retry_after_seconds: Optional[int] = None
    fallback_hint: Optional[str] = None


class ErrorEnvelope(BaseModel):
    error: ErrorBody = Field(...)

    @classmethod
    def build(
        cls,
        code: ErrorCode,
        message: str,
        retry_after_seconds: Optional[int] = None,
    ) -> "ErrorEnvelope":
        hint: Optional[str] = None
        if code in _FALLBACK_USE_ONDEVICE:
            hint = "USE_ONDEVICE"
        elif code in _FALLBACK_RETRY_ONLINE:
            hint = "RETRY_ONLINE_LATER"
        return cls(
            error=ErrorBody(
                code=code,
                message=message,
                retryable=code in _RETRYABLE,
                retry_after_seconds=retry_after_seconds,
                fallback_hint=hint,
            )
        )


class KrishiHTTPException(HTTPException):
    """HTTPException that emits a typed error envelope."""

    def __init__(
        self,
        status_code: int,
        code: ErrorCode,
        message: str,
        retry_after_seconds: Optional[int] = None,
    ) -> None:
        super().__init__(status_code=status_code, detail=message)
        self.code = code
        self.retry_after_seconds = retry_after_seconds


_STATUS_TO_CODE = {
    400: ErrorCode.VALIDATION_ERROR,
    404: ErrorCode.FARMER_NOT_FOUND,
    408: ErrorCode.LLM_TIMEOUT,
    413: ErrorCode.IMAGE_TOO_LARGE,
    415: ErrorCode.IMAGE_UNSUPPORTED_TYPE,
    422: ErrorCode.VALIDATION_ERROR,
    429: ErrorCode.UPSTREAM_RATE_LIMIT,
    503: ErrorCode.UPSTREAM_UNAVAILABLE,
}


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(KrishiHTTPException)
    async def _krishi(_: Request, exc: KrishiHTTPException) -> JSONResponse:
        env = ErrorEnvelope.build(exc.code, str(exc.detail), exc.retry_after_seconds)
        headers = {"Retry-After": str(exc.retry_after_seconds)} if exc.retry_after_seconds else None
        return JSONResponse(status_code=exc.status_code, content=env.model_dump(), headers=headers)

    @app.exception_handler(HTTPException)
    async def _http(_: Request, exc: HTTPException) -> JSONResponse:
        code = _STATUS_TO_CODE.get(exc.status_code, ErrorCode.INTERNAL_ERROR)
        env = ErrorEnvelope.build(code, str(exc.detail))
        return JSONResponse(status_code=exc.status_code, content=env.model_dump())

    @app.exception_handler(RequestValidationError)
    async def _val(_: Request, exc: RequestValidationError) -> JSONResponse:
        env = ErrorEnvelope.build(ErrorCode.VALIDATION_ERROR, "invalid request")
        return JSONResponse(status_code=400, content=env.model_dump())

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled: %s", exc)
        env = ErrorEnvelope.build(ErrorCode.INTERNAL_ERROR, "internal error")
        return JSONResponse(status_code=500, content=env.model_dump())
