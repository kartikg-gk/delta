"""Redact-safe formatting of HTTP failures from model providers."""

from __future__ import annotations

from collections.abc import Mapping
from json import JSONDecodeError, loads
from typing import Any

_MAX_ERROR_DETAIL_LENGTH = 1000


def format_http_error(
    *,
    provider_name: str,
    status_code: int,
    body: str,
    model: str | None = None,
) -> str:
    """Format a user-visible, credential-free error line from a failed provider HTTP call."""
    prefix = f"{provider_name} request failed with status {status_code}"
    if model:
        prefix = f"{prefix} for model {model}"
    detail = extract_http_detail(body)
    if detail:
        return f"{prefix}: {detail}"
    return prefix


def extract_http_detail(body: str) -> str:
    """Pull the most informative snippet from a raw HTTP error body."""
    parsed = _parse_object(body)
    if parsed is not None:
        detail = extract_mapping_detail(parsed)
        if detail:
            return detail
    return body.strip()[:_MAX_ERROR_DETAIL_LENGTH]


def extract_mapping_detail(value: Mapping[str, Any]) -> str:
    """Dig through nested dicts for the best human-readable error string."""
    error = value.get("error")
    if isinstance(error, Mapping):
        message = error.get("message")
        if isinstance(message, str) and message:
            return message
        code = error.get("code")
        if isinstance(code, str) and code:
            return code
    for key in ("message", "detail", "error"):
        detail = value.get(key)
        if isinstance(detail, str) and detail:
            return detail
        if isinstance(detail, Mapping):
            nested = extract_mapping_detail(detail)
            if nested:
                return nested
    return ""


def _parse_object(value: str) -> Mapping[str, Any] | None:
    try:
        parsed = loads(value)
    except JSONDecodeError:
        return None
    return parsed if isinstance(parsed, Mapping) else None
