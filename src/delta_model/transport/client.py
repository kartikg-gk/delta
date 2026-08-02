"""Proxy-aware HTTP utilities for model adapters."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import httpx

_PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def sanitize_proxy_url(proxy_url: str) -> str:
    """Rewrite ambiguous ``socks://`` schemes to the explicit ``socks5://`` that httpx requires."""

    if proxy_url.lower().startswith("socks://"):
        return f"socks5://{proxy_url[len('socks://') :]}"
    return proxy_url


@contextmanager
def patched_proxy_environment() -> Iterator[None]:
    """Swap proxy env vars to httpx-safe values for the duration of the block."""

    original: dict[str, str | None] = {}
    changed = False
    for name in _PROXY_ENV_VARS:
        value = os.environ.get(name)
        if value is None:
            continue
        normalized = sanitize_proxy_url(value)
        if normalized == value:
            continue
        original[name] = value
        os.environ[name] = normalized
        changed = True

    try:
        yield
    finally:
        if changed:
            for name, value in original.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def build_async_client(**kwargs: Any) -> httpx.AsyncClient:
    """Construct an ``httpx.AsyncClient`` with sanitized proxy settings."""

    with patched_proxy_environment():
        return httpx.AsyncClient(**kwargs)


def fetch_json(url: str, *, timeout: float, follow_redirects: bool = False) -> dict[str, object]:
    """GET a URL and return the parsed JSON dict, using sanitized proxy settings."""

    with patched_proxy_environment():
        response = httpx.get(url, timeout=timeout, follow_redirects=follow_redirects)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object in the response body")
    return data
