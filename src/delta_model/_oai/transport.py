"""Streaming HTTP transport for OpenAI-compatible API endpoints.

Responsibilities are deliberately narrow: execute one streaming POST request,
parse the response body as a Server-Sent Events stream, and surface HTTP
failures as a typed exception.  Retry logic and credential management live in
the provider adapter which composes this module.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx

from delta_harness.contracts.tooling import CancelToken
from delta_model._oai.helpers import ServerSentEvent, parse_retry_after


# ---------------------------------------------------------------------------
# Retriable status codes
# ---------------------------------------------------------------------------

_RETRIABLE_CODES = frozenset({429, 500, 502, 503, 504, 529})


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------



class MalformedPayload(ValueError):
    """Raised when an SSE data line cannot be decoded as a valid JSON object."""

    _TRUNCATE = 200

    def __init__(self, raw: str) -> None:
        self.raw = raw
        preview = raw[:self._TRUNCATE] + "…" if len(raw) > self._TRUNCATE else raw
        super().__init__(f"Invalid JSON payload: {preview}")


class HttpStreamError(Exception):
    """Raised when a streaming HTTP request receives a non-200 status."""

    def __init__(
        self,
        status: int,
        body: str,
        *,
        retry_after: float | None = None,
    ) -> None:
        self.status = status
        self.body = body
        self.retry_after_seconds = retry_after
        super().__init__(f"HTTP {status}")

    @property
    def retriable(self) -> bool:
        """Return whether this status code warrants an automatic reattempt."""
        return self.status in _RETRIABLE_CODES


# ---------------------------------------------------------------------------
# SSE stream reader
# ---------------------------------------------------------------------------


async def _read_sse(response: httpx.Response) -> AsyncIterator[ServerSentEvent]:
    """Parse an HTTP response body as a Server-Sent Events stream.

    Yields one ``ServerSentEvent`` per dispatched event.  The OpenAI-specific
    ``[DONE]`` sentinel is consumed silently and terminates the iterator.
    """
    event_type = ""
    data_lines: list[str] = []

    async for raw_line in response.aiter_lines():
        line = raw_line.rstrip("\r\n")

        if not line:
            # Empty line — dispatch the accumulated event
            if data_lines:
                payload = "\n".join(data_lines)
                if payload == "[DONE]":
                    return
                yield ServerSentEvent(event=event_type, data=payload)
                event_type = ""
                data_lines = []
            continue

        if line.startswith(":"):
            continue  # comment

        if line.startswith("event:"):
            event_type = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())


# ---------------------------------------------------------------------------
# Public streaming request
# ---------------------------------------------------------------------------


async def open_event_stream(
    client: httpx.AsyncClient,
    url: str,
    payload: dict,
    headers: dict[str, str],
    *,
    signal: CancelToken | None = None,
) -> AsyncIterator[ServerSentEvent]:
    """Execute a streaming POST and yield parsed SSE events.

    Raises ``HttpStreamError`` on any non-200 status code.  The caller is
    responsible for retry orchestration.
    """
    async with client.stream(
        "POST",
        url,
        json=payload,
        headers=headers,
    ) as response:
        if response.status_code != 200:
            body = (await response.aread()).decode("utf-8", errors="replace")
            raise HttpStreamError(
                response.status_code,
                body,
                retry_after=parse_retry_after(
                    response.headers.get("retry-after")
                ),
            )

        async for sse in _read_sse(response):
            if signal is not None and signal.is_cancelled():
                return
            yield sse
