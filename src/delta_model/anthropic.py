"""Anthropic Messages API provider adapter for the Delta agent harness.

``AnthropicProvider`` implements the ``ModelProvider`` protocol by streaming
responses from the ``/v1/messages`` endpoint.  It composes the courier
(HTTP transport) and emitter (state-machine parser) sub-modules, adding
request-body construction, credential management, and retry orchestration.

Usage::

    from delta_model.settings import load_anthropic_profile
    from delta_model.anthropic import AnthropicProvider

    profile = load_anthropic_profile()
    provider = AnthropicProvider(profile)

    async for event in provider.stream_response(
        model="claude-sonnet-4-20250514",
        system="You are a helpful assistant.",
        messages=transcript,
        tools=tool_specs,
    ):
        handle(event)
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx

from delta_harness.contracts.tooling import CancelToken, ToolSpec
from delta_harness.contracts.transcript import (
    CallBlock,
    ExtensionEntry,
    ForkSummaryEntry,
    HumanEntry,
    ImageSegment,
    ModelEntry,
    PruneSummaryEntry,
    ShellResultEntry,
    TextSegment,
    ThoughtSegment,
    ToolOutcomeEntry,
    TranscriptEntry,
    entry_to_human,
)
from delta_harness.provider.wire import StreamFaultEvent, WireEvent
from delta_model._claude.courier import ApiRejection, relay_sse
from delta_model._claude.emitter import ResponseMachine
from delta_model.settings import AnthropicProfile, Credential, ReasoningPolicy
from delta_model.transport.backoff import compute_delay, pause_for_retry
from delta_model.transport.faults import format_http_error
from delta_model.transport.http import build_async_client

_API_VERSION = "2023-06-01"


# ---------------------------------------------------------------------------
# Request body construction
# ---------------------------------------------------------------------------


def _user_content(entry: HumanEntry) -> list[dict[str, Any]]:
    """Convert a ``HumanEntry`` to Anthropic content blocks."""
    if isinstance(entry.content, str):
        return [{"type": "text", "text": entry.content}]
    parts: list[dict[str, Any]] = []
    for seg in entry.content:
        if isinstance(seg, TextSegment):
            parts.append({"type": "text", "text": seg.text})
        elif isinstance(seg, ImageSegment):
            parts.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": seg.mime_type,
                    "data": seg.data,
                },
            })
    return parts or [{"type": "text", "text": ""}]


def _assistant_blocks(entry: ModelEntry) -> list[dict[str, Any]]:
    """Convert a ``ModelEntry`` to Anthropic assistant content blocks."""
    blocks: list[dict[str, Any]] = []
    for seg in entry.content:
        if isinstance(seg, ThoughtSegment):
            if seg.thinking_signature:
                blocks.append({
                    "type": "thinking",
                    "thinking": seg.thinking,
                    "signature": seg.thinking_signature,
                })
        elif isinstance(seg, TextSegment):
            block: dict[str, Any] = {"type": "text", "text": seg.text}
            if seg.text_signature:
                block["signature"] = seg.text_signature
            blocks.append(block)
        elif isinstance(seg, CallBlock):
            blocks.append({
                "type": "tool_use",
                "id": seg.id,
                "name": seg.name,
                "input": seg.arguments,
            })
    return blocks or [{"type": "text", "text": ""}]


def _tool_result_block(entry: ToolOutcomeEntry) -> dict[str, Any]:
    """Convert a ``ToolOutcomeEntry`` to an Anthropic tool_result block."""
    block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": entry.tool_call_id,
        "content": entry.text or "",
    }
    if entry.is_error:
        block["is_error"] = True
    return block


def _convert_entry(entry: TranscriptEntry) -> tuple[str, list[dict[str, Any]]]:
    """Map a transcript entry to ``(anthropic_role, content_blocks)``."""
    if isinstance(entry, HumanEntry):
        return ("user", _user_content(entry))
    if isinstance(entry, ModelEntry):
        return ("assistant", _assistant_blocks(entry))
    if isinstance(entry, ToolOutcomeEntry):
        return ("user", [_tool_result_block(entry)])
    if isinstance(entry, (ExtensionEntry, ShellResultEntry, PruneSummaryEntry, ForkSummaryEntry)):
        return ("user", _user_content(entry_to_human(entry)))
    return ("user", [])


def _compile_messages(entries: Sequence[TranscriptEntry]) -> list[dict[str, Any]]:
    """Build a properly alternating Anthropic message list from transcript entries.

    Consecutive entries with the same role are merged into a single message
    to satisfy the API's strict alternation requirement.
    """
    result: list[dict[str, Any]] = []
    for entry in entries:
        role, blocks = _convert_entry(entry)
        if not blocks:
            continue
        if result and result[-1]["role"] == role:
            result[-1]["content"].extend(blocks)
        else:
            result.append({"role": role, "content": list(blocks)})
    return result


def _tool_schema(tool: ToolSpec) -> dict[str, Any]:
    """Convert a ``ToolSpec`` to an Anthropic tool definition."""
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": dict(tool.parameters),
    }


def _thinking_section(
    policy: ReasoningPolicy, max_tokens: int,
) -> dict[str, Any] | None:
    """Build the ``thinking`` parameter for the Messages API."""
    if not policy.enabled:
        return None
    budget = policy.budget_tokens
    if budget is None:
        budget = max(1024, max_tokens - 1024)
    return {"type": "enabled", "budget_tokens": budget}


def _compose_body(
    *,
    model: str,
    system: str,
    messages: Sequence[TranscriptEntry],
    tools: Sequence[ToolSpec],
    max_tokens: int,
    thinking: ReasoningPolicy,
) -> dict[str, Any]:
    """Assemble the complete Messages API request body."""
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": _compile_messages(messages),
        "stream": True,
    }
    if tools:
        body["tools"] = [_tool_schema(t) for t in tools]
    thinking_cfg = _thinking_section(thinking, max_tokens)
    if thinking_cfg is not None:
        body["thinking"] = thinking_cfg
    return body


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


def _auth_headers(cred: Credential) -> dict[str, str]:
    """Build HTTP headers from a credential snapshot."""
    headers: dict[str, str] = {
        "content-type": "application/json",
        "anthropic-version": _API_VERSION,
    }
    if cred.api_key:
        headers["x-api-key"] = cred.api_key
    for key, value in cred.extra_headers:
        headers[key] = value
    return headers


# ---------------------------------------------------------------------------
# Provider adapter
# ---------------------------------------------------------------------------


class AnthropicProvider:
    """Streaming model adapter for Anthropic's Messages API.

    The provider owns an ``httpx.AsyncClient`` and drives each streaming
    request through the courier module.  Transient HTTP failures are
    retried with exponential backoff.  An optional ``CredentialResolver``
    is invoked before each attempt to support dynamic token refresh.
    """

    def __init__(self, profile: AnthropicProfile) -> None:
        self._profile = profile
        self._client: httpx.AsyncClient = build_async_client(
            timeout=httpx.Timeout(profile.timeout_seconds),
        )

    async def close(self) -> None:
        """Release the underlying HTTP client resources."""
        await self._client.aclose()

    # --- ModelProvider protocol ---------------------------------------------

    def stream_response(
        self,
        *,
        model: str,
        system: str,
        messages: Sequence[TranscriptEntry],
        tools: Sequence[ToolSpec],
        signal: CancelToken | None = None,
    ) -> AsyncIterator[WireEvent]:
        """Stream wire events for one model round-trip."""
        return self._stream(
            model=model,
            system=system,
            messages=messages,
            tools=tools,
            signal=signal,
        )

    # --- core streaming loop ------------------------------------------------

    async def _stream(
        self,
        *,
        model: str,
        system: str,
        messages: Sequence[TranscriptEntry],
        tools: Sequence[ToolSpec],
        signal: CancelToken | None,
    ) -> AsyncIterator[WireEvent]:
        cred = await self._fresh_credential()
        base = cred.base_url.rstrip("/")
        url = f"{base}/v1/messages"

        body = _compose_body(
            model=model,
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=self._profile.max_tokens,
            thinking=self._profile.reasoning,
        )

        retry = self._profile.retry
        ceiling = retry.max_retries + 1

        for attempt in range(ceiling):
            if signal is not None and signal.is_cancelled():
                yield self._cancelled(model)
                return

            if attempt > 0 and self._profile.resolver is not None:
                cred = await self._profile.resolver()

            headers = _auth_headers(cred)
            machine = ResponseMachine(model=model, provider=self._profile.name)

            try:
                async for event_name, payload in relay_sse(
                    self._client, url, body, headers, signal=signal,
                ):
                    for wire_event in machine.ingest(event_name, payload):
                        yield wire_event

                for wire_event in machine.seal():
                    yield wire_event
                return

            except ApiRejection as exc:
                if not exc.retriable or attempt >= retry.max_retries:
                    yield self._rejection_fault(model, exc)
                    return

                delay = compute_delay(
                    attempt, max_delay_seconds=retry.max_delay_seconds,
                )
                if exc.wait_hint is not None:
                    delay = max(delay, exc.wait_hint)

                alive = await pause_for_retry(delay, signal=signal)
                if not alive:
                    yield self._cancelled(model)
                    return

            except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError) as exc:
                if attempt >= retry.max_retries:
                    yield self._network_fault(model, exc)
                    return

                delay = compute_delay(
                    attempt, max_delay_seconds=retry.max_delay_seconds,
                )
                alive = await pause_for_retry(delay, signal=signal)
                if not alive:
                    yield self._cancelled(model)
                    return

    # --- credential helpers -------------------------------------------------

    async def _fresh_credential(self) -> Credential:
        if self._profile.resolver is not None:
            return await self._profile.resolver()
        return self._profile.credential

    # --- fault builders -----------------------------------------------------

    def _cancelled(self, model: str) -> StreamFaultEvent:
        entry = ModelEntry(
            model=model,
            provider=self._profile.name,
            api="messages",
            content=[],
            stop_reason="aborted",
        )
        return StreamFaultEvent(reason="aborted", error=entry)

    def _rejection_fault(self, model: str, exc: ApiRejection) -> StreamFaultEvent:
        msg = format_http_error(
            provider_name=self._profile.name,
            status_code=exc.status,
            body=exc.detail,
            model=model,
        )
        entry = ModelEntry(
            model=model,
            provider=self._profile.name,
            api="messages",
            content=[],
            stop_reason="error",
            error_message=msg,
        )
        return StreamFaultEvent(reason="error", error=entry)

    def _network_fault(self, model: str, exc: Exception) -> StreamFaultEvent:
        entry = ModelEntry(
            model=model,
            provider=self._profile.name,
            api="messages",
            content=[],
            stop_reason="error",
            error_message=f"{self._profile.name} network error: {exc}",
        )
        return StreamFaultEvent(reason="error", error=entry)


__all__ = ["AnthropicProvider"]
