"""Anthropic Messages API provider adapter for the Delta agent harness.

``AnthropicProvider`` implements the ``ModelProvider`` protocol by streaming
responses from the ``/v1/messages`` endpoint.  It composes the courier
(HTTP transport) and emitter (state-machine parser) sub-modules, adding
request-body construction, credential management, and retry orchestration.

Usage::

    from delta_model.settings import load_anthropic_profile
    from delta_model.claude import AnthropicProvider

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

from collections.abc import AsyncIterator, Sequence
from contextlib import aclosing
from dataclasses import replace
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
from delta_harness.provider.wire import StreamFaultEvent, StreamOpenEvent, WireEvent
from delta_model._claude.courier import ApiRejection, relay_sse
from delta_model._claude.emitter import ResponseMachine
from delta_model.correlation import wire_call_id
from delta_model.settings import AnthropicProfile, Credential, ReasoningPolicy
from delta_model.transport.backoff import build_retry_event, compute_delay, pause_for_retry
from delta_model.transport.client import build_async_client
from delta_model.transport.faults import format_http_error

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
    # Reasoning signatures are opaque state owned by the vendor that minted
    # them; replaying another vendor's as Anthropic thinking fails validation.
    own_reasoning = entry.api == "messages"
    for seg in entry.content:
        if isinstance(seg, ThoughtSegment):
            if not (own_reasoning and seg.thinking_signature):
                continue
            if seg.redacted:
                blocks.append({"type": "redacted_thinking", "data": seg.thinking_signature})
            else:
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
                "id": wire_call_id(seg.id),
                "name": seg.name,
                "input": seg.arguments,
            })
    # A turn left empty (e.g. reasoning-only from another vendor) is dropped by
    # the caller: the API rejects empty assistant content.
    return blocks


def _tool_result_block(entry: ToolOutcomeEntry) -> dict[str, Any]:
    """Convert a ``ToolOutcomeEntry`` to an Anthropic tool_result block."""
    block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": wire_call_id(entry.tool_call_id),
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


# ---------------------------------------------------------------------------
# Prompt caching
# ---------------------------------------------------------------------------
#
# The API allows at most four cache markers and evaluates the cached prefix in
# tools -> system -> messages order. All four are spent: the last tool, the
# system prompt, this request's tail, and the previous request's tail. The
# second message marker exists because the API looks back at most 20 blocks
# from a marker; a turn with many parallel tool calls appends enough blocks to
# push the previous cache entry out of a single tail marker's window.

_CACHEABLE_BLOCKS = frozenset({"text", "image", "tool_result"})
_FIRST_PARTY_HOST = "api.anthropic.com"


def resolve_cache_retention(configured: str | None, base_url: str) -> str:
    """Pick the effective cache lifetime for an endpoint."""
    if configured is not None:
        return configured
    host = httpx.URL(base_url).host
    return "short" if host == _FIRST_PARTY_HOST else "none"


def _cache_marker(retention: str) -> dict[str, Any] | None:
    """The ``cache_control`` value for a retention choice; ``None`` disables caching."""
    if retention == "none":
        return None
    if retention == "long":
        return {"type": "ephemeral", "ttl": "1h"}
    return {"type": "ephemeral"}


def _system_field(system: str, marker: dict[str, Any] | None) -> Any:
    """The ``system`` value; a cached prompt must be sent as a marked text block."""
    if marker is None or not system:
        # Caching off keeps the plain-string shape; an empty block with a
        # marker would be rejected outright.
        return system
    return [{"type": "text", "text": system, "cache_control": dict(marker)}]


def _place_message_breakpoints(
    messages: list[dict[str, Any]], marker: dict[str, Any] | None,
) -> None:
    """Mark this request's tail and the previous request's tail, in place."""
    if marker is None or not messages:
        return
    positions = {len(messages) - 1}
    earlier = _previous_request_tail(messages)
    if earlier is not None:
        positions.add(earlier)
    for index in positions:
        _mark_block(messages[index], marker)


def _previous_request_tail(messages: list[dict[str, Any]]) -> int | None:
    """Index where the previous request's message list ended.

    History is append-only and each request stops just before the assistant
    turn it produces, so the last user message before the final assistant turn
    is where the previous request's tail marker sat. When that turn is missing
    the result is an older position, which only shortens the reusable prefix.
    """
    last_reply = next(
        (i for i in range(len(messages) - 1, -1, -1) if messages[i]["role"] == "assistant"),
        None,
    )
    if last_reply is None:
        return None
    return next(
        (i for i in range(last_reply - 1, -1, -1) if messages[i]["role"] == "user"),
        None,
    )


def _mark_block(message: dict[str, Any], marker: dict[str, Any]) -> None:
    """Attach ``marker`` to a user message's final block when that block may carry one."""
    if message.get("role") != "user":
        return
    content = message.get("content")
    if not isinstance(content, list) or not content:
        return
    last = content[-1]
    kind = last.get("type")
    if kind not in _CACHEABLE_BLOCKS:
        return
    # A marker on an empty block is rejected by the API.
    if kind == "text" and not last.get("text"):
        return
    if kind == "tool_result" and not last.get("content"):
        return
    last["cache_control"] = dict(marker)


def _compose_body(
    *,
    model: str,
    system: str,
    messages: Sequence[TranscriptEntry],
    tools: Sequence[ToolSpec],
    max_tokens: int,
    thinking: ReasoningPolicy,
    cache_retention: str = "none",
) -> dict[str, Any]:
    """Assemble the complete Messages API request body."""
    marker = _cache_marker(cache_retention)
    compiled = _compile_messages(messages)
    _place_message_breakpoints(compiled, marker)
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": _system_field(system, marker),
        "messages": compiled,
        "stream": True,
    }
    if tools:
        body["tools"] = [_tool_schema(t) for t in tools]
        if marker is not None:
            # One marker on the final tool caches the whole schema block.
            body["tools"][-1]["cache_control"] = dict(marker)
    thinking_cfg = _thinking_section(thinking, max_tokens)
    if thinking_cfg is not None:
        body["thinking"] = thinking_cfg
    return body


# Error types the Messages API sends inside an otherwise healthy stream when
# the condition is temporary. Anything else (auth, invalid request) is final.
_PASSING_STREAM_ERRORS = frozenset({"api_error", "overloaded_error", "rate_limit_error"})


def _is_passing_stream_error(payload: dict[str, Any]) -> bool:
    """Whether an in-stream ``error`` event is transient and safe to reissue."""
    error = payload.get("error")
    if not isinstance(error, dict):
        return False
    kind = error.get("type")
    return isinstance(kind, str) and kind.lower() in _PASSING_STREAM_ERRORS


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

    def set_reasoning(self, policy: ReasoningPolicy) -> None:
        """Swap the extended-thinking policy for subsequent requests.

        The profile is frozen, so a replacement is built rather than mutated.
        This lets a session change thinking depth mid-conversation without
        tearing down the HTTP client.
        """
        self._profile = replace(self._profile, reasoning=policy)

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
        cache_key: str | None = None,
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
            cache_retention=resolve_cache_retention(
                self._profile.cache_retention, cred.base_url,
            ),
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

            # The stream-open event is held back until real content arrives.
            # Until then nothing has reached the caller, so a transient error
            # event can be retried invisibly; after that it must surface, or
            # visible output and tool calls could be replayed.
            held: list[WireEvent] = []
            committed = False
            passing_failure = False
            failure_kind = ""
            try:
                async with aclosing(relay_sse(
                    self._client, url, body, headers, signal=signal,
                )) as events:
                    async for event_name, payload in events:
                        if (
                            event_name == "error"
                            and not committed
                            and attempt < retry.max_retries
                            and _is_passing_stream_error(payload)
                        ):
                            passing_failure = True
                            failure_kind = payload.get("error", {}).get("type", "stream error")
                            break
                        for wire_event in machine.ingest(event_name, payload):
                            if not committed and isinstance(wire_event, StreamOpenEvent):
                                held.append(wire_event)
                                continue
                            if not committed:
                                committed = True
                                for early in held:
                                    yield early
                                held.clear()
                            yield wire_event

                if not passing_failure:
                    for wire_event in [*held, *machine.seal()]:
                        yield wire_event
                    return

                delay = compute_delay(attempt, max_delay_seconds=retry.max_delay_seconds)
                yield build_retry_event(
                    attempt=attempt, max_retries=retry.max_retries,
                    delay_seconds=delay, reason=f"a transient {failure_kind}",
                )
                alive = await pause_for_retry(delay, signal=signal)
                if not alive:
                    yield self._cancelled(model)
                    return
                continue

            except ApiRejection as exc:
                if not exc.retriable or attempt >= retry.max_retries:
                    yield self._rejection_fault(model, exc)
                    return

                delay = compute_delay(
                    attempt, max_delay_seconds=retry.max_delay_seconds,
                )
                if exc.wait_hint is not None:
                    delay = max(delay, exc.wait_hint)

                yield build_retry_event(
                    attempt=attempt, max_retries=retry.max_retries,
                    delay_seconds=delay, reason=f"HTTP {exc.status}",
                )
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
                yield build_retry_event(
                    attempt=attempt, max_retries=retry.max_retries,
                    delay_seconds=delay, reason="a network error",
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
