"""OpenAI-compatible provider adapter for the Delta agent harness.

``OpenAIProvider`` implements the ``ModelProvider`` protocol by composing the
internal sub-modules for payload construction, SSE-based stream decoding, event
normalisation, and HTTP transport with retry.  It supports both the Chat
Completions and Responses APIs, selecting the appropriate endpoint automatically
based on model name and base URL.

Usage::

    from delta_model.settings import ReasoningPolicy, load_openai_profile
    from delta_model.oai_compatible import OpenAIProvider

    profile = load_openai_profile()
    provider = OpenAIProvider(profile)

    async for event in provider.stream_response(
        model="gpt-4o",
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

import httpx

from delta_harness.contracts.tooling import CancelToken, ToolSpec
from delta_harness.contracts.transcript import ModelEntry, TranscriptEntry
from delta_harness.provider.wire import StreamFaultEvent, WireEvent
from delta_model._oai.helpers import (
    build_reasoning_extra,
    is_passing_failure,
    pick_endpoint,
    stream_failure,
    try_parse_json,
)
from delta_model._oai.normalize import EventAssembler
from delta_model._oai.parsers import ChatDecoder, ParseFault, ResponsesDecoder, StreamDecoder
from delta_model._oai.payloads import build_chat_payload, build_responses_payload
from delta_model._oai.transport import HttpStreamError, MalformedPayload, open_event_stream
from delta_model.settings import Credential, OpenAIProfile, ReasoningPolicy
from delta_model.transport.backoff import build_retry_event, compute_delay, pause_for_retry
from delta_model.transport.client import build_async_client
from delta_model.transport.faults import format_http_error

_FIRST_PARTY_HOST = "api.openai.com"
# Routers that front several backends name the one that answered here.
_SERVED_BY_HEADER = "x-inference-provider"
_ROUTING_KEY_LIMIT = 64


class OpenAIProvider:
    """Streaming model adapter for OpenAI-compatible API endpoints.

    The provider owns an ``httpx.AsyncClient`` and delegates each streaming
    request through the transport layer.  Transient HTTP failures are retried
    with exponential backoff according to the profile's ``RetryPolicy``.
    An optional ``CredentialResolver`` on the profile is called before each
    attempt to support token rotation or secrets-manager integration.
    """

    def __init__(self, profile: OpenAIProfile) -> None:
        self._profile = profile
        self._client: httpx.AsyncClient = build_async_client(
            timeout=httpx.Timeout(profile.timeout_seconds),
        )
        #: Backend that served the latest accepted response, when reported.
        self.served_by: str | None = None
        #: HTTP status behind the latest request that ended in a fault.
        self.last_failure_status: int | None = None

    def set_reasoning(self, policy: ReasoningPolicy) -> None:
        """Swap the reasoning-effort policy for subsequent requests.

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
            cache_key=cache_key,
        )

    # --- internals ----------------------------------------------------------

    async def _resolve_credential(self) -> Credential:
        """Obtain authentication material, refreshing if a resolver is configured."""
        if self._profile.resolver is not None:
            return await self._profile.resolver()
        return self._profile.credential

    def _build_headers(self, cred: Credential) -> dict[str, str]:
        """Assemble HTTP headers from the credential and profile."""
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if cred.api_key:
            headers["Authorization"] = f"Bearer {cred.api_key}"
        if self._profile.organization:
            headers["OpenAI-Organization"] = self._profile.organization
        for key, value in cred.extra_headers:
            headers[key] = value
        return headers

    async def _stream(
        self,
        *,
        model: str,
        system: str,
        messages: Sequence[TranscriptEntry],
        tools: Sequence[ToolSpec],
        signal: CancelToken | None,
        cache_key: str | None = None,
    ) -> AsyncIterator[WireEvent]:
        """Execute the streaming request with retry and normalisation."""
        self.served_by = None
        self.last_failure_status = None
        cred = await self._resolve_credential()
        base_url = cred.base_url.rstrip("/")
        endpoint = pick_endpoint(model, base_url)
        reasoning_extra = build_reasoning_extra(
            self._profile.reasoning, model=model, endpoint=endpoint,
        )

        if endpoint == "responses":
            url = f"{base_url}/responses"
            payload = build_responses_payload(
                model=model,
                system=system,
                messages=messages,
                tools=tools,
                extra=reasoning_extra or None,
            )
        else:
            url = f"{base_url}/chat/completions"
            payload = build_chat_payload(
                model=model,
                system=system,
                messages=messages,
                tools=tools,
                extra=reasoning_extra or None,
            )

        routing_key = self._routing_key(cache_key, base_url)
        if routing_key is not None:
            payload["prompt_cache_key"] = routing_key

        retry = self._profile.retry
        max_attempts = retry.max_retries + 1

        for attempt in range(max_attempts):
            if signal is not None and signal.is_cancelled():
                yield self._abort_event(model, endpoint)
                return

            # Refresh credentials on retries
            if attempt > 0:
                if self._profile.resolver is not None:
                    cred = await self._profile.resolver()
                headers = self._build_headers(cred)
            else:
                headers = self._build_headers(cred)
            if routing_key is not None and endpoint == "responses":
                headers["session_id"] = routing_key

            try:
                decoder: StreamDecoder
                if endpoint == "chat":
                    decoder = ChatDecoder()
                else:
                    decoder = ResponsesDecoder()

                assembler = EventAssembler(
                    model=model,
                    provider=self._profile.name,
                    api=endpoint,
                )

                # Once anything reaches the caller the attempt is committed: a
                # later failure is surfaced, never replayed, so visible output
                # and tool calls cannot be duplicated.
                committed = False
                passing_failure = False
                failure_kind = ""
                try:
                    async with aclosing(open_event_stream(
                        self._client, url, payload, headers, signal=signal,
                        on_accept=self._note_server,
                    )) as events:
                        async for sse in events:
                            data = try_parse_json(sse.data)
                            if data is None:
                                raise MalformedPayload(sse.data)
                            # Some compatible servers omit the `event:` line;
                            # the payload's own `type` names the event then.
                            name = sse.event or str(data.get("type") or "")
                            if not committed and attempt < retry.max_retries:
                                failure = stream_failure(name, data)
                                if failure is not None and is_passing_failure(*failure):
                                    passing_failure = True
                                    failure_kind = failure[0] or "stream error"
                                    break
                            for sig in decoder.decode(name, data):
                                for wire_event in assembler.accept(sig):
                                    committed = True
                                    yield wire_event
                except MalformedPayload as exc:
                    for wire_event in assembler.accept(ParseFault(message=str(exc))):
                        yield wire_event

                if not passing_failure:
                    for wire_event in assembler.finalize():
                        yield wire_event
                    return  # success

                delay = compute_delay(attempt, max_delay_seconds=retry.max_delay_seconds)
                yield build_retry_event(
                    attempt=attempt, max_retries=retry.max_retries,
                    delay_seconds=delay, reason=f"a transient {failure_kind}",
                )
                still_alive = await pause_for_retry(delay, signal=signal)
                if not still_alive:
                    yield self._abort_event(model, endpoint)
                    return
                continue

            except HttpStreamError as exc:
                if not exc.retriable or attempt >= retry.max_retries:
                    self.last_failure_status = exc.status
                    yield self._http_error_event(model, endpoint, exc)
                    return

                delay = compute_delay(
                    attempt, max_delay_seconds=retry.max_delay_seconds,
                )
                if exc.retry_after_seconds is not None:
                    delay = max(delay, exc.retry_after_seconds)

                yield build_retry_event(
                    attempt=attempt, max_retries=retry.max_retries,
                    delay_seconds=delay, reason=f"HTTP {exc.status}",
                )
                still_alive = await pause_for_retry(delay, signal=signal)
                if not still_alive:
                    yield self._abort_event(model, endpoint)
                    return

            except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError) as exc:
                if attempt >= retry.max_retries:
                    yield self._network_error_event(model, endpoint, exc)
                    return

                delay = compute_delay(
                    attempt, max_delay_seconds=retry.max_delay_seconds,
                )
                yield build_retry_event(
                    attempt=attempt, max_retries=retry.max_retries,
                    delay_seconds=delay, reason="a network error",
                )
                still_alive = await pause_for_retry(delay, signal=signal)
                if not still_alive:
                    yield self._abort_event(model, endpoint)
                    return

    def _note_server(self, response: httpx.Response) -> None:
        self.served_by = response.headers.get(_SERVED_BY_HEADER) or None

    def _routing_key(self, cache_key: str | None, base_url: str) -> str | None:
        """The prompt-cache routing value for this request, if it should be sent."""
        if not cache_key:
            return None
        enabled = self._profile.cache_affinity
        if enabled is None:
            enabled = httpx.URL(base_url).host == _FIRST_PARTY_HOST
        # The API caps the key length; the session id is clamped to fit.
        return cache_key[:_ROUTING_KEY_LIMIT] if enabled else None

    # --- error event builders -----------------------------------------------

    def _abort_event(self, model: str, endpoint: str) -> StreamFaultEvent:
        """Build an abort event for a cancelled request."""
        entry = ModelEntry(
            model=model,
            provider=self._profile.name,
            api=endpoint,
            content=[],
            stop_reason="aborted",
        )
        return StreamFaultEvent(reason="aborted", error=entry)

    def _http_error_event(
        self,
        model: str,
        endpoint: str,
        exc: HttpStreamError,
    ) -> StreamFaultEvent:
        """Build a fault event from an HTTP error response."""
        msg = format_http_error(
            provider_name=self._profile.name,
            status_code=exc.status,
            body=exc.body,
            model=model,
        )
        entry = ModelEntry(
            model=model,
            provider=self._profile.name,
            api=endpoint,
            content=[],
            stop_reason="error",
            error_message=msg,
        )
        return StreamFaultEvent(reason="error", error=entry)

    def _network_error_event(
        self,
        model: str,
        endpoint: str,
        exc: Exception,
    ) -> StreamFaultEvent:
        """Build a fault event from a network-level exception."""
        entry = ModelEntry(
            model=model,
            provider=self._profile.name,
            api=endpoint,
            content=[],
            stop_reason="error",
            error_message=f"{self._profile.name} network error: {exc}",
        )
        return StreamFaultEvent(reason="error", error=entry)


__all__ = ["OpenAIProvider"]
