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
from dataclasses import replace

import httpx

from delta_harness.contracts.tooling import CancelToken, ToolSpec
from delta_harness.contracts.transcript import ModelEntry, TranscriptEntry
from delta_harness.provider.wire import StreamFaultEvent, WireEvent
from delta_model._oai.helpers import build_reasoning_extra, pick_endpoint, try_parse_json
from delta_model._oai.normalize import EventAssembler
from delta_model._oai.parsers import ChatDecoder, ParseFault, ResponsesDecoder, StreamDecoder
from delta_model._oai.payloads import build_chat_payload, build_responses_payload
from delta_model._oai.transport import HttpStreamError, MalformedPayload, open_event_stream
from delta_model.settings import Credential, OpenAIProfile, ReasoningPolicy
from delta_model.transport.backoff import compute_delay, pause_for_retry
from delta_model.transport.client import build_async_client
from delta_model.transport.faults import format_http_error


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
    ) -> AsyncIterator[WireEvent]:
        """Stream wire events for one model round-trip."""
        return self._stream(
            model=model,
            system=system,
            messages=messages,
            tools=tools,
            signal=signal,
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
    ) -> AsyncIterator[WireEvent]:
        """Execute the streaming request with retry and normalisation."""
        cred = await self._resolve_credential()
        base_url = cred.base_url.rstrip("/")
        endpoint = pick_endpoint(model, base_url)
        reasoning_extra = build_reasoning_extra(self._profile.reasoning)

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

                try:
                    async for sse in open_event_stream(
                        self._client, url, payload, headers, signal=signal,
                    ):
                        data = try_parse_json(sse.data)
                        if data is None:
                            raise MalformedPayload(sse.data)
                        for sig in decoder.decode(sse.event, data):
                            for wire_event in assembler.accept(sig):
                                yield wire_event
                except MalformedPayload as exc:
                    for wire_event in assembler.accept(ParseFault(message=str(exc))):
                        yield wire_event

                for wire_event in assembler.finalize():
                    yield wire_event
                return  # success

            except HttpStreamError as exc:
                if not exc.retriable or attempt >= retry.max_retries:
                    yield self._http_error_event(model, endpoint, exc)
                    return

                delay = compute_delay(
                    attempt, max_delay_seconds=retry.max_delay_seconds,
                )
                if exc.retry_after_seconds is not None:
                    delay = max(delay, exc.retry_after_seconds)

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
                still_alive = await pause_for_retry(delay, signal=signal)
                if not still_alive:
                    yield self._abort_event(model, endpoint)
                    return

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
