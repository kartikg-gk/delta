"""Request preparation, decode pipeline, and fault builders for the OpenAI provider.

Extracts the mechanical concerns of request construction, SSE-to-WireEvent
conversion, and error packaging from the provider adapter, leaving it free
to focus on retry orchestration and credential lifecycle.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any

from delta_harness.contracts.tooling import ToolSpec
from delta_harness.contracts.transcript import ModelEntry, TranscriptEntry
from delta_harness.provider.wire import StreamFaultEvent, WireEvent
from delta_model._oai.helpers import build_reasoning_extra, pick_endpoint
from delta_model._oai.normalize import EventAssembler
from delta_model._oai.parsers import ChatDecoder, ResponsesDecoder
from delta_model._oai.payloads import build_chat_payload, build_responses_payload
from delta_model._oai.parsers import ParseFault
from delta_model._oai.transport import HttpStreamError, MalformedPayload
from delta_model.settings import Credential, OpenAIProfile
from delta_model.transport.faults import format_http_error


# ---------------------------------------------------------------------------
# Prepared request
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PreparedRequest:
    """Immutable bundle of everything needed for one streaming API call."""

    url: str
    payload: dict[str, Any]
    headers: dict[str, str]
    endpoint: str  # "chat" | "responses"


# ---------------------------------------------------------------------------
# Header construction
# ---------------------------------------------------------------------------


def _auth_headers(cred: Credential, profile: OpenAIProfile) -> dict[str, str]:
    """Build HTTP headers from credential and profile settings."""
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if cred.api_key:
        headers["Authorization"] = f"Bearer {cred.api_key}"
    if profile.organization:
        headers["OpenAI-Organization"] = profile.organization
    for key, value in cred.extra_headers:
        headers[key] = value
    return headers


# ---------------------------------------------------------------------------
# Request preparation
# ---------------------------------------------------------------------------


def prepare_request(
    cred: Credential,
    profile: OpenAIProfile,
    *,
    model: str,
    system: str,
    messages: Sequence[TranscriptEntry],
    tools: Sequence[ToolSpec],
) -> PreparedRequest:
    """Build URL, payload, and headers for one streaming request."""
    base_url = cred.base_url.rstrip("/")
    endpoint = pick_endpoint(model, base_url)
    reasoning_extra = build_reasoning_extra(profile.reasoning) or None

    if endpoint == "responses":
        url = f"{base_url}/responses"
        payload = build_responses_payload(
            model=model, system=system, messages=messages,
            tools=tools, extra=reasoning_extra,
        )
    else:
        url = f"{base_url}/chat/completions"
        payload = build_chat_payload(
            model=model, system=system, messages=messages,
            tools=tools, extra=reasoning_extra,
        )

    return PreparedRequest(
        url=url,
        payload=payload,
        headers=_auth_headers(cred, profile),
        endpoint=endpoint,
    )


def refresh_headers(
    request: PreparedRequest,
    cred: Credential,
    profile: OpenAIProfile,
) -> PreparedRequest:
    """Return a copy of *request* with freshly built auth headers."""
    return PreparedRequest(
        url=request.url,
        payload=request.payload,
        headers=_auth_headers(cred, profile),
        endpoint=request.endpoint,
    )


# ---------------------------------------------------------------------------
# Decode pipeline
# ---------------------------------------------------------------------------


async def decode_stream(
    sse_source: AsyncIterator[tuple[str, dict[str, Any]]],
    *,
    endpoint: str,
    model: str,
    provider: str,
) -> AsyncIterator[WireEvent]:
    """Transform a parsed SSE stream into Delta wire events.

    Composes the appropriate API decoder with an ``EventAssembler`` to
    produce the full ``WireEvent`` sequence the harness loop expects.
    A ``MalformedPayload`` from the transport layer is caught and
    converted into a ``StreamFaultEvent``.
    """
    decoder = ChatDecoder() if endpoint == "chat" else ResponsesDecoder()
    assembler = EventAssembler(model=model, provider=provider, api=endpoint)

    try:
        async for event_name, data in sse_source:
            for signal in decoder.decode(event_name, data):
                for event in assembler.accept(signal):
                    yield event
    except MalformedPayload as exc:
        for event in assembler.accept(ParseFault(message=str(exc))):
            yield event

    for event in assembler.finalize():
        yield event


# ---------------------------------------------------------------------------
# Fault builders
# ---------------------------------------------------------------------------


def abort_fault(model: str, provider: str, endpoint: str) -> StreamFaultEvent:
    """Build an abort event for a cancelled request."""
    entry = ModelEntry(
        model=model, provider=provider, api=endpoint,
        content=[], stop_reason="aborted",
    )
    return StreamFaultEvent(reason="aborted", error=entry)


def http_fault(
    model: str,
    provider: str,
    endpoint: str,
    exc: HttpStreamError,
) -> StreamFaultEvent:
    """Build a fault event from an HTTP error response."""
    msg = format_http_error(
        provider_name=provider, status_code=exc.status,
        body=exc.body, model=model,
    )
    entry = ModelEntry(
        model=model, provider=provider, api=endpoint,
        content=[], stop_reason="error", error_message=msg,
    )
    return StreamFaultEvent(reason="error", error=entry)


def network_fault(
    model: str, provider: str, endpoint: str, exc: Exception,
) -> StreamFaultEvent:
    """Build a fault event from a network-level exception."""
    entry = ModelEntry(
        model=model, provider=provider, api=endpoint,
        content=[], stop_reason="error",
        error_message=f"{provider} network error: {exc}",
    )
    return StreamFaultEvent(reason="error", error=entry)
