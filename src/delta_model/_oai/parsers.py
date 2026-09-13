"""Streaming response decoders for Chat Completions and Responses APIs.

Each decoder processes pre-parsed SSE payloads (an event name and a JSON
dict) and produces a sequence of provider-neutral ``ParsedSignal`` values.
These signals carry only the semantic content of each chunk — text deltas,
reasoning deltas, incremental tool-call fragments, metadata, and completion
markers — with no dependency on Delta's wire-event model.  The
normalisation layer (``normalize.py``) is responsible for converting these
signals into ``WireEvent`` instances.

Tool-call arguments arrive as incremental JSON fragments.  Both decoders
accumulate these fragments internally and emit a ``CallResolved`` signal only
when the complete argument string is available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

# ---------------------------------------------------------------------------
# Parsed signal types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TextChunk:
    """An incremental fragment of assistant text content."""

    text: str


@dataclass(frozen=True, slots=True)
class ReasoningChunk:
    """An incremental fragment of reasoning / chain-of-thought output."""

    text: str


@dataclass(frozen=True, slots=True)
class CallBegin:
    """Marks the start of a new tool call at the given parser index."""

    index: int
    call_id: str
    name: str


@dataclass(frozen=True, slots=True)
class CallArgFragment:
    """An incremental fragment of a tool call's JSON argument string."""

    index: int
    fragment: str


@dataclass(frozen=True, slots=True)
class CallResolved:
    """A fully reconstructed tool call ready for execution."""

    index: int
    call_id: str
    name: str
    arguments_json: str


@dataclass(frozen=True, slots=True)
class Metadata:
    """Response-level metadata received at the start of a stream."""

    response_id: str | None = None
    model: str | None = None


@dataclass(frozen=True, slots=True)
class Finished:
    """Marks the end of the model response."""

    finish_reason: str | None = None
    usage: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ParseFault:
    """Signals a parsing error in the response stream."""

    message: str


type ParsedSignal = (
    TextChunk
    | ReasoningChunk
    | CallBegin
    | CallArgFragment
    | CallResolved
    | Metadata
    | Finished
    | ParseFault
)


# ---------------------------------------------------------------------------
# Tool-call argument accumulator
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _ToolBuffer:
    """Accumulates incremental argument fragments for one tool call."""

    index: int
    call_id: str
    name: str
    fragments: list[str] = field(default_factory=list)

    @property
    def arguments_json(self) -> str:
        return "".join(self.fragments)


# ---------------------------------------------------------------------------
# Decoder protocol
# ---------------------------------------------------------------------------


class StreamDecoder(Protocol):
    """Common interface implemented by both API-specific decoders."""

    def decode(self, event_name: str, data: dict[str, Any]) -> list[ParsedSignal]:
        """Process one parsed SSE payload and return any resulting signals."""
        ...


# ---------------------------------------------------------------------------
# Chat Completions decoder
# ---------------------------------------------------------------------------


class ChatDecoder:
    """Decodes Chat Completions streaming chunks into parsed signals.

    Each SSE ``data:`` line is a JSON object with a ``choices`` array
    containing a single ``delta`` object.  Tool-call arguments are
    accumulated across multiple chunks and emitted as a ``CallResolved``
    signal when the ``finish_reason`` arrives.
    """

    def __init__(self) -> None:
        self._tools: dict[int, _ToolBuffer] = {}

    def decode(self, event_name: str, data: dict[str, Any]) -> list[ParsedSignal]:
        # Top-level error object (non-streaming error response)
        error = data.get("error")
        if isinstance(error, dict):
            return [ParseFault(message=error.get("message", str(error)))]

        signals: list[ParsedSignal] = []

        # Response metadata
        resp_id = data.get("id")
        resp_model = data.get("model")
        if resp_id or resp_model:
            signals.append(Metadata(response_id=resp_id, model=resp_model))

        choices = data.get("choices")
        if not choices:
            # Usage-only chunk (sent with stream_options.include_usage)
            usage = data.get("usage")
            if usage:
                signals.append(Finished(usage=usage))
            return signals

        choice = choices[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")

        # Text content
        content = delta.get("content")
        if content:
            signals.append(TextChunk(text=content))

        # Reasoning content (extended-thinking models)
        reasoning = delta.get("reasoning_content")
        if reasoning:
            signals.append(ReasoningChunk(text=reasoning))

        # Tool call deltas
        for tc in delta.get("tool_calls", []):
            idx = tc.get("index", 0)
            tc_id = tc.get("id")
            func = tc.get("function", {})
            name = func.get("name")
            args_fragment = func.get("arguments", "")

            if tc_id and name:
                # First chunk — open a new tool buffer
                buf = _ToolBuffer(index=idx, call_id=tc_id, name=name)
                self._tools[idx] = buf
                signals.append(CallBegin(index=idx, call_id=tc_id, name=name))
                if args_fragment:
                    buf.fragments.append(args_fragment)
                    signals.append(CallArgFragment(index=idx, fragment=args_fragment))
            elif args_fragment and idx in self._tools:
                # Continuation — accumulate arguments
                self._tools[idx].fragments.append(args_fragment)
                signals.append(CallArgFragment(index=idx, fragment=args_fragment))

        # Stream completion
        if finish_reason is not None:
            # Resolve any pending tool calls
            for buf in self._tools.values():
                signals.append(CallResolved(
                    index=buf.index,
                    call_id=buf.call_id,
                    name=buf.name,
                    arguments_json=buf.arguments_json,
                ))
            self._tools.clear()

            usage = data.get("usage")
            signals.append(Finished(finish_reason=finish_reason, usage=usage))

        return signals


# ---------------------------------------------------------------------------
# Responses API decoder
# ---------------------------------------------------------------------------


class ResponsesDecoder:
    """Decodes Responses API streaming events into parsed signals.

    The Responses API uses named SSE event types (``response.created``,
    ``response.output_text.delta``, etc.) rather than the unnamed ``data:``
    chunks of Chat Completions.
    """

    def __init__(self) -> None:
        self._tools: dict[int, _ToolBuffer] = {}

    def decode(self, event_name: str, data: dict[str, Any]) -> list[ParsedSignal]:
        signals: list[ParsedSignal] = []

        match event_name:
            # --- lifecycle ----------------------------------------------------
            case "response.created":
                resp = data.get("response", {})
                signals.append(Metadata(
                    response_id=resp.get("id"),
                    model=resp.get("model"),
                ))

            # --- text content -------------------------------------------------
            case "response.output_text.delta":
                delta = data.get("delta", "")
                if delta:
                    signals.append(TextChunk(text=delta))

            # --- reasoning content -------------------------------------------
            case "response.reasoning_summary_text.delta":
                delta = data.get("delta", "")
                if delta:
                    signals.append(ReasoningChunk(text=delta))

            # --- tool calls ---------------------------------------------------
            case "response.output_item.added":
                item = data.get("item", {})
                idx = data.get("output_index", 0)
                if item.get("type") == "function_call":
                    call_id = item.get("call_id", "")
                    fn_name = item.get("name", "")
                    self._tools[idx] = _ToolBuffer(
                        index=idx, call_id=call_id, name=fn_name,
                    )
                    signals.append(CallBegin(index=idx, call_id=call_id, name=fn_name))

            case "response.function_call_arguments.delta":
                idx = data.get("output_index", 0)
                delta = data.get("delta", "")
                if delta and idx in self._tools:
                    self._tools[idx].fragments.append(delta)
                    signals.append(CallArgFragment(index=idx, fragment=delta))

            case "response.function_call_arguments.done":
                idx = data.get("output_index", 0)
                buf = self._tools.pop(idx, None)
                if buf is not None:
                    # Prefer the server's complete string over our accumulation
                    full_args = data.get("arguments", buf.arguments_json)
                    signals.append(CallResolved(
                        index=buf.index,
                        call_id=buf.call_id,
                        name=buf.name,
                        arguments_json=full_args,
                    ))

            # --- completion ---------------------------------------------------
            case "response.completed":
                resp = data.get("response", {})
                usage = resp.get("usage")
                status = resp.get("status", "completed")
                reason = "stop" if status == "completed" else status
                signals.append(Finished(finish_reason=reason, usage=usage))

            # --- errors -------------------------------------------------------
            case "error":
                err = data.get("error", {})
                msg = err.get("message", str(data))
                signals.append(ParseFault(message=msg))

            case "response.failed":
                resp = data.get("response", {})
                err = resp.get("last_error", {})
                msg = err.get("message", "Response failed")
                signals.append(ParseFault(message=msg))

        return signals
