"""State-machine parser that converts Anthropic SSE events into Delta wire events.

``ResponseMachine`` processes the typed events produced by the Messages API
streaming endpoint (``message_start``, ``content_block_start``,
``content_block_delta``, ``content_block_stop``, ``message_delta``,
``message_stop``) and emits the corresponding ``WireEvent`` instances that
the harness loop consumes.

The machine maintains a running ``ModelEntry`` snapshot and tracks one
in-progress content block at a time.  Completed blocks are folded into
the snapshot immediately, so every emitted event carries an up-to-date
partial view of the assistant response.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from delta_harness.contracts.transcript import (
    CallBlock,
    ModelEntry,
    ReplyContent,
    TextSegment,
    ThoughtSegment,
    UsageStats,
)
from delta_harness.provider.wire import (
    CallChunkEvent,
    CallCloseEvent,
    CallOpenEvent,
    CompletionCause,
    ContentChunkEvent,
    ContentCloseEvent,
    ContentOpenEvent,
    FaultCause,
    ReasoningChunkEvent,
    ReasoningCloseEvent,
    ReasoningOpenEvent,
    StreamCloseEvent,
    StreamFaultEvent,
    StreamOpenEvent,
    WireEvent,
)

# ---------------------------------------------------------------------------
# Phase enum
# ---------------------------------------------------------------------------


class Phase(Enum):
    """Lifecycle phase of the streaming response."""

    PENDING = auto()
    ACTIVE = auto()
    SEALED = auto()


# ---------------------------------------------------------------------------
# Content accumulator
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _ContentAccumulator:
    """Mutable tracker for one in-progress content block."""

    slot: int
    variety: str  # "text" | "thinking" | "tool_use"
    chunks: list[str] = field(default_factory=list)
    tool_id: str = ""
    tool_name: str = ""

    @property
    def assembled(self) -> str:
        """Return the concatenated content gathered so far."""
        return "".join(self.chunks)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _map_stop_reason(reason: str) -> str:
    """Translate an Anthropic stop reason to Delta's halt vocabulary."""
    match reason:
        case "end_turn" | "stop_sequence":
            return "stop"
        case "max_tokens":
            return "length"
        case "tool_use":
            return "toolUse"
        case _:
            return "stop"


def _safe_json_args(text: str) -> dict[str, Any]:
    """Parse tool-call input JSON, falling back to an empty dict."""
    if not text or not text.strip():
        return {}
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except (json.JSONDecodeError, ValueError):
        return {}


def _tally_usage(
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read: int = 0,
    cache_write: int = 0,
) -> UsageStats:
    """Build a ``UsageStats`` from Anthropic token counts."""
    return UsageStats(
        total_tokens=input_tokens + output_tokens,
        input=input_tokens,
        output=output_tokens,
        cache_read=cache_read,
        cache_write=cache_write,
    )


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


class ResponseMachine:
    """Stateful parser that ingests Anthropic SSE events and emits wire events.

    Call ``ingest`` for each ``(event_type, payload)`` pair received from the
    streaming endpoint.  Call ``seal`` after the stream closes to ensure a
    terminal event is emitted even if ``message_stop`` was never received.
    """

    def __init__(self, *, model: str, provider: str) -> None:
        self._model = model
        self._provider = provider
        self._phase = Phase.PENDING

        # Response-level metadata
        self._response_id: str | None = None
        self._response_model: str | None = None
        self._halt: str = "stop"

        # Token accounting
        self._input_tokens = 0
        self._output_tokens = 0
        self._cache_read = 0
        self._cache_write = 0

        # Completed content blocks
        self._thoughts: list[ThoughtSegment] = []
        self._texts: list[TextSegment] = []
        self._calls: list[CallBlock] = []

        # Active block (at most one at a time)
        self._accumulator: _ContentAccumulator | None = None

    # --- snapshot -----------------------------------------------------------

    def _snapshot(self) -> ModelEntry:
        """Build a ``ModelEntry`` reflecting the current accumulated state."""
        blocks: list[ReplyContent] = []
        blocks.extend(self._thoughts)
        blocks.extend(self._texts)
        blocks.extend(self._calls)

        # Include partial content from the active block
        if self._accumulator is not None:
            body = self._accumulator.assembled
            if self._accumulator.variety == "thinking" and body:
                blocks.append(ThoughtSegment(thinking=body))
            elif self._accumulator.variety == "text" and body:
                blocks.append(TextSegment(text=body))

        return ModelEntry(
            model=self._response_model or self._model,
            provider=self._provider,
            api="messages",
            response_id=self._response_id,
            response_model=self._response_model,
            content=blocks,
            stop_reason=self._halt,  # type: ignore[arg-type]
            usage=_tally_usage(
                input_tokens=self._input_tokens,
                output_tokens=self._output_tokens,
                cache_read=self._cache_read,
                cache_write=self._cache_write,
            ),
        )

    # --- public interface ---------------------------------------------------

    def ingest(self, kind: str, data: dict[str, Any]) -> list[WireEvent]:
        """Process one SSE event and return any resulting wire events."""
        match kind:
            case "message_start":
                return self._on_message_start(data)
            case "content_block_start":
                return self._on_block_open(data)
            case "content_block_delta":
                return self._on_block_delta(data)
            case "content_block_stop":
                return self._on_block_close(data)
            case "message_delta":
                return self._on_message_delta(data)
            case "message_stop":
                return self.seal()
            case "ping":
                return []
            case "error":
                return self._on_error(data)
            case _:
                return []

    def seal(self) -> list[WireEvent]:
        """Emit the terminal event if the machine has not already been sealed."""
        if self._phase == Phase.SEALED:
            return []
        self._phase = Phase.SEALED

        message = self._snapshot()

        if self._halt in ("error", "aborted"):
            cause: FaultCause = "aborted" if self._halt == "aborted" else "error"
            return [StreamFaultEvent(reason=cause, error=message)]

        reason: CompletionCause = "stop"
        if self._halt == "length":
            reason = "length"
        elif self._halt == "toolUse":
            reason = "toolUse"
        return [StreamCloseEvent(reason=reason, message=message)]

    # --- event handlers -----------------------------------------------------

    def _on_message_start(self, data: dict[str, Any]) -> list[WireEvent]:
        msg = data.get("message", {})
        self._response_id = msg.get("id")
        self._response_model = msg.get("model")

        usage = msg.get("usage", {})
        self._input_tokens = usage.get("input_tokens", 0)
        self._cache_read = usage.get("cache_read_input_tokens", 0)
        self._cache_write = usage.get("cache_creation_input_tokens", 0)

        self._phase = Phase.ACTIVE
        return [StreamOpenEvent(partial=self._snapshot())]

    def _on_block_open(self, data: dict[str, Any]) -> list[WireEvent]:
        block = data.get("content_block", {})
        slot = data.get("index", 0)
        variety = block.get("type", "text")

        if variety == "tool_use":
            self._accumulator = _ContentAccumulator(
                slot=slot,
                variety="tool_use",
                tool_id=block.get("id", ""),
                tool_name=block.get("name", ""),
            )
            return [CallOpenEvent(content_index=slot, partial=self._snapshot())]

        if variety == "thinking":
            self._accumulator = _ContentAccumulator(slot=slot, variety="thinking")
            return [ReasoningOpenEvent(content_index=slot, partial=self._snapshot())]

        # Default: text
        self._accumulator = _ContentAccumulator(slot=slot, variety="text")
        return [ContentOpenEvent(content_index=slot, partial=self._snapshot())]

    def _on_block_delta(self, data: dict[str, Any]) -> list[WireEvent]:
        if self._accumulator is None:
            return []

        delta = data.get("delta", {})
        delta_type = delta.get("type", "")
        slot = data.get("index", self._accumulator.slot)

        if delta_type == "text_delta":
            fragment = delta.get("text", "")
            if fragment:
                self._accumulator.chunks.append(fragment)
                return [ContentChunkEvent(
                    content_index=slot, delta=fragment, partial=self._snapshot(),
                )]

        elif delta_type == "thinking_delta":
            fragment = delta.get("thinking", "")
            if fragment:
                self._accumulator.chunks.append(fragment)
                return [ReasoningChunkEvent(
                    content_index=slot, delta=fragment, partial=self._snapshot(),
                )]

        elif delta_type == "input_json_delta":
            fragment = delta.get("partial_json", "")
            if fragment:
                self._accumulator.chunks.append(fragment)
                return [CallChunkEvent(
                    content_index=slot, delta=fragment, partial=self._snapshot(),
                )]

        return []

    def _on_block_close(self, data: dict[str, Any]) -> list[WireEvent]:
        acc = self._accumulator
        if acc is None:
            return []
        self._accumulator = None
        slot = data.get("index", acc.slot)
        body = acc.assembled

        if acc.variety == "text":
            self._texts.append(TextSegment(text=body))
            return [ContentCloseEvent(
                content_index=slot, content=body, partial=self._snapshot(),
            )]

        if acc.variety == "thinking":
            self._thoughts.append(ThoughtSegment(thinking=body))
            return [ReasoningCloseEvent(
                content_index=slot, content=body, partial=self._snapshot(),
            )]

        if acc.variety == "tool_use":
            args = _safe_json_args(body)
            call = CallBlock(id=acc.tool_id, name=acc.tool_name, arguments=args)
            self._calls.append(call)
            return [CallCloseEvent(
                content_index=slot, tool_call=call, partial=self._snapshot(),
            )]

        return []

    def _on_message_delta(self, data: dict[str, Any]) -> list[WireEvent]:
        delta = data.get("delta", {})
        stop = delta.get("stop_reason")
        if stop:
            self._halt = _map_stop_reason(stop)

        usage = data.get("usage", {})
        out = usage.get("output_tokens")
        if out is not None:
            self._output_tokens = int(out)
        return []

    def _on_error(self, data: dict[str, Any]) -> list[WireEvent]:
        was_pending = self._phase == Phase.PENDING
        self._phase = Phase.SEALED
        self._halt = "error"

        error_obj = data.get("error", {})
        msg = error_obj.get("message", str(data))
        entry = ModelEntry(
            model=self._response_model or self._model,
            provider=self._provider,
            api="messages",
            content=[],
            stop_reason="error",
            error_message=msg,
        )

        events: list[WireEvent] = []
        if was_pending:
            events.append(StreamOpenEvent(partial=entry))
        events.append(StreamFaultEvent(reason="error", error=entry))
        return events
