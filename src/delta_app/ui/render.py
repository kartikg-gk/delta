"""Event rendering layer — pure consumers of the harness event stream.

Three renderers share a common protocol and differ only in presentation:

- **JsonRenderer**: serialises every event as one JSON object per line.
- **FinalTextRenderer**: prints only the final assistant response.
- **TranscriptRenderer**: streams text deltas and tool activity in real time.
"""

from __future__ import annotations

import json
import sys
from enum import Enum
from typing import Protocol, TextIO

from delta_harness.contracts.stream import (
    AgentEvent,
    MessageEndEvent,
    MessageUpdateEvent,
    RetryEvent,
    ToolRunEndEvent,
    ToolRunStartEvent,
)
from delta_harness.contracts.transcript import ModelEntry, TextSegment

# ---------------------------------------------------------------------------
# Output mode enum
# ---------------------------------------------------------------------------


class OutputMode(Enum):
    """Supported rendering modes."""

    TEXT = "text"
    JSON = "json"
    TRANSCRIPT = "transcript"


# ---------------------------------------------------------------------------
# Renderer protocol
# ---------------------------------------------------------------------------


class EventRenderer(Protocol):
    """Minimal contract every renderer implements."""

    def render(self, event: AgentEvent) -> None: ...
    def finish(self) -> bool: ...


# ---------------------------------------------------------------------------
# JSON renderer
# ---------------------------------------------------------------------------


class JsonRenderer:
    """Serialises every event as a single JSON line."""

    __slots__ = ("_out", "_ok")

    def __init__(self, out: TextIO | None = None) -> None:
        self._out = out if out is not None else sys.stdout
        self._ok = True

    def render(self, event: AgentEvent) -> None:
        line = json.dumps(event.model_dump(by_alias=True), default=str)
        self._out.write(line + "\n")
        self._out.flush()
        if isinstance(event, MessageEndEvent):
            msg = event.message
            if isinstance(msg, ModelEntry) and msg.stop_reason == "error":
                self._ok = False

    def finish(self) -> bool:
        return self._ok


# ---------------------------------------------------------------------------
# Final-text renderer
# ---------------------------------------------------------------------------


class FinalTextRenderer:
    """Prints only the final assistant response after the run completes."""

    __slots__ = ("_out", "_last_text", "_errors")

    def __init__(self, out: TextIO | None = None) -> None:
        self._out = out if out is not None else sys.stdout
        self._last_text: str | None = None
        self._errors: list[str] = []

    def render(self, event: AgentEvent) -> None:
        if not isinstance(event, MessageEndEvent):
            return
        msg = event.message
        if not isinstance(msg, ModelEntry):
            return
        text = msg.text.strip()
        if text:
            self._last_text = text
        if msg.error_message:
            self._errors.append(msg.error_message)

    def finish(self) -> bool:
        if self._errors:
            for err in self._errors:
                self._out.write(f"Error: {err}\n")
            return False
        if self._last_text:
            self._out.write(self._last_text + "\n")
        return True


# ---------------------------------------------------------------------------
# Transcript renderer
# ---------------------------------------------------------------------------


class TranscriptRenderer:
    """Streams assistant text and tool activity as it happens."""

    __slots__ = ("_out", "_ok", "_prev_len", "_needs_nl")

    def __init__(self, out: TextIO | None = None) -> None:
        self._out = out if out is not None else sys.stdout
        self._ok = True
        self._prev_len = 0
        self._needs_nl = False

    def render(self, event: AgentEvent) -> None:
        if isinstance(event, MessageUpdateEvent):
            self._on_update(event)
        elif isinstance(event, MessageEndEvent):
            self._on_end(event)
        elif isinstance(event, RetryEvent):
            self._ensure_newline()
            self._out.write(f"[retry] {event.message}\n")
            self._out.flush()
        elif isinstance(event, ToolRunStartEvent):
            self._ensure_newline()
            self._out.write(f"[tool:{event.tool_name}] running...\n")
        elif isinstance(event, ToolRunEndEvent):
            status = "error" if event.is_error else "done"
            result_preview = event.result.text[:120].replace("\n", " ")
            self._out.write(f"[tool:{event.tool_name}] {status}")
            if result_preview:
                self._out.write(f" — {result_preview}")
            self._out.write("\n")
            self._out.flush()

    def finish(self) -> bool:
        self._ensure_newline()
        return self._ok

    # -- private ------------------------------------------------------------

    def _on_update(self, event: MessageUpdateEvent) -> None:
        msg = event.message
        if not isinstance(msg, ModelEntry):
            return
        full = "".join(
            block.text for block in msg.content if isinstance(block, TextSegment)
        )
        delta = full[self._prev_len:]
        if delta:
            self._out.write(delta)
            self._out.flush()
            self._prev_len = len(full)
            self._needs_nl = not delta.endswith("\n")

    def _on_end(self, event: MessageEndEvent) -> None:
        msg = event.message
        if not isinstance(msg, ModelEntry):
            return
        # Flush any remaining text not yet streamed
        full = msg.text
        delta = full[self._prev_len:]
        if delta:
            self._out.write(delta)
            self._needs_nl = not delta.endswith("\n")
        self._prev_len = 0
        self._ensure_newline()
        if msg.stop_reason == "error":
            self._ok = False
            if msg.error_message:
                self._out.write(f"Error: {msg.error_message}\n")
        self._out.flush()

    def _ensure_newline(self) -> None:
        if self._needs_nl:
            self._out.write("\n")
            self._needs_nl = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def make_renderer(mode: OutputMode, out: TextIO | None = None) -> EventRenderer:
    """Create a renderer for the given output mode."""
    if mode is OutputMode.JSON:
        return JsonRenderer(out)
    if mode is OutputMode.TRANSCRIPT:
        return TranscriptRenderer(out)
    return FinalTextRenderer(out)


__all__ = [
    "EventRenderer",
    "FinalTextRenderer",
    "JsonRenderer",
    "OutputMode",
    "TranscriptRenderer",
    "make_renderer",
]
