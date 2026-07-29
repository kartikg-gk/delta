"""Main transcript area — scrollable conversation view.

Owns the mounted message blocks and autoscrolls as content streams in.
Holds no runtime references; the app drives it.
"""

from __future__ import annotations

from textual.containers import VerticalScroll

from delta_app.tui.adapter import summarize_call
from delta_app.tui.messages import (
    AssistantBlock,
    NoticeBlock,
    ToolLine,
    ToolOutput,
    UserBlock,
    should_collapse,
)


class TranscriptView(VerticalScroll):
    """Scrollable region holding the conversation."""

    def __init__(self) -> None:
        super().__init__()
        self._live: AssistantBlock | None = None
        self._tools: dict[str, ToolLine] = {}

    # -- appending ----------------------------------------------------------

    def add_user(self, text: str) -> None:
        """Append a user turn."""
        self.mount(UserBlock(text))
        self._scroll()

    def add_notice(self, text: str, *, error: bool = False) -> None:
        """Append a notice (command output, error, cancellation)."""
        self.mount(NoticeBlock(text, error=error))
        self._scroll()

    def start_tool(self, call_id: str, summary: str) -> None:
        """Append a pending tool line."""
        line = ToolLine(call_id, summary)
        self._tools[call_id] = line
        self.mount(line)
        self._scroll()

    def finish_tool(self, call_id: str, output: str, *, is_error: bool) -> None:
        """Resolve a pending tool line and attach large output."""
        line = self._tools.pop(call_id, None)
        if line is not None:
            line.mark_done(is_error=is_error)
        if output and should_collapse(output):
            self.mount(ToolOutput(output))
        self._scroll()

    # -- streaming ----------------------------------------------------------

    def stream_delta(self, delta: str) -> None:
        """Append streamed assistant text, opening a block on first delta."""
        if self._live is None:
            self._live = AssistantBlock()
            self.mount(self._live)
        self._live.append(delta)
        self._scroll()

    def finish_assistant(self, text: str) -> None:
        """Close the live assistant block."""
        if self._live is None:
            if text:
                self.mount(AssistantBlock(text))
        else:
            self._live.finalize(text)
        self._live = None
        self._scroll()

    def reset(self) -> None:
        """Clear the transcript (used when switching sessions)."""
        self._live = None
        self._tools.clear()
        self.remove_children()

    def replay(self, entries) -> None:
        """Repaint the view from a session's stored transcript."""
        self.reset()
        for entry in entries:
            role = getattr(entry, "role", None)
            if role == "user":
                self.mount(UserBlock(entry.text))
            elif role == "assistant":
                for call in entry.tool_calls:
                    line = ToolLine(call.id, summarize_call(call.name, call.arguments))
                    line.mark_done(is_error=False)
                    self.mount(line)
                if entry.text:
                    self.mount(AssistantBlock(entry.text))
            elif role == "compactionSummary":
                self.mount(NoticeBlock(f"[compacted] {entry.summary}"))
        self._scroll()

    # -- internal -----------------------------------------------------------

    def _scroll(self) -> None:
        """Keep the newest content in view."""
        self.scroll_end(animate=False)


__all__ = ["TranscriptView"]
