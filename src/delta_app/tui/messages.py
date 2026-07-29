"""Transcript block widgets: user turns, assistant turns, and tool lines.

Pure presentation — every widget renders state handed to it and holds no
runtime references.
"""

from __future__ import annotations

from rich.markup import escape
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Collapsible, Static

_MAX_INLINE_OUTPUT = 400

DELTA_MARK = "δ"  # prompt marker for user turns
ACCENT = "#c9a4ff"  # violet accent, shared by markup and delta.tcss


class UserBlock(Static):
    """One user message."""

    def __init__(self, text: str) -> None:
        super().__init__(f"{DELTA_MARK} {escape(text)}", classes="user-block")
        self._text = text

    @property
    def text(self) -> str:
        """The message this block displays."""
        return self._text


class AssistantBlock(Static):
    """One assistant message, appended to incrementally while streaming."""

    def __init__(self, text: str = "") -> None:
        super().__init__(escape(text), classes="assistant-block")
        self._buffer = text

    @property
    def text(self) -> str:
        """Accumulated text so far."""
        return self._buffer

    def append(self, delta: str) -> None:
        """Append streamed text and re-render."""
        self._buffer += delta
        self.update(escape(self._buffer))

    def finalize(self, text: str) -> None:
        """Replace the buffer with the completed message."""
        self._buffer = text
        self.update(escape(text))


class ToolLine(Static):
    """A compact one-line record of a tool invocation."""

    def __init__(self, call_id: str, summary: str) -> None:
        super().__init__(f"· {escape(summary)}", classes="tool-line")
        self.call_id = call_id
        self._summary = summary

    def mark_done(self, *, is_error: bool) -> None:
        """Flip the pending marker to success or failure."""
        if is_error:
            self.update(f"✗ {escape(self._summary)}")
            self.add_class("tool-error")
        else:
            self.update(f"✓ {escape(self._summary)}")


class ToolOutput(Vertical):
    """Collapsible container for a tool's stdout when it is large."""

    def __init__(self, output: str) -> None:
        super().__init__(classes="tool-output")
        self._output = output

    def compose(self) -> ComposeResult:
        lines = self._output.splitlines()
        title = f"{len(lines)} lines of output"
        with Collapsible(title=title, collapsed=True):
            yield Static(escape(self._output))


def should_collapse(output: str) -> bool:
    """Whether *output* is large enough to warrant a collapsible block."""
    return len(output) > _MAX_INLINE_OUTPUT or output.count("\n") > 6


class NoticeBlock(Static):
    """A non-conversational notice: command output, errors, cancellations."""

    def __init__(self, text: str, *, error: bool = False) -> None:
        super().__init__(escape(text), classes="notice-block")
        if error:
            self.add_class("tool-error")


__all__ = [
    "AssistantBlock",
    "NoticeBlock",
    "ToolLine",
    "ToolOutput",
    "UserBlock",
    "should_collapse",
]
