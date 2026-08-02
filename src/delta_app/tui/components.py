"""Public widget surface for Delta's TUI.

Each widget lives in its own module; this re-exports them so callers can
import from one place.
"""

from delta_app.tui.messages import (
    AssistantBlock,
    NoticeBlock,
    ToolLine,
    ToolOutput,
    UserBlock,
)
from delta_app.tui.prompt import PromptInput
from delta_app.tui.sidebar import Sidebar
from delta_app.tui.status import StatusBar
from delta_app.tui.transcript import TranscriptView

__all__ = [
    "AssistantBlock",
    "NoticeBlock",
    "PromptInput",
    "Sidebar",
    "StatusBar",
    "ToolLine",
    "ToolOutput",
    "TranscriptView",
    "UserBlock",
]
