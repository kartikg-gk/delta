"""The composer row: prompt marker, input, and a model/effort readout."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Static

from delta_app.tui.messages import DELTA_MARK
from delta_app.tui.prompt import PromptInput


class ModelBadge(Static):
    """Right-hand readout showing the active model and effort level."""

    def __init__(self) -> None:
        super().__init__("", id="model-badge")
        self._line = ""

    @property
    def line(self) -> str:
        """The text currently displayed."""
        return self._line

    def show(self, model: str, effort: str) -> None:
        """Render *model* and *effort*."""
        self._line = f"{model} · {effort}"
        self.update(self._line)


class Composer(Horizontal):
    """Prompt marker + input + model badge on one row."""

    def compose(self) -> ComposeResult:
        yield Static(DELTA_MARK, id="prompt-mark")
        yield PromptInput()
        yield ModelBadge()


__all__ = ["Composer", "ModelBadge", "DELTA_MARK"]
