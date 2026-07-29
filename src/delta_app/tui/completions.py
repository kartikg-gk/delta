"""Slash-command autocomplete bar shown above the composer.

Pure presentation: it is given candidates, renders them, and reports the
chosen one. Filtering lives in :func:`match_commands` so it stays testable
without a running app.
"""

from __future__ import annotations

from dataclasses import dataclass

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from delta_app.tui.messages import ACCENT

_MAX_VISIBLE = 6


@dataclass(frozen=True, slots=True)
class Completion:
    """One selectable slash command."""

    name: str
    description: str

    @property
    def insertion(self) -> str:
        """Text that replaces the composer contents when chosen."""
        return f"/{self.name} "


def match_commands(
    text: str, commands: tuple[tuple[str, str], ...],
) -> list[Completion]:
    """Return commands matching *text*, or ``[]`` when not completing.

    Only a leading ``/`` with no whitespace yet is a completion context —
    once the user types an argument the bar goes away.
    """
    if not text.startswith("/") or " " in text:
        return []
    fragment = text[1:].lower()
    matches = [
        Completion(name, description)
        for name, description in commands
        if name.startswith(fragment)
    ]
    return matches[:_MAX_VISIBLE]


class CompletionBar(Vertical):
    """Dropdown of slash-command candidates."""

    def __init__(self) -> None:
        super().__init__(id="completion-bar")
        self._items: list[Completion] = []
        self._index = 0
        self.display = False

    # -- state --------------------------------------------------------------

    @property
    def items(self) -> list[Completion]:
        """Currently offered completions."""
        return self._items

    @property
    def active(self) -> bool:
        """Whether the bar is showing anything."""
        return bool(self._items)

    @property
    def selected(self) -> Completion | None:
        """The highlighted completion, if any."""
        if not self._items:
            return None
        return self._items[self._index % len(self._items)]

    # -- updates ------------------------------------------------------------

    def offer(self, items: list[Completion]) -> None:
        """Replace the candidate list and repaint."""
        self._items = items
        self._index = 0
        self.display = bool(items)
        self._repaint()

    def dismiss(self) -> None:
        """Hide the bar."""
        self.offer([])

    def move(self, delta: int) -> None:
        """Move the highlight, wrapping at both ends."""
        if not self._items:
            return
        self._index = (self._index + delta) % len(self._items)
        self._repaint()

    # -- rendering ----------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Static("", id="completion-body")

    def _repaint(self) -> None:
        try:
            body = self.query_one("#completion-body", Static)
        except Exception:  # noqa: BLE001 - not mounted yet
            return
        if not self._items:
            body.update("")
            return
        lines = []
        for i, item in enumerate(self._items):
            if i == self._index:
                lines.append(
                    f"[{ACCENT}]› /{item.name:<10} {item.description}[/{ACCENT}]"
                )
            else:
                lines.append(f"  /{item.name:<10} {item.description}")
        body.update("\n".join(lines))


__all__ = ["Completion", "CompletionBar", "match_commands"]
