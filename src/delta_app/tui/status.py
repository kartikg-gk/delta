"""Footer status bar: provider, model, session, context usage, git branch."""

from __future__ import annotations

from textual.widgets import Static

from delta_app.tui.adapter import StatusSnapshot


class StatusBar(Static):
    """One-line status readout pinned above the keybinding footer."""

    def __init__(self) -> None:
        super().__init__("", id="status-bar")
        self._line = ""

    @property
    def line(self) -> str:
        """The markup currently displayed."""
        return self._line

    def show(self, snapshot: StatusSnapshot, *, busy: bool = False) -> None:
        """Render *snapshot*."""
        parts = [
            snapshot.provider,
            snapshot.model,
            f"ctx {snapshot.context_percent}%",
        ]
        if snapshot.branch:
            parts.append(f"⎇ {snapshot.branch}")
        if snapshot.plan_mode:
            parts.append("PLAN")
        if busy:
            parts.append("■ generating — press Esc to stop")
        self._line = "  ·  ".join(parts)
        # Highlight the whole bar while a generation can be stopped.
        self.set_class(busy, "busy")
        self.update(self._line)


__all__ = ["StatusBar"]
