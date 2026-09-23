"""Startup question: may this folder's project inputs load?

Runs as its own short-lived app before the main UI exists, because the answer
decides what the session is built from. The choices and their meaning come
from ``delta_app.trust``; this module only presents them. Escape dismisses
without an answer, which ends startup.
"""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Label, ListItem, ListView

from delta_app.runtime import trust_choices, trust_question
from delta_app.trust import Choice, TrustRequest


class _Option(ListItem):
    def __init__(self, label: str, choice: Choice) -> None:
        super().__init__(Label(label))
        self.choice = choice


class TrustDialog(App[Choice | None]):
    """A single list of trust choices under a short explanation."""

    BINDINGS = [("escape", "dismiss_question", "Exit")]
    CSS = """
    #trust-box { padding: 1 2; height: auto; }
    #trust-text { margin-bottom: 1; }
    #trust-options { height: auto; }
    """

    def __init__(self, request: TrustRequest) -> None:
        super().__init__()
        self._request = request

    def compose(self) -> ComposeResult:
        with Vertical(id="trust-box"):
            yield Label(trust_question(self._request), id="trust-text")
            yield ListView(
                *(_Option(label, choice) for choice, label in trust_choices(self._request)),
                id="trust-options",
            )

    def on_mount(self) -> None:
        self.query_one(ListView).focus()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        item = event.item
        if isinstance(item, _Option):
            self.exit(item.choice)

    def action_dismiss_question(self) -> None:
        self.exit(None)


async def ask_in_dialog(request: TrustRequest) -> Choice | None:
    """Show the dialog and return the chosen answer, or ``None`` if dismissed."""
    return await TrustDialog(request).run_async()


__all__ = ["TrustDialog", "ask_in_dialog"]
