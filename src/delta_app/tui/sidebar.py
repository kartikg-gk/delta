"""Left sidebar — the session switcher.

Renders the session list and emits selection intents.  Holds no runtime
references; the app performs the actual switch through the adapter.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Label, ListItem, ListView

from delta_app.tui.adapter import SessionEntry


class SessionItem(ListItem):
    """A session row. Identity is carried as an attribute, not a widget id.

    Widget ids must be unique across the whole DOM, which collides with the
    deferred removal ``ListView.clear()`` performs; an attribute sidesteps it.
    """

    def __init__(self, label: str, session_id: str | None) -> None:
        super().__init__(Label(label))
        self.session_id = session_id

    @property
    def is_new_action(self) -> bool:
        """Whether this row creates a new session."""
        return self.session_id is None


class SessionList(ListView):
    """Selectable list of sessions, newest first."""

    def __init__(self) -> None:
        super().__init__(id="session-list")


class Sidebar(Vertical):
    """Fixed-width column holding the session switcher."""

    class SwitchRequested(Message):
        """User picked an existing session."""

        def __init__(self, session_id: str) -> None:
            super().__init__()
            self.session_id = session_id

    class NewRequested(Message):
        """User asked for a fresh session."""

    def __init__(self) -> None:
        super().__init__()
        #: Session ids currently rendered, in order — used to decide whether a
        #: refresh can update in place instead of rebuilding.
        self._ids: list[str] = []

    def compose(self) -> ComposeResult:
        yield Label("Sessions", id="sidebar-title")
        yield SessionList()

    async def show(self, entries: list[SessionEntry]) -> None:
        """Repaint the list, highlighting the active session.

        Rows are updated in place whenever the set of sessions is unchanged.
        Rebuilding on every refresh made the list visibly flicker and yanked
        the cursor away from wherever the user had left it.
        """
        listing = self.query_one(SessionList)
        ids = [e.session_id for e in entries]

        if ids == self._ids:
            self._restyle(listing, entries)
            return

        # Membership actually changed — rebuild. clear() removes children
        # lazily, so await it or the appends race still-mounted widgets.
        await listing.clear()
        await listing.append(SessionItem("+ New session", None))
        for entry in entries:
            await listing.append(
                SessionItem(self._label_for(entry), entry.session_id)
            )
        self._ids = ids
        self._restyle(listing, entries)
        listing.index = self._active_index(entries)

    @staticmethod
    def _label_for(entry: SessionEntry) -> str:
        marker = "●" if entry.is_active else " "
        return f"{marker} {entry.label}"

    @staticmethod
    def _active_index(entries: list[SessionEntry]) -> int:
        for i, entry in enumerate(entries, start=1):
            if entry.is_active:
                return i
        return 0

    def _restyle(self, listing: SessionList, entries: list[SessionEntry]) -> None:
        """Refresh labels and the active marker without touching the cursor."""
        by_id = {e.session_id: e for e in entries}
        for child in listing.children:
            if not isinstance(child, SessionItem) or child.is_new_action:
                continue
            entry = by_id.get(child.session_id or "")
            if entry is None:
                continue
            child.set_class(entry.is_active, "active-session")
            label = child.query(Label).first()
            label.update(self._label_for(entry))

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Translate a row activation into an intent message."""
        event.stop()
        item = event.item
        if not isinstance(item, SessionItem):
            return
        if item.is_new_action:
            self.post_message(self.NewRequested())
        else:
            self.post_message(self.SwitchRequested(item.session_id or ""))


__all__ = ["SessionItem", "SessionList", "Sidebar"]
