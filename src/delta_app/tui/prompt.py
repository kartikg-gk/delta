"""Bottom multiline prompt input.

Enter submits; Shift+Enter (and Ctrl+J, for terminals that cannot
distinguish the two) inserts a newline.  The widget emits a ``Submitted``
message and never talks to the runtime itself.
"""

from __future__ import annotations

from textual.message import Message
from textual.widgets import TextArea


class PromptInput(TextArea):
    """Multiline input pinned below the transcript."""

    class Submitted(Message):
        """Posted when the user presses Enter on a non-empty prompt."""

        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    class TextChanged(Message):
        """Posted whenever the composer contents change."""

        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    class Navigate(Message):
        """Posted on up/down while a completion list is open."""

        def __init__(self, delta: int) -> None:
            super().__init__()
            self.delta = delta

    class Complete(Message):
        """Posted on Tab to accept a completion."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(id="prompt-input", **kwargs)  # type: ignore[arg-type]

    def on_mount(self) -> None:
        """Take focus so the user can type immediately."""
        self.focus()

    #: Keys that insert a literal newline instead of submitting.
    #: ``shift+enter`` only reaches the app under the kitty keyboard
    #: protocol; ``ctrl+j`` is the universal fallback.
    NEWLINE_KEYS = frozenset({"shift+enter", "alt+enter", "ctrl+j"})

    async def _on_key(self, event) -> None:
        """Enter submits; newline keys break the line; Tab/arrows complete."""
        key = event.key
        if key in self.NEWLINE_KEYS:
            event.prevent_default()
            event.stop()
            self.insert("\n")
            self.post_message(self.TextChanged(self.text))
            return
        if key == "enter":
            event.prevent_default()
            event.stop()
            self.submit()
            return
        if key == "tab" and self._completing:
            event.prevent_default()
            event.stop()
            self.post_message(self.Complete())
            return
        if key in ("up", "down") and self._completing:
            event.prevent_default()
            event.stop()
            self.post_message(self.Navigate(1 if key == "down" else -1))
            return
        await super()._on_key(event)
        self.post_message(self.TextChanged(self.text))

    @property
    def _completing(self) -> bool:
        """Whether the current text is a bare slash command."""
        text = self.text
        return text.startswith("/") and " " not in text

    def submit(self) -> None:
        """Emit the composer's contents if non-empty."""
        text = self.text.strip()
        if text:
            self.post_message(self.Submitted(text))

    def clear_prompt(self) -> None:
        """Empty the composer after a submission."""
        self.text = ""


__all__ = ["PromptInput"]
