"""Manual OAuth redirect entry while the browser callback is pending."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label


class ClaudeLoginScreen(ModalScreen[str]):
    BINDINGS = [("escape", "cancel", "Cancel login")]
    DEFAULT_CSS = """
    ClaudeLoginScreen { align: center middle; }
    ClaudeLoginScreen > Vertical {
        width: 80%; height: auto; padding: 1 2;
        border: round $accent; background: $surface;
    }
    """

    def __init__(self, url: str) -> None:
        super().__init__()
        self._url = url

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Sign in to Claude in your browser")
            yield Label(self._url, markup=False)
            yield Label("Waiting for the browser. Or paste the redirect URL/code below.")
            yield Input(placeholder="Redirect URL or code", password=True)

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.value.strip():
            self.dismiss(event.value)

    def action_cancel(self) -> None:
        self.dismiss("")
