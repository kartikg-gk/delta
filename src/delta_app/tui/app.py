"""Textual application for Delta.

Presentation layer only.  All conversation work goes through
``SessionBridge`` -> ``CodingSession``; the app runs no agent loop, calls no
provider, and executes no tools.
"""

from __future__ import annotations

import argparse
from contextlib import suppress
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.css.query import NoMatches
from textual.widgets import Footer, Header

from delta_app.tui.adapter import (
    RetryNotice,
    SessionBridge,
    TextDelta,
    ToolFinished,
    ToolStarted,
    TurnFinished,
)
from delta_app.tui.completions import CompletionBar, match_commands
from delta_app.tui.composer import Composer, ModelBadge
from delta_app.tui.messages import ACCENT
from delta_app.tui.model_screen import ModelChoice, ModelScreen
from delta_app.tui.prompt import PromptInput
from delta_app.tui.sidebar import Sidebar
from delta_app.tui.status import StatusBar
from delta_app.tui.transcript import TranscriptView

_CSS_PATH = Path(__file__).with_name("delta.tcss")


def build_id() -> str:
    """Short fingerprint of the UI source actually loaded.

    Shown in the header so a stale binary or a second install is visible at a
    glance rather than guessed at.
    """
    import hashlib

    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.read_bytes())
    if _CSS_PATH.exists():
        digest.update(_CSS_PATH.read_bytes())
    return digest.hexdigest()[:7]



def _is_registered_command(text: str) -> bool:
    """Whether *text* names a slash command rather than being a prompt.

    Skills, prompt templates, unknown names, and absolute paths such as
    ``/tmp/shot.png`` are prompts: the session expands the first two.
    """
    from delta_app.conversation import COMMAND_REGISTRY
    from delta_app.directives import parse_command

    parsed = parse_command(text)
    return parsed is not None and COMMAND_REGISTRY.get(parsed[0]) is not None

class DeltaApp(App[None]):
    """Delta's terminal UI."""

    CSS_PATH = _CSS_PATH
    TITLE = "δelta"

    BINDINGS = [
        ("escape", "stop", "Stop"),
        ("ctrl+c", "cancel", "Stop / Quit"),
        ("ctrl+n", "new_session", "New session"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, bridge: SessionBridge | None = None) -> None:
        super().__init__()
        self._bridge = bridge
        self._busy = False
        #: Set by Ctrl+C so updates still in flight are not rendered.
        self._cancelled = False

    # -- composition --------------------------------------------------------

    def format_title(self, title: str, sub_title: str) -> Content:
        """Render the header title with a violet delta and no build suffix."""
        return Content.from_markup(f"[{ACCENT}]δ[/]elta")

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="body"):
            yield Sidebar()
            with Vertical(id="main"):
                yield TranscriptView()
                yield CompletionBar()
                yield Composer()
        yield StatusBar()
        yield Footer()

    async def on_mount(self) -> None:
        await self._refresh_sidebar()
        self._refresh_status()
        if self._bridge is not None:
            self.query_one(TranscriptView).replay(self._bridge.session.transcript)

    # -- user actions -------------------------------------------------------

    def on_prompt_input_text_changed(self, event: PromptInput.TextChanged) -> None:
        """Offer slash-command completions as the user types."""
        event.stop()
        commands = (
            self._bridge.session.COMMANDS if self._bridge is not None else ()
        )
        bar = self.query_one(CompletionBar)
        bar.offer(match_commands(event.text, commands))

    def _is_exact_command(self, text: str) -> bool:
        """Whether *text* is already a complete slash command name."""
        if self._bridge is None or not text.startswith("/"):
            return False
        name = text[1:].strip().lower()
        return any(name == cmd for cmd, _ in self._bridge.session.COMMANDS)

    def on_prompt_input_navigate(self, event: PromptInput.Navigate) -> None:
        """Arrow keys move the completion highlight when the bar is open."""
        event.stop()
        self.query_one(CompletionBar).move(event.delta)

    def on_prompt_input_complete(self, event: PromptInput.Complete) -> None:
        """Tab accepts the highlighted completion."""
        event.stop()
        bar = self.query_one(CompletionBar)
        chosen = bar.selected
        if chosen is None:
            return
        prompt = self.query_one(PromptInput)
        prompt.text = chosen.insertion
        prompt.cursor_location = (0, len(prompt.text))
        bar.dismiss()

    async def on_prompt_input_submitted(self, event: PromptInput.Submitted) -> None:
        """Handle Enter in the composer."""
        event.stop()
        bar = self.query_one(CompletionBar)
        if bar.active and not self._is_exact_command(event.text):
            # Still a partial word — Enter fills it in. A fully typed command
            # submits straight away rather than needing a second Enter.
            self.on_prompt_input_complete(PromptInput.Complete())
            return
        bar.dismiss()
        if self._busy:
            return
        text = event.text
        self.query_one(PromptInput).clear_prompt()

        if self._bridge is None:
            self.query_one(TranscriptView).add_notice(
                "No session attached — running in layout-only mode.", error=True,
            )
            return

        if _is_registered_command(text):
            await self._run_command(text)
            return

        self.query_one(TranscriptView).add_user(text)
        self.run_worker(self._stream(text), exclusive=True)

    def action_stop(self) -> bool:
        """Stop the running generation. Returns whether anything was stopped."""
        if not self._busy or self._bridge is None:
            return False
        self._cancelled = True
        self._bridge.cancel()
        transcript = self.query_one(TranscriptView)
        transcript.finish_assistant(transcript.live_text)
        transcript.add_notice("Stopped.")
        self.query_one(PromptInput).focus()
        self._refresh_status()
        return True

    def action_cancel(self) -> None:
        """Ctrl+C: stop the generation if running, otherwise quit."""
        if not self.action_stop():
            self.exit()

    def action_new_session(self) -> None:
        """Ctrl+N: start a fresh session."""
        self.post_message(Sidebar.NewRequested())

    async def on_sidebar_switch_requested(
        self, event: Sidebar.SwitchRequested,
    ) -> None:
        """Switch to an existing session."""
        event.stop()
        await self._swap(lambda b: b.switch_to(event.session_id))

    async def on_sidebar_new_requested(self, event: Sidebar.NewRequested) -> None:
        """Create a fresh session."""
        event.stop()
        await self._swap(lambda b: b.start_new())

    async def _swap(self, action) -> None:
        """Run a session change, then repaint every dependent surface."""
        if self._bridge is None:
            return
        transcript = self.query_one(TranscriptView)
        if self._busy:
            self._bridge.cancel()
            self._busy = False

        try:
            await action(self._bridge)
        except Exception as exc:  # noqa: BLE001 - keep the UI alive on failure
            transcript.add_notice(f"Could not switch session: {exc}", error=True)
            return

        # Stale state must not survive the swap.
        transcript.replay(self._bridge.session.transcript)
        self.query_one(PromptInput).clear_prompt()
        await self._refresh_sidebar()
        self._refresh_status()

    # -- runtime plumbing ---------------------------------------------------

    async def _run_command(self, text: str) -> None:
        """Delegate a slash command to the runtime."""
        transcript = self.query_one(TranscriptView)
        if text in {"/quit", "/exit", "/q"}:
            self.exit()
            return
        if text.strip() == "/version":
            from delta_app.cli.main import get_version

            transcript.add_notice(f"Delta v{get_version()}")
            return
        if text.strip() in {"/continue", "/resume"}:
            if self._busy:
                return
            self.run_worker(self._stream("", resume=True), exclusive=True)
            return
        if text.strip() in {"/model", "/provider"}:
            # push_screen_wait requires a worker context.
            self.run_worker(self._open_model_picker())
            return
        assert self._bridge is not None
        try:
            response = await self._bridge.handle_command(text)
        except Exception as exc:  # noqa: BLE001 - a bad command must not kill the UI
            transcript.add_notice(f"Command failed: {exc}", error=True)
            return
        if response is not None:
            transcript.add_notice(response)
        await self._refresh_sidebar()
        self._refresh_status()

    async def _open_model_picker(self) -> None:
        """Show the provider/model dialog and apply whatever comes back."""
        if self._bridge is None:
            return
        session = self._bridge.session
        choice = await self.push_screen_wait(
            ModelScreen(session.provider_name, session.model)
        )
        if choice is None:
            return
        await self._apply_model_choice(choice)

    async def _apply_model_choice(self, choice: ModelChoice) -> None:
        """Persist the selection and switch the runtime onto it."""
        assert self._bridge is not None
        transcript = self.query_one(TranscriptView)
        try:
            summary = await self._bridge.apply_model_choice(
                choice.provider, choice.model, choice.api_key,
            )
        except Exception as exc:  # noqa: BLE001 - a bad key must not kill the UI
            transcript.add_notice(f"Could not switch: {exc}", error=True)
            return
        transcript.add_notice(summary)
        self._refresh_status()

    async def _stream(self, text: str, *, resume: bool = False) -> None:
        """Consume the runtime's event stream and render it."""
        assert self._bridge is not None
        transcript = self.query_one(TranscriptView)
        self._busy = True
        self._cancelled = False
        self._refresh_status()
        try:
            source = (
                self._bridge.resume() if resume else self._bridge.submit(text)
            )
            async for update in source:
                if self._cancelled:
                    # Ctrl+C already closed the block and posted a notice;
                    # trailing updates must not reopen it.
                    break
                if isinstance(update, TextDelta):
                    transcript.stream_delta(update.text)
                elif isinstance(update, ToolStarted):
                    transcript.start_tool(update.call_id, update.summary)
                elif isinstance(update, ToolFinished):
                    transcript.finish_tool(
                        update.call_id, update.output, is_error=update.is_error,
                    )
                elif isinstance(update, RetryNotice):
                    transcript.add_notice(update.message)
                elif isinstance(update, TurnFinished):
                    transcript.finish_assistant(update.text)
                    if update.error:
                        transcript.add_notice(update.error, error=True)
        except Exception as exc:  # noqa: BLE001 - surface provider errors in-UI
            transcript.finish_assistant("")
            transcript.add_notice(str(exc), error=True)
        finally:
            self._busy = False
            # The app may be tearing down (quit during generation), in which
            # case these widgets are already gone.
            with suppress(NoMatches):
                await self._refresh_sidebar()
                self._refresh_status()

    def _refresh_status(self) -> None:
        """Update the footer readout and the composer's model/effort badge."""
        if self._bridge is None:
            return
        from delta_app.reasoning import DEFAULT_THINKING_LEVEL

        session = self._bridge.session
        self.query_one(StatusBar).show(self._bridge.status(), busy=self._busy)
        effort = session.thinking_level or DEFAULT_THINKING_LEVEL
        self.query_one(ModelBadge).show(session.model, effort)

    async def _refresh_sidebar(self) -> None:
        """Repaint the session list from the catalog."""
        if self._bridge is None:
            return
        await self.query_one(Sidebar).show(self._bridge.sessions())


async def run_app(ns: argparse.Namespace) -> int:
    """Build the shared runtime session and run the TUI over it."""
    from delta_app.runtime import build_session, settle_project_trust
    from delta_app.tui.trust_dialog import ask_in_dialog

    # Settled before the session exists: its answer decides what gets loaded.
    await settle_project_trust(ns, ask=ask_in_dialog)
    bridge = SessionBridge(await build_session(ns))
    app = DeltaApp(bridge)
    try:
        await app.run_async()
    finally:
        # The bridge's session changes when the user switches; shut down
        # whichever one is current.
        await bridge.shutdown()
    return 0


def run_tui(ns: argparse.Namespace | None = None) -> int:
    """Launch the TUI. Returns a process exit code."""
    import asyncio

    if ns is None:
        DeltaApp().run()
        return 0
    return asyncio.run(run_app(ns))


__all__ = ["DeltaApp", "run_app", "run_tui"]
