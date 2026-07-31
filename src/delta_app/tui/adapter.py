"""Adapter boundary: ``AgentEvent`` stream -> TUI state.

The only place the TUI touches the runtime.  ``SessionBridge`` owns a
``CodingSession`` and translates its provider-neutral event stream into
view-model updates.  It runs no agent loop, calls no provider, and executes
no tools — it delegates entirely to ``CodingSession.submit``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass

from delta_harness.contracts.stream import (
    AgentEvent,
    MessageEndEvent,
    MessageUpdateEvent,
    ToolRunEndEvent,
    ToolRunStartEvent,
)
from delta_harness.contracts.transcript import ModelEntry

from delta_app.conversation import CodingSession

# ---------------------------------------------------------------------------
# View-model updates
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TextDelta:
    """Incremental assistant text to append to the live block."""

    text: str


@dataclass(frozen=True, slots=True)
class TurnFinished:
    """The assistant message completed."""

    text: str
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ToolStarted:
    """A tool invocation began."""

    call_id: str
    name: str
    summary: str


@dataclass(frozen=True, slots=True)
class ToolFinished:
    """A tool invocation completed."""

    call_id: str
    name: str
    output: str
    is_error: bool


Update = TextDelta | TurnFinished | ToolStarted | ToolFinished


# ---------------------------------------------------------------------------
# Status snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SessionEntry:
    """One row in the session sidebar."""

    session_id: str
    title: str | None
    updated_at: float
    is_active: bool

    @property
    def label(self) -> str:
        """Display name — the title, or a neutral placeholder when unnamed.

        A session only stays untitled until its first prompt, so this shows
        for empty sessions rather than exposing a raw id.
        """
        return self.title or "Untitled session"


@dataclass(frozen=True, slots=True)
class StatusSnapshot:
    """Everything the footer displays."""

    provider: str
    model: str
    session_id: str
    context_used: int
    context_limit: int
    branch: str | None

    @property
    def context_percent(self) -> int:
        """Context consumption as a whole percentage."""
        if self.context_limit <= 0:
            return 0
        return min(100, round(self.context_used * 100 / self.context_limit))


# ---------------------------------------------------------------------------
# Tool call summaries
# ---------------------------------------------------------------------------

_PATH_ARGS = ("file_path", "path", "notebook_path")


def summarize_call(name: str, args: dict) -> str:
    """One-line description of a tool call, e.g. ``Read foo.py``."""
    for key in _PATH_ARGS:
        value = args.get(key)
        if isinstance(value, str) and value:
            return f"{name} {value.rsplit('/', 1)[-1].rsplit(chr(92), 1)[-1]}"
    command = args.get("command")
    if isinstance(command, str) and command:
        first = command.strip().splitlines()[0]
        return f"{name} {first[:60]}"
    pattern = args.get("pattern")
    if isinstance(pattern, str) and pattern:
        return f"{name} {pattern[:40]}"
    return name


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------


class SessionBridge:
    """Translates a ``CodingSession`` event stream into view-model updates."""

    __slots__ = ("_session", "_streamed")

    def __init__(self, session: CodingSession) -> None:
        self._session = session
        self._streamed = 0

    # -- read-only views ----------------------------------------------------

    @property
    def session(self) -> CodingSession:
        """The underlying runtime session."""
        return self._session

    @property
    def active(self) -> bool:
        """Whether a generation is currently running."""
        return self._session.active

    def status(self) -> StatusSnapshot:
        """Snapshot the footer state."""
        from delta_app.runtime import git_branch

        usage = self._session.usage
        return StatusSnapshot(
            provider=self._session.provider_name,
            model=self._session.model,
            session_id=self._session.session_id,
            context_used=usage.estimated_context_tokens,
            context_limit=usage.context_limit,
            branch=git_branch(self._session.cwd),
        )

    def sessions(self) -> list[SessionEntry]:
        """List every known session in a stable order, newest first.

        Ordered by creation time, not last use: sorting by ``updated_at``
        made the selected session jump to the top of the sidebar every time
        it was used.
        """
        catalog = self._session.catalog
        if catalog is None:
            return []
        active = self._session.session_id
        metas = sorted(catalog.list_all(), key=lambda m: m.created_at, reverse=True)
        return [
            SessionEntry(
                session_id=meta.session_id,
                title=meta.title,
                updated_at=meta.updated_at,
                is_active=meta.session_id == active,
            )
            for meta in metas
        ]

    # -- actions ------------------------------------------------------------

    def cancel(self) -> None:
        """Cancel the active generation via the runtime's own mechanism."""
        self._session.abort()

    async def switch_to(self, session_id: str) -> None:
        """Switch to an existing session via the runtime, reusing the provider."""
        if session_id == self._session.session_id:
            return
        self._session = await self._session.replace_session(session_id)
        self._streamed = 0

    async def start_new(self) -> None:
        """Create a fresh session via the runtime, reusing the provider."""
        self._session = await self._session.new_session()
        self._streamed = 0

    async def apply_model_choice(
        self, provider: str, model: str, api_key: str | None,
    ) -> str:
        """Persist credentials, rebuild the provider, and switch the session.

        Returns a one-line summary for the transcript.
        """
        from delta_app.config.config import resolve_provider
        from delta_app.config.loader import apply_config
        from delta_app.config.store import load_config, provider_info, save_config

        try:
            info = provider_info(provider)
        except KeyError:
            # Not one of the configurable backends (e.g. a scripted provider
            # in tests). Only the model can change.
            await self._session.switch_model(model)
            return f"Switched to {provider} · {model}"

        # Save the key (and the selection) so the next launch reuses them.
        config = load_config()
        if api_key:
            config.api_keys[provider] = api_key
        config.provider = provider
        config.model = model
        save_config(config)

        # Export credentials for this process. These are set unconditionally:
        # an explicit switch must override whatever the previous provider left
        # behind (notably the base URL shared by OpenAI-compatible backends).
        import os

        if provider == "anthropic":
            key_var, url_var = "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"
        else:
            key_var, url_var = "OPENAI_API_KEY", "OPENAI_BASE_URL"

        key = api_key or config.api_keys.get(provider) or os.environ.get(info.key_env)
        if key:
            os.environ[key_var] = key
        elif not info.needs_key:
            os.environ[key_var] = provider  # e.g. Ollama needs a placeholder
        os.environ[url_var] = config.base_urls.get(provider) or info.base_url
        apply_config(provider, config)

        if provider == self._session.provider_name:
            await self._session.switch_model(model)
        else:
            await self._session.switch_provider(
                resolve_provider(provider), provider, model,
            )
        return f"Switched to {info.label} · {model}"

    async def shutdown(self) -> None:
        """Shut down the session currently held."""
        await self._session.shutdown()

    async def handle_command(self, text: str) -> str | None:
        """Delegate a slash command to the runtime."""
        return await self._session.handle_command(text)

    async def submit(self, text: str) -> AsyncIterator[Update]:
        """Send *text* and yield view-model updates as the runtime streams."""
        self._streamed = 0
        async for event in self._session.submit(text):
            update = self._translate(event)
            if update is not None:
                yield update

    # -- translation --------------------------------------------------------

    def _translate(self, event: AgentEvent) -> Update | None:
        """Map one ``AgentEvent`` onto a view-model update."""
        if isinstance(event, MessageUpdateEvent):
            return self._on_update(event)
        if isinstance(event, MessageEndEvent):
            return self._on_end(event)
        if isinstance(event, ToolRunStartEvent):
            return ToolStarted(
                call_id=event.tool_call_id,
                name=event.tool_name,
                summary=summarize_call(event.tool_name, event.args),
            )
        if isinstance(event, ToolRunEndEvent):
            return ToolFinished(
                call_id=event.tool_call_id,
                name=event.tool_name,
                output=event.result.text,
                is_error=event.is_error,
            )
        return None

    def _on_update(self, event: MessageUpdateEvent) -> Update | None:
        """Emit only the newly streamed characters."""
        message = event.message
        if not isinstance(message, ModelEntry):
            return None
        full = message.text
        if len(full) <= self._streamed:
            return None
        delta = full[self._streamed:]
        self._streamed = len(full)
        return TextDelta(delta)

    def _on_end(self, event: MessageEndEvent) -> Update | None:
        """Finalize the assistant block, flushing any untransmitted tail."""
        message = event.message
        if not isinstance(message, ModelEntry):
            return None
        self._streamed = 0
        return TurnFinished(text=message.text, error=message.error_message)


__all__ = [
    "SessionBridge",
    "SessionEntry",
    "StatusSnapshot",
    "TextDelta",
    "ToolFinished",
    "ToolStarted",
    "TurnFinished",
    "Update",
    "summarize_call",
]
