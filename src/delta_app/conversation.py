"""CodingSession — the single stateful runtime object for a persistent coding session.

``CodingSession`` is the integration layer between Delta's pure agent runtime
(``RuntimeHarness`` → ``run_agent_loop``) and the rest of the application.  The
harness stays focused on inference and tool execution; CodingSession owns every
application-level concern:

* Persistence — durable append-only JSONL via ``JsonlVault``
* Branching / rewinding — session-tree navigation via lineage + projection
* Compaction — manual and automatic context summarisation
* Model / provider switching — live, mid-session
* Thinking mode management
* Token-budget tracking and context-limit enforcement
* Mid-run steering and follow-up queues (delegated to the harness)
* Automatic session naming
* Slash-command dispatch
* Terminal command execution with optional context injection
* Multi-session lifecycle (create / resume / replace / shutdown)
* Extension / plugin discovery and hot reload
* Diagnostics, exports, and a clean read-only API
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import os
import subprocess
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import time

from delta_harness.contracts.stream import (
    AgentEvent,
    MessageEndEvent,
    RunEndEvent,
)
from delta_harness.contracts.tooling import ToolSpec
from delta_harness.contracts.transcript import (
    HumanEntry,
    ModelEntry,
    PruneSummaryEntry,
    ShellResultEntry,
    TranscriptEntry,
    surface_text,
)
from delta_harness.contracts.values import JValue
from delta_harness.harness import (
    PendingBatch,
    QueueShiftEvent,
    RuntimeConfig,
    RuntimeHarness,
)
from delta_harness.provider.base import ModelProvider
from delta_harness.provider.wire import StreamCloseEvent
from delta_harness.session.index import SessionCatalog, SessionMeta
from delta_harness.session.records import (
    ForkSummaryRecord,
    ModelSwapRecord,
    PruneRecord,
    ReasoningLevelRecord,
    SessionMetaRecord,
    SessionRecord,
    TagRecord,
    TipRecord,
    TranscriptRecord,
    mint_id,
)
from delta_harness.session.replay import (
    project_records,
    trace_to_entry,
)
from delta_harness.session.store import JsonlVault

# ── context-window budget constants ────────────────────────────────────────

_MODEL_CONTEXT_LIMITS: dict[str, int] = {
    "claude-sonnet-4-20250514": 200_000,
    "claude-opus-4-20250514": 200_000,
    "claude-opus-4-6": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-haiku-4-5-20251001": 200_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
}
_DEFAULT_CONTEXT_LIMIT = 128_000
_COMPACTION_RATIO = 0.75
_MIN_ENTRIES_TO_COMPACT = 6
_COMPACTION_KEEP_RECENT = 4

_EXTENSION_GROUP = "delta.extensions"

_SUMMARY_SYSTEM = (
    "You are a concise summarizer. Produce a brief summary of the conversation "
    "that preserves key decisions, code changes, file paths, and open questions. "
    "Keep it under 500 words."
)

_NAMING_SYSTEM = (
    "Generate a short, descriptive title (3-8 words) for this coding conversation. "
    "Return only the title, nothing else."
)

_TITLE_MAX_CHARS = 48

#: Re-summarise the session title once it has grown this many entries since
#: the last automatic naming.
_RENAME_EVERY_N_ENTRIES = 8


def summarize_prompt(text: str, *, limit: int = _TITLE_MAX_CHARS) -> str:
    """Condense a prompt into a one-line session title.

    Collapses whitespace, strips a leading slash-command marker, and clips on
    a word boundary. Returns ``""`` when *text* has no usable content.
    """
    flat = " ".join(text.split())
    if flat.startswith("/"):
        flat = flat.lstrip("/")
    if not flat:
        return ""
    if len(flat) <= limit:
        return flat
    clipped = flat[:limit]
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return clipped.rstrip(" ,.;:-") + "…"


# ── session statistics ─────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SessionStats:
    """Cumulative usage statistics for a coding session."""

    turn_count: int = 0
    message_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    estimated_context_tokens: int = 0
    context_limit: int = _DEFAULT_CONTEXT_LIMIT


# ── main class ─────────────────────────────────────────────────────────────


class CodingSession:
    """Single stateful runtime object for a persistent, resumable coding session.

    Wraps ``RuntimeHarness`` and adds persistence, branching, compaction,
    model/provider switching, thinking-mode management, extensions, diagnostics,
    and session lifecycle management.
    """

    def __init__(
        self,
        *,
        session_id: str,
        vault: JsonlVault | None,
        catalog: SessionCatalog | None = None,
        provider: ModelProvider,
        provider_name: str,
        model: str,
        system: str,
        tools: list[ToolSpec],
        transcript: list[TranscriptEntry] | None = None,
        record_ids: list[str] | None = None,
        title: str | None = None,
        thinking_level: str | None = None,
        cwd: str | None = None,
        tools_loader: Callable[[], list[ToolSpec]] | None = None,
    ) -> None:
        self._session_id = session_id
        self._vault = vault
        self._catalog = catalog
        self._provider = provider
        self._provider_name = provider_name
        self._model = model
        self._system = system
        self._title = title
        self._thinking_level = thinking_level
        self._cwd = cwd or os.getcwd()
        self._record_ids: list[str] = list(record_ids) if record_ids else []
        self._tip_id: str | None = self._record_ids[-1] if self._record_ids else None
        self._extensions: list[object] = []
        self._named = title is not None
        self._named_at = 0
        self._renamed_manually = False
        self._tools_loader = tools_loader

        # Cumulative stats
        self._turn_count = 0
        self._total_input = 0
        self._total_output = 0
        self._total_cost = 0.0

        # Build harness
        config = RuntimeConfig(
            provider=provider,
            model=model,
            system=system,
            tools=list(tools),
        )
        self._harness = RuntimeHarness(config, messages=transcript or [])

      
        self._unsubscribe = self._harness.on_event(self._on_harness_event)

        
        if transcript:
            for entry in transcript:
                if isinstance(entry, ModelEntry):
                    self._turn_count += 1
                    self._total_input += entry.usage.input
                    self._total_output += entry.usage.output
                    self._total_cost += entry.usage.cost.total

    

    @classmethod
    async def create(
        cls,
        *,
        provider: ModelProvider,
        provider_name: str,
        model: str,
        system: str,
        tools: list[ToolSpec] | None = None,
        sessions_dir: Path | None = None,
        session_id: str | None = None,
        cwd: str | None = None,
        tools_loader: Callable[[], list[ToolSpec]] | None = None,
    ) -> CodingSession:
        """Create a new coding session with optional persistence."""
        sid = session_id or mint_id()[:12]
        vault: JsonlVault | None = None
        catalog: SessionCatalog | None = None
        effective_cwd = cwd or os.getcwd()
        if sessions_dir is not None:
            vault = JsonlVault(sessions_dir / f"{sid}.jsonl")
            await vault.append(SessionMetaRecord(cwd=effective_cwd))
            catalog = SessionCatalog(sessions_dir)
            catalog.upsert(
                SessionCatalog.prepare(
                    session_id=sid,
                    vault_path=vault.path,
                    cwd=effective_cwd,
                    model=model,
                    provider=provider_name,
                )
            )
        return cls(
            session_id=sid,
            vault=vault,
            catalog=catalog,
            provider=provider,
            provider_name=provider_name,
            model=model,
            system=system,
            tools=tools or [],
            cwd=cwd,
            tools_loader=tools_loader,
        )

    @classmethod
    async def resume(
        cls,
        session_id: str,
        *,
        provider: ModelProvider,
        provider_name: str,
        model: str,
        system: str,
        tools: list[ToolSpec] | None = None,
        sessions_dir: Path,
        tools_loader: Callable[[], list[ToolSpec]] | None = None,
    ) -> CodingSession:
        """Load an existing session from persistent storage."""
        vault = JsonlVault(sessions_dir / f"{session_id}.jsonl")
        if not vault.path.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")

        records = await vault.read_all()
        state = project_records(records)

        catalog = SessionCatalog(sessions_dir)
        catalog.touch(session_id)

        return cls(
            session_id=session_id,
            vault=vault,
            catalog=catalog,
            provider=provider,
            provider_name=provider_name,
            model=state.model or model,
            system=system,
            tools=tools or [],
            transcript=state.transcript,
            record_ids=state.record_ids,
            title=state.title,
            thinking_level=state.thinking_level,
            cwd=state.cwd,
            tools_loader=tools_loader,
        )

    # ── run lifecycle ──────────────────────────────────────────────────

    def submit(self, text: str) -> AsyncIterator[AgentEvent]:
        """Submit a user prompt and stream agent events.

        Persistence, stats tracking, auto-naming, and auto-compaction are
        handled automatically by the internal event subscriber and the
        post-run hook.
        """
        # Name the session from its first prompt straight away, so listings
        # never show a bare id while waiting on the model-generated title.
        if not self._title:
            self.set_provisional_title(text)
        return self._run_wrapped(self._harness.submit(text))

    def _should_rename(self, entries: int) -> bool:
        """Whether the title should be regenerated at *entries* messages.

        The opening prompt stops describing a conversation once it has moved
        on, so re-summarise as it grows rather than naming once forever.
        """
        if self._renamed_manually:
            return False
        return entries >= self._named_at + _RENAME_EVERY_N_ENTRIES

    def set_provisional_title(self, text: str) -> None:
        """Derive a placeholder title from *text* and publish it immediately.

        Does not set ``_named``, so ``auto_name`` still refines it later.
        """
        title = summarize_prompt(text)
        if not title:
            return
        self._title = title
        self._sync_title_to_catalog(title)

    def resume_run(self) -> AsyncIterator[AgentEvent]:
        """Continue the agent loop without appending a new user message."""
        return self._run_wrapped(self._harness.resume())

    def abort(self) -> None:
        """Cancel the currently running agent loop."""
        self._harness.abort()

    async def shutdown(self) -> None:
        """Flush pending state and release resources."""
        self._unsubscribe()
        if self._vault and self._tip_id:
            try:
                await self._vault.append(TipRecord(entry_id=self._tip_id))
            except Exception:  # noqa: BLE001
                pass

    # ── mid-run queues ─────────────────────────────────────────────────

    def inject(self, content: str) -> QueueShiftEvent:
        """Queue a steering message for the active or next run."""
        return self._harness.inject(content)

    def enqueue(self, content: str) -> QueueShiftEvent:
        """Queue a follow-up message for when the active run stops."""
        return self._harness.enqueue(content)

    @property
    def pending(self) -> PendingBatch:
        """Snapshot of queued injection and follow-up messages."""
        return self._harness.pending

    def flush_queues(self) -> PendingBatch:
        """Clear and return all queued messages."""
        return self._harness.flush_queues()

    # ── model / provider switching ─────────────────────────────────────

    async def switch_model(self, model: str) -> None:
        """Switch to a different model identifier (takes effect next turn)."""
        self._model = model
        self._harness.settings.model = model
        if self._vault:
            await self._vault.append(ModelSwapRecord(model=model))

    async def switch_provider(
        self,
        provider: ModelProvider,
        provider_name: str,
        model: str | None = None,
    ) -> None:
        """Replace the active model provider (takes effect next turn)."""
        self._provider = provider
        self._provider_name = provider_name
        self._harness.settings.provider = provider
        if model:
            await self.switch_model(model)

    # ── thinking mode ──────────────────────────────────────────────────

    async def set_thinking(self, level: str | None) -> None:
        """Set the thinking/reasoning depth level (or disable with ``None``)."""
        self._thinking_level = level
        if self._vault:
            await self._vault.append(ReasoningLevelRecord(thinking_level=level))

    @property
    def thinking_level(self) -> str | None:
        """The current thinking/reasoning depth, or ``None`` if disabled."""
        return self._thinking_level

    # ── context compaction ─────────────────────────────────────────────

    async def compact(self) -> PruneSummaryEntry:
        """Compact the transcript by summarising older entries.

        Keeps the most recent entries intact and replaces everything before
        them with a single ``PruneSummaryEntry``.  The compaction is persisted
        as a ``PruneRecord`` so that resumption reconstructs the same state.

        Raises ``ValueError`` if the transcript is too short to compact.
        """
        transcript = list(self._harness.transcript)
        if len(transcript) < _MIN_ENTRIES_TO_COMPACT:
            raise ValueError(
                f"Transcript has {len(transcript)} entries; "
                f"need at least {_MIN_ENTRIES_TO_COMPACT} to compact."
            )

        keep_count = max(_COMPACTION_KEEP_RECENT, len(transcript) // 4)
        to_summarize = transcript[:-keep_count]
        to_keep = transcript[-keep_count:]

        summary_text = await self._generate_summary(to_summarize)
        tokens_before = self._estimate_context_tokens()

        summary_entry = PruneSummaryEntry(
            summary=summary_text,
            tokens_before=tokens_before,
        )

        compacted_record_ids = self._record_ids[: len(to_summarize)]
        remaining_record_ids = self._record_ids[len(to_summarize) :]

        prune_record = PruneRecord(
            summary=summary_text,
            replaces_entry_ids=compacted_record_ids,
        )
        if self._vault:
            await self._vault.append(prune_record)

        new_transcript: list[TranscriptEntry] = [summary_entry, *to_keep]
        self._harness.set_messages(new_transcript)
        self._record_ids = [prune_record.id, *remaining_record_ids]

        return summary_entry

    def should_compact(self) -> bool:
        """Return whether automatic compaction is recommended."""
        estimated = self._estimate_context_tokens()
        limit = self._context_limit()
        return estimated > limit * _COMPACTION_RATIO

    # ── automatic session naming ───────────────────────────────────────

    async def auto_name(self) -> str:
        """Generate and persist an automatic session title from the first exchange."""
        transcript = self._harness.transcript
        if len(transcript) < 2:
            raise ValueError("Need at least one exchange to generate a title.")

        context = "\n".join(
            f"[{entry.role}] {surface_text(entry)}"
            for entry in transcript[:4]
        )

        title = await self._utility_completion(
            f"Conversation so far:\n{context}",
            _NAMING_SYSTEM,
        )
        title = title.strip().strip('"').strip("'")[:80]
        if not title:
            # Never fall back to the raw session id — that is what surfaced as
            # "random numbers" in session listings. Keep whatever we derived
            # from the first prompt instead.
            title = self._title or summarize_prompt(surface_text(transcript[0]))
        if not title:
            return self._session_id

        self._title = title
        self._named = True
        if self._vault:
            await self._vault.append(TagRecord(label=title))
        self._sync_title_to_catalog(title)
        return title

    # ── catalog sync ───────────────────────────────────────────────────

    @property
    def catalog(self) -> SessionCatalog | None:
        """The session catalog, or ``None`` for ephemeral sessions."""
        return self._catalog

    def _sync_title_to_catalog(self, title: str) -> None:
        """Push a title change to the session catalog, if one is active."""
        if self._catalog is None:
            return
        existing = self._catalog.get(self._session_id)
        if existing is None:
            return
        self._catalog.upsert(
            SessionMeta(
                session_id=existing.session_id,
                vault_path=existing.vault_path,
                cwd=existing.cwd,
                model=existing.model,
                provider=existing.provider,
                title=title,
                created_at=existing.created_at,
                updated_at=time(),
            )
        )

    # ── slash-command dispatch ─────────────────────────────────────────

    #: Slash commands, with one-line help. Drives autocomplete and /help.
    COMMANDS: tuple[tuple[str, str], ...] = (
        ("model", "Show or switch the model"),
        ("provider", "Show or switch the provider"),
        ("think", "Set the thinking/effort level"),
        ("compact", "Summarise older context"),
        ("stats", "Show token, turn, and cost stats"),
        ("name", "Show or set the session title"),
        ("export", "Export the transcript"),
        ("branch", "Fork the conversation here"),
        ("rewind", "Rewind to a prior entry"),
        ("shell", "Run a shell command"),
        ("reload", "Reload tools and extensions"),
        ("diag", "Dump session diagnostics"),
    )

    async def handle_command(self, text: str) -> str | None:
        """Handle a ``/``-prefixed command.

        Returns a human-readable response string if the input was a recognised
        command, or ``None`` if it was not (so the caller can treat it as a
        normal prompt).
        """
        if not text.startswith("/"):
            return None

        parts = text[1:].split(None, 1)
        cmd = parts[0].lower() if parts else ""
        arg = parts[1].strip() if len(parts) > 1 else ""

        handlers: dict[str, Callable[[str], object]] = {
            "model": self._cmd_model,
            "provider": self._cmd_provider,
            "think": self._cmd_think,
            "thinking": self._cmd_think,
            "compact": self._cmd_compact,
            "stats": self._cmd_stats,
            "name": self._cmd_name,
            "export": self._cmd_export,
            "branch": self._cmd_branch,
            "rewind": self._cmd_rewind,
            "shell": self._cmd_shell,
            "reload": self._cmd_reload,
            "diag": self._cmd_diag,
        }

        handler = handlers.get(cmd)
        if handler is None:
            return None
        result = handler(arg)
        if asyncio.iscoroutine(result):
            return await result  # type: ignore[misc]
        return result  # type: ignore[return-value]

    # ── terminal command execution ─────────────────────────────────────

    async def run_shell(
        self,
        command: str,
        *,
        inject: bool = False,
        timeout: float = 30.0,
    ) -> ShellResultEntry:
        """Execute a shell command and optionally inject output into context."""
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    subprocess.run,
                    command,
                    shell=True,  # noqa: S602 — intentional user-controlled execution
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                ),
                timeout=timeout + 5,
            )
            output = result.stdout
            if result.stderr:
                output += ("\n" if output else "") + result.stderr
            entry = ShellResultEntry(
                command=command,
                output=output,
                exit_code=result.returncode,
            )
        except (TimeoutError, subprocess.TimeoutExpired):
            entry = ShellResultEntry(
                command=command,
                output="Command timed out",
                exit_code=None,
                cancelled=True,
            )

        if inject:
            self._harness.push_message(entry)
            await self._persist_entry(entry)
        return entry

    # ── session tree: branching and rewinding ───────────────────────────

    async def branch(self, summary: str | None = None) -> str:
        """Create a conversation branch from the current point.

        Persists a ``ForkSummaryRecord`` and returns its ID.  Subsequent
        entries will descend from the fork.
        """
        if summary is None:
            context = "\n".join(
                f"[{e.role}] {surface_text(e)}"
                for e in self._harness.transcript[-4:]
            )
            summary = await self._utility_completion(
                f"Summarize this conversation branch concisely:\n{context}",
                "Produce a 1-2 sentence summary of the conversation so far.",
            )

        record = ForkSummaryRecord(
            summary=summary.strip(),
            branch_root_id=self._tip_id,
        )
        if self._vault:
            await self._vault.append(record)
        self._tip_id = record.id
        return record.id

    async def rewind(self, entry_id: str) -> None:
        """Rewind the session to a previous point in the conversation tree.

        Loads the lineage from root to *entry_id*, projects it into a
        transcript, and replaces the live state.
        """
        if self._vault is None:
            raise RuntimeError("Rewind requires a persistent session.")

        records = await self._vault.read_all()
        path = trace_to_entry(records, entry_id)
        state = project_records(path)

        self._harness.set_messages(state.transcript)
        self._record_ids = state.record_ids
        self._tip_id = entry_id
        if state.model:
            self._model = state.model
            self._harness.settings.model = state.model
        if state.thinking_level is not None:
            self._thinking_level = state.thinking_level
        if state.title:
            self._title = state.title

        await self._vault.append(TipRecord(entry_id=entry_id))

    # ── multi-session lifecycle ────────────────────────────────────────

    async def new_session(
        self,
        sessions_dir: Path | None = None,
    ) -> CodingSession:
        """Create a fresh session, preserving the current provider/model/tools."""
        await self.shutdown()
        base = sessions_dir
        if base is None and self._vault:
            base = self._vault.path.parent
        return await CodingSession.create(
            provider=self._provider,
            provider_name=self._provider_name,
            model=self._model,
            system=self._system,
            tools=list(self._harness.settings.tools),
            sessions_dir=base,
            tools_loader=self._tools_loader,
        )

    async def replace_session(self, session_id: str) -> CodingSession:
        """Switch to a different existing session.

        Shuts down the current session and loads the target.
        """
        if self._vault is None:
            raise RuntimeError("Replace requires a persistent session.")
        await self.shutdown()
        return await CodingSession.resume(
            session_id,
            provider=self._provider,
            provider_name=self._provider_name,
            model=self._model,
            system=self._system,
            tools=list(self._harness.settings.tools),
            sessions_dir=self._vault.path.parent,
            tools_loader=self._tools_loader,
        )

    # ── extensions and hot reload ──────────────────────────────────────

    def load_extensions(self) -> list[object]:
        """Discover and activate extensions from entry points."""
        loaded: list[object] = []
        try:
            eps = importlib.metadata.entry_points(group=_EXTENSION_GROUP)
        except TypeError:
            eps = importlib.metadata.entry_points().get(  # type: ignore[assignment]
                _EXTENSION_GROUP, []
            )
        for ep in eps:
            try:
                ext = ep.load()()
                if ext is not None:
                    loaded.append(ext)
            except Exception:  # noqa: BLE001
                pass
        self._extensions = loaded
        return loaded

    async def reload(self) -> None:
        """Hot-reload tools, extensions, and system prompt."""
        self._extensions = self.load_extensions()
        if self._tools_loader:
            self._harness.settings.tools = self._tools_loader()
        self._harness.settings.system = self._system

    # ── exports ────────────────────────────────────────────────────────

    async def export(self, fmt: str = "text") -> str:
        """Export the session transcript in the given format (text/json/jsonl)."""
        if self._vault is not None:
            return await self._export_from_vault(fmt)
        return self._export_from_transcript(fmt)

    async def _export_from_vault(self, fmt: str) -> str:
        assert self._vault is not None
        records = await self._vault.read_all()
        if fmt == "jsonl":
            from delta_harness.session.store import serialize_record

            return "".join(serialize_record(r) for r in records)
        if fmt == "json":
            from pydantic import TypeAdapter

            adapter: TypeAdapter[list[SessionRecord]] = TypeAdapter(list[SessionRecord])
            return adapter.dump_json(records, indent=2).decode()
        lines: list[str] = []
        for r in records:
            if isinstance(r, TranscriptRecord):
                lines.append(f"[{r.message.role}] {surface_text(r.message)}")
        return "\n\n".join(lines)

    def _export_from_transcript(self, fmt: str) -> str:
        transcript = self._harness.transcript
        if fmt == "json":
            from pydantic import TypeAdapter

            adapter: TypeAdapter[list[TranscriptEntry]] = TypeAdapter(list[TranscriptEntry])
            return adapter.dump_json(list(transcript), indent=2).decode()
        return "\n\n".join(
            f"[{e.role}] {surface_text(e)}" for e in transcript
        )

    # ── diagnostics ────────────────────────────────────────────────────

    def diagnostics(self) -> dict[str, JValue]:
        """Return a snapshot of session state for debugging."""
        return {
            "session_id": self._session_id,
            "model": self._model,
            "provider": self._provider_name,
            "thinking_level": self._thinking_level,
            "title": self._title,
            "message_count": len(self._harness.transcript),
            "turn_count": self._turn_count,
            "total_input_tokens": self._total_input,
            "total_output_tokens": self._total_output,
            "total_cost_usd": self._total_cost,
            "estimated_context_tokens": self._estimate_context_tokens(),
            "context_limit": self._context_limit(),
            "active": self._harness.active,
            "pending_count": self._harness.pending_count,
            "persistent": self._vault is not None,
            "cwd": self._cwd,
            "extensions_loaded": len(self._extensions),
        }

    # ── read-only API ──────────────────────────────────────────────────

    @property
    def session_id(self) -> str:
        """Unique identifier for this session."""
        return self._session_id

    @property
    def transcript(self) -> tuple[TranscriptEntry, ...]:
        """Immutable snapshot of the current conversation."""
        return self._harness.transcript

    @property
    def model(self) -> str:
        """The active model identifier."""
        return self._model

    @property
    def provider_name(self) -> str:
        """The active provider name."""
        return self._provider_name

    @property
    def tools(self) -> tuple[ToolSpec, ...]:
        """The currently registered tool specifications."""
        return tuple(self._harness.settings.tools)

    @property
    def active(self) -> bool:
        """Whether an agent run is in progress."""
        return self._harness.active

    @property
    def title(self) -> str | None:
        """The session display name, if set."""
        return self._title

    @property
    def cwd(self) -> str:
        """Working directory associated with this session."""
        return self._cwd

    @property
    def usage(self) -> SessionStats:
        """Cumulative usage statistics."""
        return SessionStats(
            turn_count=self._turn_count,
            message_count=len(self._harness.transcript),
            total_input_tokens=self._total_input,
            total_output_tokens=self._total_output,
            total_cost_usd=self._total_cost,
            estimated_context_tokens=self._estimate_context_tokens(),
            context_limit=self._context_limit(),
        )

    @property
    def harness(self) -> RuntimeHarness:
        """Direct access to the underlying harness (advanced use)."""
        return self._harness

    # ── internal: event handling ────────────────────────────────────────

    async def _on_harness_event(self, event: AgentEvent) -> None:
        """Subscriber called by the harness for every streamed event."""
        if not isinstance(event, MessageEndEvent):
            return
        record_id = await self._persist_entry(event.message)
        if record_id:
            self._record_ids.append(record_id)
            self._tip_id = record_id
        if isinstance(event.message, ModelEntry):
            self._turn_count += 1
            self._total_input += event.message.usage.input
            self._total_output += event.message.usage.output
            self._total_cost += event.message.usage.cost.total

    async def _run_wrapped(
        self, stream: AsyncIterator[AgentEvent]
    ) -> AsyncIterator[AgentEvent]:
        """Wrap a harness event stream with post-run processing."""
        completed = False
        try:
            async for event in stream:
                if isinstance(event, RunEndEvent):
                    completed = True
                yield event
        finally:
            if completed:
                await self._post_run()

    async def _post_run(self) -> None:
        """Housekeeping after a completed agent run."""
        if self._vault and self._tip_id:
            try:
                await self._vault.append(TipRecord(entry_id=self._tip_id))
            except Exception:  # noqa: BLE001
                pass

        # Rename from the model as soon as there is an exchange to summarise,
        # then re-summarise once the conversation has grown enough that the
        # opening prompt no longer describes it. A provisional title (derived
        # from the first prompt) does not count as named.
        entries = len(self._harness.transcript)
        if entries >= 2 and (not self._named or self._should_rename(entries)):
            try:
                await self.auto_name()
                self._named_at = entries
            except Exception:  # noqa: BLE001
                pass

        if (
            self.should_compact()
            and len(self._harness.transcript) >= _MIN_ENTRIES_TO_COMPACT
        ):
            try:
                await self.compact()
            except Exception:  # noqa: BLE001
                pass

    # ── internal: persistence ──────────────────────────────────────────

    async def _persist_entry(self, entry: TranscriptEntry) -> str | None:
        """Append a transcript record to the vault, returning its record ID."""
        if self._vault is None:
            return None
        record = TranscriptRecord(message=entry, parent_id=self._tip_id)
        try:
            await self._vault.append(record)
            return record.id
        except Exception:  # noqa: BLE001
            return None

    # ── internal: token estimation ─────────────────────────────────────

    def _estimate_context_tokens(self) -> int:
        """Estimate the current context size in tokens."""
        last_input = 0
        last_idx = -1
        transcript = self._harness.transcript
        for i, entry in enumerate(transcript):
            if isinstance(entry, ModelEntry):
                last_input = entry.usage.input
                last_idx = i

        if last_idx < 0:
            return sum(len(surface_text(e)) // 4 for e in transcript)

        extra = sum(
            len(surface_text(e)) // 4 for e in transcript[last_idx + 1 :]
        )
        return last_input + extra

    def _context_limit(self) -> int:
        """Return the context token limit for the current model."""
        return _MODEL_CONTEXT_LIMITS.get(self._model, _DEFAULT_CONTEXT_LIMIT)

    # ── internal: utility completions ──────────────────────────────────

    async def _generate_summary(
        self, entries: Sequence[TranscriptEntry]
    ) -> str:
        """Generate a concise summary of transcript entries using the provider."""
        text = "\n".join(
            f"[{entry.role}] {surface_text(entry)}" for entry in entries
        )
        if len(text) > 50_000:
            text = text[:50_000] + "\n[...truncated...]"

        return await self._utility_completion(
            f"Summarize this conversation:\n\n{text}",
            _SUMMARY_SYSTEM,
        )

    async def _utility_completion(self, prompt: str, system: str) -> str:
        """Get a simple text completion from the provider for internal use."""
        messages: list[TranscriptEntry] = [HumanEntry(content=prompt)]
        async for event in self._provider.stream_response(
            model=self._model,
            system=system,
            messages=messages,
            tools=[],
        ):
            if isinstance(event, StreamCloseEvent):
                return event.message.text
        return ""

    # ── internal: command handlers ─────────────────────────────────────

    async def _cmd_model(self, arg: str) -> str:
        if not arg:
            return f"Current model: {self._model}"
        await self.switch_model(arg)
        return f"Switched to model: {arg}"

    async def _cmd_provider(self, arg: str) -> str:
        if not arg:
            return f"Current provider: {self._provider_name}"
        from delta_app.config.config import resolve_provider

        provider = resolve_provider(arg)
        await self.switch_provider(provider, arg)
        return f"Switched to provider: {arg}"

    async def _cmd_think(self, arg: str) -> str:
        if not arg or arg.lower() in {"off", "none", "disable"}:
            await self.set_thinking(None)
            return "Thinking disabled."
        await self.set_thinking(arg.split()[0])
        return f"Thinking set to: {arg.split()[0]}"

    async def _cmd_compact(self, _arg: str) -> str:
        try:
            entry = await self.compact()
            preview = entry.summary[:100]
            return f"Compacted. Summary: {preview}..."
        except ValueError as exc:
            return str(exc)

    async def _cmd_stats(self, _arg: str) -> str:
        s = self.usage
        return (
            f"Turns: {s.turn_count}  Messages: {s.message_count}\n"
            f"Input tokens: {s.total_input_tokens:,}  "
            f"Output tokens: {s.total_output_tokens:,}\n"
            f"Cost: ${s.total_cost_usd:.4f}\n"
            f"Context: ~{s.estimated_context_tokens:,} / "
            f"{s.context_limit:,} tokens"
        )

    async def _cmd_name(self, arg: str) -> str:
        if not arg:
            return f"Session: {self._title or self._session_id}"
        self._title = arg
        self._named = True
        # An explicit name is final — stop auto-renaming over it.
        self._renamed_manually = True
        if self._vault:
            await self._vault.append(TagRecord(label=arg))
        self._sync_title_to_catalog(arg)
        return f"Session named: {arg}"

    async def _cmd_export(self, arg: str) -> str:
        fmt = arg or "text"
        return await self.export(fmt)

    async def _cmd_branch(self, arg: str) -> str:
        summary = arg or None
        branch_id = await self.branch(summary)
        return f"Branch created: {branch_id}"

    async def _cmd_rewind(self, arg: str) -> str:
        if not arg:
            return "Usage: /rewind <entry_id>"
        try:
            await self.rewind(arg)
            return f"Rewound to: {arg}"
        except Exception as exc:  # noqa: BLE001
            return f"Rewind failed: {exc}"

    async def _cmd_shell(self, arg: str) -> str:
        if not arg:
            return "Usage: /shell <command>"
        entry = await self.run_shell(arg)
        return f"Exit {entry.exit_code}:\n{entry.output}"

    async def _cmd_reload(self, _arg: str) -> str:
        await self.reload()
        return "Reloaded."

    async def _cmd_diag(self, _arg: str) -> str:
        return json.dumps(self.diagnostics(), indent=2)


__all__ = [
    "CodingSession",
    "SessionStats",
]
