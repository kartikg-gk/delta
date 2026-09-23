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
import re
import subprocess
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, replace
from pathlib import Path
from time import time
from typing import Any

from delta_app.context.budget import (
    DEFAULT_WINDOW,
    ContextEstimate,
    ContextLimits,
    apply_compaction,
    build_summary_prompts,
    estimate_context,
    exceeds_threshold,
    plan_compaction,
    resolve_window,
)
from delta_app.directives import build_default_registry, parse_command
from delta_app.instructions import PromptSection
from delta_app.routing import ROUTED_PROVIDER, RoutePin, is_route_failure, pin_from_storage
from delta_harness.contracts.stream import (
    AgentEvent,
    MessageEndEvent,
    MessageStartEvent,
    RetryEvent,
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
from delta_harness.driver import (
    PendingBatch,
    QueueShiftEvent,
    RuntimeConfig,
    RuntimeHarness,
)
from delta_harness.mending import HistoryMend, mend_tool_history
from delta_harness.provider.base import ModelProvider
from delta_harness.provider.wire import StreamCloseEvent
from delta_harness.session.index import SessionCatalog
from delta_harness.session.records import (
    ExtensionRecord,
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
    TreeIntegrityError,
    find_tip,
    project_active_branch,
    project_records,
    trace_to_entry,
)
from delta_harness.session.store import JsonlVault

# ── context-window budget ──────────────────────────────────────────────────

#: Context accounting and compaction policy live in ``delta_app.context.budget``.
#: The session owns one immutable ``ContextLimits`` and delegates every estimate,
#: threshold check, and compaction plan to that module.
_CONTEXT_LIMITS = ContextLimits()

#: The single slash-command catalog. Names, aliases, descriptions, and usage
#: strings all come from here; the session binds its own handlers to them.
COMMAND_REGISTRY = build_default_registry()

#: Registry commands a coding session dispatches, plus the ones a frontend runs
#: on its behalf: ``/continue`` resumes the loop (which yields events, not a
#: string), and ``/quit`` and ``/version`` belong to the frontend entirely.
#: They are listed here so completion and ``/help`` describe everything a user
#: can actually type. Anything outside this set stays in the registry for other
#: callers but is never offered here.
_RUNNABLE_COMMANDS: frozenset[str] = frozenset({
    "branch", "branches", "compact", "continue", "create-skill", "diag",
    "export", "help", "model", "name", "plan", "provider", "quit", "reload",
    "rewind", "session", "shell", "skills", "stats", "system", "think", "version",
})

_RUNNABLE_COMMAND_TABLE: tuple[tuple[str, str], ...] = tuple(
    (command.name, command.description)
    for command in COMMAND_REGISTRY.all_commands()
    if command.name in _RUNNABLE_COMMANDS
)

_EXTENSION_GROUP = "delta.extensions"

_NAMING_SYSTEM = (
    "Generate a short, descriptive title (3-8 words) for this coding conversation. "
    "Return only the title, nothing else."
)

_EXPORT_FORMATS = ("text", "json", "jsonl")
_EXPORT_SUFFIX = {"text": "txt", "json": "json", "jsonl": "jsonl"}


def parse_export_arg(arg: str) -> tuple[str, str | None]:
    """Split ``/export`` arguments into ``(format, destination)``.

    Accepts a format, a path, or both in either order::

        ""                     -> ("text", None)
        "json"                 -> ("json",  None)
        "out.json"             -> ("json",  "out.json")     # inferred
        "json ~/notes/a.json"  -> ("json",  "~/notes/a.json")
        "./exports/"           -> ("text",  "./exports/")
    """
    parts = arg.split()
    if not parts:
        return "text", None

    if parts[0].lower() in _EXPORT_FORMATS:
        fmt = parts[0].lower()
        rest = " ".join(parts[1:]).strip()
        return fmt, rest or None

    destination = arg.strip()

    # A lone bare word with no separator and no extension is a mistyped
    # format, not a path — returning it as the format lets the caller reject
    # it instead of silently creating a junk file.
    if (
        len(parts) == 1
        and not Path(destination).suffix
        and not any(sep in destination for sep in ("/", "\\", "~", "."))
    ):
        return destination.lower(), None

    # Otherwise infer the format from the file extension.
    suffix = Path(destination).suffix.lstrip(".").lower()
    for name, ext in _EXPORT_SUFFIX.items():
        if suffix == ext:
            return name, destination
    return "text", destination


_SKILL_NAME_RE = re.compile(r"\A[a-z0-9][a-z0-9._-]*\Z", re.IGNORECASE)
_SKILL_DESC_MAX = 100


def parse_create_skill_arg(arg: str) -> tuple[str, str]:
    """Split ``/create-skill`` arguments into ``(name, body)``.

    The first whitespace-delimited token is the skill name; everything after
    it — including newlines — is the body.
    """
    stripped = arg.strip()
    if not stripped:
        return "", ""
    parts = stripped.split(None, 1)
    name = parts[0]
    body = parts[1].strip() if len(parts) > 1 else ""
    return name, body


def derive_skill_description(body: str) -> str:
    """Use the first non-empty line of *body* as the skill description."""
    for line in body.splitlines():
        line = line.strip().lstrip("#").strip()
        if line:
            return line[:_SKILL_DESC_MAX]
    return ""


def render_skill_file(name: str, description: str, body: str) -> str:
    """Render a ``SKILL.md`` with front matter."""
    return (
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body.rstrip()}\n"
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


# Phrases providers use when a request exceeds the model's context window.
_OVERFLOW_PHRASES = (
    "context length",
    "context window",
    "context limit",
    "maximum context",
    "max context",
    "input is too long",
    "input length",
    "prompt is too long",
    "too many tokens",
    "token limit",
    "exceeds the limit",
    "exceeded the limit",
)


def _is_context_overflow(reply: ModelEntry) -> bool:
    """Whether a failed reply reports that the prompt did not fit."""
    if reply.stop_reason != "error":
        return False
    text = (reply.error_message or "").lower()
    return any(phrase in text for phrase in _OVERFLOW_PHRASES)


_HISTORY_REPAIR_NAMESPACE = "delta.history-repair"


def _records_after(
    records: list[SessionRecord], tip_id: str | None, ancestor_id: str | None,
) -> list[SessionRecord]:
    """Records on the active path strictly after ``ancestor_id``, root-first."""
    if tip_id is None:
        return []
    try:
        path = trace_to_entry(records, tip_id)
    except TreeIntegrityError:
        return []
    ids = [record.id for record in path]
    start = ids.index(ancestor_id) + 1 if ancestor_id in ids else 0
    return path[start:]


@dataclass(frozen=True, slots=True)
class SessionTreeChoice:
    """One branchable entry in the active session tree."""

    entry_id: str
    label: str
    active: bool = False
    is_tool_call: bool = False


def _short_preview(text: str, *, limit: int = 72) -> str:
    """Collapse whitespace and clip *text* for single-line display."""
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized or "(empty)"
    return f"{normalized[: limit - 1]}..."


def _tree_entry_title(record: SessionRecord) -> str:
    """Human-readable one-line title for a tree row."""
    match record.type:
        case "message":
            message = record.message
            if (
                isinstance(message, ModelEntry)
                and message.tool_calls
                and not message.text.strip()
            ):
                names = ", ".join(call.name for call in message.tool_calls)
                return f"tool call: {names}"
            return f"{message.role}: {_short_preview(surface_text(message))}"
        case "compaction":
            return f"compaction summary: {_short_preview(record.summary)}"
        case "branch_summary":
            return f"branch summary: {_short_preview(record.summary)}"
        case _:
            return record.type


def _is_branchable_record(record: SessionRecord) -> bool:
    """Whether a rewind/branch may target *record*."""
    if record.type in {"compaction", "branch_summary"}:
        return True
    if record.type != "message":
        return False
    return isinstance(record.message, HumanEntry | ModelEntry)


def _is_tool_call_record(record: SessionRecord) -> bool:
    """Whether *record* holds an assistant turn that requested tools."""
    return (
        record.type == "message"
        and isinstance(record.message, ModelEntry)
        and bool(record.message.tool_calls)
    )


def _ordered_tree_records(records: list[SessionRecord]) -> tuple[SessionRecord, ...]:
    """Depth-first ordering of the record tree, roots first.

    Pointer (``leaf``) records are skipped — they mark the head rather than
    forming part of the conversation. Orphans whose parent is missing are
    appended after the connected tree so nothing is silently dropped.
    """
    children: dict[str | None, list[SessionRecord]] = {}
    for record in records:
        if record.type != "leaf":
            children.setdefault(record.parent_id, []).append(record)

    ordered: list[SessionRecord] = []
    seen: set[str] = set()
    expanded: set[str | None] = set()

    def walk(root_parent_id: str | None) -> None:
        stack: list[str | None] = [root_parent_id]
        while stack:
            parent_id = stack.pop()
            if parent_id in expanded:
                continue
            expanded.add(parent_id)
            kids = children.get(parent_id, [])
            for child in kids:
                if child.id not in seen:
                    ordered.append(child)
                    seen.add(child.id)
            for child in reversed(kids):
                stack.append(child.id)

    walk(None)
    for record in records:
        if record.type != "leaf" and record.id not in seen:
            ordered.append(record)
            seen.add(record.id)
            walk(record.id)
    return tuple(ordered)


def _tree_branch_indents(records: list[SessionRecord]) -> dict[str, int]:
    """Indent depth per record: only forks (2nd+ sibling) add a level."""
    children: dict[str | None, list[str]] = {}
    for record in records:
        if record.type != "leaf":
            children.setdefault(record.parent_id, []).append(record.id)

    sibling_index = {
        child_id: index
        for kids in children.values()
        for index, child_id in enumerate(kids)
    }
    indents: dict[str, int] = {}
    for record in records:
        if record.type == "leaf":
            continue
        parent_indent = (
            indents.get(record.parent_id, 0) if record.parent_id is not None else 0
        )
        indents[record.id] = parent_indent + (
            1 if sibling_index.get(record.id, 0) > 0 else 0
        )
    return indents


def _tree_choice_label(record: SessionRecord, *, branch_indent: int = 0) -> str:
    """Indented display label for one tree row."""
    return f"{'  ' * branch_indent}{_tree_entry_title(record)}"


@dataclass(frozen=True, slots=True)
class SessionStats:
    """Cumulative usage statistics for a coding session."""

    turn_count: int = 0
    message_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    estimated_context_tokens: int = 0
    context_limit: int = DEFAULT_WINDOW
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    # Prompt size and cached share of the most recent model request.
    latest_prompt_tokens: int = 0
    latest_cache_read_tokens: int = 0

    @property
    def cache_hit_rate(self) -> float | None:
        """Share of all prompt tokens this session served from the cache.

        ``None`` while no provider has reported any cache activity, so a
        backend without prompt caching never shows a misleading 0%.
        """
        if not self._cache_active or self.total_input_tokens <= 0:
            return None
        return self.cache_read_tokens / self.total_input_tokens

    @property
    def latest_cache_hit_rate(self) -> float | None:
        """Share of the latest request's prompt served from the cache."""
        if not self._cache_active or self.latest_prompt_tokens <= 0:
            return None
        return self.latest_cache_read_tokens / self.latest_prompt_tokens

    @property
    def _cache_active(self) -> bool:
        return self.cache_read_tokens > 0 or self.cache_write_tokens > 0


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
        tip_id: str | None = None,
        title: str | None = None,
        thinking_level: str | None = None,
        cwd: str | None = None,
        tools_loader: Callable[[], list[ToolSpec]] | None = None,
        skills_loader: Callable[[], list[Any]] | None = None,
        templates_loader: Callable[[], list[Any]] | None = None,
    ) -> None:
        self._session_id = session_id
        self._vault = vault
        self._catalog = catalog
        self._pending_header: SessionMetaRecord | None = None
        self._provider = provider
        self._provider_name = provider_name
        self._model = model
        self._system = system
        self._title = title
        self._thinking_level = thinking_level
        self._cwd = cwd or os.getcwd()
        self._record_ids: list[str] = list(record_ids) if record_ids else []
        # The tip is the head of the active branch. It is not always the last
        # transcript record — metadata records (title, model, prune) also chain
        # onto it — so callers pass it explicitly when known.
        self._tip_id: str | None = tip_id or (
            self._record_ids[-1] if self._record_ids else None
        )
        self._extensions: list[object] = []
        self._named = title is not None
        self._named_at = 0
        self._renamed_manually = False
        self._tools_loader = tools_loader
        self._skills_loader = skills_loader
        self._skills: list[Any] = list(skills_loader() if skills_loader else [])
        self._templates_loader = templates_loader
        self._templates: list[Any] = list(
            templates_loader() if templates_loader else []
        )
        # Unrestricted tool set. Plan mode narrows what the harness sees, so
        # the full list is kept here to restore from and to hand to new
        # sessions.
        self._all_tools: list[ToolSpec] = list(tools)
        self._plan_mode = False
        # Backend pin for routed providers; see ``delta_app.routing``.
        self._route = RoutePin()

        # Cumulative stats
        self._turn_count = 0
        self._total_input = 0
        self._total_output = 0
        self._total_cost = 0.0
        self._cache_read = 0
        self._cache_write = 0
        self._latest_prompt = 0
        self._latest_cache_read = 0

        # Build harness
        config = RuntimeConfig(
            provider=provider,
            model=model,
            system=system,
            tools=list(tools),
            cache_key=session_id,
        )
        self._harness = RuntimeHarness(config, messages=transcript or [])

      
        self._unsubscribe = self._harness.on_event(self._on_harness_event)

        # A resumed session restores its thinking level from the record log;
        # apply it so the first request matches what the transcript claims.
        self._apply_thinking_level()

        
        if transcript:
            for entry in transcript:
                if isinstance(entry, ModelEntry):
                    self._tally(entry)

    

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
        skills_loader: Callable[[], list[Any]] | None = None,
        templates_loader: Callable[[], list[Any]] | None = None,
    ) -> CodingSession:
        """Create a new coding session with optional persistence."""
        sid = session_id or mint_id()[:12]
        vault: JsonlVault | None = None
        catalog: SessionCatalog | None = None
        root_id: str | None = None
        header: SessionMetaRecord | None = None
        effective_cwd = cwd or os.getcwd()
        if sessions_dir is not None:
            vault = JsonlVault(sessions_dir / f"{sid}.jsonl")
            # The metadata record is the tree root; everything else descends
            # from it so a parent-chain walk always reaches session cwd/title.
            # Nothing is written until the first record, so a session that is
            # opened and left never shows up as an empty resumable one.
            header = SessionMetaRecord(cwd=effective_cwd)
            root_id = header.id
            catalog = SessionCatalog(sessions_dir)
        session = cls(
            session_id=sid,
            vault=vault,
            catalog=catalog,
            provider=provider,
            provider_name=provider_name,
            model=model,
            system=system,
            tools=tools or [],
            tip_id=root_id,
            cwd=cwd,
            tools_loader=tools_loader,
            skills_loader=skills_loader,
            templates_loader=templates_loader,
        )
        if vault is not None:
            session._pending_header = header
        return session

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
        skills_loader: Callable[[], list[Any]] | None = None,
        templates_loader: Callable[[], list[Any]] | None = None,
        keep_model: bool = False,
    ) -> CodingSession:
        """Load an existing session from persistent storage.

        *keep_model* uses *model* even when the session recorded another,
        for a model the caller chose explicitly.
        """
        vault = JsonlVault(sessions_dir / f"{session_id}.jsonl")
        if not vault.path.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")

        records = await vault.read_all()
        # Reconstruct along the active-leaf path, not raw write order: once a
        # session has branched, the file also holds records from abandoned
        # branches. Walking parent pointers from the tip keeps those out, and
        # a damaged chain degrades to a flat replay rather than failing to load.
        state = project_active_branch(records)
        tip_id = find_tip(records)

        catalog = SessionCatalog(sessions_dir)
        catalog.touch(session_id)
        meta = catalog.get(session_id)
        # The recorded model belongs to the provider the session last used.
        # Under a different provider it would be rejected, so keep the
        # active provider's model; an explicit choice always wins.
        stored = meta.provider if meta is not None else None
        if not keep_model and (stored is None or stored == provider_name):
            # The starting model lives in the index; later switches in the log.
            model = state.model or (meta.model if meta is not None else None) or model

        session = cls(
            session_id=session_id,
            vault=vault,
            catalog=catalog,
            provider=provider,
            provider_name=provider_name,
            model=model,
            system=system,
            tools=tools or [],
            transcript=state.transcript,
            record_ids=state.record_ids,
            tip_id=tip_id,
            title=state.title,
            thinking_level=state.thinking_level,
            cwd=state.cwd,
            tools_loader=tools_loader,
            skills_loader=skills_loader,
            templates_loader=templates_loader,
        )
        if meta is not None:
            session._route = pin_from_storage(
                meta.inference_provider, meta.inference_provider_mode,
            )
            session._apply_route()
        await session._mend_stored_history()
        return session

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
        # `/skill:<name>` and `/<template-name>` are expanded into their full
        # text before reaching the model; the title keeps the short form.
        # Skills are checked first: their `skill:` prefix cannot collide with a
        # template name, so the cheaper, unambiguous match goes ahead.
        prompt = self.expand_skill(text) or self.expand_template(text) or text
        return self._run_wrapped(self._harness.submit(prompt))

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
        if self._tip_id and self._pending_header is None:
            await self._append_record(
                TipRecord(entry_id=self._tip_id), advance_tip=False
            )

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
        if self._route.mode == "automatic":
            # A learned backend was learned for the previous model.
            self._route = RoutePin()
            self._sync_route_to_catalog()
        self._apply_route()
        await self._append_record(ModelSwapRecord(model=model))

    async def switch_provider(
        self,
        provider: ModelProvider,
        provider_name: str,
        model: str | None = None,
    ) -> None:
        """Replace the active model provider (takes effect next turn)."""
        self._provider = provider
        self._provider_name = provider_name
        self._route = RoutePin()
        self._apply_route()
        self._sync_route_to_catalog()
        if self._catalog is not None:
            existing = self._catalog.get(self._session_id)
            if existing is not None:
                self._catalog.upsert(replace(
                    existing, provider=provider_name, updated_at=time(),
                ))
        # A fresh provider carries its own configured reasoning policy; the
        # session's chosen thinking level has to be re-applied on top.
        self._apply_thinking_level()
        self._harness.settings.provider = provider
        if model:
            await self.switch_model(model)

    # ── thinking mode ──────────────────────────────────────────────────

    async def set_thinking(self, level: str | None) -> None:
        """Set the thinking/reasoning depth level (or disable with ``None``)."""
        self._thinking_level = level
        self._apply_thinking_level()
        await self._append_record(ReasoningLevelRecord(thinking_level=level))

    def _apply_thinking_level(self) -> None:
        """Push the active thinking level onto the provider.

        Providers hold a frozen reasoning policy chosen when they were built,
        so without this the level would be recorded and replayed but never
        actually change a request. Providers that expose no ``set_reasoning``
        simply keep their configured policy.
        """
        setter = getattr(self._provider, "set_reasoning", None)
        if setter is None:
            return
        from delta_app.reasoning import (
            normalize_thinking_level,
            thinking_to_budget,
            thinking_to_effort,
        )
        from delta_model.settings import ReasoningPolicy

        if self._thinking_level is None:
            setter(ReasoningPolicy(enabled=False))
            return
        level = normalize_thinking_level(self._thinking_level)
        setter(ReasoningPolicy(
            enabled=True,
            budget_tokens=thinking_to_budget(level),
            effort=thinking_to_effort(level),
        ))

    @property
    def thinking_level(self) -> str | None:
        """The current thinking/reasoning depth, or ``None`` if disabled."""
        return self._thinking_level

    # ── plan mode ──────────────────────────────────────────────────────

    @property
    def plan_mode(self) -> bool:
        """Whether the session is restricted to research-and-propose."""
        return self._plan_mode

    def set_plan_mode(self, enabled: bool) -> None:
        """Enter or leave plan mode (takes effect on the next turn).

        Not persisted: plan mode is a stance for the current sitting, so a
        resumed session always starts able to make changes.
        """
        self._plan_mode = enabled
        self._apply_plan_mode()

    def _apply_plan_mode(self) -> None:
        """Push the tool set and system prompt matching the current mode."""
        from delta_app.planning import plan_system_prompt, restrict_tools

        if self._plan_mode:
            self._harness.settings.tools = restrict_tools(self._all_tools)
            self._harness.settings.system = plan_system_prompt(self._system)
        else:
            self._harness.settings.tools = list(self._all_tools)
            self._harness.settings.system = self._system

    # ── context compaction ─────────────────────────────────────────────

    async def compact(self) -> PruneSummaryEntry:
        """Compact the transcript by summarising older entries.

        Keeps the most recent entries intact and replaces everything before
        them with a single ``PruneSummaryEntry``.  The compaction is persisted
        as a ``PruneRecord`` so that resumption reconstructs the same state.

        Raises ``ValueError`` if the transcript is too short to compact.
        """
        transcript = list(self._harness.transcript)
        plan = plan_compaction(transcript, self._record_ids, limits=_CONTEXT_LIMITS)
        if plan is None:
            raise ValueError(
                f"Transcript has {len(transcript)} entries; "
                f"need at least {_CONTEXT_LIMITS.min_entries} to compact."
            )

        # A transcript that was compacted before carries its earlier summary
        # into this one, so repeated compactions merge rather than stack.
        system_prompt, user_prompt = build_summary_prompts(
            plan,
            system=_CONTEXT_LIMITS.summary_system,
            max_chars=_CONTEXT_LIMITS.max_summary_chars,
        )
        summary_text = await self._utility_completion(user_prompt, system_prompt)

        prune_record = PruneRecord(
            summary=summary_text,
            replaces_entry_ids=list(plan.summarize_ids),
        )
        await self._append_record(prune_record)

        result = apply_compaction(
            plan, summary_text, summary_record_id=prune_record.id
        )
        self._harness.set_messages(list(result.transcript))
        self._record_ids = list(result.record_ids)

        return result.summary_entry

    def should_compact(self) -> bool:
        """Return whether automatic compaction is recommended."""
        return exceeds_threshold(self.context_estimate())

    def context_estimate(self) -> ContextEstimate:
        """Full context accounting for the session as it stands.

        Counts the system prompt and tool schemas alongside the transcript, so
        the utilisation figure reflects everything actually sent to the model.
        """
        return estimate_context(
            self._harness.transcript,
            system=self._system,
            tools=self._harness.settings.tools,
            limits=_CONTEXT_LIMITS,
            model=self._model,
        )

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
        await self._append_record(TagRecord(label=title))
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
        self._catalog.upsert(replace(existing, title=title, updated_at=time()))

    # ── inference routing ──────────────────────────────────────────────

    @property
    def inference_route(self) -> RoutePin:
        """The backend pin in effect for a routed provider."""
        return self._route

    def set_inference_route(self, backend: str | None) -> None:
        """Choose a backend (fixed) or hand the choice back to the router."""
        self._route = RoutePin("fixed", backend) if backend else RoutePin()
        self._apply_route()
        self._sync_route_to_catalog()

    def _apply_route(self) -> None:
        """Send the routed model id when this provider routes, else the plain one."""
        routed = self._provider_name == ROUTED_PROVIDER
        self._harness.settings.model = (
            self._route.model_for(self._model) if routed else self._model
        )

    def _sync_route_to_catalog(self) -> None:
        if self._catalog is None:
            return
        existing = self._catalog.get(self._session_id)
        if existing is None:
            return
        self._catalog.upsert(replace(
            existing,
            inference_provider=self._route.backend,
            inference_provider_mode=self._route.mode,
            updated_at=time(),
        ))

    def _learn_route(self) -> None:
        """Pin the backend that just answered, for an automatic route."""
        if self._provider_name != ROUTED_PROVIDER or self._route.backend:
            return
        if self._route.mode != "automatic":
            return
        served = getattr(self._provider, "served_by", None)
        if served:
            self._route = RoutePin("automatic", served)
            self._apply_route()
            self._sync_route_to_catalog()

    def _should_reroute(self, reply: ModelEntry) -> bool:
        """Whether a failed reply should retry on the router's own choice.

        Only a learned pin is abandoned, only for a status that points at
        that backend being unhealthy, and only when the failure produced no
        output — replaying partial text or tool calls would duplicate them.
        """
        return (
            self._provider_name == ROUTED_PROVIDER
            and self._route.mode == "automatic"
            and self._route.backend is not None
            and reply.stop_reason == "error"
            and not reply.content
            and is_route_failure(getattr(self._provider, "last_failure_status", None))
        )

    # ── slash-command dispatch ─────────────────────────────────────────

    #: Registry commands this session offers. See ``_RUNNABLE_COMMANDS``.
    RUNNABLE_COMMANDS: frozenset[str] = _RUNNABLE_COMMANDS

    #: Slash commands, with one-line help. Drives autocomplete and /help.
    #: Derived from the shared registry so the CLI, the TUI, and dispatch all
    #: read one list — a command cannot be completable but unrunnable, or
    #: runnable but undiscoverable.
    COMMANDS: tuple[tuple[str, str], ...] = _RUNNABLE_COMMAND_TABLE

    async def handle_command(self, text: str) -> str | None:
        """Handle a ``/``-prefixed command.

        Returns a human-readable response string if the input was a recognised
        command, or ``None`` if it was not (so the caller can treat it as a
        normal prompt).
        """
        parsed = parse_command(text)
        if parsed is None:
            return None

        name, arg = parsed
        # Resolve through the registry so aliases (/m, /fork, /title, /sh …)
        # reach the same handler as their primary name.
        command = COMMAND_REGISTRY.get(name)
        if command is None:
            return None

        handler = self._command_handlers().get(command.name)
        if handler is None:
            # A registered command the session itself does not run (/quit,
            # /help, pickers). The frontend owns those; returning None lets it
            # decide rather than sending the text to the model as a prompt.
            return None

        result = handler(arg)
        if asyncio.iscoroutine(result):
            return await result  # type: ignore[misc]
        return result  # type: ignore[return-value]

    def _command_handlers(self) -> dict[str, Callable[[str], object]]:
        """Map registry command names to this session's bound handlers."""
        return {
            "model": self._cmd_model,
            "provider": self._cmd_provider,
            "think": self._cmd_think,
            "plan": self._cmd_plan,
            "compact": self._cmd_compact,
            "stats": self._cmd_stats,
            "name": self._cmd_name,
            "export": self._cmd_export,
            "branch": self._cmd_branch,
            "branches": self._cmd_branches,
            "rewind": self._cmd_rewind,
            "shell": self._cmd_shell,
            "skills": self._cmd_skills,
            "create-skill": self._cmd_create_skill,
            "reload": self._cmd_reload,
            "diag": self._cmd_diag,
            "system": self._cmd_system,
            "session": self._cmd_session,
            "help": self._cmd_help,
        }

    def _cmd_help(self, _arg: str) -> str:
        """List every command available here, with usage and aliases.

        Covers frontend-owned commands (``/quit``, ``/version``, ``/continue``)
        as well as the session's own, so this is the one help text a user sees.
        """
        lines = ["Commands:"]
        for command in COMMAND_REGISTRY.all_commands():
            if command.name not in _RUNNABLE_COMMANDS:
                continue
            aliases = (
                "  (" + " ".join(f"/{a}" for a in command.aliases) + ")"
                if command.aliases
                else ""
            )
            lines.append(f"  {command.display_usage:<28}{command.description}{aliases}")
        return "\n".join(lines)

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
                    stdin=subprocess.DEVNULL,
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

    async def tree_choices(self) -> tuple[SessionTreeChoice, ...]:
        """Return branchable session entries for a tree picker.

        Rows are depth-first over the record tree, indented where a branch
        diverged, so a caller can render a selectable list instead of asking
        the user for a raw record id.
        """
        if self._vault is None:
            return ()
        records = await self._vault.read_all()
        indents = _tree_branch_indents(records)
        return tuple(
            SessionTreeChoice(
                entry_id=record.id,
                label=_tree_choice_label(
                    record, branch_indent=indents.get(record.id, 0)
                ),
                active=record.id == self._tip_id,
                is_tool_call=_is_tool_call_record(record),
            )
            for record in _ordered_tree_records(records)
            if _is_branchable_record(record)
        )

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
        await self._append_record(record)
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
        # The active model stays: an older point may predate a model or
        # provider switch, and its model would not fit the current provider.
        if state.thinking_level is not None:
            self._thinking_level = state.thinking_level
        if state.title:
            self._title = state.title

        await self._append_record(
            TipRecord(entry_id=entry_id), advance_tip=False
        )
        await self._mend_stored_history()

    async def _mend_stored_history(self) -> HistoryMend | None:
        """Repair malformed tool-call pairing in the active branch, durably.

        The log is append-only, so nothing is rewritten: a diagnostic record
        and the repaired messages are appended as a new branch, and the tip
        moves there. Model, reasoning level, and title are snapshotted onto
        the branch so it projects to the same session state. Valid history is
        left alone, so repeated loads never stack up repair branches.
        """
        transcript = list(self._harness.transcript)
        mend = mend_tool_history(transcript)
        if not mend.changed:
            return None
        self._harness.set_messages(list(mend.entries))
        if self._vault is None:
            return mend

        records = await self._vault.read_all()
        parent_id, fresh = self._repair_branch_point(records, transcript, mend.entries)
        dropped = _records_after(records, self._tip_id, parent_id)

        staged: list[SessionRecord] = [
            ExtensionRecord(
                parent_id=parent_id,
                namespace=_HISTORY_REPAIR_NAMESPACE,
                data={"version": 1, **mend.counters()},
            )
        ]
        staged += [TranscriptRecord(message=entry) for entry in fresh]
        staged.append(ModelSwapRecord(model=self._model))
        staged.append(ReasoningLevelRecord(thinking_level=self._thinking_level))
        if self._title:
            staged.append(TagRecord(label=self._title))
        # Application records that sat on the abandoned stretch travel along.
        staged += [
            ExtensionRecord(namespace=r.namespace, data=r.data)
            for r in dropped
            if isinstance(r, ExtensionRecord) and r.namespace != _HISTORY_REPAIR_NAMESPACE
        ]
        for previous, record in zip(staged, staged[1:], strict=False):
            record.parent_id = previous.id
        for record in staged:
            await self._vault.append(record)
        self._tip_id = staged[-1].id
        await self._append_record(TipRecord(entry_id=self._tip_id), advance_tip=False)

        state = project_records(trace_to_entry([*records, *staged], self._tip_id))
        self._record_ids = state.record_ids
        return mend

    @staticmethod
    def _repair_branch_point(
        records: list[SessionRecord],
        original: list[TranscriptEntry],
        repaired: tuple[TranscriptEntry, ...],
    ) -> tuple[str | None, list[TranscriptEntry]]:
        """Where the repair branch attaches, and which messages it must carry.

        It attaches after the longest unchanged prefix when the log up to that
        record projects to exactly that prefix. Compaction can break that
        (messages it kept may precede the compaction record itself), so the
        fallback re-roots at the session header and carries the whole
        repaired history.
        """
        prefix = 0
        for before, after in zip(original, repaired, strict=False):
            if before is not after:
                break
            prefix += 1
        by_id = {record.id: record for record in records}
        state = project_active_branch(records)
        if prefix and len(state.record_ids) >= prefix:
            candidate = state.record_ids[prefix - 1]
            if candidate in by_id:
                try:
                    projected = project_records(trace_to_entry(records, candidate)).transcript
                except TreeIntegrityError:
                    projected = []
                if projected == original[:prefix]:
                    return candidate, list(repaired[prefix:])
        header = next((r for r in records if isinstance(r, SessionMetaRecord)), None)
        return (header.id if header else None), list(repaired)

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
            tools=list(self._all_tools),
            sessions_dir=base,
            tools_loader=self._tools_loader,
            skills_loader=self._skills_loader,
            templates_loader=self._templates_loader,
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
            tools=list(self._all_tools),
            sessions_dir=self._vault.path.parent,
            tools_loader=self._tools_loader,
            skills_loader=self._skills_loader,
            templates_loader=self._templates_loader,
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
        """Hot-reload tools, skills, extensions, and system prompt."""
        self._extensions = self.load_extensions()
        if self._tools_loader:
            self._all_tools = self._tools_loader()
        self.reload_skills()
        self._apply_plan_mode()

    # ── skills ─────────────────────────────────────────────────────────

    @property
    def skills(self) -> tuple[Any, ...]:
        """Skills currently available to this session."""
        return tuple(self._skills)

    def get_skill(self, name: str) -> Any | None:
        """Look up a loaded skill by name, case-insensitively."""
        wanted = name.strip().lstrip("/").removeprefix("skill:").lower()
        for skill in self._skills:
            if skill.name.lower() == wanted:
                return skill
        return None

    def reload_skills(self) -> int:
        """Re-read skills from disk, returning how many are now loaded."""
        if self._skills_loader is not None:
            self._skills = list(self._skills_loader())
        return len(self._skills)

    def expand_skill(self, text: str) -> str | None:
        """Expand ``/skill:<name> [args]`` into an injectable prompt block.

        Returns ``None`` when *text* is not a skill invocation.
        """
        from delta_app.skillset import expand_command

        index = {s.name: s for s in self._skills}
        try:
            return expand_command(text, index)
        except KeyError:
            return None

    # ── prompt templates ───────────────────────────────────────────────

    @property
    def prompt_templates(self) -> tuple[Any, ...]:
        """Markdown prompt templates available to this session."""
        return tuple(self._templates)

    def reload_prompt_templates(self) -> int:
        """Re-read prompt templates from disk, returning how many are loaded."""
        if self._templates_loader is not None:
            self._templates = list(self._templates_loader())
        return len(self._templates)

    def expand_template(self, text: str) -> str | None:
        """Expand ``/<template-name> [args]`` into its rendered body.

        Returns ``None`` when *text* names no loaded template, so a genuine
        slash command is never swallowed by a same-named template.
        """
        from delta_app.prompts import expand_slash_command

        index = {t.name: t for t in self._templates}
        if not index:
            return None
        try:
            return expand_slash_command(text, index)
        except KeyError:
            return None

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
            cache_read_tokens=self._cache_read,
            cache_write_tokens=self._cache_write,
            latest_prompt_tokens=self._latest_prompt,
            latest_cache_read_tokens=self._latest_cache_read,
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
        if isinstance(event.message, ModelEntry):
            self._tally(event.message)

    def _tally(self, reply: ModelEntry) -> None:
        """Fold one model reply's usage into the session totals."""
        usage = reply.usage
        prompt = usage.input + usage.cache_read + usage.cache_write
        self._turn_count += 1
        self._total_input += prompt
        self._total_output += usage.output
        self._total_cost += usage.cost.total
        self._cache_read += usage.cache_read
        self._cache_write += usage.cache_write
        self._latest_prompt = prompt
        self._latest_cache_read = usage.cache_read

    async def _run_wrapped(
        self, stream: AsyncIterator[AgentEvent]
    ) -> AsyncIterator[AgentEvent]:
        """Wrap a harness event stream with post-run processing.

        Two failures get one automatic continuation of the same run, with no
        extra user message, and the intermediate run end is swallowed so the
        caller sees a single run:

        - the request overflowed the model's context window: older history
          is compacted first (if that fails, the original error stands);
        - a routed provider's learned backend failed before answering: the
          pin is dropped so the router picks again.

        Each recovery is tried at most once per run.
        """
        completed = False
        used: set[str] = set()
        try:
            while True:
                recovery: str | None = None
                # Events for the failed reply a recovery replaces. A reply's
                # start is held until its outcome is known, so a recovered
                # failure never reaches the caller as an error.
                withheld: list[AgentEvent] = []
                async for event in stream:
                    if recovery is not None:
                        # The rest of a run being recovered is not shown.
                        withheld.append(event)
                        continue
                    if isinstance(event, MessageStartEvent) and isinstance(
                        event.message, ModelEntry
                    ):
                        withheld = [event]
                        continue
                    if isinstance(event, MessageEndEvent) and isinstance(
                        event.message, ModelEntry
                    ):
                        reply = event.message
                        if "overflow" not in used and _is_context_overflow(reply):
                            recovery = "overflow"
                        elif "reroute" not in used and self._should_reroute(reply):
                            recovery = "reroute"
                        elif reply.stop_reason not in ("error", "aborted"):
                            self._learn_route()
                        if recovery is not None:
                            withheld.append(event)
                            continue
                    for early in withheld:
                        yield early
                    withheld = []
                    if isinstance(event, RunEndEvent):
                        completed = True
                    yield event
                if recovery is None:
                    for early in withheld:
                        yield early
                    break
                used.add(recovery)
                if recovery == "overflow":
                    if not await self._compact_for_overflow():
                        # Recovery failed: the original error is the answer.
                        for early in withheld:
                            yield early
                        completed = True
                        break
                    notice = "Context window exceeded; compacted older history and retrying."
                else:
                    failed = self._route.backend
                    self._route = RoutePin()
                    self._apply_route()
                    self._sync_route_to_catalog()
                    notice = f"Inference provider {failed} failed; retrying on the router's choice."
                yield RetryEvent(attempt=2, max_attempts=2, message=notice)
                stream = self._harness.resume()
        finally:
            if completed:
                await self._post_run()

    async def _compact_for_overflow(self) -> bool:
        """Compact after a context-overflow error; ``False`` if that is impossible."""
        try:
            await self.compact()
        except Exception:  # noqa: BLE001 - the overflow error itself stays visible
            return False
        return True

    async def _post_run(self) -> None:
        """Housekeeping after a completed agent run."""
        if self._tip_id:
            await self._append_record(
                TipRecord(entry_id=self._tip_id), advance_tip=False
            )

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
            and len(self._harness.transcript) >= _CONTEXT_LIMITS.min_entries
        ):
            try:
                await self.compact()
            except Exception:  # noqa: BLE001
                pass

    # ── internal: persistence ──────────────────────────────────────────

    async def _append_record(
        self,
        record: SessionRecord,
        *,
        advance_tip: bool = True,
    ) -> str | None:
        """Chain *record* onto the active branch and persist it.

        Every durable record is linked to the current tip via ``parent_id`` so
        that resuming can walk the parent chain and replay only the active
        branch.  ``advance_tip=False`` is for pointer records (``TipRecord``)
        that mark the head without becoming it.
        """
        if self._vault is None:
            return None
        if self._pending_header is not None:
            if not await self._write_header():
                return None
        if record.parent_id is None:
            record.parent_id = self._tip_id
        try:
            await self._vault.append(record)
        except Exception:  # noqa: BLE001
            return None
        if advance_tip:
            self._tip_id = record.id
        return record.id

    async def _write_header(self) -> bool:
        """Write the deferred session header and index row, once."""
        header = self._pending_header
        assert header is not None and self._vault is not None
        try:
            await self._vault.append(header)
        except Exception:  # noqa: BLE001
            return False
        self._pending_header = None
        if self._catalog is not None:
            self._catalog.upsert(replace(
                SessionCatalog.prepare(
                    session_id=self._session_id,
                    vault_path=self._vault.path,
                    cwd=header.cwd or self._cwd,
                    model=self._model,
                    provider=self._provider_name,
                ),
                title=self._title,
                inference_provider=self._route.backend,
                inference_provider_mode=self._route.mode,
            ))
        return True

    async def _persist_entry(self, entry: TranscriptEntry) -> str | None:
        """Append a transcript record to the vault, returning its record ID."""
        return await self._append_record(TranscriptRecord(message=entry))

    # ── internal: token estimation ─────────────────────────────────────

    def _estimate_context_tokens(self) -> int:
        """Estimated tokens currently occupying the context window."""
        return self.context_estimate().used

    def _context_limit(self) -> int:
        """Return the context token limit for the current model."""
        return resolve_window(self._model)

    # ── internal: utility completions ──────────────────────────────────

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

    def _cmd_plan(self, arg: str) -> str:
        """Toggle plan mode, or set it explicitly with ``on``/``off``."""
        argument = arg.strip().lower()
        if not argument:
            enabled = not self._plan_mode
        elif argument in {"on", "start", "enable"}:
            enabled = True
        elif argument in {"off", "stop", "disable", "exit"}:
            enabled = False
        else:
            return f"Unknown argument: {arg!r}. Use /plan [on|off]."

        if enabled is self._plan_mode:
            return f"Plan mode already {'on' if enabled else 'off'}."

        self.set_plan_mode(enabled)
        if enabled:
            return (
                "Plan mode on — read-only tools, no edits. "
                "Use /plan off to resume editing."
            )
        return "Plan mode off — editing tools restored."

    async def _cmd_compact(self, _arg: str) -> str:
        try:
            entry = await self.compact()
            preview = entry.summary[:100]
            return f"Compacted. Summary: {preview}..."
        except ValueError as exc:
            return str(exc)

    async def _cmd_stats(self, _arg: str) -> str:
        s = self.usage
        report = (
            f"Turns: {s.turn_count}  Messages: {s.message_count}\n"
            f"Input tokens: {s.total_input_tokens:,}  "
            f"Output tokens: {s.total_output_tokens:,}\n"
            f"Cost: ${s.total_cost_usd:.4f}\n"
            f"Context: ~{s.estimated_context_tokens:,} / "
            f"{s.context_limit:,} tokens"
        )
        latest, overall = s.latest_cache_hit_rate, s.cache_hit_rate
        if latest is not None and overall is not None:
            report += (
                f"\nCache hit: latest {latest:.0%}  session {overall:.0%}  "
                f"(read {s.cache_read_tokens:,}, written {s.cache_write_tokens:,})"
            )
        return report

    async def _cmd_name(self, arg: str) -> str:
        if not arg:
            return f"Session: {self._title or self._session_id}"
        self._title = arg
        self._named = True
        # An explicit name is final — stop auto-renaming over it.
        self._renamed_manually = True
        await self._append_record(TagRecord(label=arg))
        self._sync_title_to_catalog(arg)
        return f"Session named: {arg}"

    async def _cmd_export(self, arg: str) -> str:
        fmt, destination = parse_export_arg(arg)
        if fmt not in _EXPORT_FORMATS:
            valid = ", ".join(sorted(_EXPORT_FORMATS))
            return f"Unknown format {fmt!r}. Choose one of: {valid}"

        content = await self.export(fmt)
        if destination is None:
            return content

        try:
            path = self.export_to_path(content, destination, fmt)
        except OSError as exc:
            return f"Could not write export: {exc}"
        return f"Exported {fmt} to {path}"

    def export_to_path(self, content: str, destination: str, fmt: str) -> Path:
        """Write *content* to *destination*, resolving directories and ``~``.

        A destination that is (or ends like) a directory receives a file named
        after the session; parent directories are created as needed.
        """
        target = Path(destination).expanduser()
        if not target.is_absolute():
            target = Path(self.cwd) / target

        looks_like_dir = target.is_dir() or destination.endswith(("/", "\\"))
        if looks_like_dir:
            target = target / f"{self._session_id}.{_EXPORT_SUFFIX[fmt]}"

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target

    async def _cmd_branch(self, arg: str) -> str:
        summary = arg or None
        branch_id = await self.branch(summary)
        return f"Branch created: {branch_id}"

    async def _cmd_rewind(self, arg: str) -> str:
        """Rewind by row number from ``/rewind``, or by raw entry id."""
        choices = await self.tree_choices()
        if not choices:
            return "Nothing to rewind to — this session has no stored history."

        if not arg:
            return self._render_tree(choices, header="Rewind to which entry?")

        target = arg.strip()
        if target.isdigit():
            index = int(target) - 1
            if not 0 <= index < len(choices):
                return f"No entry {target}. Run /rewind to list them."
            target = choices[index].entry_id

        try:
            await self.rewind(target)
        except Exception as exc:  # noqa: BLE001
            return f"Rewind failed: {exc}"
        return f"Rewound to: {target}"

    async def _cmd_branches(self, _arg: str) -> str:
        """List fork points recorded in this session."""
        choices = await self.tree_choices()
        forks = tuple(c for c in choices if c.label.lstrip().startswith("branch summary:"))
        if not forks:
            return "No branches yet. Use /branch to fork the conversation here."
        return self._render_tree(forks, header="Branches:")

    @staticmethod
    def _render_tree(
        choices: tuple[SessionTreeChoice, ...],
        *,
        header: str,
    ) -> str:
        """Render tree rows as a numbered, selectable list."""
        lines = [header]
        for number, choice in enumerate(choices, start=1):
            marker = "*" if choice.active else " "
            lines.append(f"{marker}{number:>3}. {choice.label}")
        lines.append("")
        lines.append("Pick with /rewind <number>.")
        return "\n".join(lines)

    async def _cmd_shell(self, arg: str) -> str:
        if not arg:
            return "Usage: /shell <command>"
        entry = await self.run_shell(arg)
        return f"Exit {entry.exit_code}:\n{entry.output}"

    async def _cmd_create_skill(self, arg: str) -> str:
        """Write everything after the name into a new project skill."""
        name, body = parse_create_skill_arg(arg)

        if not name:
            return (
                "Usage: /create-skill <name> <instructions>\n"
                "Everything after the name becomes the skill body "
                "(Ctrl+J for multi-line)."
            )
        if not _SKILL_NAME_RE.match(name):
            return (
                f"Invalid skill name {name!r}. Use letters, digits, "
                "dots, dashes or underscores — no path separators."
            )
        if not body:
            return f"Nothing to save. Add the instructions after '{name}'."

        try:
            path, existed = self.write_skill(name, body)
        except OSError as exc:
            return f"Could not write skill: {exc}"

        count = self.reload_skills()
        verb = "Updated" if existed else "Created"
        return (
            f"{verb} skill '{name}' at {path}\n"
            f"{count} skill(s) loaded. Invoke it with /skill:{name}"
        )

    def write_skill(self, name: str, body: str) -> tuple[Path, bool]:
        """Write ``<cwd>/.delta/skills/<name>/SKILL.md``.

        Returns the path and whether it already existed.
        """
        target = Path(self._cwd) / ".delta" / "skills" / name / "SKILL.md"
        existed = target.exists()
        target.parent.mkdir(parents=True, exist_ok=True)
        description = derive_skill_description(body)
        target.write_text(
            render_skill_file(name, description, body), encoding="utf-8",
        )
        return target, existed

    async def _cmd_reload(self, _arg: str) -> str:
        await self.reload()
        return "Reloaded."

    async def _cmd_skills(self, arg: str) -> str:
        """List loaded skills, show one, or reload them from disk."""
        argument = arg.strip()

        if argument in {"reload", "refresh"}:
            count = self.reload_skills()
            return f"Reloaded {count} skill(s)."

        if argument:
            skill = self.get_skill(argument)
            if skill is None:
                known = ", ".join(s.name for s in self._skills) or "none"
                return f"Unknown skill {argument!r}. Loaded: {known}"
            return (
                f"{skill.name} — {skill.description}\n"
                f"source: {skill.source}\n\n{skill.body}"
            )

        if not self._skills:
            return (
                "No skills loaded.\n"
                "Create .delta/skills/<name>/SKILL.md (or .agents/skills/<name>/, "
                "or the same under ~/), then run /skills reload."
            )
        lines = [f"{len(self._skills)} skill(s) loaded:"]
        for skill in self._skills:
            lines.append(f"  /skill:{skill.name}  —  {skill.description}")
        lines.append("")
        lines.append("Use /skills <name> to view one, /skills reload to re-read.")
        return "\n".join(lines)

    def system_sections(self) -> list[PromptSection]:
        """The live system prompt split into attributed sections.

        Sections are rebuilt from the same inputs the prompt was assembled
        from and trusted only if they join back to the exact live text; if
        anything drifted (an override, a changed date, an edited file) the
        whole prompt is shown as one unattributed section rather than with
        provenance that might be wrong. Display only: nothing here is sent to
        the model or persisted.
        """
        from delta_app.instructions import prompt_sections

        attributed: list[PromptSection] | None = None
        for cwd in dict.fromkeys((self._cwd, os.getcwd())):
            for dated in (True, False):
                candidate = prompt_sections(
                    tools=self._all_tools, skills=self._skills, cwd=cwd, include_date=dated,
                )
                if "\n\n".join(part.text for part in candidate) == self._system:
                    attributed = candidate
                    break
            if attributed is not None:
                break
        if attributed is None:
            attributed = [PromptSection("System prompt", "runtime-composed", self._system)]

        live = self._harness.settings.system
        if live != self._system and live.startswith(self._system + "\n\n"):
            attributed.append(PromptSection(
                "Plan mode", "plan mode (active)", live[len(self._system) + 2:],
            ))
        return attributed

    def _cmd_session(self, _arg: str) -> str:
        lines = [
            f"Session:  {self._session_id}",
            f"Title:    {self._title or '(untitled)'}",
            f"Model:    {self._model}",
            f"Provider: {self._provider_name}",
            f"CWD:      {self._cwd}",
        ]
        if self._provider_name == ROUTED_PROVIDER:
            lines.append(f"Hugging Face inference provider: {self._route.describe()}")
        return "\n".join(lines)

    def _cmd_system(self, _arg: str) -> str:
        parts = []
        for number, section in enumerate(self.system_sections(), start=1):
            parts.append(f"── {number}. {section.title}  [{section.origin}]\n{section.text}")
        return "\n\n".join(parts)

    async def _cmd_diag(self, _arg: str) -> str:
        return json.dumps(self.diagnostics(), indent=2)


__all__ = [
    "CodingSession",
    "SessionStats",
]
