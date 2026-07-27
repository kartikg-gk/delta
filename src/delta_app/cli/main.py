"""CLI entry point for Delta: argument parsing, mode dispatch, and lifecycle management.

The CLI is a pure orchestrator — all business logic lives in ``delta_harness``,
``delta_model``, and the sibling ``delta_app`` modules.  This file owns only:

- argument parsing with implicit ``run`` subcommand
- provider / model resolution
- session creation, resumption, listing, and export
- extension discovery via file-based loading (``delta_app.extensions``)
- startup banner and notices
- signal handling for graceful shutdown
- dispatch to interactive or one-shot run mode
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import os
import signal
import sys
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn, TextIO

if TYPE_CHECKING:
    from delta_app.conversation import CodingSession

from delta_harness.contracts.stream import (
    AgentEvent,
    MessageEndEvent,
    MessageUpdateEvent,
    ToolRunEndEvent,
    ToolRunStartEvent,
)
from delta_harness.contracts.tooling import ToolSpec
from delta_harness.contracts.transcript import (
    ModelEntry,
    surface_text,
)
from delta_harness.harness import RuntimeHarness
from delta_harness.provider.base import ModelProvider
from delta_harness.session.index import SessionCatalog
from delta_harness.session.records import (
    SessionMetaRecord,
    TagRecord,
    TranscriptRecord,
)
from delta_harness.session.store import JsonlVault

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PACKAGE = "delta"
_DEFAULT_MODEL = "claude-sonnet-4-20250514"
_DEFAULT_SYSTEM = "You are a helpful coding assistant."
_SUBCOMMANDS = frozenset({"provider", "session"})


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------


def get_version() -> str:
    """Resolve the installed package version, falling back to ``dev``."""
    try:
        return importlib.metadata.version(_PACKAGE)
    except importlib.metadata.PackageNotFoundError:
        return "dev"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _info(msg: str) -> None:
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def _warn(msg: str) -> None:
    sys.stderr.write(f"warning: {msg}\n")
    sys.stderr.flush()


def _die(msg: str) -> NoReturn:
    sys.stderr.write(f"error: {msg}\n")
    sys.stderr.flush()
    sys.exit(1)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _sessions_dir(override: str | None = None) -> Path:
    """Resolve the session directory from flag, env, or default.

    Delegates to ``DeltaPaths`` for the canonical default location
    (``~/.delta/sessions``), with flag and env-var overrides.
    """
    if override:
        return Path(override)
    from delta_app.resources import default_paths

    return default_paths().sessions


def _session_path(base: Path, session_id: str) -> Path:
    return base / f"{session_id}.jsonl"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_run_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the default run mode."""
    p = argparse.ArgumentParser(
        prog="delta",
        description="Delta — a small, readable coding-agent harness.",
    )
    p.add_argument(
        "--version", "-V", action="version", version=f"delta {get_version()}",
    )
    p.add_argument(
        "prompt", nargs="?", default=None,
        help="One-shot prompt (omit for interactive mode).",
    )
    p.add_argument(
        "--print", "-p", dest="print_mode", action="store_true",
        help="Non-interactive: stream final text only, then exit.",
    )
    p.add_argument("--model", "-m", default=None, help="Override model identifier.")
    p.add_argument("--provider", default=None, help="Override provider backend.")
    p.add_argument(
        "--resume", "-r", metavar="ID", default=None,
        help="Resume a previous session by its ID.",
    )
    p.add_argument("--session-dir", default=None, help="Session storage directory.")
    p.add_argument("--system-prompt", default=None, help="Override system prompt.")
    p.add_argument(
        "--max-turns", type=int, default=None,
        help="Limit agent turns per submission.",
    )
    p.add_argument("--verbose", "-v", action="store_true", help="Verbose diagnostics.")
    p.add_argument(
        "--no-session", action="store_true", help="Disable session persistence.",
    )
    return p


def _build_provider_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="delta provider")
    sub = p.add_subparsers(dest="action", required=True)
    sub.add_parser("list", help="List configured providers.")
    sub.add_parser("setup", help="Interactive provider setup.")
    sel = sub.add_parser("select", help="Set the active provider.")
    sel.add_argument("name", help="Provider name to activate.")
    return p


def _build_session_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="delta session")
    p.add_argument("--session-dir", default=None)
    sub = p.add_subparsers(dest="action", required=True)
    ls = sub.add_parser("list", help="List saved sessions.")
    ls.add_argument("--session-dir", default=None, dest="sub_session_dir")
    exp = sub.add_parser("export", help="Export a session to stdout.")
    exp.add_argument("session_id", help="Session ID to export.")
    exp.add_argument(
        "--format", "-f", choices=["jsonl", "json", "text"], default="jsonl",
    )
    exp.add_argument("--session-dir", default=None, dest="sub_session_dir")
    return p


# ---------------------------------------------------------------------------
# Provider / model resolution
# ---------------------------------------------------------------------------


def _resolve_provider(name: str | None) -> ModelProvider:
    """Build a ``ModelProvider`` from the given name, env, or config module."""
    from delta_app.config.config import resolve_provider
    from delta_model.settings import ConfigError

    try:
        return resolve_provider(name)
    except ConfigError as exc:
        _die(str(exc))


def _resolve_provider_name(explicit: str | None) -> str:
    """Best-effort provider name for session metadata (mirrors config detection)."""
    if explicit:
        return explicit.strip().lower()
    env = os.environ.get("DELTA_PROVIDER")
    if env:
        return env.strip().lower()
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return "unknown"


def _resolve_model(override: str | None) -> str:
    return override or os.environ.get("DELTA_MODEL") or _DEFAULT_MODEL


def _resolve_system(
    override: str | None,
    *,
    tools: Sequence[ToolSpec] = (),
    skills: Sequence[object] = (),
) -> str:
    if override:
        return override
    try:
        from delta_app.instructions import system_prompt
        return system_prompt(tools=tools, skills=skills)  # type: ignore[arg-type]
    except (ImportError, AttributeError):
        return _DEFAULT_SYSTEM


# ---------------------------------------------------------------------------
# Tool loading
# ---------------------------------------------------------------------------


def _load_tools(verbose: bool = False) -> list[ToolSpec]:
    try:
        from delta_app.tools import build_tool_registry  # type: ignore[import-not-found]
        tools: list[ToolSpec] = build_tool_registry()
        if verbose:
            _info(f"Loaded {len(tools)} tool(s).")
        return tools
    except (ImportError, AttributeError):
        if verbose:
            _info("Tool module not available; running without tools.")
        return []


# ---------------------------------------------------------------------------
# Skill loading
# ---------------------------------------------------------------------------


def _load_skills(cwd: str, verbose: bool = False) -> list[object]:
    """Load skills from all resource directories with full precedence.

    Search order (highest-precedence first):
    ``<cwd>/.agents/skills``, ``<cwd>/.delta/skills``,
    ``~/.agents/skills``, ``~/.delta/skills``.
    """
    try:
        from delta_app.resources import default_paths, skill_search_paths
        from delta_app.skills import load_skills

        paths = default_paths(project=Path(cwd))
        dirs = skill_search_paths(paths)
        skills = load_skills(dirs)
        if verbose:
            searched = ", ".join(str(d) for d in dirs)
            _info(f"Loaded {len(skills)} skill(s) (searched: {searched}).")
        return list(skills)
    except (ImportError, AttributeError):
        if verbose:
            _info("Skill module not available; running without skills.")
        return []


# ---------------------------------------------------------------------------
# Hook wiring
# ---------------------------------------------------------------------------


def _install_hooks(harness: RuntimeHarness, verbose: bool = False) -> None:
    try:
        from delta_app.hooks.hooks import install_hooks  # type: ignore[import-not-found]
        install_hooks(harness)
        if verbose:
            _info("Hooks installed.")
    except (ImportError, AttributeError):
        pass


# ---------------------------------------------------------------------------
# Extension loading
# ---------------------------------------------------------------------------


def _load_extensions(verbose: bool = False) -> list[object]:
    """Discover and activate extensions from the extension directories.

    Uses file-based discovery (``delta_app.extensions``) rather than installed
    entry points, so files written after install — including ones the agent
    authors itself — are picked up on the next load or ``/reload``.
    """
    from delta_app.extensions import load_extensions

    result = load_extensions()

    for failure in result.failures:
        _warn(f"Extension {failure.name!r} failed: {failure.error}")

    if verbose:
        searched = ", ".join(str(path) for path in result.searched)
        _info(f"Extensions: {result.count} loaded (searched: {searched})")
        for extension in result.loaded:
            _info(f"  {extension.name}  <- {extension.path}")

    return [extension.value for extension in result.loaded]


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------


async def _list_sessions(base: Path, out: TextIO) -> None:
    if not base.exists():
        _info("No sessions found.")
        return

    # Prefer the catalog index when available
    catalog = SessionCatalog(base)
    entries = catalog.list_all()
    if entries:
        for meta in entries:
            label = meta.title or meta.session_id
            out.write(f"  {meta.session_id}  {label}  ({meta.provider}/{meta.model})\n")
        return

    # Fallback: scan vault files directly (pre-index sessions)
    files = sorted(base.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    # Exclude the index file itself
    files = [f for f in files if f.name != "index.jsonl"]
    if not files:
        _info("No sessions found.")
        return
    for path in files:
        vault = JsonlVault(path)
        try:
            records = await vault.read_all()
        except Exception:  # noqa: BLE001
            out.write(f"  {path.stem}  (unreadable)\n")
            continue
        title = path.stem
        n_messages = 0
        for r in records:
            if isinstance(r, TagRecord):
                title = r.label
            elif isinstance(r, SessionMetaRecord) and r.title:
                title = r.title
            elif isinstance(r, TranscriptRecord):
                n_messages += 1
        out.write(f"  {path.stem}  {title}  ({n_messages} messages)\n")


async def _export_session(base: Path, sid: str, fmt: str, out: TextIO) -> None:
    path = _session_path(base, sid)
    if not path.exists():
        _die(f"Session not found: {sid}")
    records = await JsonlVault(path).read_all()
    if fmt == "jsonl":
        from delta_harness.session.store import serialize_record
        for r in records:
            out.write(serialize_record(r))
    elif fmt == "json":
        from pydantic import TypeAdapter

        from delta_harness.session.records import SessionRecord
        out.write(
            TypeAdapter(list[SessionRecord]).dump_json(records, indent=2).decode() + "\n"
        )
    elif fmt == "text":
        for r in records:
            if isinstance(r, TranscriptRecord):
                out.write(f"[{r.message.role}] {surface_text(r.message)}\n\n")


# ---------------------------------------------------------------------------
# Provider subcommand
# ---------------------------------------------------------------------------


def _handle_provider(argv: list[str]) -> int:
    ns = _build_provider_parser().parse_args(argv)

    if ns.action == "list":
        from delta_app.config.config import list_providers
        list_providers()
        return 0

    if ns.action == "setup":
        from delta_app.config.config import setup_provider
        setup_provider()
        return 0

    if ns.action == "select":
        from delta_app.config.config import select_provider
        from delta_model.settings import ConfigError
        try:
            select_provider(ns.name)
            _info(f"Active provider: {ns.name}")
        except ConfigError as exc:
            _die(str(exc))
        return 0

    return 1


# ---------------------------------------------------------------------------
# Session subcommand
# ---------------------------------------------------------------------------


def _handle_session(argv: list[str]) -> int:
    ns = _build_session_parser().parse_args(argv)
    session_dir = ns.session_dir or getattr(ns, "sub_session_dir", None)
    base = _sessions_dir(session_dir)

    if ns.action == "list":
        asyncio.run(_list_sessions(base, sys.stdout))
        return 0

    if ns.action == "export":
        asyncio.run(_export_session(base, ns.session_id, ns.format, sys.stdout))
        return 0

    return 1


# ---------------------------------------------------------------------------
# Event rendering
# ---------------------------------------------------------------------------


def _extract_text_delta(event: MessageUpdateEvent) -> str | None:
    """Pull a text delta from a streaming update, if present."""
    wire = event.assistant_message_event
    if wire is None:
        return None
    # The wire payload may arrive as a dict (Pydantic coercion) or a model object
    if isinstance(wire, dict):
        if wire.get("type") == "text_delta":
            d = wire.get("delta")
            return str(d) if d else None
    elif hasattr(wire, "type") and getattr(wire, "type", None) == "text_delta":
        d = getattr(wire, "delta", None)
        return str(d) if d else None
    return None


class _FallbackRenderer:
    """Minimal stateful event renderer used when the full UI module is unavailable.

    Tracks whether streaming text deltas were emitted so it can fall back to
    printing the full model text on ``MessageEndEvent`` when no deltas arrived
    (e.g. with non-streaming providers or the test replay provider).
    """

    __slots__ = ("_saw_delta",)

    def __init__(self) -> None:
        self._saw_delta = False

    def __call__(self, event: AgentEvent, *, print_mode: bool) -> None:
        if isinstance(event, MessageUpdateEvent) and isinstance(event.message, ModelEntry):
            delta = _extract_text_delta(event)
            if delta:
                self._saw_delta = True
                sys.stdout.write(delta)
                sys.stdout.flush()
        elif isinstance(event, MessageEndEvent) and isinstance(event.message, ModelEntry):
            if not self._saw_delta:
                text = event.message.text
                if text:
                    sys.stdout.write(text)
                    sys.stdout.flush()
            self._saw_delta = False
            if not print_mode:
                sys.stdout.write("\n")
                sys.stdout.flush()
        elif not print_mode:
            if isinstance(event, ToolRunStartEvent):
                _info(f"  [{event.tool_name}] running...")
            elif isinstance(event, ToolRunEndEvent):
                _info(f"  [{event.tool_name}] {'error' if event.is_error else 'done'}")


async def _consume_events(
    events: AsyncIterator[AgentEvent],
    *,
    print_mode: bool,
) -> None:
    """Consume the event stream and render output.

    Persistence, stats, auto-naming, and compaction are handled inside
    ``CodingSession`` (via its harness event subscriber), so this only renders.
    """
    render = None
    try:
        from delta_app.ui.render import render_event  # type: ignore[import-not-found]
        render = render_event
    except (ImportError, AttributeError):
        pass
    fallback = _FallbackRenderer()

    async for event in events:
        if render is not None:
            render(event)
        else:
            fallback(event, print_mode=print_mode)


# ---------------------------------------------------------------------------
# Run modes
# ---------------------------------------------------------------------------


_HELP_TEXT = """Commands:
  /help              Show this help
  /quit /exit /q     End the session
  /version           Show version info
  /model [name]      Show or switch model
  /provider [name]   Show or switch provider
  /think [level]     Set thinking level (off to disable)
  /compact           Summarise older context
  /stats             Show token/turn/cost stats
  /name [title]      Show or set session title
  /export [fmt]      Export transcript (text/json/jsonl)
  /branch [summary]  Fork the conversation here
  /rewind <id>       Rewind to a prior entry
  /shell <cmd>       Run a shell command
  /reload            Reload tools/extensions
  /diag              Dump session diagnostics"""


async def _run_one_shot(
    session: CodingSession,
    prompt: str,
    *,
    print_mode: bool,
) -> int:
    """Submit a single prompt and exit."""
    events = session.submit(prompt)
    await _consume_events(events, print_mode=print_mode)
    if print_mode:
        sys.stdout.write("\n")
        sys.stdout.flush()
    return 0


async def _run_interactive(session: CodingSession) -> int:
    """Run the interactive prompt loop until the user exits."""
    loop = asyncio.get_running_loop()
    _info("Type /help for commands, /quit to exit.\n")

    while True:
        try:
            line = await loop.run_in_executor(None, _prompt_user)
        except (EOFError, KeyboardInterrupt):
            break

        if line is None:
            break
        text = line.strip()
        if not text:
            continue

        # CLI-level built-ins
        if text in {"/quit", "/exit", "/q"}:
            break
        if text == "/help":
            _info(_HELP_TEXT)
            continue
        if text == "/version":
            _info(f"Delta v{get_version()}")
            continue

        # Session-level slash commands (/model, /compact, /stats, /export, ...)
        try:
            response = await session.handle_command(text)
        except Exception as exc:  # noqa: BLE001 - a bad command must not kill the REPL
            _warn(f"Command failed: {exc}")
            continue
        if response is not None:
            _info(response)
            continue

        # Ordinary prompt
        try:
            events = session.submit(text)
        except RuntimeError as exc:
            _warn(str(exc))
            continue
        await _consume_events(events, print_mode=False)

    return 0


def _prompt_user() -> str | None:
    """Read a line of input from the user, returning ``None`` on EOF."""
    try:
        return input("\ndelta> ")
    except EOFError:
        return None


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------


def _wire_signals(session: CodingSession) -> None:
    """Install a SIGINT handler: first press aborts the active run, idle press
    raises ``KeyboardInterrupt`` for the normal exit path."""

    def _handler(_signum: int, _frame: object) -> None:
        if session.active:
            session.abort()
            return
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handler)


# ---------------------------------------------------------------------------
# Startup banner
# ---------------------------------------------------------------------------


def _banner(model: str, session_id: str | None, verbose: bool) -> None:
    parts = [f"Delta v{get_version()}  model={model}"]
    if session_id:
        parts.append(f"  session: {session_id}")
    if verbose:
        parts.append(f"  python:  {sys.version.split()[0]}")
        parts.append(f"  cwd:     {os.getcwd()}")
    _info("\n".join(parts))


# ---------------------------------------------------------------------------
# Async orchestrator
# ---------------------------------------------------------------------------


async def _async_main(ns: argparse.Namespace) -> int:
    """Wire everything together and dispatch to the chosen run mode."""
    from delta_app.conversation import CodingSession

    model = _resolve_model(ns.model)
    provider = _resolve_provider(ns.provider)
    provider_name = _resolve_provider_name(ns.provider)

    # Piped stdin -> one-shot print mode
    if ns.prompt is None and not sys.stdin.isatty():
        ns.prompt = sys.stdin.read().strip()
        if not ns.prompt:
            _die("Empty input from stdin.")
        ns.print_mode = True

    if ns.print_mode and ns.prompt is None:
        _die("Print mode requires a prompt (positional argument or stdin).")

    # --- extensions, tools, skills, system prompt ----------------------

    _load_extensions(verbose=ns.verbose)
    tools = _load_tools(verbose=ns.verbose)
    skills = _load_skills(os.getcwd(), verbose=ns.verbose)
    system = _resolve_system(ns.system_prompt, tools=tools, skills=skills)

    def _tools_loader() -> list[ToolSpec]:
        return _load_tools()

    # --- session (CodingSession owns persistence/stats/compaction) -----

    sessions_dir = None if ns.no_session else _sessions_dir(ns.session_dir)

    if ns.resume:
        if sessions_dir is None:
            _die("Cannot use --resume together with --no-session.")
        try:
            session = await CodingSession.resume(
                ns.resume,
                provider=provider,
                provider_name=provider_name,
                model=model,
                system=system,
                tools=tools,
                sessions_dir=sessions_dir,
                tools_loader=_tools_loader,
            )
        except FileNotFoundError:
            _die(f"Session not found: {ns.resume}")
        if ns.verbose:
            _info(f"Resumed {len(session.transcript)} messages from {ns.resume}")
    else:
        session = await CodingSession.create(
            provider=provider,
            provider_name=provider_name,
            model=model,
            system=system,
            tools=tools,
            sessions_dir=sessions_dir,
            tools_loader=_tools_loader,
        )

    if ns.max_turns is not None:
        session.harness.settings.max_turns = ns.max_turns

    _install_hooks(session.harness, verbose=ns.verbose)
    _wire_signals(session)

    if not ns.print_mode:
        _banner(model, session.session_id, ns.verbose)

    # --- dispatch ------------------------------------------------------

    try:
        if ns.prompt:
            return await _run_one_shot(session, ns.prompt, print_mode=ns.print_mode)
        return await _run_interactive(session)
    finally:
        await session.shutdown()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Top-level CLI entry point for Delta."""
    args = list(argv if argv is not None else sys.argv[1:])

    # Subcommand dispatch (checked before argparse to avoid positional conflicts)
    if args and args[0] in _SUBCOMMANDS:
        cmd, rest = args[0], args[1:]
        if cmd == "provider":
            return _handle_provider(rest)
        if cmd == "session":
            return _handle_session(rest)

    # Default: run mode
    ns = _build_run_parser().parse_args(args)
    try:
        return asyncio.run(_async_main(ns))
    except KeyboardInterrupt:
        _info("\nInterrupted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
