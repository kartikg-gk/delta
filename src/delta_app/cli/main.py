"""CLI entry point for Delta: argument parsing, mode dispatch, and lifecycle management.

The CLI is a pure orchestrator — all business logic lives in ``delta_harness``,
``delta_model``, and the sibling ``delta_app`` modules.  This file owns only:

- argument parsing with implicit ``run`` subcommand
- provider / model resolution
- session creation, resumption, listing, and export
- extension discovery via file-based loading (``delta_app.plugins``)
- startup banner and notices
- signal handling for graceful shutdown
- dispatch to interactive or one-shot run mode
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import importlib.util
import os
import signal
import sys
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn, TextIO

if TYPE_CHECKING:
    from delta_app.conversation import CodingSession

from delta_app.safety import (
    ApprovalContext,
    ApprovalDecision,
    ApprovalManager,
    ApprovalPolicy,
    clean_tool_result,
    mark_untrusted,
)
from delta_app.ui.render import OutputMode, make_renderer
from delta_harness.contracts.stream import AgentEvent
from delta_harness.contracts.tooling import ToolOutcome, ToolSpec
from delta_harness.contracts.transcript import CallBlock, surface_text
from delta_harness.driver import RuntimeHarness
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

_PACKAGE = "deltaa"
_DEFAULT_ANTHROPIC_MODEL = "claude-opus-5"
_DEFAULT_OPENAI_MODEL = "gpt-5"
_DEFAULT_SYSTEM = "You are a helpful coding assistant."
_SUBCOMMANDS = frozenset({"config", "provider", "session", "tui"})


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
    from delta_app.discovery import default_paths

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
        description="Delta - a small, readable coding-agent harness.",
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
    p.add_argument(
        "--repl", action="store_true",
        help="Use the line-based REPL instead of the Textual UI.",
    )
    trust = p.add_mutually_exclusive_group()
    trust.add_argument(
        "--approve", "-a", action="store_true",
        help="Load this folder's project instructions, skills and plugins for this run.",
    )
    trust.add_argument(
        "--no-approve", "-na", dest="no_approve", action="store_true",
        help="Skip this folder's project inputs for this run.",
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
    st = sub.add_parser("stats", help="Print a session's token and cost totals as JSON.")
    st.add_argument(
        "session_id", nargs="?", default=None,
        help="Session ID to summarize (default: the most recent session).",
    )
    st.add_argument("--session-dir", default=None, dest="sub_session_dir")
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
    """Provider name from the flag, then ``DELTA_PROVIDER``, then saved config."""
    if explicit:
        return explicit.strip().lower()
    from delta_app.config.loader import load_provider
    return load_provider()


def _resolve_model(override: str | None, provider_name: str = "anthropic") -> str:
    """Model from the ``--model`` flag, ``DELTA_MODEL``, or the provider's default."""
    explicit = override or os.environ.get("DELTA_MODEL")
    if explicit:
        return explicit
    if provider_name == "openai":
        return _DEFAULT_OPENAI_MODEL
    return _DEFAULT_ANTHROPIC_MODEL


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
        from delta_app.discovery import default_paths, skill_search_paths
        from delta_app.skillset import load_skills

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


def _load_prompt_templates(cwd: str, verbose: bool = False) -> list[object]:
    """Load markdown prompt templates from all resource directories.

    Same four-level precedence as skills: ``<cwd>/.agents/prompts``,
    ``<cwd>/.delta/prompts``, ``~/.agents/prompts``, ``~/.delta/prompts``.
    """
    try:
        from delta_app.discovery import default_paths, prompt_search_paths
        from delta_app.prompts import load_prompt_templates

        paths = default_paths(project=Path(cwd))
        dirs = prompt_search_paths(paths)
        templates = load_prompt_templates(dirs)
        if verbose:
            searched = ", ".join(str(d) for d in dirs)
            _info(f"Loaded {len(templates)} prompt template(s) (searched: {searched}).")
        return list(templates)
    except (ImportError, AttributeError):
        if verbose:
            _info("Prompt template module not available; running without templates.")
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


_MUTATING_TOOLS = frozenset({"Write", "Edit", "Bash"})


def _approval_context(call: CallBlock) -> ApprovalContext:
    """Describe a pending tool call for the approval manager."""
    args = call.arguments
    command = args.get("command")
    path = args.get("file_path")
    return ApprovalContext(
        tool_name=call.name,
        is_mutating=call.name in _MUTATING_TOOLS,
        affected_paths=(str(path),) if isinstance(path, str) else (),
        command=command if isinstance(command, str) else None,
    )


def _install_safety(harness: RuntimeHarness, policy: ApprovalPolicy) -> None:
    """Gate tool calls through the approval manager and sanitize their results."""
    manager = ApprovalManager(policy)

    async def before_tool_call(call: CallBlock) -> tuple[bool, str | None]:
        decision = await manager.check(_approval_context(call))
        if decision is ApprovalDecision.APPROVED:
            return False, None
        return True, f"Blocked by safety policy ({decision.value}): {call.name}"

    async def after_tool_call(
        call: CallBlock, result: ToolOutcome, is_error: bool,
    ) -> tuple[ToolOutcome, bool]:
        return mark_untrusted(clean_tool_result(result)), is_error

    harness.settings.before_tool_call = before_tool_call
    harness.settings.after_tool_call = after_tool_call


# ---------------------------------------------------------------------------
# Extension loading
# ---------------------------------------------------------------------------


def _load_extensions(verbose: bool = False) -> list[object]:
    """Discover and activate extensions from the extension directories.

    Uses file-based discovery (``delta_app.plugins``) rather than installed
    entry points, so files written after install — including ones the agent
    authors itself — are picked up on the next load or ``/reload``.
    """
    from delta_app.plugins import load_extensions

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


def _latest_session_id(base: Path) -> str | None:
    """Session id of the most recently modified vault file, if any."""
    if not base.exists():
        return None
    files = [p for p in base.glob("*.jsonl") if p.name != "index.jsonl"]
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime).stem


async def _session_stats(base: Path, sid: str | None, out: TextIO) -> None:
    """Write one JSON object of token and cost totals for a session."""
    import json
    from dataclasses import asdict

    from delta_harness.session.summary import summarize_records

    resolved = sid or _latest_session_id(base)
    if resolved is None:
        _die("No sessions found.")
    path = _session_path(base, resolved)
    if not path.exists():
        _die(f"Session not found: {resolved}")
    records = await JsonlVault(path).read_all()
    out.write(json.dumps(asdict(summarize_records(resolved, records))) + "\n")


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


def _textual_available() -> bool:
    """Whether the optional ``textual`` extra is importable."""
    return importlib.util.find_spec("textual") is not None


def _handle_tui(ns: argparse.Namespace | None = None) -> int:
    """Launch the Textual UI over the shared runtime session."""
    if not _textual_available():
        _die(
            "The TUI requires the 'textual' extra.\n"
            "  Install it with: pip install delta[tui]"
        )
    from delta_app.tui.app import run_tui
    return run_tui(ns)


def _should_launch_tui(ns: argparse.Namespace) -> bool:
    """Whether a bare ``delta`` invocation should open the Textual UI.

    Only plain interactive startup qualifies: a one-shot prompt, print mode,
    a resumed session, ``--repl``, or a non-terminal stdio all stay on the
    line-based REPL.
    """
    if ns.repl or ns.print_mode or ns.prompt is not None or ns.resume:
        return False
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return False
    return _textual_available()


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

    if ns.action == "stats":
        asyncio.run(_session_stats(base, ns.session_id, sys.stdout))
        return 0

    return 1


# ---------------------------------------------------------------------------
# Event rendering
# ---------------------------------------------------------------------------


async def _consume_events(
    events: AsyncIterator[AgentEvent],
    *,
    print_mode: bool,
) -> bool:
    """Consume the event stream and render output, returning overall success.

    Persistence, stats, auto-naming, and compaction are handled inside
    ``CodingSession`` (via its harness event subscriber), so this only renders.
    """
    mode = OutputMode.TEXT if print_mode else OutputMode.TRANSCRIPT
    renderer = make_renderer(mode)
    async for event in events:
        renderer.render(event)
    return renderer.finish()


# ---------------------------------------------------------------------------
# Run modes
# ---------------------------------------------------------------------------


#: ``/help`` is rendered from the shared command registry by the session, so
#: this file keeps no command list of its own to drift out of date. Only the
#: skill-invocation form, which is not a registered command, is appended.
_HELP_SUFFIX = "  /skill:<name> [args]      Invoke a skill"


async def _run_one_shot(
    session: CodingSession,
    prompt: str,
    *,
    print_mode: bool,
) -> int:
    """Submit a single prompt and exit."""
    events = session.submit(prompt)
    ok = await _consume_events(events, print_mode=print_mode)
    sys.stdout.flush()
    return 0 if ok else 1


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
        if text == "/version":
            _info(f"Delta v{get_version()}")
            continue

        # Resuming yields an event stream rather than a string, so it cannot go
        # through handle_command like the other session commands.
        if text in {"/continue", "/resume"}:
            if session.active:
                _warn("Agent is already running.")
                continue
            try:
                events = session.resume_run()
            except RuntimeError as exc:
                _warn(str(exc))
                continue
            await _consume_events(events, print_mode=False)
            continue

        # Session-level slash commands (/model, /compact, /stats, /export, ...)
        try:
            response = await session.handle_command(text)
        except Exception as exc:  # noqa: BLE001 - a bad command must not kill the REPL
            _warn(f"Command failed: {exc}")
            continue
        if response is not None:
            if text == "/help":
                response = f"{response}\n{_HELP_SUFFIX}"
            _info(response)
            continue

        # A registered command the session does not run (terminal-UI pickers and
        # the like) must not reach the model as a prompt.
        from delta_app.conversation import COMMAND_REGISTRY
        from delta_app.directives import parse_command

        parsed = parse_command(text)
        command = COMMAND_REGISTRY.get(parsed[0]) if parsed is not None else None
        if command is not None:
            _warn(f"/{command.name.lstrip('/')} is not available in the line REPL.")
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


def _banner(provider: str, model: str, session_id: str | None, verbose: bool) -> None:
    from delta_app.config.store import PROVIDERS

    label = next((p.label for p in PROVIDERS if p.key == provider), provider)
    parts = [
        f"Delta v{get_version()}",
        f"Provider: {label}",
        f"Model: {model}",
    ]
    if session_id:
        parts.append(f"Session: {session_id}")
    if verbose:
        parts.append(f"Python: {sys.version.split()[0]}")
        parts.append(f"Cwd: {os.getcwd()}")
    _info("\n".join(parts))


# ---------------------------------------------------------------------------
# Async orchestrator
# ---------------------------------------------------------------------------


async def _async_main(ns: argparse.Namespace) -> int:
    """Wire everything together and dispatch to the chosen run mode."""
    from delta_app.runtime import build_session

    # Piped stdin -> one-shot print mode
    if ns.prompt is None and not sys.stdin.isatty():
        ns.prompt = sys.stdin.read().strip()
        if not ns.prompt:
            _die("Empty input from stdin.")
        ns.print_mode = True

    if ns.print_mode and ns.prompt is None:
        _die("Print mode requires a prompt (positional argument or stdin).")

    # Shared construction — identical to the path the TUI uses.
    session = await build_session(ns)
    _wire_signals(session)

    if not ns.print_mode:
        _banner(session.provider_name, session.model, session.session_id, ns.verbose)

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
    # Model output is Unicode; a redirected Windows stream defaults to a legacy codepage.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = list(argv if argv is not None else sys.argv[1:])

    # Subcommand dispatch (checked before argparse to avoid positional conflicts)
    if args and args[0] in _SUBCOMMANDS:
        cmd, rest = args[0], args[1:]
        if cmd == "config":
            from delta_app.cli.config_cmd import handle_config
            return handle_config(rest)
        if cmd == "tui":
            return _handle_tui(_build_run_parser().parse_args(rest))
        if cmd == "provider":
            return _handle_provider(rest)
        if cmd == "session":
            return _handle_session(rest)

    # Default: run mode
    ns = _build_run_parser().parse_args(args)

    # A bare interactive `delta` opens the Textual UI. It needs no provider or
    # session, so this runs before any credential resolution.
    if _should_launch_tui(ns):
        return _handle_tui(ns)

    try:
        return asyncio.run(_async_main(ns))
    except KeyboardInterrupt:
        _info("\nInterrupted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
