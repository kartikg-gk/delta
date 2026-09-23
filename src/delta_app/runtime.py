"""Shared session construction for every Delta frontend.

Both the line-based REPL and the Textual UI build their ``CodingSession``
here, so there is exactly one setup path and one conversation engine.  The
CLI helpers are imported lazily inside the function body because
``delta_app.cli.main`` imports this module.
"""

from __future__ import annotations

import argparse
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from delta_app.conversation import CodingSession
    from delta_app.trust import Asker, Choice, TrustOutcome, TrustRequest


async def build_session(ns: argparse.Namespace) -> CodingSession:
    """Resolve configuration and return a ready ``CodingSession``.

    Applies saved config, resolves provider/model, loads extensions, tools
    and skills, then creates or resumes the session and installs hooks.
    """
    from delta_app.cli.main import (
        _die,
        _info,
        _install_hooks,
        _install_safety,
        _load_extensions,
        _load_prompt_templates,
        _load_skills,
        _load_tools,
        _resolve_model,
        _resolve_provider,
        _resolve_provider_name,
        _resolve_system,
        _sessions_dir,
    )
    from delta_app.config.loader import apply_config, load_model
    from delta_app.conversation import CodingSession
    from delta_app.safety import ApprovalPolicy

    provider_name = _resolve_provider_name(getattr(ns, "provider", None))
    # Export saved credentials for this process so the provider loaders
    # (which read the environment) see them. Real env vars keep precedence.
    apply_config(provider_name)
    model = (
        getattr(ns, "model", None)
        or load_model(provider_name)
        or _resolve_model(None, provider_name)
    )
    provider = _resolve_provider(getattr(ns, "provider", None))

    verbose = getattr(ns, "verbose", False)
    await settle_project_trust(ns)
    _load_extensions(verbose=verbose)
    tools = _load_tools(verbose=verbose)
    skills = _load_skills(os.getcwd(), verbose=verbose)
    system = _resolve_system(getattr(ns, "system_prompt", None), tools=tools, skills=skills)

    def _tools_loader():
        return _load_tools()

    def _skills_loader():
        return _load_skills(os.getcwd())

    def _templates_loader():
        return _load_prompt_templates(os.getcwd())

    no_session = getattr(ns, "no_session", False)
    sessions_dir = None if no_session else _sessions_dir(getattr(ns, "session_dir", None))

    resume = getattr(ns, "resume", None)
    if resume:
        if sessions_dir is None:
            _die("Cannot use --resume together with --no-session.")
        try:
            session = await CodingSession.resume(
                resume,
                provider=provider,
                provider_name=provider_name,
                model=model,
                system=system,
                tools=tools,
                sessions_dir=sessions_dir,
                tools_loader=_tools_loader,
                skills_loader=_skills_loader,
                templates_loader=_templates_loader,
                keep_model=bool(getattr(ns, "model", None)),
            )
        except FileNotFoundError:
            _die(f"Session not found: {resume}")
        if verbose:
            _info(f"Resumed {len(session.transcript)} messages from {resume}")
    else:
        session = await CodingSession.create(
            provider=provider,
            provider_name=provider_name,
            model=model,
            system=system,
            tools=tools,
            sessions_dir=sessions_dir,
            tools_loader=_tools_loader,
            skills_loader=_skills_loader,
            templates_loader=_templates_loader,
        )

    max_turns = getattr(ns, "max_turns", None)
    if max_turns is not None:
        session.harness.settings.max_turns = max_turns

    _install_hooks(session.harness, verbose=verbose)
    _install_safety(session.harness, ApprovalPolicy.AUTO)
    attach_logger(session)
    return session


def _is_interactive(ns: argparse.Namespace) -> bool:
    """A human is at a terminal and this run is not one-shot or piped."""
    import sys

    if getattr(ns, "print_mode", False) or getattr(ns, "prompt", None) is not None:
        return False
    return sys.stdin.isatty() and sys.stdout.isatty()


async def settle_project_trust(
    ns: argparse.Namespace, *, ask: Asker | None = None,
) -> TrustOutcome:
    """Decide whether this folder's project inputs load, before any are read.

    ``ask`` lets a frontend supply its own question; without one, an
    interactive terminal is asked on the console and anything else declines.
    Cancelling the question ends startup.
    """
    from delta_app.cli.main import _die, _info, _warn
    from delta_app.config.store import load_config
    from delta_app.trust import StartupCancelled, resolve_project_trust

    override: bool | None = None
    if getattr(ns, "approve", False):
        override = True
    elif getattr(ns, "no_approve", False):
        override = False
    if ask is None and _is_interactive(ns):
        ask = ask_on_console
    default = load_config().project_trust or "ask"

    try:
        outcome = await resolve_project_trust(
            os.getcwd(), override=override, default=default, ask=ask,  # type: ignore[arg-type]
        )
    except StartupCancelled:
        _die("No project trust decision made; exiting.")
    if outcome.notice:
        _warn(outcome.notice)
    if not outcome.trusted and outcome.inputs is not None and not outcome.inputs.empty:
        _info(
            f"Project inputs not loaded ({outcome.inputs.describe()}): {outcome.reason}. "
            "Run with --approve to load them this once."
        )
    return outcome


def trust_choices(request: TrustRequest) -> list[tuple[Choice, str]]:
    from delta_app.trust import Choice

    menu = [(Choice.TRUST_FOLDER, "Trust this folder")]
    if request.parent is not None:
        menu.append((Choice.TRUST_PARENT, f"Trust parent folder ({request.parent})"))
    menu += [
        (Choice.TRUST_ONCE, "Trust for this run only"),
        (Choice.DISTRUST_FOLDER, "Do not trust this folder"),
        (Choice.DISTRUST_ONCE, "Do not trust for this run only"),
    ]
    return menu


def trust_question(request: TrustRequest) -> str:
    """The explanation shown above the choices, shared by every frontend."""
    lines = [
        f"{request.folder} contains project inputs: {request.inputs.describe()}.",
        "They shape the agent's instructions and can add plugins that run as code.",
        "This controls project inputs; it is not a sandbox.",
    ]
    if request.store_problem:
        lines.append(f"Saved decisions are unavailable: {request.store_problem}")
    return "\n".join(lines)


async def ask_on_console(request: TrustRequest) -> Choice | None:
    """Ask on the terminal; EOF or Ctrl+C cancels."""
    import asyncio
    import sys

    menu = trust_choices(request)
    sys.stderr.write(trust_question(request) + "\n")
    for number, (_choice, label) in enumerate(menu, start=1):
        sys.stderr.write(f"  {number}. {label}\n")
    loop = asyncio.get_running_loop()
    while True:
        try:
            answer = await loop.run_in_executor(None, input, f"Choose 1-{len(menu)}: ")
        except (EOFError, KeyboardInterrupt):
            return None
        answer = answer.strip()
        if answer.isdigit() and 1 <= int(answer) <= len(menu):
            return menu[int(answer) - 1][0]


def attach_logger(session: CodingSession) -> None:
    """Subscribe a ``RunLogger`` to the session's event stream.

    Wired here so every frontend — TUI, REPL, one-shot — logs identically.
    """
    from delta_app.logs import RunLogger

    logger = RunLogger.create(
        session.session_id,
        provider=session.provider_name,
        model=session.model,
        cwd=session.cwd,
    )
    if logger is None:
        return
    logger.stats_source = lambda: session.usage
    session.harness.on_event(logger.record_event)


def git_branch(cwd: str | None = None) -> str | None:
    """Current git branch name, or ``None`` outside a repository."""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=cwd or os.getcwd(),
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    branch = out.stdout.strip()
    return branch or None


__all__ = [
    "ask_on_console",
    "build_session",
    "git_branch",
    "settle_project_trust",
    "trust_choices",
    "trust_question",
]
