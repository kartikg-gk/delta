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
    _load_extensions(verbose=verbose)
    tools = _load_tools(verbose=verbose)
    skills = _load_skills(os.getcwd(), verbose=verbose)
    system = _resolve_system(getattr(ns, "system_prompt", None), tools=tools, skills=skills)

    def _tools_loader():
        return _load_tools()

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
        )

    max_turns = getattr(ns, "max_turns", None)
    if max_turns is not None:
        session.harness.settings.max_turns = max_turns

    _install_hooks(session.harness, verbose=verbose)
    _install_safety(session.harness, ApprovalPolicy.AUTO)
    return session


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


__all__ = ["build_session", "git_branch"]
