"""Coding tools (files, bash) + tool registry/assembly."""

from __future__ import annotations

from delta_app.tools._shared import (
    CallBlock,
    CancelToken,
    InputShaper,
    InvocationPrinter,
    JValue,
    OutcomePresenter,
    Parallelism,
    ProgressNotifier,
    RunHandler,
    ToolOutcome,
    ToolSpec,
)
from delta_app.tools.files import file_tools


def build_tool_registry() -> list[ToolSpec]:
    """Assemble all available coding tools.

    Called by ``cli/main.py`` at startup.  As new tool modules are added
    (bash, subagent, etc.), their factory functions get called here.
    """
    tools: list[ToolSpec] = []
    tools.extend(file_tools())
    # Future: tools.extend(bash_tools())  — when bash.py is implemented
    return tools


__all__ = [
    "CallBlock",
    "CancelToken",
    "InputShaper",
    "InvocationPrinter",
    "JValue",
    "OutcomePresenter",
    "Parallelism",
    "ProgressNotifier",
    "RunHandler",
    "ToolOutcome",
    "ToolSpec",
    "build_tool_registry",
    "file_tools",
]
