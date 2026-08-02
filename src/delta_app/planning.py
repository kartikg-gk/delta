"""Plan mode: a read-only stance for the agent.

While plan mode is active the session hands the model a restricted tool set
and an extra system-prompt block telling it to research and propose rather
than act.  Two things change:

* Mutating tools (``Write``, ``Edit``) are removed from the tool list, so the
  model cannot call them at all.
* ``Bash`` is wrapped so only commands ``safety.is_safe_command`` recognises
  as read-only are executed; anything else is refused without running.

Nothing here is stateful — ``restrict_tools`` returns a new list and the
caller decides when to apply it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from delta_app.safety import is_safe_command
from delta_harness.contracts.tooling import (
    CancelToken,
    ProgressNotifier,
    ToolOutcome,
    ToolSpec,
)
from delta_harness.contracts.values import JValue

#: Tools removed entirely while plan mode is active.
MUTATING_TOOLS = frozenset({"Write", "Edit"})

#: Tool whose commands are filtered rather than removed.
_SHELL_TOOL = "Bash"

_REFUSAL = (
    "Plan mode is active: {tool} may only run read-only commands. "
    "Refused: {command!r}. Present your plan and ask the user to leave "
    "plan mode before running this."
)

PLAN_MODE_INSTRUCTIONS = """## Plan mode

Plan mode is active. Research the codebase and produce a plan — do not \
change anything.

- Read files, inspect the tree, and run read-only shell commands freely.
- Do not create, modify, or delete files, and do not run commands with side \
effects. The tools to do so are unavailable or will refuse.
- Finish by presenting a concrete, ordered implementation plan: the files to \
touch, the change in each, and how the result gets verified.
- Ask the user to leave plan mode (`/plan off`) before implementing anything."""


def _guard_shell(tool: ToolSpec) -> ToolSpec:
    """Wrap a shell tool so only read-only commands reach the real handler."""
    inner = tool.run

    async def run(
        tool_call_id: str,
        arguments: Mapping[str, JValue],
        signal: CancelToken | None = None,
        on_update: ProgressNotifier | None = None,
    ) -> ToolOutcome:
        command = arguments.get("command")
        if not isinstance(command, str) or not is_safe_command(command):
            return ToolOutcome(
                content=_REFUSAL.format(tool=tool.name, command=command),
                details={"error": "plan_mode"},
            )
        return await inner(tool_call_id, arguments, signal, on_update)

    return replace(tool, run=run)


def restrict_tools(tools: Sequence[ToolSpec]) -> list[ToolSpec]:
    """Return *tools* with mutating entries dropped and the shell guarded."""
    restricted: list[ToolSpec] = []
    for tool in tools:
        if tool.name in MUTATING_TOOLS:
            continue
        restricted.append(_guard_shell(tool) if tool.name == _SHELL_TOOL else tool)
    return restricted


def plan_system_prompt(system: str) -> str:
    """Append the plan-mode instruction block to a base system prompt."""
    return f"{system}\n\n{PLAN_MODE_INSTRUCTIONS}"


__all__ = [
    "MUTATING_TOOLS",
    "PLAN_MODE_INSTRUCTIONS",
    "plan_system_prompt",
    "restrict_tools",
]
