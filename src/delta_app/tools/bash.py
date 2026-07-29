"""Bash tool: execute shell commands with timeout, cancellation, and output truncation.

The tool runs commands via ``asyncio.create_subprocess_shell``, captures
stdout+stderr, enforces a caller-supplied timeout, and truncates output
to a configurable line limit before returning a ``ToolOutcome``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

from delta_app.tools._shared import (
    CancelToken,
    JValue,
    Parallelism,
    ProgressNotifier,
    ToolOutcome,
    ToolSpec,
    fault,
    optional_int,
    require_str,
    text_outcome,
    truncate_output,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT_MS: int = 120_000
"""Default command timeout in milliseconds (2 minutes)."""

MAX_TIMEOUT_MS: int = 600_000
"""Hard ceiling for caller-requested timeouts (10 minutes)."""

# ---------------------------------------------------------------------------
# Parameter schema
# ---------------------------------------------------------------------------

_BASH_PARAMS: Mapping[str, JValue] = {
    "type": "object",
    "properties": {
        "command": {
            "type": "string",
            "description": "The shell command to execute.",
        },
        "timeout": {
            "type": "integer",
            "description": (
                "Optional timeout in milliseconds (default 120 000, max 600 000)."
            ),
        },
    },
    "required": ["command"],
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# Tool implementation
# ---------------------------------------------------------------------------


async def _run_bash(
    tool_call_id: str,
    arguments: Mapping[str, JValue],
    signal: CancelToken | None = None,
    on_update: ProgressNotifier | None = None,
) -> ToolOutcome:
    """Execute a shell command and return its combined output."""
    try:
        command = require_str(arguments, "command")
    except ValueError as exc:
        return fault(str(exc))

    timeout_ms = optional_int(arguments, "timeout") or DEFAULT_TIMEOUT_MS
    timeout_ms = max(1, min(timeout_ms, MAX_TIMEOUT_MS))
    timeout_s = timeout_ms / 1000.0

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        return fault(f"Failed to start command: {exc}")

    try:
        stdout_raw, stderr_raw = await asyncio.wait_for(
            proc.communicate(), timeout=timeout_s,
        )
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return fault(
            f"Command timed out after {timeout_ms}ms and was killed.\n"
            f"Command: {command}"
        )

    if signal is not None and signal.is_cancelled():
        return fault("Command cancelled.")

    stdout = stdout_raw.decode("utf-8", errors="replace")
    stderr = stderr_raw.decode("utf-8", errors="replace")

    output = stdout
    if stderr:
        output += ("\n" if output else "") + stderr

    output = truncate_output(output)

    exit_code = proc.returncode
    header = f"Exit code: {exit_code}\n" if exit_code else ""
    return text_outcome(header + output)


def _format_bash_call(arguments: Mapping[str, JValue]) -> str | None:
    return str(arguments.get("command", "?"))


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def make_bash_tool() -> ToolSpec:
    """Create the Bash shell tool."""
    return ToolSpec(
        name="Bash",
        label="Shell Command",
        description=(
            "Execute a shell command and return its output. "
            "Commands run with a default timeout of 2 minutes. "
            "Use the timeout parameter for long-running commands (max 10 minutes)."
        ),
        parameters=_BASH_PARAMS,
        run=_run_bash,
        parallelism="sequential",
        format_call=_format_bash_call,
    )


def bash_tools() -> list[ToolSpec]:
    """Build all bash-related tools."""
    return [make_bash_tool()]


__all__ = ["bash_tools", "make_bash_tool"]
