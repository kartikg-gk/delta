"""Shared tool helpers: output truncation, per-path async locks, arg parsing, error builders."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from delta_harness.contracts.tooling import (
    CancelToken,
    InputShaper,
    InvocationPrinter,
    OutcomePresenter,
    Parallelism,
    ProgressNotifier,
    RunHandler,
    ToolOutcome,
    ToolSpec,
)
from delta_harness.contracts.transcript import CallBlock, TextSegment
from delta_harness.contracts.values import JValue

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_OUTPUT_LINES: int = 2000
TRUNCATION_NOTICE = "\n... (output truncated — {shown}/{total} lines shown)"

IMAGE_EXTENSIONS: frozenset[str] = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".svg", ".ico", ".tiff",
})

BINARY_EXTENSIONS: frozenset[str] = frozenset({
    ".pdf", ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
    ".exe", ".dll", ".so", ".dylib",
    ".pyc", ".pyo", ".class",
    ".whl", ".egg",
    ".sqlite", ".db",
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
})

# ---------------------------------------------------------------------------
# Per-path async locks
# ---------------------------------------------------------------------------

_path_locks: dict[Path, asyncio.Lock] = {}


def path_lock(path: Path) -> asyncio.Lock:
    """Return a per-path async lock (created on first access)."""
    resolved = path.resolve()
    if resolved not in _path_locks:
        _path_locks[resolved] = asyncio.Lock()
    return _path_locks[resolved]


# ---------------------------------------------------------------------------
# Output truncation
# ---------------------------------------------------------------------------


def truncate_output(text: str, *, max_lines: int = MAX_OUTPUT_LINES) -> str:
    """Truncate text to *max_lines*, appending a notice if clipped."""
    lines = text.split("\n")
    if len(lines) <= max_lines:
        return text
    kept = "\n".join(lines[:max_lines])
    return kept + TRUNCATION_NOTICE.format(shown=max_lines, total=len(lines))


# ---------------------------------------------------------------------------
# Line-numbered formatting (cat -n style)
# ---------------------------------------------------------------------------


def format_with_line_numbers(
    text: str, *, start: int = 1
) -> str:
    """Format text with right-aligned line numbers and a tab separator."""
    lines = text.split("\n")
    width = len(str(start + len(lines) - 1))
    numbered = []
    for i, line in enumerate(lines):
        num = start + i
        numbered.append(f"{num:>{width}}\t{line}")
    return "\n".join(numbered)


# ---------------------------------------------------------------------------
# Argument extraction helpers
# ---------------------------------------------------------------------------


def require_str(args: Mapping[str, JValue], key: str) -> str:
    """Extract a required string argument, raising on missing/wrong type."""
    val = args.get(key)
    if not isinstance(val, str):
        raise ValueError(f"Missing or invalid required argument: {key}")
    return val


def optional_str(args: Mapping[str, JValue], key: str) -> str | None:
    """Extract an optional string argument."""
    val = args.get(key)
    return val if isinstance(val, str) else None


def optional_int(args: Mapping[str, JValue], key: str) -> int | None:
    """Extract an optional integer argument."""
    val = args.get(key)
    if val is None:
        return None
    if isinstance(val, int) and not isinstance(val, bool):
        return val
    if isinstance(val, float) and val == int(val):
        return int(val)
    return None


def optional_bool(args: Mapping[str, JValue], key: str) -> bool | None:
    """Extract an optional boolean argument."""
    val = args.get(key)
    return val if isinstance(val, bool) else None


# ---------------------------------------------------------------------------
# Error / fault outcome builders
# ---------------------------------------------------------------------------


def fault(message: str, **extra: Any) -> ToolOutcome:
    """Build a ToolOutcome representing a tool-level error."""
    return ToolOutcome(
        content=[TextSegment(text=message)],
        details=extra if extra else None,
    )


def text_outcome(text: str) -> ToolOutcome:
    """Build a ToolOutcome containing a single text block."""
    return ToolOutcome(content=[TextSegment(text=text)])


# ---------------------------------------------------------------------------
# Path validation
# ---------------------------------------------------------------------------


def validate_file_path(raw: str) -> Path:
    """Validate and resolve a file path. Raises ValueError on issues."""
    if not raw.strip():
        raise ValueError("File path must not be empty.")
    path = Path(raw)
    if not path.is_absolute():
        raise ValueError(
            f"File path must be absolute, got relative path: {raw}"
        )
    return path


__all__ = [
    # re-exports from contracts
    "CallBlock",
    "CancelToken",
    "InputShaper",
    "InvocationPrinter",
    "JValue",
    "OutcomePresenter",
    "Parallelism",
    "ProgressNotifier",
    "RunHandler",
    "TextSegment",
    "ToolOutcome",
    "ToolSpec",
    # shared helpers
    "BINARY_EXTENSIONS",
    "IMAGE_EXTENSIONS",
    "MAX_OUTPUT_LINES",
    "fault",
    "format_with_line_numbers",
    "optional_bool",
    "optional_int",
    "optional_str",
    "path_lock",
    "require_str",
    "text_outcome",
    "truncate_output",
    "validate_file_path",
]
