"""File tools: Read, Write, and Edit.

Three ToolSpec factories that produce the core filesystem tools for the agent.
Each tool is async, uses per-path locking where needed, and returns ToolOutcome.

- **Read**: read text files (with line numbers) or images (as base64).
- **Write**: create or overwrite a file, auto-creating parent directories.
- **Edit**: exact-match string replacement with a uniqueness invariant.
"""

from __future__ import annotations

import base64
import mimetypes
from collections.abc import Mapping
from pathlib import Path

from delta_app.tools._shared import (
    BINARY_EXTENSIONS,
    IMAGE_EXTENSIONS,
    MAX_OUTPUT_LINES,
    CancelToken,
    JValue,
    ProgressNotifier,
    ToolOutcome,
    ToolSpec,
    fault,
    format_with_line_numbers,
    optional_bool,
    optional_int,
    path_lock,
    require_str,
    text_outcome,
    validate_file_path,
)
from delta_harness.contracts.transcript import ImageSegment

# ═══════════════════════════════════════════════════════════════════════════
# Read tool
# ═══════════════════════════════════════════════════════════════════════════

_READ_PARAMS: Mapping[str, JValue] = {
    "type": "object",
    "properties": {
        "file_path": {
            "type": "string",
            "description": "Absolute path to the file to read.",
        },
        "offset": {
            "type": "integer",
            "description": "1-based line number to start reading from.",
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of lines to read.",
        },
    },
    "required": ["file_path"],
    "additionalProperties": False,
}


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def _is_binary(path: Path) -> bool:
    return path.suffix.lower() in BINARY_EXTENSIONS


async def _run_read(
    tool_call_id: str,
    arguments: Mapping[str, JValue],
    signal: CancelToken | None = None,
    on_update: ProgressNotifier | None = None,
) -> ToolOutcome:
    """Read a file and return its contents with line numbers, or base64 for images."""
    try:
        raw_path = require_str(arguments, "file_path")
        path = validate_file_path(raw_path)
    except ValueError as exc:
        return fault(str(exc))

    if not path.exists():
        return fault(f"File not found: {path}")

    if not path.is_file():
        return fault(
            f"Not a regular file: {path}. "
            "Use a shell command to list directory contents."
        )

    # Image files → base64 image segment
    if _is_image(path):
        return _read_image(path)

    # Binary files → reject with helpful message
    if _is_binary(path):
        return fault(
            f"Cannot read binary file: {path} "
            f"(extension: {path.suffix})"
        )

    # Text files → line-numbered output
    return _read_text(path, arguments)


_MAX_IMAGE_BYTES = 5_000_000


def _read_image(path: Path) -> ToolOutcome:
    """Read an image file and return as a base64 ImageSegment."""
    try:
        size = path.stat().st_size
        if size > _MAX_IMAGE_BYTES:
            # Providers reject oversized images, and an attached image is resent
            # on every later turn, so one bad attachment would break the session.
            return fault(
                f"Image is {size / 1_000_000:.1f} MB; the limit is "
                f"{_MAX_IMAGE_BYTES // 1_000_000} MB. Resize or crop it first."
            )
        raw = path.read_bytes()
    except OSError as exc:
        return fault(f"Failed to read image: {exc}")

    mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    encoded = base64.b64encode(raw).decode("ascii")
    return ToolOutcome(content=[ImageSegment(mime_type=mime, data=encoded)])


def _read_text(path: Path, arguments: Mapping[str, JValue]) -> ToolOutcome:
    """Read a text file with optional offset/limit, returning line-numbered output."""
    offset = optional_int(arguments, "offset")
    limit = optional_int(arguments, "limit")

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return fault(f"Failed to read file: {exc}")

    if not text:
        return text_outcome(f"(empty file: {path})")

    lines = text.split("\n")
    total = len(lines)

    # Apply offset (1-based)
    start_line = 1
    if offset is not None and offset > 1:
        start_line = min(offset, total + 1)
        lines = lines[start_line - 1:]

    # Apply limit
    effective_limit = limit if limit is not None else MAX_OUTPUT_LINES
    truncated = len(lines) > effective_limit
    if truncated:
        lines = lines[:effective_limit]

    body = format_with_line_numbers(
        "\n".join(lines), start=start_line,
    )

    if truncated:
        body += (
            f"\n\n... ({effective_limit} of {total} lines shown. "
            f"Use offset/limit to read more.)"
        )

    return text_outcome(body)


def _format_read_call(arguments: Mapping[str, JValue]) -> str | None:
    path = arguments.get("file_path", "?")
    parts = [str(path)]
    offset = arguments.get("offset")
    limit = arguments.get("limit")
    if offset:
        parts.append(f"from line {offset}")
    if limit:
        parts.append(f"({limit} lines)")
    return " ".join(parts)


def make_read_tool() -> ToolSpec:
    """Create the Read file tool."""
    return ToolSpec(
        name="Read",
        label="Read File",
        description=(
            "Read a file from the filesystem. Returns text with line numbers, "
            "or the image content for image files. Use offset/limit for large files."
        ),
        parameters=_READ_PARAMS,
        run=_run_read,
        parallelism="parallel",
        format_call=_format_read_call,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Write tool
# ═══════════════════════════════════════════════════════════════════════════

_WRITE_PARAMS: Mapping[str, JValue] = {
    "type": "object",
    "properties": {
        "file_path": {
            "type": "string",
            "description": "Absolute path to the file to write.",
        },
        "content": {
            "type": "string",
            "description": "The full content to write to the file.",
        },
    },
    "required": ["file_path", "content"],
    "additionalProperties": False,
}


async def _run_write(
    tool_call_id: str,
    arguments: Mapping[str, JValue],
    signal: CancelToken | None = None,
    on_update: ProgressNotifier | None = None,
) -> ToolOutcome:
    """Write content to a file, creating parent directories as needed."""
    try:
        raw_path = require_str(arguments, "file_path")
        content = require_str(arguments, "content")
        path = validate_file_path(raw_path)
    except ValueError as exc:
        return fault(str(exc))

    async with path_lock(path):
        return _write_file(path, content)


def _write_file(path: Path, content: str) -> ToolOutcome:
    """Perform the actual file write."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return fault(f"Failed to create directories: {exc}")

    is_new = not path.exists()

    try:
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        return fault(f"Failed to write file: {exc}")

    line_count = content.count("\n") + (1 if content else 0)
    verb = "Created" if is_new else "Updated"
    return text_outcome(f"{verb} {path} ({line_count} lines)")


def _format_write_call(arguments: Mapping[str, JValue]) -> str | None:
    path = arguments.get("file_path", "?")
    content = arguments.get("content", "")
    lines = str(content).count("\n") + 1 if content else 0
    return f"{path} ({lines} lines)"


def make_write_tool() -> ToolSpec:
    """Create the Write file tool."""
    return ToolSpec(
        name="Write",
        label="Write File",
        description=(
            "Write content to a file. Creates parent directories automatically. "
            "Overwrites existing files. Use the Edit tool for partial modifications."
        ),
        parameters=_WRITE_PARAMS,
        run=_run_write,
        parallelism="sequential",
        format_call=_format_write_call,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Edit tool
# ═══════════════════════════════════════════════════════════════════════════

_EDIT_PARAMS: Mapping[str, JValue] = {
    "type": "object",
    "properties": {
        "file_path": {
            "type": "string",
            "description": "Absolute path to the file to edit.",
        },
        "old_string": {
            "type": "string",
            "description": "The exact text to find and replace.",
        },
        "new_string": {
            "type": "string",
            "description": "The replacement text.",
        },
        "replace_all": {
            "type": "boolean",
            "description": (
                "If true, replace all occurrences. "
                "If false (default), old_string must appear exactly once."
            ),
            "default": False,
        },
    },
    "required": ["file_path", "old_string", "new_string"],
    "additionalProperties": False,
}


async def _run_edit(
    tool_call_id: str,
    arguments: Mapping[str, JValue],
    signal: CancelToken | None = None,
    on_update: ProgressNotifier | None = None,
) -> ToolOutcome:
    """Replace exact text in a file with match-uniqueness validation."""
    try:
        raw_path = require_str(arguments, "file_path")
        old_string = require_str(arguments, "old_string")
        new_string = require_str(arguments, "new_string")
    except ValueError as exc:
        return fault(str(exc))

    replace_all = optional_bool(arguments, "replace_all") or False

    if old_string == new_string:
        return fault("old_string and new_string are identical — nothing to change.")

    try:
        path = validate_file_path(raw_path)
    except ValueError as exc:
        return fault(str(exc))

    if not path.exists():
        return fault(f"File not found: {path}")

    if not path.is_file():
        return fault(f"Not a regular file: {path}")

    async with path_lock(path):
        return _apply_edit(path, old_string, new_string, replace_all=replace_all)


def _apply_edit(
    path: Path,
    old_string: str,
    new_string: str,
    *,
    replace_all: bool,
) -> ToolOutcome:
    """Validate and apply an exact-match replacement."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return fault(f"Failed to read file: {exc}")

    count = content.count(old_string)

    if count == 0:
        return _no_match_hint(content, old_string, path)

    if count > 1 and not replace_all:
        return fault(
            f"old_string matches {count} locations in {path}. "
            "Provide more surrounding context to make the match unique, "
            "or set replace_all to true."
        )

    updated = content.replace(old_string, new_string, -1 if replace_all else 1)

    try:
        path.write_text(updated, encoding="utf-8")
    except OSError as exc:
        return fault(f"Failed to write file: {exc}")

    replacements = count if replace_all else 1
    suffix = "s" if replacements > 1 else ""
    return text_outcome(
        f"Applied {replacements} replacement{suffix} in {path}"
    )


def _no_match_hint(content: str, needle: str, path: Path) -> ToolOutcome:
    """Produce a helpful error when old_string is not found."""
    lines = content.split("\n")
    # Try to find a partial match (first line of needle) to help locate the issue
    first_line = needle.split("\n")[0].strip()
    hints: list[str] = []
    if first_line:
        for i, line in enumerate(lines, 1):
            if first_line in line:
                hints.append(f"  line {i}: {line.rstrip()}")
            if len(hints) >= 3:
                break

    msg = f"old_string not found in {path}."
    if hints:
        msg += (
            f"\n\nPartial matches for the first line "
            f"({first_line!r}):\n" + "\n".join(hints)
        )
        msg += (
            "\n\nCommon causes: wrong indentation, "
            "extra/missing whitespace, or stale content. "
            "Read the file first to verify the exact text."
        )
    else:
        msg += " Read the file first to verify its current content."

    return fault(msg)


def _format_edit_call(arguments: Mapping[str, JValue]) -> str | None:
    path = arguments.get("file_path", "?")
    return str(path)


def make_edit_tool() -> ToolSpec:
    """Create the Edit file tool."""
    return ToolSpec(
        name="Edit",
        label="Edit File",
        description=(
            "Make exact string replacements in a file. "
            "By default, old_string must appear exactly once (prevents ambiguous edits). "
            "Set replace_all to true for global replacement."
        ),
        parameters=_EDIT_PARAMS,
        run=_run_edit,
        parallelism="sequential",
        format_call=_format_edit_call,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════


def file_tools() -> list[ToolSpec]:
    """Build all file-related tools."""
    return [
        make_read_tool(),
        make_write_tool(),
        make_edit_tool(),
    ]


__all__ = [
    "file_tools",
    "make_edit_tool",
    "make_read_tool",
    "make_write_tool",
]
