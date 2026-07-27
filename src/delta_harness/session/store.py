"""Append-only JSONL persistence for session records.

One place for the on-disk side of sessions: the vault (append/read) plus record
(de)serialization and forward-migration of older on-disk shapes. Runtime models
stay strict; every legacy fixup lives here at the persistence boundary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from pydantic import TypeAdapter, ValidationError

from delta_harness.session.records import SessionRecord

# ---------------------------------------------------------------------------
# Serialization + migration
# ---------------------------------------------------------------------------

_RECORD_ADAPTER: TypeAdapter[SessionRecord] = TypeAdapter(SessionRecord)


class RecordParseError(ValueError):
    """A JSONL line could not be decoded into a valid session record."""


def serialize_record(entry: SessionRecord) -> str:
    """Encode a single record to a JSONL line with no None fields."""
    return _RECORD_ADAPTER.dump_json(entry, exclude_none=True).decode() + "\n"


def deserialize_record(line: str, *, line_number: int | None = None) -> SessionRecord:
    """Decode a single JSONL line, upgrading legacy shapes on the fly."""
    location = f" on line {line_number}" if line_number is not None else ""
    try:
        raw = json.loads(line)
        upgraded = _upgrade_entry(raw)
        return _RECORD_ADAPTER.validate_python(upgraded)
    except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
        raise RecordParseError(f"Unreadable session record{location}: {exc}") from exc


def deserialize_records(lines: list[str]) -> list[SessionRecord]:
    """Decode all non-blank JSONL lines into session records."""
    results: list[SessionRecord] = []
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        results.append(deserialize_record(line, line_number=index))
    return results


def _upgrade_entry(value: Any) -> Any:
    """Normalize a raw dict into the current record schema.

    Application-facing models evolve freely, but saved sessions must remain
    loadable.  All shape fixups live here at the persistence boundary so the
    runtime layer stays strict and unaware of legacy formats.
    """
    if not isinstance(value, dict) or value.get("type") != "message":
        return value
    migrated = dict(value)
    migrated["message"] = _upgrade_message(value.get("message"))
    return migrated


def _upgrade_message(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    message = dict(value)
    role = message.get("role")

    if role == "user" and ("custom_type" in message or "customType" in message):
        message["role"] = "custom"
        message["customType"] = message.pop("custom_type", message.get("customType"))
        message.pop("custom_type", None)
        message.setdefault("display", True)
        return message

    if role == "assistant":
        usage = message.get("usage")
        if isinstance(usage, dict) and usage.get("cost") is None:
            usage = dict(usage)
            usage["cost"] = {}
            message["usage"] = usage

        content = message.get("content", "")
        if isinstance(content, str):
            blocks: list[Any] = []
            if content:
                blocks.append({"type": "text", "text": content})
            blocks.extend(message.pop("tool_calls", message.pop("toolCalls", [])) or [])
            message["content"] = blocks
        elif "tool_calls" in message or "toolCalls" in message:
            blocks = list(content or [])
            blocks.extend(message.pop("tool_calls", message.pop("toolCalls", [])) or [])
            message["content"] = blocks
        return message

    if role == "tool":
        message["role"] = "toolResult"
        message["toolName"] = message.pop("name", message.get("toolName", "unknown"))
        message["toolCallId"] = message.pop("tool_call_id", message.get("toolCallId", ""))
        message["isError"] = not bool(message.pop("ok", True))
        content = message.get("content", "")
        if isinstance(content, str):
            message["content"] = [{"type": "text", "text": content}] if content else []
        data = message.pop("data", None)
        details = message.get("details")
        if isinstance(data, dict) and isinstance(details, dict):
            message["details"] = {**data, **details}
        elif details is None and data is not None:
            message["details"] = data
        error = message.pop("error", None)
        if error and not message["content"]:
            message["content"] = [{"type": "text", "text": str(error)}]
        return message

    return message


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


class SessionVault(Protocol):
    """Contract for any backend that persists session records."""

    async def append(self, entry: SessionRecord) -> None:
        """Write a single record to the end of the log."""
        ...

    async def read_all(self) -> list[SessionRecord]:
        """Load every record in the order it was written."""
        ...


class JsonlVault:
    """File-backed vault that stores one JSON object per line."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    async def append(self, entry: SessionRecord) -> None:
        """Write a record, creating parent directories on first use."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(serialize_record(entry))

    async def read_all(self) -> list[SessionRecord]:
        """Load all records from disk. A missing file yields an empty list."""
        if not self.path.exists():
            return []
        return deserialize_records(self.path.read_text(encoding="utf-8").splitlines())
