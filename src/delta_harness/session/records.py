"""Discriminated record types for the append-only session log."""

from __future__ import annotations

from time import time
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from delta_harness.contracts.transcript import TranscriptEntry
from delta_harness.contracts.values import JValue


def mint_id() -> str:
    """Generate a hex identifier for a new session record."""
    return uuid4().hex


def now_epoch() -> float:
    """Capture the wall-clock time as a float epoch."""
    return time()


class EntryBase(BaseModel):
    """Shared header present on every record in the session log."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=mint_id)
    parent_id: str | None = None
    timestamp: float = Field(default_factory=now_epoch)


class TranscriptRecord(EntryBase):
    """Wraps a single harness transcript entry for persistence."""

    type: Literal["message"] = "message"
    message: TranscriptEntry


class ModelSwapRecord(EntryBase):
    """Records a switch to a different model identifier."""

    type: Literal["model_change"] = "model_change"
    model: str


class ReasoningLevelRecord(EntryBase):
    """Records an adjustment to the reasoning depth setting."""

    type: Literal["thinking_level_change"] = "thinking_level_change"
    thinking_level: str | None = None


class PruneRecord(EntryBase):
    """A condensed summary that stands in for pruned earlier records on replay."""

    type: Literal["compaction"] = "compaction"
    summary: str
    replaces_entry_ids: list[str] = Field(default_factory=list)


class ForkSummaryRecord(EntryBase):
    """Captures the synopsis of a diverged conversation branch."""

    type: Literal["branch_summary"] = "branch_summary"
    summary: str
    branch_root_id: str | None = None


class TagRecord(EntryBase):
    """A user-assigned display name for the session."""

    type: Literal["label"] = "label"
    label: str


class TipRecord(EntryBase):
    """Points to the current head of the active conversation branch."""

    type: Literal["leaf"] = "leaf"
    entry_id: str | None = None


class SessionMetaRecord(EntryBase):
    """Top-level metadata stamped when a session is first created."""

    type: Literal["session_info"] = "session_info"
    created_at: float = Field(default_factory=now_epoch)
    cwd: str | None = None
    title: str | None = None


class ExtensionRecord(EntryBase):
    """Opaque payload owned by an application-level plugin."""

    type: Literal["custom"] = "custom"
    namespace: str
    data: dict[str, JValue] = Field(default_factory=dict)


type SessionRecord = Annotated[
    TranscriptRecord
    | ModelSwapRecord
    | ReasoningLevelRecord
    | PruneRecord
    | ForkSummaryRecord
    | TagRecord
    | TipRecord
    | SessionMetaRecord
    | ExtensionRecord,
    Field(discriminator="type"),
]
