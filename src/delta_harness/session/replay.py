"""Reconstruct live conversation state from the session record tree.

Three steps of one pipeline:

1. **lineage** — walk parent pointers from a leaf back to the root, yielding the
   active-leaf path in root-first order.
2. **projection** — fold that ordered record path into a live transcript, applying
   compaction summaries and extracting runtime settings.
3. **find_tip** — locate the latest ``TipRecord`` leaf in a record list.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from delta_harness.contracts.transcript import (
    PruneSummaryEntry,
    TranscriptEntry,
)
from delta_harness.session.records import (
    ModelSwapRecord,
    PruneRecord,
    ReasoningLevelRecord,
    SessionMetaRecord,
    SessionRecord,
    TagRecord,
    TipRecord,
    TranscriptRecord,
)


class TreeIntegrityError(ValueError):
    """The record graph contains duplicates, cycles, or dangling references."""


# ---------------------------------------------------------------------------
# Lineage — walk the parent chain
# ---------------------------------------------------------------------------


def index_by_id(entries: list[SessionRecord]) -> dict[str, SessionRecord]:
    """Map records by their id, raising on collisions."""
    result: dict[str, SessionRecord] = {}
    for entry in entries:
        if entry.id in result:
            raise TreeIntegrityError(f"Colliding record id: {entry.id}")
        result[entry.id] = entry
    return result


def trace_to_entry(entries: list[SessionRecord], leaf_id: str) -> list[SessionRecord]:
    """Walk parent pointers from ``leaf_id`` back to the root and return the path in root-first order."""
    by_id = index_by_id(entries)
    path: list[SessionRecord] = []
    seen: set[str] = set()
    current_id: str | None = leaf_id

    while current_id is not None:
        if current_id in seen:
            raise TreeIntegrityError(f"Parent-chain cycle at record: {current_id}")
        seen.add(current_id)
        entry = by_id.get(current_id)
        if entry is None:
            raise TreeIntegrityError(f"Dangling parent reference: {current_id}")
        path.append(entry)
        current_id = entry.parent_id

    path.reverse()
    return path


# ---------------------------------------------------------------------------
# Projection — fold records -> live state
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProjectedState:
    """Result of projecting an ordered record list into live session state."""

    transcript: list[TranscriptEntry] = field(default_factory=list)
    record_ids: list[str] = field(default_factory=list)
    model: str | None = None
    thinking_level: str | None = None
    title: str | None = None
    cwd: str | None = None


def project_records(records: list[SessionRecord]) -> ProjectedState:
    """Fold an ordered list of session records into live conversation state.

    Handles ``TranscriptRecord`` entries, ``PruneRecord`` compaction summaries,
    ``ModelSwapRecord`` / ``ReasoningLevelRecord`` runtime switches, and
    ``TagRecord`` / ``SessionMetaRecord`` metadata.

    Records whose IDs appear in any ``PruneRecord.replaces_entry_ids`` are
    excluded from the resulting transcript; a ``PruneSummaryEntry`` takes their
    place.
    """
    pruned_ids: set[str] = set()
    for record in records:
        if isinstance(record, PruneRecord):
            pruned_ids.update(record.replaces_entry_ids)

    transcript: list[TranscriptEntry] = []
    record_ids: list[str] = []
    model: str | None = None
    thinking_level: str | None = None
    title: str | None = None
    cwd: str | None = None

    for record in records:
        if isinstance(record, TranscriptRecord):
            if record.id not in pruned_ids:
                transcript.append(record.message)
                record_ids.append(record.id)
        elif isinstance(record, PruneRecord):
            transcript.append(PruneSummaryEntry(
                summary=record.summary,
                tokens_before=0,
            ))
            record_ids.append(record.id)
        elif isinstance(record, ModelSwapRecord):
            model = record.model
        elif isinstance(record, ReasoningLevelRecord):
            thinking_level = record.thinking_level
        elif isinstance(record, TagRecord):
            title = record.label
        elif isinstance(record, SessionMetaRecord):
            if record.title:
                title = record.title
            if record.cwd:
                cwd = record.cwd

    return ProjectedState(
        transcript=transcript,
        record_ids=record_ids,
        model=model,
        thinking_level=thinking_level,
        title=title,
        cwd=cwd,
    )


# ---------------------------------------------------------------------------
# Tip discovery
# ---------------------------------------------------------------------------


def find_tip(records: list[SessionRecord]) -> str | None:
    """Return the entry_id from the most recent ``TipRecord``, or ``None``."""
    for record in reversed(records):
        if isinstance(record, TipRecord):
            return record.entry_id
    return None
