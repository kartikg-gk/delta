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
    """Walk parent pointers from ``leaf_id`` to the root, returned root-first."""
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


def _apply_prune(
    rows: list[tuple[str, TranscriptEntry]],
    record: PruneRecord,
) -> list[tuple[str, TranscriptEntry]]:
    """Replace the rows superseded by *record* with a single summary row.

    The summary takes the **position of the first row it replaces**, so the
    condensed history stays where it happened chronologically rather than
    surfacing after the messages that outlived it.
    """
    superseded = set(record.replaces_entry_ids)
    summary = PruneSummaryEntry(summary=record.summary, tokens_before=0)

    kept: list[tuple[str, TranscriptEntry]] = []
    placed = False
    for row_id, message in rows:
        if row_id not in superseded:
            kept.append((row_id, message))
            continue
        if not placed:
            kept.append((record.id, summary))
            placed = True

    # A compaction that matched nothing still belongs in the transcript.
    if not placed:
        kept.append((record.id, summary))
    return kept


def project_records(records: list[SessionRecord]) -> ProjectedState:
    """Fold an ordered list of session records into live conversation state.

    Handles ``TranscriptRecord`` entries, ``PruneRecord`` compaction summaries,
    ``ModelSwapRecord`` / ``ReasoningLevelRecord`` runtime switches, and
    ``TagRecord`` / ``SessionMetaRecord`` metadata.

    Compactions are applied **in order, as they are encountered**, so a later
    compaction can supersede an earlier summary just as it supersedes ordinary
    messages.
    """
    rows: list[tuple[str, TranscriptEntry]] = []
    model: str | None = None
    thinking_level: str | None = None
    title: str | None = None
    cwd: str | None = None

    for record in records:
        if isinstance(record, TranscriptRecord):
            rows.append((record.id, record.message))
        elif isinstance(record, PruneRecord):
            rows = _apply_prune(rows, record)
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
        transcript=[message for _row_id, message in rows],
        record_ids=[row_id for row_id, _message in rows],
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


# ---------------------------------------------------------------------------
# Active-branch projection — the loader entry point
# ---------------------------------------------------------------------------


def project_active_branch(records: list[SessionRecord]) -> ProjectedState:
    """Project only the active branch of the record tree.

    Locates the newest ``TipRecord``, walks parent pointers back to the root,
    and projects that path.  Records abandoned by a ``rewind`` are therefore
    excluded, which a flat replay of the file would wrongly resurrect.

    Falls back to a flat projection when there is no tip (a session that never
    completed a run) or when the parent chain is unusable, so a damaged log
    still loads rather than failing outright.
    """
    tip = find_tip(records)
    if tip is None:
        return project_records(records)
    try:
        return project_records(trace_to_entry(records, tip))
    except TreeIntegrityError:
        return project_records(records)
