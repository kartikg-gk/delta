"""Deterministic repair of malformed tool-call history.

Providers require that every tool call in an assistant turn is answered by
exactly one tool result, placed right after that turn. Interrupted runs and
older persistence paths can break this in several ways: a call with no
result, a result with no call, a result stranded after unrelated messages,
duplicate results for one call, or parallel results saved out of order.

``mend_tool_history`` restores the invariant without guessing:

1. existing results move directly after their calls, in call order;
2. an unanswered call receives an interruption error result;
3. a result whose call is gone is dropped (its arguments cannot be rebuilt);
4. duplicates collapse to one, and a real result beats a synthetic one.

Running it on already-valid history is a no-op, so it is safe to apply on
every load and as a last check before each provider request.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from delta_harness.contracts.transcript import (
    CallBlock,
    ModelEntry,
    TextSegment,
    ToolOutcomeEntry,
    TranscriptEntry,
)

INTERRUPTED_RESULT_TEXT = "Tool call interrupted by user"

# A call is identified by (index of its assistant entry, 1-based position
# among that entry's calls); its ideal result slot is entry index + position.
type _CallSlot = tuple[int, int]
type _Pick = tuple[int | None, ToolOutcomeEntry]


@dataclass(frozen=True, slots=True)
class HistoryMend:
    """Repaired history plus counters describing what was changed."""

    entries: tuple[TranscriptEntry, ...]
    changed: bool = False
    synthesized: int = 0
    dropped_orphans: int = 0
    dropped_duplicates: int = 0
    reordered: int = 0

    def counters(self) -> dict[str, int]:
        """JSON-safe summary for a durable diagnostic record."""
        return {
            "synthesized": self.synthesized,
            "dropped_orphans": self.dropped_orphans,
            "dropped_duplicates": self.dropped_duplicates,
            "reordered": self.reordered,
        }


def mend_tool_history(entries: Sequence[TranscriptEntry]) -> HistoryMend:
    """Return history in which every tool call has exactly one adjacent result."""
    history = tuple(entries)
    calls = _call_slots(history)
    results: dict[str, list[tuple[int, ToolOutcomeEntry]]] = defaultdict(list)
    for position, entry in enumerate(history):
        if isinstance(entry, ToolOutcomeEntry):
            results[entry.tool_call_id].append((position, entry))

    chosen: dict[_CallSlot, _Pick] = {}
    taken: set[int] = set()

    # Pairs that are already adjacent are claimed first, so a legitimately
    # reused id stays with its own turn instead of an earlier call taking it.
    for slot, call, ideal in calls:
        if ideal < len(history) and ideal not in taken:
            candidate = history[ideal]
            if isinstance(candidate, ToolOutcomeEntry) and candidate.tool_call_id == call.id:
                chosen[slot] = (ideal, candidate)
                taken.add(ideal)

    synthesized = 0
    for slot, call, _ideal in calls:
        if slot in chosen:
            continue
        free = [c for c in results.get(call.id, []) if c[0] not in taken]
        if not free:
            chosen[slot] = (None, _interruption(call))
            synthesized += 1
            continue
        pool = [c for c in free if c[0] > slot[0]] or free
        pick = next((c for c in pool if not _is_interruption(c[1])), pool[0])
        chosen[slot] = pick
        taken.add(pick[0])

    # A call paired with a synthetic interruption switches to a real result
    # for the same id if one is still unclaimed.
    for slot, call, _ideal in calls:
        claimed, result = chosen[slot]
        if claimed is None or not _is_interruption(result):
            continue
        better = next(
            (c for c in results.get(call.id, [])
             if c[0] not in taken and not _is_interruption(c[1])),
            None,
        )
        if better is not None:
            taken.discard(claimed)
            taken.add(better[0])
            chosen[slot] = better

    mended: list[TranscriptEntry] = []
    reordered = 0
    for position, entry in enumerate(history):
        if isinstance(entry, ToolOutcomeEntry):
            continue
        mended.append(entry)
        if not isinstance(entry, ModelEntry):
            continue
        for offset, _call in enumerate(entry.tool_calls, start=1):
            source, result = chosen[(position, offset)]
            mended.append(result)
            if source is not None and source != position + offset:
                reordered += 1

    known_ids = {call.id for _slot, call, _ideal in calls}
    leftovers = [
        result
        for group in results.values()
        for position, result in group
        if position not in taken
    ]
    changed = len(mended) != len(history) or any(
        a is not b for a, b in zip(mended, history, strict=False)
    )
    return HistoryMend(
        entries=tuple(mended),
        changed=changed,
        synthesized=synthesized,
        dropped_orphans=sum(r.tool_call_id not in known_ids for r in leftovers),
        dropped_duplicates=sum(r.tool_call_id in known_ids for r in leftovers),
        reordered=reordered,
    )


def _call_slots(history: tuple[TranscriptEntry, ...]) -> list[tuple[_CallSlot, CallBlock, int]]:
    slots: list[tuple[_CallSlot, CallBlock, int]] = []
    for position, entry in enumerate(history):
        if isinstance(entry, ModelEntry):
            for offset, call in enumerate(entry.tool_calls, start=1):
                slots.append(((position, offset), call, position + offset))
    return slots


def _interruption(call: CallBlock) -> ToolOutcomeEntry:
    return ToolOutcomeEntry(
        tool_call_id=call.id,
        tool_name=call.name,
        content=[TextSegment(text=INTERRUPTED_RESULT_TEXT)],
        is_error=True,
    )


def _is_interruption(result: ToolOutcomeEntry) -> bool:
    return result.is_error and result.text == INTERRUPTED_RESULT_TEXT


__all__ = ["INTERRUPTED_RESULT_TEXT", "HistoryMend", "mend_tool_history"]
