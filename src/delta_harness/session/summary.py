"""Aggregate a stored session into token and cost totals.

The summary is derived from persisted records rather than a live harness, so a
finished session file can be summarized by tooling that never ran the agent.
This is the stable shape exported by ``delta session stats``.
"""

from __future__ import annotations

from dataclasses import dataclass

from delta_harness.contracts.transcript import ModelEntry
from delta_harness.session.records import SessionRecord, TranscriptRecord


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Token and cost totals for a single session."""

    session_id: str
    model: str | None
    provider: str | None
    turns: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    total_tokens: int
    cost_usd: float


def summarize_records(session_id: str, records: list[SessionRecord]) -> RunSummary:
    """Sum the usage of every assistant turn in *records*.

    ``model`` and ``provider`` are taken from the last assistant turn, so a
    session that switched models reports the one that finished it.  A session
    with no assistant turns yields zeroed totals and ``None`` identifiers.
    """
    replies = [
        record.message
        for record in records
        if isinstance(record, TranscriptRecord) and isinstance(record.message, ModelEntry)
    ]

    return RunSummary(
        session_id=session_id,
        model=replies[-1].model if replies else None,
        provider=replies[-1].provider if replies else None,
        turns=len(replies),
        input_tokens=sum(reply.usage.input for reply in replies),
        output_tokens=sum(reply.usage.output for reply in replies),
        cache_read_tokens=sum(reply.usage.cache_read for reply in replies),
        cache_write_tokens=sum(reply.usage.cache_write for reply in replies),
        total_tokens=sum(reply.usage.total_tokens for reply in replies),
        cost_usd=sum(reply.usage.cost.total for reply in replies),
    )
