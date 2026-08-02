"""Context accounting, threshold detection, and compaction planning.

Provides deterministic, provider-neutral token estimation for conversation
transcripts and tool definitions, configurable compaction thresholds, plan
generation for context compaction, message serialization for summarizers,
and helpers to apply a compaction result back onto a transcript.

This module is pure — it performs no I/O, no provider calls, no async work.
Summarization itself is the caller's responsibility; this module supplies
the plan and the prompt, the caller executes the LLM call and feeds the
summary back in.

Lifecycle (as used by ``CodingSession``)::

    limits = ContextLimits(window=200_000)
    estimate = estimate_context(transcript, system=sys, tools=tools, limits=limits)

    if exceeds_threshold(estimate):
        plan = plan_compaction(transcript, record_ids, limits=limits)
        if plan is not None:
            system_prompt, user_prompt = build_summary_prompts(plan)
            summary_text = await provider.complete(system_prompt, user_prompt)
            new_transcript, new_ids = apply_compaction(plan, summary_text)
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from delta_harness.contracts.transcript import (
    HumanEntry,
    ModelEntry,
    PruneSummaryEntry,
    ToolOutcomeEntry,
    TranscriptEntry,
    surface_text,
)

if TYPE_CHECKING:
    from delta_harness.contracts.tooling import ToolSpec

# ── constants ───────────────────────────────────────────────────────────

CHARS_PER_TOKEN = 4
"""Rough character-to-token ratio for estimation when no usage data exists."""

TOOL_SCHEMA_BASELINE = 200
"""Estimated token overhead per tool definition (name + description + schema)."""

DEFAULT_WINDOW = 128_000
"""Fallback context-window size when the model is not in ``MODEL_WINDOWS``."""

MODEL_WINDOWS: dict[str, int] = {
    "claude-sonnet-4-20250514": 200_000,
    "claude-opus-4-20250514": 200_000,
    "claude-opus-4-6": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-haiku-4-5-20251001": 200_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
}
"""Known model context-window sizes in tokens."""

DEFAULT_SUMMARY_SYSTEM = (
    "You are a concise summarizer. Produce a brief summary of the conversation "
    "that preserves key decisions, code changes, file paths, and open questions. "
    "Keep it under 500 words."
)
"""Default system prompt sent to the model when generating compaction summaries."""

_INCREMENTAL_PREAMBLE = (
    "The conversation was already compacted once. The previous summary is "
    "included below as context. Merge it with the new material into a single "
    "cohesive summary.\n\n"
    "Previous summary:\n{prior_summary}\n\n"
    "New material:\n"
)
"""Template prepended to the user prompt when a prior compaction summary exists."""


# ── configuration ───────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ContextLimits:
    """Configurable knobs governing context accounting and compaction policy.

    All fields have production-ready defaults and can be overridden per-session.
    """

    window: int = DEFAULT_WINDOW
    """Maximum context-window size in tokens."""

    compaction_threshold: float = 0.75
    """Utilization ratio above which automatic compaction is triggered."""

    keep_recent: int = 4
    """Minimum number of recent entries preserved across a compaction."""

    min_entries: int = 6
    """Transcript must be at least this long before compaction is allowed."""

    max_summary_chars: int = 50_000
    """Cap on the serialized text sent to the summarizer."""

    summary_system: str = DEFAULT_SUMMARY_SYSTEM
    """System prompt for the summarizer model call."""


# ── context estimation ──────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ContextEstimate:
    """Snapshot of how much context is consumed and what the limits are.

    All values are approximate token counts derived deterministically from
    transcript content and tool schemas.
    """

    used: int
    """Total estimated tokens currently in the context window."""

    system: int
    """Tokens consumed by the system prompt and tool definitions."""

    messages: int
    """Tokens consumed by transcript entries alone."""

    limit: int
    """The effective context-window size in tokens."""

    threshold: float
    """The compaction-trigger ratio (copied from ``ContextLimits``)."""

    @property
    def utilization(self) -> float:
        """Fraction of the context window in use (0.0–1.0+)."""
        if self.limit <= 0:
            return 0.0
        return self.used / self.limit

    @property
    def headroom(self) -> int:
        """Tokens remaining before the hard window limit."""
        return max(0, self.limit - self.used)

    @property
    def needs_compaction(self) -> bool:
        """Whether utilization exceeds the compaction threshold."""
        return self.utilization > self.threshold


def resolve_window(model: str, *, override: int | None = None) -> int:
    """Return the context-window size for *model*.

    An explicit *override* takes precedence, then the ``MODEL_WINDOWS``
    lookup table, and finally ``DEFAULT_WINDOW``.
    """
    if override is not None:
        return override
    return MODEL_WINDOWS.get(model, DEFAULT_WINDOW)


def _chars_to_tokens(n: int) -> int:
    """Convert a character count to an approximate token count."""
    return max(1, n // CHARS_PER_TOKEN) if n > 0 else 0


def estimate_entry_tokens(entry: TranscriptEntry) -> int:
    """Approximate the token footprint of a single transcript entry.

    For ``ModelEntry`` objects that carry reported usage, the provider's
    ``usage.output`` is authoritative.  For everything else, we fall back
    to dividing the visible text length by ``CHARS_PER_TOKEN``.

    Tool-call arguments are serialized as compact JSON and counted
    separately from the text body.
    """
    if isinstance(entry, ModelEntry):
        # Output tokens are what the model *produced*.  Input tokens
        # reflect the cumulative context at that turn, not the entry's
        # own weight.  Use output as the entry size.
        reported = entry.usage.output
        if reported > 0:
            return reported

        # No reported usage — estimate from content.
        text_chars = len(entry.text) + len(entry.thinking_text)
        call_chars = sum(
            len(call.name) + len(json.dumps(dict(call.arguments), separators=(",", ":")))
            for call in entry.tool_calls
        )
        return _chars_to_tokens(text_chars + call_chars)

    if isinstance(entry, ToolOutcomeEntry):
        text = surface_text(entry)
        overhead = len(entry.tool_name) + len(entry.tool_call_id)
        return _chars_to_tokens(len(text) + overhead)

    if isinstance(entry, PruneSummaryEntry):
        return _chars_to_tokens(len(entry.summary))

    # HumanEntry, ShellResultEntry, ForkSummaryEntry, ExtensionEntry
    return _chars_to_tokens(len(surface_text(entry)))


def estimate_tool_tokens(tools: Sequence[ToolSpec]) -> int:
    """Approximate the tokens consumed by tool definitions in the system prompt.

    Each tool contributes a baseline overhead plus the serialized size of
    its JSON schema, description, and any prompt snippets.
    """
    total = 0
    for tool in tools:
        schema_chars = len(json.dumps(dict(tool.parameters), separators=(",", ":")))
        desc_chars = len(tool.description) + len(tool.name)
        snippet_chars = len(tool.prompt_snippet or "")
        guideline_chars = sum(len(g) for g in tool.prompt_guidelines)
        total += TOOL_SCHEMA_BASELINE + _chars_to_tokens(
            schema_chars + desc_chars + snippet_chars + guideline_chars
        )
    return total


def estimate_system_tokens(system: str, tools: Sequence[ToolSpec] = ()) -> int:
    """Token estimate for the system prompt plus all tool definitions."""
    return _chars_to_tokens(len(system)) + estimate_tool_tokens(tools)


def estimate_transcript_tokens(
    transcript: Sequence[TranscriptEntry],
) -> int:
    """Sum the per-entry token estimates for the full transcript.

    When the transcript contains a ``ModelEntry`` with reported
    ``usage.input``, that value subsumes all prior entries.  We use the
    most recent such value as a floor and add estimates only for entries
    that come after it.
    """
    last_input = 0
    last_idx = -1
    for i, entry in enumerate(transcript):
        if isinstance(entry, ModelEntry) and entry.usage.input > 0:
            last_input = entry.usage.input
            last_idx = i

    if last_idx < 0:
        # No provider-reported usage at all — estimate everything.
        return sum(estimate_entry_tokens(e) for e in transcript)

    # Entries after the last-reported turn are new; estimate them.
    tail = sum(estimate_entry_tokens(e) for e in transcript[last_idx + 1 :])
    return last_input + tail


def _has_reported_usage(transcript: Sequence[TranscriptEntry]) -> bool:
    """Whether any turn carries provider-reported input token counts."""
    return any(
        isinstance(entry, ModelEntry) and entry.usage.input > 0 for entry in transcript
    )


def estimate_context(
    transcript: Sequence[TranscriptEntry],
    *,
    system: str = "",
    tools: Sequence[ToolSpec] = (),
    limits: ContextLimits | None = None,
    model: str = "",
    window_override: int | None = None,
) -> ContextEstimate:
    """Build a full ``ContextEstimate`` for the current session state.

    This is the primary entry point for context accounting.
    """
    cfg = limits or ContextLimits()
    window = window_override or resolve_window(
        model,
        override=cfg.window if cfg.window != DEFAULT_WINDOW else None,
    )
    if window == DEFAULT_WINDOW and model:
        window = resolve_window(model)

    sys_tokens = estimate_system_tokens(system, tools)
    msg_tokens = estimate_transcript_tokens(transcript)

    if _has_reported_usage(transcript):
        # A provider's reported ``input`` count already covers the system
        # prompt and tool schemas that were sent with that turn, so adding the
        # local system estimate on top would bill them twice. Keep the reported
        # total authoritative and derive the message share from it.
        used = msg_tokens
        message_tokens = max(0, msg_tokens - sys_tokens)
    else:
        used = sys_tokens + msg_tokens
        message_tokens = msg_tokens

    return ContextEstimate(
        used=used,
        system=sys_tokens,
        messages=message_tokens,
        limit=window,
        threshold=cfg.compaction_threshold,
    )


# ── threshold detection ─────────────────────────────────────────────────


def exceeds_threshold(estimate: ContextEstimate) -> bool:
    """Return ``True`` when *estimate* indicates compaction should trigger."""
    return estimate.needs_compaction


# ── compaction planning ─────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CompactionPlan:
    """Describes exactly what a compaction operation will do.

    Produced by ``plan_compaction``; consumed by ``build_summary_prompts``
    and ``apply_compaction``.  Immutable, contains no transcript mutations.
    """

    to_summarize: tuple[TranscriptEntry, ...]
    """Older entries that will be replaced by a summary."""

    to_keep: tuple[TranscriptEntry, ...]
    """Recent entries that survive the compaction unmodified."""

    summarize_ids: tuple[str, ...]
    """Record IDs corresponding to *to_summarize* (for persistence)."""

    keep_ids: tuple[str, ...]
    """Record IDs corresponding to *to_keep*."""

    tokens_before: int
    """Estimated message tokens at the time the plan was created."""

    has_prior_summary: bool
    """Whether the first summarized entry is itself a compaction summary."""

    prior_summary_text: str
    """Text of the prior summary, or empty string."""


def plan_compaction(
    transcript: Sequence[TranscriptEntry],
    record_ids: Sequence[str],
    *,
    limits: ContextLimits | None = None,
) -> CompactionPlan | None:
    """Build a compaction plan, or ``None`` if the transcript is too short.

    The plan partitions the transcript into *to_summarize* (older entries
    that will be replaced by a summary) and *to_keep* (recent entries that
    are preserved).  It never mutates the input.
    """
    cfg = limits or ContextLimits()

    if len(transcript) < cfg.min_entries:
        return None

    keep_count = max(cfg.keep_recent, len(transcript) // 4)
    # Never keep more than we have.
    keep_count = min(keep_count, len(transcript) - 1)

    split = len(transcript) - keep_count
    if split < 1:
        return None

    to_summarize = tuple(transcript[:split])
    to_keep = tuple(transcript[split:])

    # Align record IDs with the same split.
    ids = list(record_ids)
    # Pad or truncate to match transcript length.
    while len(ids) < len(transcript):
        ids.append("")
    summarize_ids = tuple(ids[:split])
    keep_ids = tuple(ids[split:])

    # Detect prior compaction summary for incremental updates.
    has_prior = isinstance(to_summarize[0], PruneSummaryEntry)
    prior_text = to_summarize[0].summary if has_prior else ""

    return CompactionPlan(
        to_summarize=to_summarize,
        to_keep=to_keep,
        summarize_ids=summarize_ids,
        keep_ids=keep_ids,
        tokens_before=estimate_transcript_tokens(transcript),
        has_prior_summary=has_prior,
        prior_summary_text=prior_text,
    )


# ── message serialization ──────────────────────────────────────────────


def _role_label(entry: TranscriptEntry) -> str:
    """Short human-readable role tag for serialization."""
    if isinstance(entry, HumanEntry):
        return "user"
    if isinstance(entry, ModelEntry):
        return "assistant"
    if isinstance(entry, ToolOutcomeEntry):
        return "tool"
    if isinstance(entry, PruneSummaryEntry):
        return "summary"
    return entry.role


def serialize_for_summary(
    entries: Sequence[TranscriptEntry],
    *,
    max_chars: int = 50_000,
) -> str:
    """Render transcript entries as plain text for the summarizer.

    Each entry is formatted as ``[role] text``.  Tool-call arguments and
    tool results are included so the summarizer can capture code changes
    and decisions.  Output is capped at *max_chars*.
    """
    parts: list[str] = []
    total = 0
    for entry in entries:
        role = _role_label(entry)
        body = surface_text(entry)

        # Include tool-call details for assistant entries.
        if isinstance(entry, ModelEntry) and entry.tool_calls:
            call_lines = []
            for call in entry.tool_calls:
                args_str = json.dumps(dict(call.arguments), separators=(",", ":"))
                if len(args_str) > 500:
                    args_str = args_str[:500] + "..."
                call_lines.append(f"  -> {call.name}({args_str})")
            body = body + "\n" + "\n".join(call_lines) if body else "\n".join(call_lines)

        line = f"[{role}] {body}"
        if total + len(line) > max_chars:
            remaining = max_chars - total
            if remaining > 20:
                parts.append(line[:remaining])
            parts.append("[...truncated...]")
            break
        parts.append(line)
        total += len(line) + 1  # +1 for newline

    return "\n".join(parts)


# ── summary prompt construction ─────────────────────────────────────────


def build_summary_prompts(
    plan: CompactionPlan,
    *,
    system: str = DEFAULT_SUMMARY_SYSTEM,
    max_chars: int = 50_000,
) -> tuple[str, str]:
    """Build the (system, user) prompts for the summarizer call.

    When the compaction plan includes a prior summary (incremental
    compaction), the user prompt incorporates it so the model can merge
    old and new material into a single cohesive summary.

    Returns a ``(system_prompt, user_prompt)`` tuple ready to pass to
    the provider.
    """
    serialized = serialize_for_summary(plan.to_summarize, max_chars=max_chars)

    if plan.has_prior_summary and plan.prior_summary_text:
        user_prompt = _INCREMENTAL_PREAMBLE.format(
            prior_summary=plan.prior_summary_text
        ) + serialized
    else:
        user_prompt = f"Summarize this conversation:\n\n{serialized}"

    return (system, user_prompt)


# ── applying compaction ─────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CompactionResult:
    """The output of applying a compaction plan with a generated summary.

    Contains the new transcript and record IDs, plus the ``PruneSummaryEntry``
    that was inserted.
    """

    transcript: tuple[TranscriptEntry, ...]
    """The compacted transcript: summary entry followed by kept entries."""

    record_ids: tuple[str, ...]
    """Record IDs aligned with the new transcript."""

    summary_entry: PruneSummaryEntry
    """The summary entry that replaced the older messages."""

    tokens_before: int
    """Message tokens at the time of planning."""

    tokens_after: int
    """Estimated message tokens after compaction."""


def apply_compaction(
    plan: CompactionPlan,
    summary: str,
    *,
    summary_record_id: str = "",
) -> CompactionResult:
    """Combine a compaction plan with a generated summary into a result.

    Creates a ``PruneSummaryEntry``, prepends it to the kept entries,
    and returns the new transcript and aligned record IDs.
    """
    summary_entry = PruneSummaryEntry(
        summary=summary,
        tokens_before=plan.tokens_before,
    )

    new_transcript = (summary_entry, *plan.to_keep)
    new_ids = (summary_record_id, *plan.keep_ids)

    tokens_after = estimate_transcript_tokens(list(new_transcript))

    return CompactionResult(
        transcript=new_transcript,
        record_ids=new_ids,
        summary_entry=summary_entry,
        tokens_before=plan.tokens_before,
        tokens_after=tokens_after,
    )


__all__ = [
    "CHARS_PER_TOKEN",
    "CompactionPlan",
    "CompactionResult",
    "ContextEstimate",
    "ContextLimits",
    "DEFAULT_SUMMARY_SYSTEM",
    "DEFAULT_WINDOW",
    "MODEL_WINDOWS",
    "TOOL_SCHEMA_BASELINE",
    "apply_compaction",
    "build_summary_prompts",
    "estimate_context",
    "estimate_entry_tokens",
    "estimate_system_tokens",
    "estimate_tool_tokens",
    "estimate_transcript_tokens",
    "exceeds_threshold",
    "plan_compaction",
    "resolve_window",
    "serialize_for_summary",
]
