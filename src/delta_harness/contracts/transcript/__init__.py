"""Provider-neutral content and transcript message models.

Split by concern: `_base` (StrictModel + timestamp), `segments` (content blocks),
`diagnostics` (usage/cost/faults), `entries` (transcript entry types). Import from
here — the public surface is unchanged:

    from delta_harness.contracts.transcript import HumanEntry, ModelEntry, TextSegment
"""

from __future__ import annotations

from delta_harness.contracts.transcript._base import StrictModel, now_ms
from delta_harness.contracts.transcript.diagnostics import (
    CostBreakdown,
    FaultInfo,
    HaltReason,
    TurnDiagnostic,
    UsageStats,
)
from delta_harness.contracts.transcript.entries import (
    ExtensionEntry,
    ForkSummaryEntry,
    HumanEntry,
    ModelEntry,
    PruneSummaryEntry,
    ShellResultEntry,
    ToolOutcomeEntry,
    TranscriptEntry,
    entry_to_human,
    surface_text,
)
from delta_harness.contracts.transcript.segments import (
    CallBlock,
    ImageSegment,
    InputContent,
    ReplyContent,
    ResultContent,
    TextSegment,
    ThoughtSegment,
    build_model_blocks,
    gather_text,
)

__all__ = [
    # base
    "StrictModel",
    "now_ms",
    # segments
    "CallBlock",
    "ImageSegment",
    "InputContent",
    "ReplyContent",
    "ResultContent",
    "TextSegment",
    "ThoughtSegment",
    "build_model_blocks",
    "gather_text",
    # diagnostics
    "CostBreakdown",
    "FaultInfo",
    "HaltReason",
    "TurnDiagnostic",
    "UsageStats",
    # entries
    "ExtensionEntry",
    "ForkSummaryEntry",
    "HumanEntry",
    "ModelEntry",
    "PruneSummaryEntry",
    "ShellResultEntry",
    "ToolOutcomeEntry",
    "TranscriptEntry",
    "entry_to_human",
    "surface_text",
]
