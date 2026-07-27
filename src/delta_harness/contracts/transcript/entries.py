"""Transcript entry types and the discriminated TranscriptEntry union."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from delta_harness.contracts.transcript._base import StrictModel, now_ms
from delta_harness.contracts.transcript.diagnostics import HaltReason, TurnDiagnostic, UsageStats
from delta_harness.contracts.transcript.segments import (
    CallBlock,
    InputContent,
    ReplyContent,
    ResultContent,
    TextSegment,
    ThoughtSegment,
    gather_text,
)
from delta_harness.contracts.values import JValue


class HumanEntry(StrictModel):
    """User-authored transcript entry."""

    role: Literal["user"] = "user"
    content: InputContent
    timestamp: int = Field(default_factory=now_ms)

    @property
    def text(self) -> str:
        return gather_text(self.content)


class ModelEntry(StrictModel):
    """An assistant message with ordered content blocks."""

    role: Literal["assistant"] = "assistant"
    content: list[ReplyContent] = Field(default_factory=list)
    model: str = "unknown"
    provider: str = "unknown"
    api: str = "unknown"
    response_model: str | None = None
    response_id: str | None = None
    stop_reason: HaltReason = "stop"
    usage: UsageStats = UsageStats()
    error_message: str | None = None
    diagnostics: list[TurnDiagnostic] | None = None
    timestamp: int = Field(default_factory=now_ms)

    @model_validator(mode="before")
    @classmethod
    def _coerce_str_blocks(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        content = data.get("content")
        if isinstance(content, str):
            data["content"] = [TextSegment(text=content)] if content else []
        usage = data.get("usage")
        if usage is None:
            data["usage"] = UsageStats()
        return data

    @property
    def text(self) -> str:
        return "".join(block.text for block in self.content if isinstance(block, TextSegment))

    @property
    def tool_calls(self) -> tuple[CallBlock, ...]:
        return tuple(block for block in self.content if isinstance(block, CallBlock))

    @property
    def thinking_text(self) -> str:
        return "".join(
            block.thinking for block in self.content if isinstance(block, ThoughtSegment)
        )


class ToolOutcomeEntry(StrictModel):
    """Result of a single tool invocation."""

    role: Literal["toolResult"] = "toolResult"
    tool_call_id: str
    tool_name: str
    content: list[ResultContent] = Field(default_factory=list)
    is_error: bool = False
    details: JValue = None
    added_tool_names: list[str] | None = None
    timestamp: int = Field(default_factory=now_ms)

    @model_validator(mode="before")
    @classmethod
    def _coerce_str_blocks(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        content = data.get("content")
        if isinstance(content, str):
            data["content"] = [TextSegment(text=content)] if content else []
        return data

    @property
    def text(self) -> str:
        return gather_text(self.content)


class ShellResultEntry(StrictModel):
    """Captured output from a shell command execution."""

    role: Literal["bashExecution"] = "bashExecution"
    command: str
    output: str
    exit_code: int | None = None
    truncated: bool = False
    cancelled: bool = False
    full_output_path: str | None = None
    exclude_from_context: bool = False
    timestamp: int = Field(default_factory=now_ms)


class ExtensionEntry(StrictModel):
    """Application-defined custom transcript entry."""

    role: Literal["custom"] = "custom"
    custom_type: str
    content: InputContent
    details: JValue = None
    display: bool = True
    timestamp: int = Field(default_factory=now_ms)

    @property
    def text(self) -> str:
        return gather_text(self.content)


class PruneSummaryEntry(StrictModel):
    """Summary produced when compacting conversation history."""

    role: Literal["compactionSummary"] = "compactionSummary"
    summary: str
    tokens_before: int
    timestamp: int = Field(default_factory=now_ms)


class ForkSummaryEntry(StrictModel):
    """Summary produced when branching a conversation."""

    role: Literal["branchSummary"] = "branchSummary"
    summary: str
    from_id: str
    timestamp: int = Field(default_factory=now_ms)


type TranscriptEntry = Annotated[
    HumanEntry
    | ModelEntry
    | ToolOutcomeEntry
    | ShellResultEntry
    | ExtensionEntry
    | PruneSummaryEntry
    | ForkSummaryEntry,
    Field(discriminator="role"),
]


def surface_text(message: TranscriptEntry) -> str:
    """Return the user-visible text of a transcript entry."""
    if isinstance(message, (HumanEntry, ModelEntry, ToolOutcomeEntry, ExtensionEntry)):
        return message.text
    if isinstance(message, (ForkSummaryEntry, PruneSummaryEntry)):
        return message.summary
    if isinstance(message, ShellResultEntry):
        return message.output
    return ""


def entry_to_human(message: TranscriptEntry) -> HumanEntry:
    """Convert custom/session-only messages to provider-compatible user context."""
    return HumanEntry(content=surface_text(message), timestamp=message.timestamp)
