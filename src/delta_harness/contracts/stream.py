"""Harness-level event types consumed by UIs and session persistence."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from delta_harness.contracts.tooling import ToolOutcome
from delta_harness.contracts.transcript import (
    StrictModel,
    ToolOutcomeEntry,
    TranscriptEntry,
)
from delta_harness.contracts.values import JValue

# --- message streaming -----------------------------------------------------


class MessageStartEvent(StrictModel):
    """Signals that a new transcript entry has opened."""

    type: Literal["message_start"] = "message_start"
    message: TranscriptEntry


class MessageUpdateEvent(StrictModel):
    """Carries a partial snapshot as the model streams content."""

    type: Literal["message_update"] = "message_update"
    message: TranscriptEntry
    assistant_message_event: JValue = Field(
        default=None, serialization_alias="assistantMessageEvent"
    )


class MessageEndEvent(StrictModel):
    """Marks a transcript entry as complete and finalized."""

    type: Literal["message_end"] = "message_end"
    message: TranscriptEntry


# --- tool execution --------------------------------------------------------


class ToolRunStartEvent(StrictModel):
    """Fired just before a tool invocation starts."""

    type: Literal["tool_execution_start"] = "tool_execution_start"
    tool_call_id: str
    tool_name: str
    args: dict[str, JValue] = Field(default_factory=dict)


class ToolRunUpdateEvent(StrictModel):
    """Delivers intermediate progress from a running tool."""

    type: Literal["tool_execution_update"] = "tool_execution_update"
    tool_call_id: str
    tool_name: str
    args: dict[str, JValue] = Field(default_factory=dict)
    partial_result: ToolOutcome


class ToolRunEndEvent(StrictModel):
    """Carries the final outcome after a tool invocation completes."""

    type: Literal["tool_execution_end"] = "tool_execution_end"
    tool_call_id: str
    tool_name: str
    result: ToolOutcome
    is_error: bool


# --- run lifecycle ---------------------------------------------------------


class TurnStartEvent(StrictModel):
    """Opens a new turn boundary in the run."""

    type: Literal["turn_start"] = "turn_start"


class TurnEndEvent(StrictModel):
    """Closes a turn, carrying the model reply and any tool outcomes."""

    type: Literal["turn_end"] = "turn_end"
    message: TranscriptEntry
    tool_results: list[ToolOutcomeEntry] = Field(default_factory=list)


class RetryEvent(StrictModel):
    """A failed attempt is about to be retried. Progress only, not a result."""

    type: Literal["retry"] = "retry"
    attempt: int
    max_attempts: int
    delay_seconds: float = 0.0
    message: str


class RunStartEvent(StrictModel):
    """Marks the beginning of an agent run."""

    type: Literal["agent_start"] = "agent_start"


class RunEndEvent(StrictModel):
    """Marks the end of an agent run, carrying all entries produced."""

    type: Literal["agent_end"] = "agent_end"
    messages: list[TranscriptEntry] = Field(default_factory=list)


type AgentEvent = Annotated[
    RunStartEvent
    | RunEndEvent
    | TurnStartEvent
    | TurnEndEvent
    | MessageStartEvent
    | MessageUpdateEvent
    | MessageEndEvent
    | ToolRunStartEvent
    | ToolRunUpdateEvent
    | ToolRunEndEvent
    | RetryEvent,
    Field(discriminator="type"),
]
