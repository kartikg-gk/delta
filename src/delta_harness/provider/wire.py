"""Typed streaming signals emitted by model adapters and the agent wire protocol."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from delta_harness.contracts.transcript import CallBlock, ModelEntry, StrictModel
from delta_harness.contracts.values import JValue


class StreamOpenEvent(StrictModel):
    type: Literal["start"] = "start"
    partial: ModelEntry


class ContentOpenEvent(StrictModel):
    type: Literal["text_start"] = "text_start"
    content_index: int
    partial: ModelEntry


class ContentChunkEvent(StrictModel):
    type: Literal["text_delta"] = "text_delta"
    content_index: int
    delta: str
    partial: ModelEntry


class ContentCloseEvent(StrictModel):
    type: Literal["text_end"] = "text_end"
    content_index: int
    content: str
    partial: ModelEntry


class ReasoningOpenEvent(StrictModel):
    type: Literal["thinking_start"] = "thinking_start"
    content_index: int
    partial: ModelEntry


class ReasoningChunkEvent(StrictModel):
    type: Literal["thinking_delta"] = "thinking_delta"
    content_index: int
    delta: str
    partial: ModelEntry


class ReasoningCloseEvent(StrictModel):
    type: Literal["thinking_end"] = "thinking_end"
    content_index: int
    content: str
    partial: ModelEntry


class CallOpenEvent(StrictModel):
    type: Literal["toolcall_start"] = "toolcall_start"
    content_index: int
    partial: ModelEntry


class CallChunkEvent(StrictModel):
    type: Literal["toolcall_delta"] = "toolcall_delta"
    content_index: int
    delta: str
    partial: ModelEntry


class CallCloseEvent(StrictModel):
    type: Literal["toolcall_end"] = "toolcall_end"
    content_index: int
    tool_call: CallBlock
    partial: ModelEntry


CompletionCause = Literal["stop", "length", "toolUse"]
FaultCause = Literal["aborted", "error"]


class StreamCloseEvent(StrictModel):
    type: Literal["done"] = "done"
    reason: CompletionCause
    message: ModelEntry


class StreamFaultEvent(StrictModel):
    type: Literal["error"] = "error"
    reason: FaultCause
    error: ModelEntry


type WireEvent = Annotated[
    StreamOpenEvent
    | StreamCloseEvent
    | StreamFaultEvent
    | ContentOpenEvent
    | ContentChunkEvent
    | ContentCloseEvent
    | ReasoningOpenEvent
    | ReasoningChunkEvent
    | ReasoningCloseEvent
    | CallOpenEvent
    | CallChunkEvent
    | CallCloseEvent,
    Field(discriminator="type"),
]


class SourceStartEvent(BaseModel):
    """Signals that a new model completion is being streamed."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["response_start"] = "response_start"
    model: str


class SourceRetryEvent(BaseModel):
    """Adapter is backing off and reattempting after a transient failure."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["retry"] = "retry"
    attempt: int
    max_attempts: int
    delay_seconds: float
    message: str
    data: dict[str, JValue] | None = None


class SourceTextDelta(BaseModel):
    """An incremental chunk of plain text from the model."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["text_delta"] = "text_delta"
    delta: str


class SourceThinkingDelta(BaseModel):
    """An incremental chunk of reasoning output from the model."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["thinking_delta"] = "thinking_delta"
    delta: str


class SourceCallEvent(BaseModel):
    """A fully resolved tool invocation emitted by the model."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["tool_call"] = "tool_call"
    tool_call: CallBlock


class SourceEndEvent(BaseModel):
    """The model response has been fully received and assembled."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["response_end"] = "response_end"
    message: ModelEntry
    finish_reason: str | None = None


class SourceErrorEvent(BaseModel):
    """An adapter-level fault surfaced to the harness for display or recovery."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["error"] = "error"
    message: str
    data: dict[str, JValue] | None = None


type SourceEvent = (
    SourceStartEvent
    | SourceRetryEvent
    | SourceTextDelta
    | SourceThinkingDelta
    | SourceCallEvent
    | SourceEndEvent
    | SourceErrorEvent
)
