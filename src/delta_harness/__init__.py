"""delta_harness: the core.

Data contracts (`values`, `transcript`, `tooling`, `stream`), the agent loop
(`engine`), the stateful harness (`driver`), session persistence (`session/`),
and the coding conversation (`conversation`). Import the vocabulary from here:

    from delta_harness import AgentEvent, ToolSpec, HumanEntry
"""

from __future__ import annotations

from delta_harness.contracts.stream import (
    AgentEvent,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    RunEndEvent,
    RunStartEvent,
    ToolRunEndEvent,
    ToolRunStartEvent,
    ToolRunUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from delta_harness.contracts.tooling import (
    CancelToken,
    RunHandler,
    ToolOutcome,
    ToolSpec,
)
from delta_harness.contracts.transcript import (
    CallBlock,
    HumanEntry,
    ModelEntry,
    TextSegment,
    ToolOutcomeEntry,
    TranscriptEntry,
)
from delta_harness.contracts.values import JObject, JPrimitive, JValue

__all__ = [
    # values
    "JObject",
    "JPrimitive",
    "JValue",
    # transcript
    "CallBlock",
    "HumanEntry",
    "ModelEntry",
    "TextSegment",
    "ToolOutcomeEntry",
    "TranscriptEntry",
    # tooling
    "CancelToken",
    "RunHandler",
    "ToolOutcome",
    "ToolSpec",
    # stream
    "AgentEvent",
    "MessageEndEvent",
    "MessageStartEvent",
    "MessageUpdateEvent",
    "RunEndEvent",
    "RunStartEvent",
    "ToolRunEndEvent",
    "ToolRunStartEvent",
    "ToolRunUpdateEvent",
    "TurnEndEvent",
    "TurnStartEvent",
]
