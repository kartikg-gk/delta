"""Provider-neutral transcript message models."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
)

from delta_harness.tooling import Invocation
from delta_harness.values import JSONValue


# --- supporting types ------------------------------------------------------


class CostBreakdown(BaseModel):
    """Cost breakdown in USD."""

    model_config = ConfigDict(extra="forbid")

    total: float = 0.0
    output: float = 0.0
    input: float = 0.0
    cache_write: float = 0.0
    cache_read: float = 0.0


class TokenUsage(BaseModel):
    """Token counts for one assistant turn."""

    model_config = ConfigDict(extra="forbid")

    total_tokens: int = 0
    output: int = 0
    input: int = 0
    cache_write: int = 0
    cache_read: int = 0
    cache_write_1h: int | None = None
    reasoning: int | None = None
    cost: CostBreakdown | None = None


# --- base with conditional serialization -----------------------------------


class _OptionalFieldModel(BaseModel):
    """Strips None-valued ``_OPTIONAL_FIELDS`` from serialized output."""

    _OPTIONAL_FIELDS: ClassVar[frozenset[str]] = frozenset()

    model_config = ConfigDict(extra="forbid")

    @model_serializer(mode="wrap")
    def _omit_null_optionals(
        self, serialize: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        payload = serialize(self)
        optionals = self._OPTIONAL_FIELDS
        return {
            k: v
            for k, v in payload.items()
            if k not in optionals or v is not None
        }


# --- message types ---------------------------------------------------------


class HumanEntry(_OptionalFieldModel):
    """User-authored transcript entry."""

    _OPTIONAL_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"custom_type", "details"}
    )

    content: str
    details: dict[str, JSONValue] | None = None
    custom_type: str | None = None
    role: Literal["user"] = "user"


class ModelEntry(_OptionalFieldModel):
    """Assistant-authored transcript entry."""

    _OPTIONAL_FIELDS: ClassVar[frozenset[str]] = frozenset({"usage"})

    tool_calls: list[Invocation] = Field(default_factory=list)
    content: str = ""
    usage: TokenUsage | None = None
    role: Literal["assistant"] = "assistant"


class ToolOutcomeEntry(BaseModel):
    """Single tool invocation outcome."""

    model_config = ConfigDict(extra="forbid")

    name: str
    tool_call_id: str
    content: str
    ok: bool = True
    error: str | None = None
    data: dict[str, JSONValue] | None = None
    details: dict[str, JSONValue] | None = None
    role: Literal["tool"] = "tool"


type TranscriptEntry = HumanEntry | ModelEntry | ToolOutcomeEntry
