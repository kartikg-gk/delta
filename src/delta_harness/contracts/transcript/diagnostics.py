"""Usage, cost, and diagnostic metadata for assistant responses."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from delta_harness.contracts.transcript._base import StrictModel, now_ms
from delta_harness.contracts.values import JValue


class CostBreakdown(StrictModel):

    total: float = 0.0
    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0


class UsageStats(StrictModel):

    total_tokens: int = 0
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    cache_write_1h: int | None = None
    reasoning: int | None = None
    cost: CostBreakdown = CostBreakdown()


class FaultInfo(StrictModel):
    

    message: str
    code: str | int | None = None
    name: str | None = None
    stack: str | None = None


class TurnDiagnostic(StrictModel):


    type: str
    error: FaultInfo | None = None
    details: dict[str, JValue] | None = None
    timestamp: int = Field(default_factory=now_ms)


HaltReason = Literal["stop", "length", "toolUse", "error", "aborted"]
