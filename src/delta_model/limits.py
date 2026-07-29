"""Runtime model limits — immutable capability data and provider discovery contract.

This module defines only two things:

1. ``ModelLimits`` — a frozen dataclass representing a model's token-budget
   constraints, supplied at runtime by the provider.
2. ``LimitsProbe`` — an optional protocol that providers implement to
   expose runtime model-limit discovery.

The module contains no model registries, no hardcoded model names, no
fallback logic, and no resolution helpers.  The provider is solely
responsible for constructing and returning ``ModelLimits``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


class LimitsError(ValueError):
    """Raised when model-limit values are invalid or inconsistent."""


@dataclass(frozen=True, slots=True)
class ModelLimits:
    """Immutable, provider-supplied token-budget constraints for a model."""

    context_window: int
    """Total context-window size in tokens (input + output)."""

    max_output: int
    """Maximum tokens the model can produce in a single response."""

    def __post_init__(self) -> None:
        if self.context_window < 1:
            raise LimitsError(f"context_window must be positive, got {self.context_window}")
        if self.max_output < 1:
            raise LimitsError(f"max_output must be positive, got {self.max_output}")
        if self.max_output >= self.context_window:
            raise LimitsError(
                f"max_output ({self.max_output:,}) must be less than "
                f"context_window ({self.context_window:,})"
            )

    @property
    def usable_context(self) -> int:
        """Effective input budget: context window minus output reservation."""
        return self.context_window - self.max_output


@runtime_checkable
class LimitsProbe(Protocol):
    """Optional capability a provider implements to report model limits."""

    async def query_limits(self, model: str) -> ModelLimits:
        """Return the runtime limits for *model*."""
        ...
