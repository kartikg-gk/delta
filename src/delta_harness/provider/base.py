"""Structural protocol for model adapters consumed by the agent loop.

Any object whose ``stream_response`` method matches the expected shape can
serve as a ``ModelProvider`` — no base class or registration required.
"""
#provider
from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Protocol

from delta_harness.contracts.tooling import ToolSpec, CancelToken
from delta_harness.contracts.transcript import TranscriptEntry
from delta_harness.provider.wire import WireEvent


class ModelProvider(Protocol):
    """Structural interface for anything that can produce model completions."""

    def stream_response(
        self,
        *,
        model: str,
        system: str,
        messages: Sequence[TranscriptEntry],
        tools: Sequence[ToolSpec],
        signal: CancelToken | None = None,
    ) -> AsyncIterator[WireEvent]:
        """Stream typed wire events for one model round-trip."""
        ...
