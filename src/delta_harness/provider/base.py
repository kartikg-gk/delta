"""Structural protocol for model adapters consumed by the agent loop.

Any object whose ``stream_response`` method matches the expected shape can
serve as a ``ModelProvider`` — no base class or registration required.
"""
#provider
from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Protocol

from delta_harness.contracts.tooling import CancelToken, ToolSpec
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
        cache_key: str | None = None,
    ) -> AsyncIterator[WireEvent]:
        """Stream typed wire events for one model round-trip.

        ``cache_key`` is a stable per-conversation identifier a provider may
        use to route successive requests toward the same prompt cache. It is
        only passed when the caller has one, so adapters may omit it.
        """
        ...
