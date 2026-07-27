from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator, Iterable

from delta_harness.contracts.tooling import CancelToken, ToolSpec
from delta_harness.contracts.transcript import TranscriptEntry
from delta_harness.provider.wire import WireEvent


class ReplayProvider:
    def __init__(self, streams: Iterable[Iterable[WireEvent]]) -> None:
        self._streams = deque(map(tuple, streams))
        self.calls: list[
            tuple[
                str,
                str,
                list[TranscriptEntry],
                list[ToolSpec],
            ]
        ] = []

    async def _emit(
        self,
        events: Iterable[WireEvent],
        signal: CancelToken | None,
    ) -> AsyncIterator[WireEvent]:
        for event in events:
            if signal is not None and signal.is_cancelled():
                break
            yield event

    def stream_response(
        self,
        *,
        model: str,
        system: str,
        messages: list[TranscriptEntry],
        tools: list[ToolSpec],
        signal: CancelToken | None = None,
    ) -> AsyncIterator[WireEvent]:

        self.calls.append(
            (
                model,
                system,
                messages.copy(),
                tools.copy(),
            )
        )

        events = self._streams.popleft() if self._streams else ()

        return self._emit(events, signal)
