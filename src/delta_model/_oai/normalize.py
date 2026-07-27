"""Convert parsed stream signals into Delta wire events.

The ``EventAssembler`` maintains a running ``ModelEntry`` snapshot and emits
the appropriate ``WireEvent`` sequence as signals arrive from the decoder.
It is the single place that translates between the provider-neutral signal
vocabulary (``parsers.py``) and Delta's internal streaming model
(``delta_harness.provider.wire``).
"""

from __future__ import annotations

from delta_harness.contracts.transcript import (
    CallBlock,
    ModelEntry,
    ReplyContent,
    TextSegment,
    ThoughtSegment,
    UsageStats,
)
from delta_harness.provider.wire import (
    CallChunkEvent,
    CallCloseEvent,
    CallOpenEvent,
    CompletionCause,
    ContentChunkEvent,
    ContentCloseEvent,
    ContentOpenEvent,
    FaultCause,
    ReasoningChunkEvent,
    ReasoningCloseEvent,
    ReasoningOpenEvent,
    StreamCloseEvent,
    StreamFaultEvent,
    StreamOpenEvent,
    WireEvent,
)

from delta_model._oai.helpers import extract_usage, map_halt_reason, parse_tool_arguments
from delta_model._oai.parsers import (
    CallArgFragment,
    CallBegin,
    CallResolved,
    Finished,
    Metadata,
    ParsedSignal,
    ParseFault,
    ReasoningChunk,
    TextChunk,
)


class EventAssembler:
    """Accumulates parsed signals and emits Delta wire events.

    The assembler builds a running ``ModelEntry`` from incremental deltas
    and yields the corresponding ``WireEvent`` instances that the harness
    loop expects.  A ``finalize`` call closes any open content blocks and
    emits the terminal ``StreamCloseEvent`` or ``StreamFaultEvent``.
    """

    def __init__(self, *, model: str, provider: str, api: str) -> None:
        self._model = model
        self._provider = provider
        self._api = api
        self._response_id: str | None = None
        self._response_model: str | None = None

        # Ordered sequence of completed content blocks
        self._blocks: list[ReplyContent] = []

        # Active text or reasoning block (at most one at a time)
        self._active_kind: str | None = None  # "text" | "reasoning"
        self._active_chunks: list[str] = []
        self._active_idx: int = -1

        # Usage and halt
        self._usage: UsageStats = UsageStats()
        self._halt: str = "stop"

        # Content-index tracking
        self._next_idx = 0
        self._call_idx_map: dict[int, int] = {}  # parser index → content index

        # State flags
        self._opened = False
        self._faulted = False

    # --- snapshot -----------------------------------------------------------

    def _snapshot(self) -> ModelEntry:
        """Build a ``ModelEntry`` reflecting the current accumulated state."""
        blocks: list[ReplyContent] = list(self._blocks)

        # Append partial content from the active block
        if self._active_kind == "text":
            text = "".join(self._active_chunks)
            if text:
                blocks.append(TextSegment(text=text))
        elif self._active_kind == "reasoning":
            thinking = "".join(self._active_chunks)
            if thinking:
                blocks.append(ThoughtSegment(thinking=thinking))

        return ModelEntry(
            model=self._response_model or self._model,
            provider=self._provider,
            api=self._api,
            response_model=self._response_model,
            response_id=self._response_id,
            content=blocks,
            stop_reason=self._halt,  # type: ignore[arg-type]
            usage=self._usage,
        )

    def _ensure_open(self) -> list[WireEvent]:
        """Emit ``StreamOpenEvent`` if the stream hasn't started yet."""
        if self._opened:
            return []
        self._opened = True
        return [StreamOpenEvent(partial=self._snapshot())]

    # --- signal dispatch ----------------------------------------------------

    def accept(self, signal: ParsedSignal) -> list[WireEvent]:
        """Process one parsed signal and return any resulting wire events."""
        if isinstance(signal, Metadata):
            return self._on_metadata(signal)
        if isinstance(signal, TextChunk):
            return self._on_text(signal)
        if isinstance(signal, ReasoningChunk):
            return self._on_reasoning(signal)
        if isinstance(signal, CallBegin):
            return self._on_call_begin(signal)
        if isinstance(signal, CallArgFragment):
            return self._on_call_arg(signal)
        if isinstance(signal, CallResolved):
            return self._on_call_resolved(signal)
        if isinstance(signal, Finished):
            return self._on_finished(signal)
        if isinstance(signal, ParseFault):
            return self._on_fault(signal)
        return []

    # --- signal handlers ----------------------------------------------------

    def _on_metadata(self, signal: Metadata) -> list[WireEvent]:
        if signal.response_id:
            self._response_id = signal.response_id
        if signal.model:
            self._response_model = signal.model
        return []

    def _on_text(self, signal: TextChunk) -> list[WireEvent]:
        events = self._ensure_open()
        if self._active_kind != "text":
            events.extend(self._close_active())
            self._active_kind = "text"
            self._active_chunks = []
            self._active_idx = self._next_idx
            self._next_idx += 1
            events.append(ContentOpenEvent(
                content_index=self._active_idx,
                partial=self._snapshot(),
            ))
        self._active_chunks.append(signal.text)
        events.append(ContentChunkEvent(
            content_index=self._active_idx,
            delta=signal.text,
            partial=self._snapshot(),
        ))
        return events

    def _on_reasoning(self, signal: ReasoningChunk) -> list[WireEvent]:
        events = self._ensure_open()
        if self._active_kind != "reasoning":
            events.extend(self._close_active())
            self._active_kind = "reasoning"
            self._active_chunks = []
            self._active_idx = self._next_idx
            self._next_idx += 1
            events.append(ReasoningOpenEvent(
                content_index=self._active_idx,
                partial=self._snapshot(),
            ))
        self._active_chunks.append(signal.text)
        events.append(ReasoningChunkEvent(
            content_index=self._active_idx,
            delta=signal.text,
            partial=self._snapshot(),
        ))
        return events

    def _on_call_begin(self, signal: CallBegin) -> list[WireEvent]:
        events = self._ensure_open()
        events.extend(self._close_active())
        idx = self._next_idx
        self._next_idx += 1
        self._call_idx_map[signal.index] = idx
        events.append(CallOpenEvent(
            content_index=idx,
            partial=self._snapshot(),
        ))
        return events

    def _on_call_arg(self, signal: CallArgFragment) -> list[WireEvent]:
        idx = self._call_idx_map.get(signal.index)
        if idx is None:
            return []
        return [CallChunkEvent(
            content_index=idx,
            delta=signal.fragment,
            partial=self._snapshot(),
        )]

    def _on_call_resolved(self, signal: CallResolved) -> list[WireEvent]:
        args = parse_tool_arguments(signal.arguments_json)
        call = CallBlock(id=signal.call_id, name=signal.name, arguments=args)
        self._blocks.append(call)
        idx = self._call_idx_map.get(signal.index)
        if idx is None:
            return []
        return [CallCloseEvent(
            content_index=idx,
            tool_call=call,
            partial=self._snapshot(),
        )]

    def _on_finished(self, signal: Finished) -> list[WireEvent]:
        if signal.finish_reason is not None:
            self._halt = map_halt_reason(signal.finish_reason)
        if signal.usage is not None:
            self._usage = extract_usage(signal.usage)
        return []

    def _on_fault(self, signal: ParseFault) -> list[WireEvent]:
        self._faulted = True
        self._halt = "error"
        events = self._ensure_open()
        error_entry = ModelEntry(
            model=self._response_model or self._model,
            provider=self._provider,
            api=self._api,
            response_model=self._response_model,
            response_id=self._response_id,
            content=list(self._snapshot().content),
            stop_reason="error",
            error_message=signal.message,
            usage=self._usage,
        )
        events.append(StreamFaultEvent(reason="error", error=error_entry))
        return events

    # --- block closing ------------------------------------------------------

    def _close_active(self) -> list[WireEvent]:
        """Close the currently active text or reasoning block."""
        if self._active_kind is None:
            return []

        content = "".join(self._active_chunks)
        idx = self._active_idx
        kind = self._active_kind

        # Commit the completed block to the ordered sequence
        if kind == "text":
            self._blocks.append(TextSegment(text=content))
        else:
            self._blocks.append(ThoughtSegment(thinking=content))

        self._active_kind = None
        self._active_chunks = []

        if kind == "text":
            return [ContentCloseEvent(
                content_index=idx,
                content=content,
                partial=self._snapshot(),
            )]
        return [ReasoningCloseEvent(
            content_index=idx,
            content=content,
            partial=self._snapshot(),
        )]

    # --- finalisation -------------------------------------------------------

    def finalize(self) -> list[WireEvent]:
        """Close all open content blocks and emit the terminal event."""
        if self._faulted:
            return []

        events: list[WireEvent] = []
        events.extend(self._close_active())

        message = self._snapshot()

        if not self._opened:
            events.append(StreamOpenEvent(partial=message))

        if self._halt in ("error", "aborted"):
            cause: FaultCause = "aborted" if self._halt == "aborted" else "error"
            events.append(StreamFaultEvent(reason=cause, error=message))
        else:
            reason: CompletionCause = "stop"
            if self._halt == "length":
                reason = "length"
            elif self._halt == "toolUse":
                reason = "toolUse"
            events.append(StreamCloseEvent(reason=reason, message=message))

        return events
