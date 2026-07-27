"""Stateful reusable agent harness built on the portable loop.

``RuntimeHarness`` wraps the pure agent loop with transcript ownership, event
subscriptions, cancellation wiring, and message queuing.  It is independent
of any coding or UI policy.
"""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from inspect import isawaitable
from typing import Literal

from delta_harness.contracts.stream import (
    AgentEvent,
    MessageEndEvent,
    MessageStartEvent,
)
from delta_harness.contracts.tooling import ToolSpec
from delta_harness.contracts.transcript import (
    ExtensionEntry,
    HumanEntry,
    InputContent,
    ModelEntry,
    ToolOutcomeEntry,
    TranscriptEntry,
)
from delta_harness.contracts.values import JValue
from delta_harness.loop import run_agent_loop
from delta_harness.provider.base import ModelProvider

# ---------------------------------------------------------------------------
# Public type aliases
# ---------------------------------------------------------------------------

EventCallback = Callable[[AgentEvent], Awaitable[None] | None]
"""Signature for event subscribers — sync or async callables."""

DrainPolicy = Literal["one_at_a_time", "all"]
"""How many queued messages to drain per opportunity."""


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


class ManualCancelFlag:
    """Concrete cancellation token triggered by external code."""

    __slots__ = ("_triggered",)

    def __init__(self) -> None:
        self._triggered = False

    def cancel(self) -> None:
        """Signal cancellation."""
        self._triggered = True

    def is_cancelled(self) -> bool:
        """Check whether cancellation was signalled."""
        return self._triggered


# ---------------------------------------------------------------------------
# Queue snapshots
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class QueueShiftEvent:
    """Lightweight snapshot emitted whenever a queue changes."""

    steering: tuple[InputContent, ...] = ()
    follow_up: tuple[InputContent, ...] = ()


@dataclass(frozen=True, slots=True)
class PendingBatch:
    """Frozen view of messages waiting in both queues."""

    steering: tuple[TranscriptEntry, ...] = ()
    follow_up: tuple[TranscriptEntry, ...] = ()

    @property
    def count(self) -> int:
        """Total number of pending messages across both queues."""
        return len(self.steering) + len(self.follow_up)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RuntimeConfig:
    """Mutable bag of knobs for a ``RuntimeHarness``."""

    provider: ModelProvider
    model: str
    system: str
    tools: list[ToolSpec] = field(default_factory=list)
    max_turns: int | None = None
    queue_mode: DrainPolicy = "one_at_a_time"


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class RuntimeHarness:
    """Reusable stateful agent brain.

    Owns the transcript, delegates execution to ``run_agent_loop``, and
    remains independent of CLI, Rich, Textual, session files, and
    coding-agent resource loading.
    """

    __slots__ = (
        "_cfg",
        "_entries",
        "_watchers",
        "_flag",
        "_running",
        "_steer_q",
        "_followup_q",
    )

    def __init__(
        self,
        config: RuntimeConfig,
        *,
        messages: Sequence[TranscriptEntry] = (),
    ) -> None:
        self._cfg = config
        self._entries: list[TranscriptEntry] = list(messages)
        self._watchers: list[EventCallback] = []
        self._flag: ManualCancelFlag | None = None
        self._running = False
        self._steer_q: deque[TranscriptEntry] = deque()
        self._followup_q: deque[TranscriptEntry] = deque()

    # -- read-only views ----------------------------------------------------

    @property
    def transcript(self) -> tuple[TranscriptEntry, ...]:
        """Immutable snapshot of the current transcript."""
        return tuple(self._entries)

    @property
    def settings(self) -> RuntimeConfig:
        """The harness configuration object."""
        return self._cfg

    @property
    def active(self) -> bool:
        """Whether a submit or resume is in progress."""
        return self._running

    @property
    def pending(self) -> PendingBatch:
        """Frozen view of both queues."""
        return PendingBatch(
            steering=tuple(self._steer_q),
            follow_up=tuple(self._followup_q),
        )

    @property
    def pending_count(self) -> int:
        """Total queued message count."""
        return self.pending.count

    # -- transcript mutation ------------------------------------------------

    def has_pending(self) -> bool:
        """True when either queue holds at least one message."""
        return bool(self._steer_q or self._followup_q)

    def push_message(self, message: TranscriptEntry) -> None:
        """Append an entry directly (e.g. when restoring session state)."""
        self._entries.append(message)

    def set_messages(self, messages: Sequence[TranscriptEntry]) -> None:
        """Replace the entire transcript (e.g. after context reconstruction)."""
        self._entries = list(messages)

    # -- subscriptions ------------------------------------------------------

    def on_event(self, listener: EventCallback) -> Callable[[], None]:
        """Register a subscriber; returns an unsubscribe handle."""
        self._watchers.append(listener)

        def _detach() -> None:
            with suppress(ValueError):
                self._watchers.remove(listener)

        return _detach

    # -- cancellation -------------------------------------------------------

    def abort(self) -> None:
        """Cancel the active loop, if any."""
        if self._flag is not None:
            self._flag.cancel()

    # -- queue operations ---------------------------------------------------

    def inject(self, content: str) -> QueueShiftEvent:
        """Convenience: queue a steering ``HumanEntry`` from plain text."""
        return self.inject_message(HumanEntry(content=content))

    def inject_message(self, message: TranscriptEntry) -> QueueShiftEvent:
        """Queue a message to be injected after the current turn/tool batch."""
        self._steer_q.append(message)
        return self.build_queue_event()

    def enqueue(self, content: str) -> QueueShiftEvent:
        """Convenience: queue a follow-up ``HumanEntry`` from plain text."""
        return self.enqueue_message(HumanEntry(content=content))

    def enqueue_message(self, message: TranscriptEntry) -> QueueShiftEvent:
        """Queue a message to be sent when the active run would otherwise stop."""
        self._followup_q.append(message)
        return self.build_queue_event()

    def flush_queues(self) -> PendingBatch:
        """Drain and return everything from both queues."""
        snap = self.pending
        self._steer_q.clear()
        self._followup_q.clear()
        return snap

    def retract_enqueued(self) -> TranscriptEntry | None:
        """Pop the newest follow-up message, or ``None``."""
        return self._followup_q.pop() if self._followup_q else None

    def retract_injected(self) -> TranscriptEntry | None:
        """Pop the newest steering message, or ``None``."""
        return self._steer_q.pop() if self._steer_q else None

    def build_queue_event(self) -> QueueShiftEvent:
        """Build a ``QueueShiftEvent`` reflecting current queue contents."""
        return QueueShiftEvent(
            steering=tuple(m.content for m in self._steer_q),
            follow_up=tuple(m.content for m in self._followup_q),
        )

    # -- run lifecycle ------------------------------------------------------

    def submit(
        self,
        content: str,
        *,
        custom_type: str | None = None,
        details: dict[str, JValue] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Append a user message and start the agent loop.

        ``custom_type``/``details`` are presentation metadata on the
        ``HumanEntry``; they don't change what the model reads.
        """
        self._assert_idle()
        self._heal_dangling_calls()
        self._running = True

        entry: TranscriptEntry
        if custom_type is not None:
            entry = ExtensionEntry(
                custom_type=custom_type,
                content=content,
                details=details,
            )
        else:
            entry = HumanEntry(content=content)
        self._entries.append(entry)
        return self._drive(prompt_entry=entry)

    def resume(self) -> AsyncIterator[AgentEvent]:
        """Continue without appending a new user message."""
        self._assert_idle()
        self._heal_dangling_calls()
        self._running = True
        return self._drive()

    # -- public repair ------------------------------------------------------

    def patch_interrupted_calls(self) -> int:
        """Repair dangling tool calls and return number of patches applied."""
        before = len(self._entries)
        self._heal_dangling_calls()
        return len(self._entries) - before

    # -- private machinery --------------------------------------------------

    async def _drive(
        self,
        *,
        prompt_entry: TranscriptEntry | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Core generator: run the loop, broadcast events, emit prompt echo."""
        token = ManualCancelFlag()
        self._flag = token
        pending_prompt = prompt_entry
        try:
            async for event in run_agent_loop(
                provider=self._cfg.provider,
                model=self._cfg.model,
                system=self._cfg.system,
                messages=self._entries,
                tools=self._cfg.tools,
                max_turns=self._cfg.max_turns,
                signal=token,
                get_steering_messages=self._pop_steered,
                get_follow_up_messages=self._pop_followups,
            ):
                await self._fan_out(event)
                yield event
                # Echo the prompt as start/end pair on the first turn boundary
                if pending_prompt is not None and event.type == "turn_start":
                    for echo in (
                        MessageStartEvent(message=pending_prompt),
                        MessageEndEvent(message=pending_prompt),
                    ):
                        await self._fan_out(echo)
                        yield echo
                    pending_prompt = None
        finally:
            if token.is_cancelled():
                self._heal_dangling_calls()
            if self._flag is token:
                self._flag = None
            self._running = False

    async def _fan_out(self, event: AgentEvent) -> None:
        """Notify every subscriber, awaiting async ones."""
        for watcher in list(self._watchers):
            rv = watcher(event)
            if isawaitable(rv):
                await rv

    def _assert_idle(self) -> None:
        """Raise if another run is already active."""
        if self._running:
            raise RuntimeError(
                "RuntimeHarness is already running; "
                "use inject() or enqueue() to queue messages."
            )

    def _pop_steered(self) -> tuple[TranscriptEntry, ...]:
        """Drain the steering queue according to the configured policy."""
        return self._drain(self._steer_q)

    def _pop_followups(self) -> tuple[TranscriptEntry, ...]:
        """Drain the follow-up queue according to the configured policy."""
        return self._drain(self._followup_q)

    def _drain(self, q: deque[TranscriptEntry]) -> tuple[TranscriptEntry, ...]:
        """Pull messages from *q* respecting ``queue_mode``."""
        if not q:
            return ()
        if self._cfg.queue_mode == "all":
            batch = tuple(q)
            q.clear()
            return batch
        return (q.popleft(),)

    def _heal_dangling_calls(self) -> None:
        """Ensure every tool call in the transcript has a matching result.

        OpenAI-compatible providers reject histories where an assistant tool
        call has no corresponding result.  If the previous run was interrupted
        mid-tool, synthesise error results to close the gaps.
        """
        answered: set[str] = {
            e.tool_call_id
            for e in self._entries
            if isinstance(e, ToolOutcomeEntry)
        }
        for entry in tuple(self._entries):
            if not isinstance(entry, ModelEntry):
                continue
            for call in entry.tool_calls:
                if call.id in answered:
                    continue
                answered.add(call.id)
                self._entries.append(
                    ToolOutcomeEntry(
                        tool_call_id=call.id,
                        tool_name=call.name,
                        content="Tool call interrupted by user",
                        is_error=True,
                    )
                )


__all__ = [
    "DrainPolicy",
    "EventCallback",
    "ManualCancelFlag",
    "PendingBatch",
    "QueueShiftEvent",
    "RuntimeConfig",
    "RuntimeHarness",
]
