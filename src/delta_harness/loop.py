"""Pure portable agent loop.

``run_agent_loop`` drives one run to completion: it asks the provider for a
response, translates the raw ``WireEvent`` stream into delta's public
``AgentEvent`` stream, runs any tools the assistant requested, and loops until
the assistant replies with no tool calls (or a limit / cancellation / error
stops it).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence

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
from delta_harness.contracts.tooling import CancelToken, ToolOutcome, ToolSpec
from delta_harness.contracts.transcript import (
    CallBlock,
    ModelEntry,
    TextSegment,
    ToolOutcomeEntry,
    TranscriptEntry,
)
from delta_harness.provider import wire
from delta_harness.provider.base import ModelProvider

PreToolHook = Callable[[CallBlock], Awaitable[tuple[bool, str | None]]]
PostToolHook = Callable[
    [CallBlock, ToolOutcome, bool],
    Awaitable[tuple[ToolOutcome, bool]],
]


# --- synthetic entries ------------------------------------------------------


def _make_abort_result(text: str) -> ToolOutcome:
    """Create an error tool outcome with the given message."""
    return ToolOutcome(content=[TextSegment(text=text)])


def _make_error_reply(model: str, reason: str) -> ModelEntry:
    """Create a synthetic assistant entry representing a fault."""
    return ModelEntry(
        model=model,
        content=[],
        stop_reason="error",
        error_message=reason,
    )


# --- provider streaming -----------------------------------------------------


async def _stream_model_reply(
    provider: ModelProvider,
    model: str,
    system: str,
    transcript: list[TranscriptEntry],
    tool_defs: list[ToolSpec],
    token: CancelToken | None,
) -> AsyncIterator[AgentEvent]:
    """Relay wire events from the provider as agent-level message events."""
    raw: AsyncIterator[wire.WireEvent] = provider.stream_response(
        model=model,
        system=system,
        messages=transcript,
        tools=tool_defs,
        signal=token,
    )
    saw_open = False
    async for evt in raw:
        if isinstance(evt, wire.StreamOpenEvent):
            saw_open = True
            yield MessageStartEvent(message=evt.partial)
        elif isinstance(evt, wire.StreamCloseEvent):
            if not saw_open:
                yield MessageStartEvent(message=evt.message)
            yield MessageEndEvent(message=evt.message)
        elif isinstance(evt, wire.StreamFaultEvent):
            if not saw_open:
                yield MessageStartEvent(message=evt.error)
            yield MessageEndEvent(message=evt.error)
        else:
            yield MessageUpdateEvent(
                message=evt.partial,
                assistant_message_event=evt.model_dump(),
            )


# --- single tool invocation -------------------------------------------------


async def _run_single_tool(
    spec: ToolSpec,
    invocation: CallBlock,
    token: CancelToken | None,
) -> tuple[ToolOutcome, bool, list[ToolOutcome]]:
    """Execute one tool, capturing progress snapshots."""
    snapshots: list[ToolOutcome] = []
    alive = True

    def _on_partial(snapshot: ToolOutcome) -> None:
        if alive:
            snapshots.append(snapshot.model_copy(deep=True))

    try:
        result = await spec.execute(invocation.id, invocation.arguments, token, _on_partial)
        return result, False, snapshots
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - tools are an isolation boundary
        return _make_abort_result(str(exc)), True, snapshots
    finally:
        alive = False


# --- tool dispatch pipeline -------------------------------------------------


async def _handle_invocation(
    invocation: CallBlock,
    registry: Mapping[str, ToolSpec],
    token: CancelToken | None,
    pre_hook: PreToolHook | None,
    post_hook: PostToolHook | None,
) -> AsyncIterator[AgentEvent]:
    """Run one tool call through the full pre-hook / execute / post-hook pipeline."""
    yield ToolRunStartEvent(
        tool_call_id=invocation.id,
        tool_name=invocation.name,
        args=invocation.arguments,
    )

    rejected = False
    rejection_msg: str | None = None
    if pre_hook is not None:
        rejected, rejection_msg = await pre_hook(invocation)

    if rejected:
        outcome = _make_abort_result(rejection_msg or "Tool execution was blocked")
        errored = True
    elif token is not None and token.is_cancelled():
        outcome = _make_abort_result("Operation aborted")
        errored = True
    else:
        spec = registry.get(invocation.name)
        if spec is None:
            outcome = _make_abort_result(f"Tool {invocation.name} not found")
            errored = True
        else:
            outcome, errored, snapshots = await _run_single_tool(spec, invocation, token)
            for snap in snapshots:
                yield ToolRunUpdateEvent(
                    tool_call_id=invocation.id,
                    tool_name=invocation.name,
                    args=invocation.arguments,
                    partial_result=snap,
                )

    if post_hook is not None:
        outcome, errored = await post_hook(invocation, outcome, errored)

    yield ToolRunEndEvent(
        tool_call_id=invocation.id,
        tool_name=invocation.name,
        result=outcome,
        is_error=errored,
    )
    record = ToolOutcomeEntry(
        tool_call_id=invocation.id,
        tool_name=invocation.name,
        content=outcome.content,
        details=outcome.details,
        added_tool_names=outcome.added_tool_names,
        is_error=errored,
    )
    yield MessageStartEvent(message=record)
    yield MessageEndEvent(message=record)


# --- parallel / sequential strategy -----------------------------------------


async def _gather_events_for(
    invocation: CallBlock,
    registry: Mapping[str, ToolSpec],
    token: CancelToken | None,
    pre_hook: PreToolHook | None,
    post_hook: PostToolHook | None,
) -> list[AgentEvent]:
    """Buffer all events from a single tool invocation (used for gather)."""
    buf: list[AgentEvent] = []
    async for ev in _handle_invocation(invocation, registry, token, pre_hook, post_hook):
        buf.append(ev)
    return buf


def _can_parallelize(
    invocations: list[CallBlock],
    registry: Mapping[str, ToolSpec],
) -> bool:
    """True when every invocation's tool permits concurrent execution."""
    return all(
        (spec := registry.get(c.name)) is None or spec.parallelism != "sequential"
        for c in invocations
    )


def _cancelled_outcome_events(invocation: CallBlock) -> list[AgentEvent]:
    """Build the event sequence for a call aborted before execution."""
    abort = _make_abort_result("Operation aborted")
    record = ToolOutcomeEntry(
        tool_call_id=invocation.id,
        tool_name=invocation.name,
        content=abort.content,
        is_error=True,
    )
    return [
        ToolRunStartEvent(
            tool_call_id=invocation.id,
            tool_name=invocation.name,
            args=invocation.arguments,
        ),
        ToolRunEndEvent(
            tool_call_id=invocation.id,
            tool_name=invocation.name,
            result=abort,
            is_error=True,
        ),
        MessageStartEvent(message=record),
        MessageEndEvent(message=record),
    ]


async def _execute_calls(
    invocations: list[CallBlock],
    registry: Mapping[str, ToolSpec],
    token: CancelToken | None,
    pre_hook: PreToolHook | None,
    post_hook: PostToolHook | None,
) -> AsyncIterator[AgentEvent]:
    """Execute tool calls, choosing parallel or sequential strategy."""
    if len(invocations) > 1 and _can_parallelize(invocations, registry):
        batches = await asyncio.gather(*(
            _gather_events_for(inv, registry, token, pre_hook, post_hook)
            for inv in invocations
        ))
        for batch in batches:
            for ev in batch:
                yield ev
    else:
        for inv in invocations:
            if token is not None and token.is_cancelled():
                for ev in _cancelled_outcome_events(inv):
                    yield ev
                continue
            async for ev in _handle_invocation(inv, registry, token, pre_hook, post_hook):
                yield ev


def _pick_outcome_entries(events: list[AgentEvent]) -> list[ToolOutcomeEntry]:
    """Extract ToolOutcomeEntry objects from MessageEndEvent events."""
    return [
        ev.message
        for ev in events
        if isinstance(ev, MessageEndEvent) and isinstance(ev.message, ToolOutcomeEntry)
    ]


# --- main loop --------------------------------------------------------------


async def run_agent_loop(
    *,
    provider: ModelProvider,
    model: str,
    system: str,
    messages: list[TranscriptEntry],
    tools: list[ToolSpec],
    prompts: Sequence[TranscriptEntry] = (),
    max_turns: int | None = None,
    signal: CancelToken | None = None,
    get_steering_messages: Callable[[], Sequence[TranscriptEntry]] | None = None,
    get_follow_up_messages: Callable[[], Sequence[TranscriptEntry]] | None = None,
    before_tool_call: PreToolHook | None = None,
    after_tool_call: PostToolHook | None = None,
) -> AsyncIterator[AgentEvent]:
    """Run the provider/tool loop and emit portable agent events."""
    emitted: list[TranscriptEntry] = list(prompts)
    if prompts:
        messages.extend(prompts)

    yield RunStartEvent()
    yield TurnStartEvent()
    for p in prompts:
        yield MessageStartEvent(message=p)
        yield MessageEndEvent(message=p)

    if max_turns is not None and max_turns < 1:
        err = _make_error_reply(model, "max_turns must be at least 1")
        messages.append(err)
        emitted.append(err)
        yield MessageStartEvent(message=err)
        yield MessageEndEvent(message=err)
        yield TurnEndEvent(message=err)
        yield RunEndEvent(messages=emitted)
        return

    lookup = {t.name: t for t in tools}
    step = 1
    first = True
    pending_steering = tuple(get_steering_messages() if get_steering_messages else ())

    while True:
        has_calls = True
        while has_calls or pending_steering:
            if not first:
                yield TurnStartEvent()
            first = False

            for entry in pending_steering:
                messages.append(entry)
                emitted.append(entry)
                yield MessageStartEvent(message=entry)
                yield MessageEndEvent(message=entry)
            pending_steering = ()

            if max_turns is not None and step > max_turns:
                err = _make_error_reply(
                    model, f"Agent stopped after max_turns={max_turns}"
                )
                messages.append(err)
                emitted.append(err)
                yield MessageStartEvent(message=err)
                yield MessageEndEvent(message=err)
                yield TurnEndEvent(message=err)
                yield RunEndEvent(messages=emitted)
                return

            assistant: ModelEntry | None = None
            async for ev in _stream_model_reply(
                provider, model, system, messages, tools, signal,
            ):
                yield ev
                if isinstance(ev, MessageEndEvent) and isinstance(ev.message, ModelEntry):
                    assistant = ev.message

            if assistant is None:
                assistant = _make_error_reply(
                    model, "Provider produced no assistant message"
                )
                yield MessageStartEvent(message=assistant)
                yield MessageEndEvent(message=assistant)

            messages.append(assistant)
            emitted.append(assistant)
            if assistant.stop_reason in {"error", "aborted"}:
                yield TurnEndEvent(message=assistant)
                yield RunEndEvent(messages=emitted)
                return

            calls = list(assistant.tool_calls)
            has_calls = bool(calls)

            tool_events: list[AgentEvent] = []
            async for ev in _execute_calls(
                calls, lookup, signal, before_tool_call, after_tool_call,
            ):
                yield ev
                tool_events.append(ev)

            outcomes = _pick_outcome_entries(tool_events)
            for oc in outcomes:
                messages.append(oc)
                emitted.append(oc)

            yield TurnEndEvent(message=assistant, tool_results=outcomes)
            step += 1
            pending_steering = tuple(
                get_steering_messages() if get_steering_messages else ()
            )

        followups = tuple(
            get_follow_up_messages() if get_follow_up_messages else ()
        )
        if followups:
            pending_steering = followups
            continue
        break

    yield RunEndEvent(messages=emitted)
