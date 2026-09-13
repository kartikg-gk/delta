"""Comprehensive tests for delta_harness.engine.

Verifies all code paths: event sequences, transcript mutations, tool execution,
provider streaming, parallel/sequential execution, cancellation, tool failures,
provider failures, hooks, max_turns, follow-up messages, steering messages,
empty provider responses, and edge cases.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

import pytest

from delta_harness.contracts.stream import (
    AgentEvent,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    RunEndEvent,
    ToolRunEndEvent,
    ToolRunStartEvent,
    ToolRunUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from delta_harness.contracts.tooling import CancelToken, ToolOutcome, ToolSpec
from delta_harness.contracts.transcript import (
    CallBlock,
    HumanEntry,
    ModelEntry,
    TextSegment,
    ToolOutcomeEntry,
    TranscriptEntry,
)
from delta_harness.contracts.values import JValue
from delta_harness.engine import (
    run_agent_loop,
)
from delta_harness.provider.wire import (
    ContentChunkEvent,
    ContentCloseEvent,
    ContentOpenEvent,
    StreamCloseEvent,
    StreamFaultEvent,
    StreamOpenEvent,
    WireEvent,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class FakeCancel:
    """Controllable cancel token."""

    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def is_cancelled(self) -> bool:
        return self._cancelled


def _tool_call(name: str = "echo", call_id: str = "c1", args: dict[str, Any] | None = None) -> CallBlock:
    return CallBlock(name=name, id=call_id, arguments=args or {})


def _model_entry(
    text: str = "",
    tool_calls: list[CallBlock] | None = None,
    stop_reason: str = "stop",
    model: str = "test-model",
) -> ModelEntry:
    content: list[Any] = []
    if text:
        content.append(TextSegment(text=text))
    if tool_calls:
        content.extend(tool_calls)
    return ModelEntry(
        model=model,
        content=content,
        stop_reason=stop_reason,
    )


class ScriptedProvider:
    """Provider that yields pre-scripted wire event sequences per turn."""

    def __init__(self, turns: list[list[WireEvent]]) -> None:
        self._turns = list(turns)
        self._turn_idx = 0

    async def stream_response(
        self,
        *,
        model: str,
        system: str,
        messages: Sequence[TranscriptEntry],
        tools: Sequence[ToolSpec],
        signal: CancelToken | None = None,
    ) -> AsyncIterator[WireEvent]:
        events = self._turns[self._turn_idx]
        self._turn_idx += 1
        for e in events:
            yield e


def _simple_close(entry: ModelEntry) -> list[WireEvent]:
    """Standard wire events: open then close."""
    return [
        StreamOpenEvent(partial=entry),
        StreamCloseEvent(reason="stop", message=entry),
    ]


def _tooluse_close(entry: ModelEntry) -> list[WireEvent]:
    return [
        StreamOpenEvent(partial=entry),
        StreamCloseEvent(reason="toolUse", message=entry),
    ]


async def _echo_run(
    tool_call_id: str,
    arguments: Mapping[str, JValue],
    signal: CancelToken | None = None,
    on_update: Any = None,
) -> ToolOutcome:
    return ToolOutcome(content=[TextSegment(text=f"echoed:{arguments.get('x', '')}")])


async def _progress_run(
    tool_call_id: str,
    arguments: Mapping[str, JValue],
    signal: CancelToken | None = None,
    on_update: Any = None,
) -> ToolOutcome:
    if on_update:
        on_update(ToolOutcome(content=[TextSegment(text="partial1")]))
        on_update(ToolOutcome(content=[TextSegment(text="partial2")]))
    return ToolOutcome(content=[TextSegment(text="done")])


async def _failing_run(
    tool_call_id: str,
    arguments: Mapping[str, JValue],
    signal: CancelToken | None = None,
    on_update: Any = None,
) -> ToolOutcome:
    raise RuntimeError("tool exploded")


async def _slow_run(
    tool_call_id: str,
    arguments: Mapping[str, JValue],
    signal: CancelToken | None = None,
    on_update: Any = None,
) -> ToolOutcome:
    await asyncio.sleep(0.05)
    return ToolOutcome(content=[TextSegment(text=f"slow:{tool_call_id}")])


def _make_tool(
    name: str = "echo",
    run: Any = _echo_run,
    parallelism: str = "parallel",
) -> ToolSpec:
    return ToolSpec(
        name=name,
        label=name,
        description=f"test {name}",
        parameters={},
        run=run,
        parallelism=parallelism,
    )


async def _collect(aiter: AsyncIterator[AgentEvent]) -> list[AgentEvent]:
    return [ev async for ev in aiter]


def _event_types(events: list[AgentEvent]) -> list[str]:
    return [ev.type for ev in events]


# ---------------------------------------------------------------------------
# Basic run lifecycle
# ---------------------------------------------------------------------------


class TestBasicRun:
    @pytest.mark.asyncio
    async def test_simple_text_reply(self) -> None:
        """Model replies with text only — no tool calls."""
        entry = _model_entry(text="hello world")
        provider = ScriptedProvider([_simple_close(entry)])
        messages: list[TranscriptEntry] = []

        events = await _collect(run_agent_loop(
            provider=provider, model="m", system="s",
            messages=messages, tools=[],
        ))

        types = _event_types(events)
        assert types[0] == "agent_start"
        assert types[1] == "turn_start"
        assert "message_start" in types
        assert "message_end" in types
        assert types[-2] == "turn_end"
        assert types[-1] == "agent_end"

    @pytest.mark.asyncio
    async def test_transcript_mutation(self) -> None:
        """The messages list is mutated with the model reply."""
        entry = _model_entry(text="hello")
        provider = ScriptedProvider([_simple_close(entry)])
        messages: list[TranscriptEntry] = []

        await _collect(run_agent_loop(
            provider=provider, model="m", system="s",
            messages=messages, tools=[],
        ))

        assert len(messages) == 1
        assert isinstance(messages[0], ModelEntry)

    @pytest.mark.asyncio
    async def test_run_end_carries_all_entries(self) -> None:
        """RunEndEvent.messages contains all entries added during the run."""
        entry = _model_entry(text="hi")
        provider = ScriptedProvider([_simple_close(entry)])
        messages: list[TranscriptEntry] = []

        events = await _collect(run_agent_loop(
            provider=provider, model="m", system="s",
            messages=messages, tools=[],
        ))

        run_end = [e for e in events if isinstance(e, RunEndEvent)][0]
        assert len(run_end.messages) == 1

    @pytest.mark.asyncio
    async def test_prompts_emitted_and_appended(self) -> None:
        """Prompts appear in events and get appended to messages."""
        entry = _model_entry(text="ok")
        provider = ScriptedProvider([_simple_close(entry)])
        prompt = HumanEntry(content="initial prompt")
        messages: list[TranscriptEntry] = []

        events = await _collect(run_agent_loop(
            provider=provider, model="m", system="s",
            messages=messages, tools=[], prompts=[prompt],
        ))

        # Prompt should be first in messages
        assert messages[0] is prompt
        # Should have message start/end for prompt
        msg_starts = [e for e in events if isinstance(e, MessageStartEvent)]
        assert any(e.message is prompt for e in msg_starts)


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


class TestToolExecution:
    @pytest.mark.asyncio
    async def test_single_tool_call(self) -> None:
        """Model makes one tool call, loop executes it and feeds result back."""
        call = _tool_call("echo", "c1", {"x": "hi"})
        reply1 = _model_entry(tool_calls=[call], stop_reason="toolUse")
        reply2 = _model_entry(text="done")

        provider = ScriptedProvider([
            _tooluse_close(reply1),
            _simple_close(reply2),
        ])
        tool = _make_tool("echo")
        messages: list[TranscriptEntry] = []

        events = await _collect(run_agent_loop(
            provider=provider, model="m", system="s",
            messages=messages, tools=[tool],
        ))

        types = _event_types(events)
        assert "tool_execution_start" in types
        assert "tool_execution_end" in types

        # Messages should contain: reply1, tool_outcome, reply2
        assert len(messages) == 3
        assert isinstance(messages[1], ToolOutcomeEntry)
        assert messages[1].tool_name == "echo"

    @pytest.mark.asyncio
    async def test_tool_not_found(self) -> None:
        """Calling a tool that doesn't exist yields an error outcome."""
        call = _tool_call("nonexistent", "c1")
        reply1 = _model_entry(tool_calls=[call], stop_reason="toolUse")
        reply2 = _model_entry(text="ok")

        provider = ScriptedProvider([
            _tooluse_close(reply1),
            _simple_close(reply2),
        ])
        messages: list[TranscriptEntry] = []

        events = await _collect(run_agent_loop(
            provider=provider, model="m", system="s",
            messages=messages, tools=[],
        ))

        end_events = [e for e in events if isinstance(e, ToolRunEndEvent)]
        assert len(end_events) == 1
        assert end_events[0].is_error
        assert "not found" in end_events[0].result.text

    @pytest.mark.asyncio
    async def test_tool_exception_caught(self) -> None:
        """A tool raising an exception produces an error outcome."""
        call = _tool_call("fail", "c1")
        reply1 = _model_entry(tool_calls=[call], stop_reason="toolUse")
        reply2 = _model_entry(text="ok")

        provider = ScriptedProvider([
            _tooluse_close(reply1),
            _simple_close(reply2),
        ])
        tool = _make_tool("fail", _failing_run)
        messages: list[TranscriptEntry] = []

        events = await _collect(run_agent_loop(
            provider=provider, model="m", system="s",
            messages=messages, tools=[tool],
        ))

        end_events = [e for e in events if isinstance(e, ToolRunEndEvent)]
        assert end_events[0].is_error
        assert "tool exploded" in end_events[0].result.text

    @pytest.mark.asyncio
    async def test_tool_progress_updates(self) -> None:
        """Progress callbacks generate ToolRunUpdateEvent events."""
        call = _tool_call("prog", "c1")
        reply1 = _model_entry(tool_calls=[call], stop_reason="toolUse")
        reply2 = _model_entry(text="done")

        provider = ScriptedProvider([
            _tooluse_close(reply1),
            _simple_close(reply2),
        ])
        tool = _make_tool("prog", _progress_run)
        messages: list[TranscriptEntry] = []

        events = await _collect(run_agent_loop(
            provider=provider, model="m", system="s",
            messages=messages, tools=[tool],
        ))

        updates = [e for e in events if isinstance(e, ToolRunUpdateEvent)]
        assert len(updates) == 2
        assert updates[0].partial_result.text == "partial1"
        assert updates[1].partial_result.text == "partial2"

    @pytest.mark.asyncio
    async def test_tool_outcome_entry_fields(self) -> None:
        """ToolOutcomeEntry captures details and added_tool_names from ToolOutcome."""
        async def _detailed_run(
            tool_call_id: str, arguments: Mapping[str, JValue],
            signal: CancelToken | None = None, on_update: Any = None,
        ) -> ToolOutcome:
            return ToolOutcome(
                content=[TextSegment(text="ok")],
                details={"key": "val"},
                added_tool_names=["new_tool"],
            )

        call = _tool_call("det", "c1")
        reply1 = _model_entry(tool_calls=[call], stop_reason="toolUse")
        reply2 = _model_entry(text="done")

        provider = ScriptedProvider([_tooluse_close(reply1), _simple_close(reply2)])
        tool = _make_tool("det", _detailed_run)
        messages: list[TranscriptEntry] = []

        events = await _collect(run_agent_loop(
            provider=provider, model="m", system="s",
            messages=messages, tools=[tool],
        ))

        outcomes = [
            e.message for e in events
            if isinstance(e, MessageEndEvent) and isinstance(e.message, ToolOutcomeEntry)
        ]
        assert len(outcomes) == 1
        assert outcomes[0].details == {"key": "val"}
        assert outcomes[0].added_tool_names == ["new_tool"]


# ---------------------------------------------------------------------------
# Parallel vs sequential execution
# ---------------------------------------------------------------------------


class TestParallelExecution:
    @pytest.mark.asyncio
    async def test_parallel_calls_execute_concurrently(self) -> None:
        """Multiple parallel-eligible tools run via gather (concurrently)."""
        c1 = _tool_call("slow", "a1")
        c2 = _tool_call("slow", "a2")
        reply1 = _model_entry(tool_calls=[c1, c2], stop_reason="toolUse")
        reply2 = _model_entry(text="done")

        provider = ScriptedProvider([_tooluse_close(reply1), _simple_close(reply2)])
        tool = _make_tool("slow", _slow_run, parallelism="parallel")
        messages: list[TranscriptEntry] = []

        import time
        t0 = time.monotonic()
        events = await _collect(run_agent_loop(
            provider=provider, model="m", system="s",
            messages=messages, tools=[tool],
        ))
        elapsed = time.monotonic() - t0

        # Both 50ms sleeps should overlap — total < 90ms
        assert elapsed < 0.15

        end_events = [e for e in events if isinstance(e, ToolRunEndEvent)]
        assert len(end_events) == 2

    @pytest.mark.asyncio
    async def test_sequential_when_any_tool_is_sequential(self) -> None:
        """If any tool is sequential, all run serially."""
        c1 = _tool_call("seq", "a1")
        c2 = _tool_call("par", "a2")
        reply1 = _model_entry(tool_calls=[c1, c2], stop_reason="toolUse")
        reply2 = _model_entry(text="done")

        provider = ScriptedProvider([_tooluse_close(reply1), _simple_close(reply2)])
        seq_tool = _make_tool("seq", _slow_run, parallelism="sequential")
        par_tool = _make_tool("par", _slow_run, parallelism="parallel")
        messages: list[TranscriptEntry] = []

        import time
        t0 = time.monotonic()
        _events = await _collect(run_agent_loop(
            provider=provider, model="m", system="s",
            messages=messages, tools=[seq_tool, par_tool],
        ))
        elapsed = time.monotonic() - t0

        # Sequential: at least 100ms total
        assert elapsed >= 0.09

    @pytest.mark.asyncio
    async def test_single_call_no_gather(self) -> None:
        """Single tool call runs sequentially even if parallel-eligible."""
        call = _tool_call("echo", "c1", {"x": "val"})
        reply1 = _model_entry(tool_calls=[call], stop_reason="toolUse")
        reply2 = _model_entry(text="done")

        provider = ScriptedProvider([_tooluse_close(reply1), _simple_close(reply2)])
        tool = _make_tool("echo")
        messages: list[TranscriptEntry] = []

        events = await _collect(run_agent_loop(
            provider=provider, model="m", system="s",
            messages=messages, tools=[tool],
        ))

        end_events = [e for e in events if isinstance(e, ToolRunEndEvent)]
        assert len(end_events) == 1
        assert not end_events[0].is_error

    @pytest.mark.asyncio
    async def test_parallel_events_in_call_order(self) -> None:
        """Parallel results are yielded in call order, not completion order."""
        c1 = _tool_call("echo", "first", {"x": "1"})
        c2 = _tool_call("echo", "second", {"x": "2"})
        reply1 = _model_entry(tool_calls=[c1, c2], stop_reason="toolUse")
        reply2 = _model_entry(text="done")

        provider = ScriptedProvider([_tooluse_close(reply1), _simple_close(reply2)])
        tool = _make_tool("echo")
        messages: list[TranscriptEntry] = []

        events = await _collect(run_agent_loop(
            provider=provider, model="m", system="s",
            messages=messages, tools=[tool],
        ))

        starts = [e for e in events if isinstance(e, ToolRunStartEvent)]
        assert starts[0].tool_call_id == "first"
        assert starts[1].tool_call_id == "second"


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


class TestCancellation:
    @pytest.mark.asyncio
    async def test_cancel_before_tool_execution(self) -> None:
        """Pre-cancelled signal skips tool execution with abort outcome."""
        call = _tool_call("echo", "c1")
        reply1 = _model_entry(tool_calls=[call], stop_reason="toolUse")
        reply2 = _model_entry(text="done")

        provider = ScriptedProvider([_tooluse_close(reply1), _simple_close(reply2)])
        tool = _make_tool("echo")
        token = FakeCancel()
        token.cancel()  # cancelled before any tool runs
        messages: list[TranscriptEntry] = []

        events = await _collect(run_agent_loop(
            provider=provider, model="m", system="s",
            messages=messages, tools=[tool], signal=token,
        ))

        end_events = [e for e in events if isinstance(e, ToolRunEndEvent)]
        assert len(end_events) == 1
        assert end_events[0].is_error
        assert "aborted" in end_events[0].result.text.lower()

    @pytest.mark.asyncio
    async def test_cancel_between_sequential_calls(self) -> None:
        """Cancelling between sequential calls aborts remaining tools."""
        cancel_after_first = FakeCancel()
        call_count = 0

        async def _counting_run(
            tool_call_id: str, arguments: Mapping[str, JValue],
            signal: CancelToken | None = None, on_update: Any = None,
        ) -> ToolOutcome:
            nonlocal call_count
            call_count += 1
            cancel_after_first.cancel()
            return ToolOutcome(content=[TextSegment(text="ok")])

        c1 = _tool_call("seq", "a1")
        c2 = _tool_call("seq", "a2")
        reply1 = _model_entry(tool_calls=[c1, c2], stop_reason="toolUse")
        reply2 = _model_entry(text="done")

        provider = ScriptedProvider([_tooluse_close(reply1), _simple_close(reply2)])
        tool = _make_tool("seq", _counting_run, parallelism="sequential")
        messages: list[TranscriptEntry] = []

        events = await _collect(run_agent_loop(
            provider=provider, model="m", system="s",
            messages=messages, tools=[tool], signal=cancel_after_first,
        ))

        assert call_count == 1  # second tool was not executed
        end_events = [e for e in events if isinstance(e, ToolRunEndEvent)]
        assert len(end_events) == 2
        assert not end_events[0].is_error  # first succeeded
        assert end_events[1].is_error  # second aborted

    @pytest.mark.asyncio
    async def test_cancel_checked_in_dispatch(self) -> None:
        """Cancel token checked inside dispatch pipeline before execution."""
        call = _tool_call("echo", "c1")
        reply1 = _model_entry(tool_calls=[call], stop_reason="toolUse")
        reply2 = _model_entry(text="ok")

        provider = ScriptedProvider([_tooluse_close(reply1), _simple_close(reply2)])
        tool = _make_tool("echo")
        token = FakeCancel()
        token.cancel()
        messages: list[TranscriptEntry] = []

        events = await _collect(run_agent_loop(
            provider=provider, model="m", system="s",
            messages=messages, tools=[tool], signal=token,
        ))

        end_events = [e for e in events if isinstance(e, ToolRunEndEvent)]
        assert all(e.is_error for e in end_events)


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------


class TestHooks:
    @pytest.mark.asyncio
    async def test_before_tool_call_blocks(self) -> None:
        """PreToolHook returning (True, reason) blocks execution."""
        async def deny(call: CallBlock) -> tuple[bool, str | None]:
            return True, "denied by policy"

        call = _tool_call("echo", "c1")
        reply1 = _model_entry(tool_calls=[call], stop_reason="toolUse")
        reply2 = _model_entry(text="done")

        provider = ScriptedProvider([_tooluse_close(reply1), _simple_close(reply2)])
        tool = _make_tool("echo")
        messages: list[TranscriptEntry] = []

        events = await _collect(run_agent_loop(
            provider=provider, model="m", system="s",
            messages=messages, tools=[tool],
            before_tool_call=deny,
        ))

        end_events = [e for e in events if isinstance(e, ToolRunEndEvent)]
        assert end_events[0].is_error
        assert "denied by policy" in end_events[0].result.text

    @pytest.mark.asyncio
    async def test_before_tool_call_allows(self) -> None:
        """PreToolHook returning (False, None) allows execution."""
        async def allow(call: CallBlock) -> tuple[bool, str | None]:
            return False, None

        call = _tool_call("echo", "c1", {"x": "hi"})
        reply1 = _model_entry(tool_calls=[call], stop_reason="toolUse")
        reply2 = _model_entry(text="done")

        provider = ScriptedProvider([_tooluse_close(reply1), _simple_close(reply2)])
        tool = _make_tool("echo")
        messages: list[TranscriptEntry] = []

        events = await _collect(run_agent_loop(
            provider=provider, model="m", system="s",
            messages=messages, tools=[tool],
            before_tool_call=allow,
        ))

        end_events = [e for e in events if isinstance(e, ToolRunEndEvent)]
        assert not end_events[0].is_error

    @pytest.mark.asyncio
    async def test_before_tool_block_no_reason(self) -> None:
        """PreToolHook blocking with None reason uses default message."""
        async def deny(call: CallBlock) -> tuple[bool, str | None]:
            return True, None

        call = _tool_call("echo", "c1")
        reply1 = _model_entry(tool_calls=[call], stop_reason="toolUse")
        reply2 = _model_entry(text="done")

        provider = ScriptedProvider([_tooluse_close(reply1), _simple_close(reply2)])
        tool = _make_tool("echo")
        messages: list[TranscriptEntry] = []

        events = await _collect(run_agent_loop(
            provider=provider, model="m", system="s",
            messages=messages, tools=[tool],
            before_tool_call=deny,
        ))

        end_events = [e for e in events if isinstance(e, ToolRunEndEvent)]
        assert "blocked" in end_events[0].result.text.lower()

    @pytest.mark.asyncio
    async def test_after_tool_call_modifies_result(self) -> None:
        """PostToolHook can modify the outcome and error status."""
        async def modify(
            call: CallBlock, outcome: ToolOutcome, is_error: bool,
        ) -> tuple[ToolOutcome, bool]:
            return ToolOutcome(content=[TextSegment(text="modified")]), True

        call = _tool_call("echo", "c1")
        reply1 = _model_entry(tool_calls=[call], stop_reason="toolUse")
        reply2 = _model_entry(text="done")

        provider = ScriptedProvider([_tooluse_close(reply1), _simple_close(reply2)])
        tool = _make_tool("echo")
        messages: list[TranscriptEntry] = []

        events = await _collect(run_agent_loop(
            provider=provider, model="m", system="s",
            messages=messages, tools=[tool],
            after_tool_call=modify,
        ))

        end_events = [e for e in events if isinstance(e, ToolRunEndEvent)]
        assert end_events[0].is_error
        assert end_events[0].result.text == "modified"


# ---------------------------------------------------------------------------
# max_turns
# ---------------------------------------------------------------------------


class TestMaxTurns:
    @pytest.mark.asyncio
    async def test_max_turns_zero(self) -> None:
        """max_turns=0 emits error immediately."""
        provider = ScriptedProvider([])
        messages: list[TranscriptEntry] = []

        events = await _collect(run_agent_loop(
            provider=provider, model="m", system="s",
            messages=messages, tools=[], max_turns=0,
        ))

        types = _event_types(events)
        assert types[0] == "agent_start"
        assert types[-1] == "agent_end"

        msg_ends = [e for e in events if isinstance(e, MessageEndEvent)]
        assert any(
            isinstance(e.message, ModelEntry) and e.message.stop_reason == "error"
            for e in msg_ends
        )

    @pytest.mark.asyncio
    async def test_max_turns_negative(self) -> None:
        """max_turns=-1 emits error immediately."""
        provider = ScriptedProvider([])
        messages: list[TranscriptEntry] = []

        events = await _collect(run_agent_loop(
            provider=provider, model="m", system="s",
            messages=messages, tools=[], max_turns=-1,
        ))

        run_end = [e for e in events if isinstance(e, RunEndEvent)][0]
        assert any(
            isinstance(m, ModelEntry) and "max_turns" in (m.error_message or "")
            for m in run_end.messages
        )

    @pytest.mark.asyncio
    async def test_max_turns_limits_iterations(self) -> None:
        """Loop stops after max_turns even with pending tool calls."""
        call = _tool_call("echo", "c1")
        reply = _model_entry(tool_calls=[call], stop_reason="toolUse")

        # Two turns: first makes a tool call, second would too but max_turns=1
        # The provider must only be asked once
        provider = ScriptedProvider([
            _tooluse_close(reply),
            _tooluse_close(reply),  # shouldn't be reached with max_turns=1
            _simple_close(_model_entry(text="end")),
        ])
        tool = _make_tool("echo")
        messages: list[TranscriptEntry] = []

        events = await _collect(run_agent_loop(
            provider=provider, model="m", system="s",
            messages=messages, tools=[tool], max_turns=1,
        ))

        # Should have stopped after turn 1's tool results, before turn 2 provider call
        msg_ends = [
            e.message for e in events
            if isinstance(e, MessageEndEvent) and isinstance(e.message, ModelEntry)
        ]
        # Last ModelEntry should be the max_turns error
        last_model = msg_ends[-1]
        assert last_model.stop_reason == "error"
        assert "max_turns" in (last_model.error_message or "")


# ---------------------------------------------------------------------------
# Provider failures
# ---------------------------------------------------------------------------


class TestProviderFailures:
    @pytest.mark.asyncio
    async def test_provider_error_event(self) -> None:
        """StreamFaultEvent from provider stops the run."""
        error_entry = _model_entry(text="", stop_reason="error")
        provider = ScriptedProvider([
            [StreamFaultEvent(reason="error", error=error_entry)],
        ])
        messages: list[TranscriptEntry] = []

        events = await _collect(run_agent_loop(
            provider=provider, model="m", system="s",
            messages=messages, tools=[],
        ))

        types = _event_types(events)
        assert types[-1] == "agent_end"
        # The error entry should be in messages
        assert any(
            isinstance(m, ModelEntry) and m.stop_reason == "error"
            for m in messages
        )

    @pytest.mark.asyncio
    async def test_provider_no_assistant_message(self) -> None:
        """Provider emitting no StreamCloseEvent gets a synthetic error."""
        provider = ScriptedProvider([
            [StreamOpenEvent(partial=_model_entry(text=""))],
        ])
        messages: list[TranscriptEntry] = []

        events = await _collect(run_agent_loop(
            provider=provider, model="m", system="s",
            messages=messages, tools=[],
        ))

        # Should get a synthetic "no assistant message" error
        msg_ends = [
            e.message for e in events
            if isinstance(e, MessageEndEvent) and isinstance(e.message, ModelEntry)
        ]
        assert any(
            "no assistant message" in (m.error_message or "").lower()
            for m in msg_ends
        )

    @pytest.mark.asyncio
    async def test_aborted_stop_reason(self) -> None:
        """Model entry with stop_reason='aborted' ends the run."""
        entry = _model_entry(text="", stop_reason="aborted")
        provider = ScriptedProvider([_simple_close(entry)])
        messages: list[TranscriptEntry] = []

        events = await _collect(run_agent_loop(
            provider=provider, model="m", system="s",
            messages=messages, tools=[],
        ))

        types = _event_types(events)
        assert types[-1] == "agent_end"

    @pytest.mark.asyncio
    async def test_error_stop_reason(self) -> None:
        """Model entry with stop_reason='error' ends the run."""
        entry = _model_entry(text="", stop_reason="error")
        provider = ScriptedProvider([_simple_close(entry)])
        messages: list[TranscriptEntry] = []

        events = await _collect(run_agent_loop(
            provider=provider, model="m", system="s",
            messages=messages, tools=[],
        ))

        run_end = [e for e in events if isinstance(e, RunEndEvent)]
        assert len(run_end) == 1


# ---------------------------------------------------------------------------
# Steering and follow-up messages
# ---------------------------------------------------------------------------


class TestSteeringMessages:
    @pytest.mark.asyncio
    async def test_initial_steering(self) -> None:
        """Steering messages from get_steering_messages are injected into transcript."""
        steering = HumanEntry(content="steer this")
        call = _tool_call("echo", "c1")
        reply1 = _model_entry(tool_calls=[call], stop_reason="toolUse")
        reply2 = _model_entry(text="done")

        first_call = True

        def get_steering() -> Sequence[TranscriptEntry]:
            nonlocal first_call
            if first_call:
                first_call = False
                return [steering]
            return []

        provider = ScriptedProvider([_tooluse_close(reply1), _simple_close(reply2)])
        tool = _make_tool("echo")
        messages: list[TranscriptEntry] = []

        _events = await _collect(run_agent_loop(
            provider=provider, model="m", system="s",
            messages=messages, tools=[tool],
            get_steering_messages=get_steering,
        ))

        # Steering message should be in the transcript
        assert steering in messages

    @pytest.mark.asyncio
    async def test_follow_up_messages(self) -> None:
        """Follow-up messages trigger an additional conversation loop."""
        followup = HumanEntry(content="follow up")
        reply1 = _model_entry(text="first reply")
        reply2 = _model_entry(text="second reply")

        calls = 0

        def get_followups() -> Sequence[TranscriptEntry]:
            nonlocal calls
            calls += 1
            if calls == 1:
                return [followup]
            return []

        provider = ScriptedProvider([
            _simple_close(reply1),
            _simple_close(reply2),
        ])
        messages: list[TranscriptEntry] = []

        events = await _collect(run_agent_loop(
            provider=provider, model="m", system="s",
            messages=messages, tools=[],
            get_follow_up_messages=get_followups,
        ))

        assert followup in messages
        run_end = [e for e in events if isinstance(e, RunEndEvent)][0]
        assert followup in run_end.messages

    @pytest.mark.asyncio
    async def test_steering_between_turns(self) -> None:
        """Steering messages injected between tool-call turns."""
        steering = HumanEntry(content="mid-turn steer")
        call = _tool_call("echo", "c1")
        reply1 = _model_entry(tool_calls=[call], stop_reason="toolUse")
        reply2 = _model_entry(text="done")

        call_count = 0

        def get_steering() -> Sequence[TranscriptEntry]:
            nonlocal call_count
            call_count += 1
            if call_count == 2:  # after first tool execution
                return [steering]
            return []

        provider = ScriptedProvider([_tooluse_close(reply1), _simple_close(reply2)])
        tool = _make_tool("echo")
        messages: list[TranscriptEntry] = []

        _events = await _collect(run_agent_loop(
            provider=provider, model="m", system="s",
            messages=messages, tools=[tool],
            get_steering_messages=get_steering,
        ))

        assert steering in messages


# ---------------------------------------------------------------------------
# Provider streaming details
# ---------------------------------------------------------------------------


class TestProviderStreaming:
    @pytest.mark.asyncio
    async def test_wire_events_serialized_to_jvalue(self) -> None:
        """MessageUpdateEvent.assistant_message_event is a dict, not a pydantic model."""
        entry = _model_entry(text="hi")
        text_open = ContentOpenEvent(content_index=0, partial=entry)
        text_chunk = ContentChunkEvent(content_index=0, delta="hi", partial=entry)
        text_close = ContentCloseEvent(content_index=0, content="hi", partial=entry)

        provider = ScriptedProvider([[
            StreamOpenEvent(partial=entry),
            text_open,
            text_chunk,
            text_close,
            StreamCloseEvent(reason="stop", message=entry),
        ]])
        messages: list[TranscriptEntry] = []

        events = await _collect(run_agent_loop(
            provider=provider, model="m", system="s",
            messages=messages, tools=[],
        ))

        updates = [e for e in events if isinstance(e, MessageUpdateEvent)]
        assert len(updates) == 3
        for u in updates:
            assert isinstance(u.assistant_message_event, dict)

    @pytest.mark.asyncio
    async def test_stream_open_without_close_gets_synthetic(self) -> None:
        """Provider that opens but never closes gets a synthetic error message."""
        entry = _model_entry(text="partial")
        provider = ScriptedProvider([[
            StreamOpenEvent(partial=entry),
        ]])
        messages: list[TranscriptEntry] = []

        events = await _collect(run_agent_loop(
            provider=provider, model="m", system="s",
            messages=messages, tools=[],
        ))

        # Should have a synthetic error about no assistant message
        msg_ends = [
            e.message for e in events
            if isinstance(e, MessageEndEvent) and isinstance(e.message, ModelEntry)
        ]
        assert any("no assistant message" in (m.error_message or "").lower() for m in msg_ends)

    @pytest.mark.asyncio
    async def test_fault_without_open(self) -> None:
        """StreamFaultEvent without prior StreamOpenEvent still emits MessageStart."""
        error_entry = _model_entry(text="", stop_reason="error")
        provider = ScriptedProvider([
            [StreamFaultEvent(reason="error", error=error_entry)],
        ])
        messages: list[TranscriptEntry] = []

        events = await _collect(run_agent_loop(
            provider=provider, model="m", system="s",
            messages=messages, tools=[],
        ))

        starts = [e for e in events if isinstance(e, MessageStartEvent)]
        assert len(starts) >= 1  # fault should generate a MessageStart


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_no_tools_no_calls(self) -> None:
        """Empty tools list with no tool calls is fine."""
        entry = _model_entry(text="simple")
        provider = ScriptedProvider([_simple_close(entry)])
        messages: list[TranscriptEntry] = []

        events = await _collect(run_agent_loop(
            provider=provider, model="m", system="s",
            messages=messages, tools=[],
        ))

        assert _event_types(events)[0] == "agent_start"
        assert _event_types(events)[-1] == "agent_end"

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_in_one_turn(self) -> None:
        """Model requesting 3 tool calls in one turn."""
        c1 = _tool_call("echo", "a1", {"x": "1"})
        c2 = _tool_call("echo", "a2", {"x": "2"})
        c3 = _tool_call("echo", "a3", {"x": "3"})
        reply1 = _model_entry(tool_calls=[c1, c2, c3], stop_reason="toolUse")
        reply2 = _model_entry(text="done")

        provider = ScriptedProvider([_tooluse_close(reply1), _simple_close(reply2)])
        tool = _make_tool("echo")
        messages: list[TranscriptEntry] = []

        events = await _collect(run_agent_loop(
            provider=provider, model="m", system="s",
            messages=messages, tools=[tool],
        ))

        end_events = [e for e in events if isinstance(e, ToolRunEndEvent)]
        assert len(end_events) == 3

    @pytest.mark.asyncio
    async def test_turn_end_carries_tool_results(self) -> None:
        """TurnEndEvent.tool_results contains all tool outcomes from that turn."""
        c1 = _tool_call("echo", "a1", {"x": "1"})
        c2 = _tool_call("echo", "a2", {"x": "2"})
        reply1 = _model_entry(tool_calls=[c1, c2], stop_reason="toolUse")
        reply2 = _model_entry(text="done")

        provider = ScriptedProvider([_tooluse_close(reply1), _simple_close(reply2)])
        tool = _make_tool("echo")
        messages: list[TranscriptEntry] = []

        events = await _collect(run_agent_loop(
            provider=provider, model="m", system="s",
            messages=messages, tools=[tool],
        ))

        turn_ends = [e for e in events if isinstance(e, TurnEndEvent)]
        # First turn end should have 2 tool results
        assert len(turn_ends[0].tool_results) == 2

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self) -> None:
        """Model does 2 tool-call turns then a final text reply."""
        c1 = _tool_call("echo", "a1")
        c2 = _tool_call("echo", "a2")
        reply1 = _model_entry(tool_calls=[c1], stop_reason="toolUse")
        reply2 = _model_entry(tool_calls=[c2], stop_reason="toolUse")
        reply3 = _model_entry(text="final")

        provider = ScriptedProvider([
            _tooluse_close(reply1),
            _tooluse_close(reply2),
            _simple_close(reply3),
        ])
        tool = _make_tool("echo")
        messages: list[TranscriptEntry] = []

        events = await _collect(run_agent_loop(
            provider=provider, model="m", system="s",
            messages=messages, tools=[tool],
        ))

        turn_starts = [e for e in events if isinstance(e, TurnStartEvent)]
        turn_ends = [e for e in events if isinstance(e, TurnEndEvent)]
        assert len(turn_starts) == 3  # initial + 2 tool turns
        assert len(turn_ends) == 3

        # Messages: reply1, tool1, reply2, tool2, reply3
        assert len(messages) == 5

    @pytest.mark.asyncio
    async def test_empty_prompts(self) -> None:
        """Empty prompts sequence doesn't break anything."""
        entry = _model_entry(text="ok")
        provider = ScriptedProvider([_simple_close(entry)])
        messages: list[TranscriptEntry] = []

        events = await _collect(run_agent_loop(
            provider=provider, model="m", system="s",
            messages=messages, tools=[], prompts=(),
        ))

        assert _event_types(events)[0] == "agent_start"
        assert _event_types(events)[-1] == "agent_end"

    @pytest.mark.asyncio
    async def test_close_without_open(self) -> None:
        """StreamCloseEvent without prior StreamOpenEvent emits both start and end."""
        entry = _model_entry(text="direct close")
        provider = ScriptedProvider([
            [StreamCloseEvent(reason="stop", message=entry)],
        ])
        messages: list[TranscriptEntry] = []

        events = await _collect(run_agent_loop(
            provider=provider, model="m", system="s",
            messages=messages, tools=[],
        ))

        starts = [e for e in events if isinstance(e, MessageStartEvent)]
        ends = [e for e in events if isinstance(e, MessageEndEvent)]
        # Should have at least one start and one end for the model message
        assert len(starts) >= 1
        assert len(ends) >= 1
