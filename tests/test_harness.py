"""Tests for delta_harness.driver."""

from __future__ import annotations

import pytest

from delta_harness.contracts.transcript import (
    CallBlock,
    HumanEntry,
    ModelEntry,
    ToolOutcomeEntry,
)
from delta_harness.driver import (
    ManualCancelFlag,
    PendingBatch,
    RuntimeConfig,
    RuntimeHarness,
)


class _StubProvider:
    """Duck-typed provider that is never actually called."""

    async def stream_response(self, **kw):
        return
        yield  # unreachable — makes it an async generator


def _cfg() -> RuntimeConfig:
    return RuntimeConfig(provider=_StubProvider(), model="m", system="s")


def _harness(**kw) -> RuntimeHarness:
    return RuntimeHarness(_cfg(), **kw)


# -- ManualCancelFlag -------------------------------------------------------


class TestManualCancelFlag:
    def test_lifecycle(self):
        flag = ManualCancelFlag()
        assert not flag.is_cancelled()
        flag.cancel()
        assert flag.is_cancelled()


# -- PendingBatch -----------------------------------------------------------


class TestPendingBatch:
    def test_count(self):
        b = PendingBatch(steering=(HumanEntry(content="a"),), follow_up=())
        assert b.count == 1


# -- RuntimeHarness ---------------------------------------------------------


class TestRuntimeHarness:
    def test_initial_state(self):
        h = _harness()
        assert h.transcript == ()
        assert not h.active
        assert not h.has_pending()

    def test_push_and_set_messages(self):
        h = _harness()
        e = HumanEntry(content="hi")
        h.push_message(e)
        assert len(h.transcript) == 1
        h.set_messages([])
        assert h.transcript == ()

    def test_restore_messages(self):
        msgs = [HumanEntry(content="a"), HumanEntry(content="b")]
        h = _harness(messages=msgs)
        assert len(h.transcript) == 2

    def test_on_event_unsubscribe(self):
        h = _harness()
        calls = []
        unsub = h.on_event(lambda e: calls.append(e))
        unsub()
        # double-unsub is safe
        unsub()

    # -- queues -------------------------------------------------------------

    def test_inject_and_retract(self):
        h = _harness()
        h.inject("steer1")
        h.inject("steer2")
        assert h.pending_count == 2
        retracted = h.retract_injected()
        assert retracted is not None and retracted.content == "steer2"

    def test_enqueue_and_retract(self):
        h = _harness()
        h.enqueue("f1")
        assert h.has_pending()
        retracted = h.retract_enqueued()
        assert retracted is not None
        assert not h.has_pending()

    def test_retract_empty(self):
        h = _harness()
        assert h.retract_injected() is None
        assert h.retract_enqueued() is None

    def test_flush_queues(self):
        h = _harness()
        h.inject("s")
        h.enqueue("f")
        batch = h.flush_queues()
        assert batch.count == 2
        assert not h.has_pending()

    # -- abort / idle guard -------------------------------------------------

    def test_abort_when_idle(self):
        h = _harness()
        h.abort()  # no-op, no crash

    def test_assert_idle_blocks_double_submit(self):
        h = _harness()
        h._running = True
        with pytest.raises(RuntimeError, match="already running"):
            h.submit("x")

    # -- dangling call healing ---------------------------------------------

    def test_heal_dangling_calls(self):
        call = CallBlock(name="bash", id="c1")
        h = _harness(messages=[
            ModelEntry(content=[call]),
        ])
        patched = h.patch_interrupted_calls()
        assert patched == 1
        outcome = h.transcript[-1]
        assert isinstance(outcome, ToolOutcomeEntry)
        assert outcome.tool_call_id == "c1"
        assert outcome.is_error

    def test_no_patch_when_already_answered(self):
        call = CallBlock(name="bash", id="c2")
        h = _harness(messages=[
            ModelEntry(content=[call]),
            ToolOutcomeEntry(tool_call_id="c2", tool_name="bash", content="ok"),
        ])
        assert h.patch_interrupted_calls() == 0

    # -- drain policy -------------------------------------------------------

    def test_drain_all(self):
        cfg = _cfg()
        cfg.queue_mode = "all"
        h = RuntimeHarness(cfg)
        h.inject("a")
        h.inject("b")
        drained = h._pop_steered()
        assert len(drained) == 2

    def test_drain_one_at_a_time(self):
        h = _harness()
        h.inject("a")
        h.inject("b")
        drained = h._pop_steered()
        assert len(drained) == 1
        assert h.pending_count == 1
