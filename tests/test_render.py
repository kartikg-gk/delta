"""Tests for delta_app.ui.render."""

from __future__ import annotations

import io
import json

import pytest

from delta_app.ui.render import (
    FinalTextRenderer,
    JsonRenderer,
    OutputMode,
    TranscriptRenderer,
    make_renderer,
)
from delta_harness.contracts.stream import (
    MessageEndEvent,
    MessageUpdateEvent,
    RunStartEvent,
    ToolRunEndEvent,
    ToolRunStartEvent,
)
from delta_harness.contracts.tooling import ToolOutcome
from delta_harness.contracts.transcript import ModelEntry, TextSegment


def _model_entry(text: str = "hello", **kw) -> ModelEntry:
    content = [TextSegment(text=text)] if text else []
    return ModelEntry(content=content, **kw)


def _end(entry: ModelEntry | None = None) -> MessageEndEvent:
    return MessageEndEvent(message=entry or _model_entry())


def _update(text: str) -> MessageUpdateEvent:
    return MessageUpdateEvent(message=_model_entry(text))


# -- OutputMode / make_renderer ---------------------------------------------


@pytest.mark.parametrize("mode,cls", [
    (OutputMode.JSON, JsonRenderer),
    (OutputMode.TEXT, FinalTextRenderer),
    (OutputMode.TRANSCRIPT, TranscriptRenderer),
])
def test_make_renderer(mode, cls):
    assert isinstance(make_renderer(mode), cls)


@pytest.mark.parametrize("mode", list(OutputMode))
def test_default_stdout_resolved_at_call_time(mode, capsys):
    """Default output must not bind sys.stdout at import time (breaks capture)."""
    r = make_renderer(mode)
    r.render(_end(_model_entry("captured")))
    r.finish()
    assert "captured" in capsys.readouterr().out


# -- JsonRenderer -----------------------------------------------------------


class TestJsonRenderer:
    def test_writes_json_lines(self):
        buf = io.StringIO()
        r = JsonRenderer(buf)
        r.render(RunStartEvent())
        line = json.loads(buf.getvalue().strip())
        assert line["type"] == "agent_start"

    def test_success_on_normal_end(self):
        r = JsonRenderer(io.StringIO())
        r.render(_end())
        assert r.finish() is True

    def test_failure_on_error(self):
        r = JsonRenderer(io.StringIO())
        r.render(_end(_model_entry(stop_reason="error")))
        assert r.finish() is False


# -- FinalTextRenderer ------------------------------------------------------


class TestFinalTextRenderer:
    def test_prints_last_text(self):
        buf = io.StringIO()
        r = FinalTextRenderer(buf)
        r.render(_end(_model_entry("first")))
        r.render(_end(_model_entry("second")))
        assert r.finish() is True
        assert "second" in buf.getvalue()
        assert "first" not in buf.getvalue()

    def test_prints_errors(self):
        buf = io.StringIO()
        r = FinalTextRenderer(buf)
        r.render(_end(_model_entry(error_message="boom")))
        assert r.finish() is False
        assert "boom" in buf.getvalue()

    def test_ignores_non_end_events(self):
        buf = io.StringIO()
        r = FinalTextRenderer(buf)
        r.render(RunStartEvent())
        assert r.finish() is True
        assert buf.getvalue() == ""


# -- TranscriptRenderer ----------------------------------------------------


class TestTranscriptRenderer:
    def test_streams_text_deltas(self):
        buf = io.StringIO()
        r = TranscriptRenderer(buf)
        r.render(_update("he"))
        r.render(_update("hello"))
        assert buf.getvalue() == "hello"

    def test_tool_events(self):
        buf = io.StringIO()
        r = TranscriptRenderer(buf)
        r.render(ToolRunStartEvent(tool_call_id="t1", tool_name="Bash"))
        r.render(ToolRunEndEvent(
            tool_call_id="t1", tool_name="Bash",
            result=ToolOutcome(content="ok"), is_error=False,
        ))
        out = buf.getvalue()
        assert "[tool:Bash] running" in out
        assert "[tool:Bash] done" in out

    def test_error_stop_reason(self):
        buf = io.StringIO()
        r = TranscriptRenderer(buf)
        r.render(_end(_model_entry(stop_reason="error", error_message="fail")))
        assert r.finish() is False
        assert "fail" in buf.getvalue()

    def test_finish_success(self):
        r = TranscriptRenderer(io.StringIO())
        r.render(_end())
        assert r.finish() is True
