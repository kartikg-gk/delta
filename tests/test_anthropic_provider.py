"""Tests for the Anthropic Messages API provider adapter and its sub-modules."""

from __future__ import annotations

from typing import Any

from delta_harness.contracts.tooling import ToolOutcome, ToolSpec
from delta_harness.contracts.transcript import (
    CallBlock,
    HumanEntry,
    ImageSegment,
    ModelEntry,
    TextSegment,
    ThoughtSegment,
    ToolOutcomeEntry,
)
from delta_harness.provider.wire import (
    CallChunkEvent,
    CallCloseEvent,
    CallOpenEvent,
    ContentChunkEvent,
    ContentCloseEvent,
    ContentOpenEvent,
    ReasoningChunkEvent,
    ReasoningCloseEvent,
    ReasoningOpenEvent,
    StreamCloseEvent,
    StreamFaultEvent,
    StreamOpenEvent,
)
from delta_model._claude.courier import ApiRejection
from delta_model._claude.emitter import (
    ResponseMachine,
    _map_stop_reason,
    _safe_json_args,
    _tally_usage,
)
from delta_model.claude import (
    _assistant_blocks,
    _compile_messages,
    _compose_body,
    _thinking_section,
    _tool_result_block,
    _tool_schema,
    _user_content,
)
from delta_model.settings import ReasoningPolicy

# ===========================================================================
# emitter.py — helpers
# ===========================================================================


class TestMapStopReason:
    def test_end_turn(self) -> None:
        assert _map_stop_reason("end_turn") == "stop"

    def test_stop_sequence(self) -> None:
        assert _map_stop_reason("stop_sequence") == "stop"

    def test_max_tokens(self) -> None:
        assert _map_stop_reason("max_tokens") == "length"

    def test_tool_use(self) -> None:
        assert _map_stop_reason("tool_use") == "toolUse"

    def test_unknown(self) -> None:
        assert _map_stop_reason("whatever") == "stop"


class TestSafeJsonArgs:
    def test_valid(self) -> None:
        assert _safe_json_args('{"x": 1}') == {"x": 1}

    def test_empty(self) -> None:
        assert _safe_json_args("") == {}

    def test_invalid(self) -> None:
        assert _safe_json_args("{broken") == {}

    def test_non_object(self) -> None:
        assert _safe_json_args("[1, 2]") == {}


class TestTallyUsage:
    def test_basic(self) -> None:
        usage = _tally_usage(input_tokens=100, output_tokens=50)
        assert usage.input == 100
        assert usage.output == 50
        assert usage.total_tokens == 150

    def test_cache(self) -> None:
        usage = _tally_usage(
            input_tokens=100, output_tokens=50,
            cache_read=20, cache_write=10,
        )
        assert usage.cache_read == 20
        assert usage.cache_write == 10


# ===========================================================================
# emitter.py — ResponseMachine
# ===========================================================================


class TestResponseMachine:
    def _machine(self) -> ResponseMachine:
        return ResponseMachine(model="claude-sonnet-4-20250514", provider="anthropic")

    def test_message_start_emits_open(self) -> None:
        m = self._machine()
        events = m.ingest("message_start", {
            "message": {
                "id": "msg_1",
                "model": "claude-sonnet-4-20250514",
                "usage": {"input_tokens": 25, "output_tokens": 1},
            },
        })
        assert len(events) == 1
        assert isinstance(events[0], StreamOpenEvent)

    def test_text_streaming(self) -> None:
        m = self._machine()
        m.ingest("message_start", {"message": {"id": "msg_1", "model": "c", "usage": {}}})

        e1 = m.ingest("content_block_start", {
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        })
        assert any(isinstance(e, ContentOpenEvent) for e in e1)

        e2 = m.ingest("content_block_delta", {
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello"},
        })
        chunks = [e for e in e2 if isinstance(e, ContentChunkEvent)]
        assert len(chunks) == 1
        assert chunks[0].delta == "Hello"

        e3 = m.ingest("content_block_stop", {"index": 0})
        closes = [e for e in e3 if isinstance(e, ContentCloseEvent)]
        assert len(closes) == 1
        assert closes[0].content == "Hello"

    def test_thinking_streaming(self) -> None:
        m = self._machine()
        m.ingest("message_start", {"message": {"id": "m1", "model": "c", "usage": {}}})

        e1 = m.ingest("content_block_start", {
            "index": 0,
            "content_block": {"type": "thinking", "thinking": ""},
        })
        assert any(isinstance(e, ReasoningOpenEvent) for e in e1)

        e2 = m.ingest("content_block_delta", {
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "Let me think..."},
        })
        assert any(isinstance(e, ReasoningChunkEvent) for e in e2)

        e3 = m.ingest("content_block_stop", {"index": 0})
        closes = [e for e in e3 if isinstance(e, ReasoningCloseEvent)]
        assert closes[0].content == "Let me think..."

    def test_tool_use_streaming(self) -> None:
        m = self._machine()
        m.ingest("message_start", {"message": {"id": "m1", "model": "c", "usage": {}}})

        e1 = m.ingest("content_block_start", {
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "get_weather",
                "input": {},
            },
        })
        assert any(isinstance(e, CallOpenEvent) for e in e1)

        e2 = m.ingest("content_block_delta", {
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"ci'},
        })
        call_chunks = [e for e in e2 if isinstance(e, CallChunkEvent)]
        assert len(call_chunks) == 1

        m.ingest("content_block_delta", {
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": 'ty":"NYC"}'},
        })

        e3 = m.ingest("content_block_stop", {"index": 0})
        closes = [e for e in e3 if isinstance(e, CallCloseEvent)]
        assert len(closes) == 1
        assert closes[0].tool_call.name == "get_weather"
        assert closes[0].tool_call.arguments == {"city": "NYC"}

    def test_message_delta_captures_stop_and_usage(self) -> None:
        m = self._machine()
        m.ingest("message_start", {"message": {"id": "m1", "model": "c", "usage": {"input_tokens": 10}}})
        m.ingest("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}})
        m.ingest("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "Hi"}})
        m.ingest("content_block_stop", {"index": 0})

        m.ingest("message_delta", {
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 5},
        })

        events = m.seal()
        close = [e for e in events if isinstance(e, StreamCloseEvent)]
        assert len(close) == 1
        assert close[0].reason == "stop"
        assert close[0].message.usage.input == 10
        assert close[0].message.usage.output == 5

    def test_message_stop_seals(self) -> None:
        m = self._machine()
        m.ingest("message_start", {"message": {"id": "m1", "model": "c", "usage": {}}})
        m.ingest("message_delta", {"delta": {"stop_reason": "end_turn"}, "usage": {}})

        events = m.ingest("message_stop", {})
        close = [e for e in events if isinstance(e, StreamCloseEvent)]
        assert len(close) == 1

    def test_seal_idempotent(self) -> None:
        m = self._machine()
        m.ingest("message_start", {"message": {"id": "m1", "model": "c", "usage": {}}})

        first = m.seal()
        second = m.seal()
        assert len(first) == 1
        assert len(second) == 0

    def test_tool_use_stop_reason(self) -> None:
        m = self._machine()
        m.ingest("message_start", {"message": {"id": "m1", "model": "c", "usage": {}}})
        m.ingest("message_delta", {"delta": {"stop_reason": "tool_use"}, "usage": {}})
        events = m.seal()
        close = [e for e in events if isinstance(e, StreamCloseEvent)]
        assert close[0].reason == "toolUse"

    def test_length_stop_reason(self) -> None:
        m = self._machine()
        m.ingest("message_start", {"message": {"id": "m1", "model": "c", "usage": {}}})
        m.ingest("message_delta", {"delta": {"stop_reason": "max_tokens"}, "usage": {}})
        events = m.seal()
        close = [e for e in events if isinstance(e, StreamCloseEvent)]
        assert close[0].reason == "length"

    def test_error_event(self) -> None:
        m = self._machine()
        events = m.ingest("error", {
            "error": {"type": "overloaded_error", "message": "Overloaded"},
        })
        # Should open + fault since we were in PENDING phase
        assert any(isinstance(e, StreamOpenEvent) for e in events)
        faults = [e for e in events if isinstance(e, StreamFaultEvent)]
        assert len(faults) == 1
        assert "Overloaded" in faults[0].error.error_message

    def test_error_after_streaming(self) -> None:
        m = self._machine()
        m.ingest("message_start", {"message": {"id": "m1", "model": "c", "usage": {}}})
        events = m.ingest("error", {
            "error": {"message": "Something went wrong"},
        })
        # Should NOT emit a second StreamOpenEvent
        opens = [e for e in events if isinstance(e, StreamOpenEvent)]
        assert len(opens) == 0
        faults = [e for e in events if isinstance(e, StreamFaultEvent)]
        assert len(faults) == 1

    def test_ping_ignored(self) -> None:
        m = self._machine()
        assert m.ingest("ping", {}) == []

    def test_snapshot_includes_active_text(self) -> None:
        m = self._machine()
        m.ingest("message_start", {"message": {"id": "m1", "model": "c", "usage": {}}})
        m.ingest("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}})
        events = m.ingest("content_block_delta", {
            "index": 0, "delta": {"type": "text_delta", "text": "partial"},
        })
        # The snapshot in the chunk event should include the partial text
        chunk = [e for e in events if isinstance(e, ContentChunkEvent)][0]
        assert chunk.partial.text == "partial"

    def test_cache_tokens_captured(self) -> None:
        m = self._machine()
        m.ingest("message_start", {
            "message": {
                "id": "m1", "model": "c",
                "usage": {
                    "input_tokens": 100,
                    "cache_read_input_tokens": 30,
                    "cache_creation_input_tokens": 10,
                },
            },
        })
        m.ingest("message_delta", {"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}})
        events = m.seal()
        close = [e for e in events if isinstance(e, StreamCloseEvent)][0]
        assert close.message.usage.cache_read == 30
        assert close.message.usage.cache_write == 10


# ===========================================================================
# anthropic.py — request construction
# ===========================================================================


class TestUserContent:
    def test_string(self) -> None:
        entry = HumanEntry(content="hello")
        blocks = _user_content(entry)
        assert blocks == [{"type": "text", "text": "hello"}]

    def test_multimodal(self) -> None:
        entry = HumanEntry(content=[
            TextSegment(text="look at this"),
            ImageSegment(mime_type="image/png", data="abc123"),
        ])
        blocks = _user_content(entry)
        assert blocks[0] == {"type": "text", "text": "look at this"}
        assert blocks[1]["type"] == "image"
        assert blocks[1]["source"]["media_type"] == "image/png"


class TestAssistantBlocks:
    def test_text_only(self) -> None:
        entry = ModelEntry(content=[TextSegment(text="Hi")])
        blocks = _assistant_blocks(entry)
        assert blocks == [{"type": "text", "text": "Hi"}]

    def test_with_tool_use(self) -> None:
        entry = ModelEntry(content=[
            TextSegment(text="Let me check."),
            CallBlock(id="t1", name="get_weather", arguments={"city": "NYC"}),
        ])
        blocks = _assistant_blocks(entry)
        assert len(blocks) == 2
        assert blocks[1]["type"] == "tool_use"
        assert blocks[1]["name"] == "get_weather"
        assert blocks[1]["input"] == {"city": "NYC"}

    def test_thinking_with_signature(self) -> None:
        entry = ModelEntry(content=[
            ThoughtSegment(thinking="I should think", thinking_signature="sig123"),
            TextSegment(text="Answer"),
        ])
        blocks = _assistant_blocks(entry)
        assert len(blocks) == 2
        assert blocks[0]["type"] == "thinking"
        assert blocks[0]["signature"] == "sig123"

    def test_thinking_without_signature_skipped(self) -> None:
        entry = ModelEntry(content=[
            ThoughtSegment(thinking="No sig"),
            TextSegment(text="Answer"),
        ])
        blocks = _assistant_blocks(entry)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"

    def test_empty_content_gets_placeholder(self) -> None:
        entry = ModelEntry(content=[])
        blocks = _assistant_blocks(entry)
        assert blocks == [{"type": "text", "text": ""}]


class TestToolResultBlock:
    def test_basic(self) -> None:
        entry = ToolOutcomeEntry(
            tool_call_id="t1",
            tool_name="get_weather",
            content=[TextSegment(text="Sunny")],
        )
        block = _tool_result_block(entry)
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "t1"
        assert block["content"] == "Sunny"
        assert "is_error" not in block

    def test_error(self) -> None:
        entry = ToolOutcomeEntry(
            tool_call_id="t1",
            tool_name="get_weather",
            content=[TextSegment(text="Failed")],
            is_error=True,
        )
        block = _tool_result_block(entry)
        assert block["is_error"] is True


class TestCompileMessages:
    def test_simple_conversation(self) -> None:
        messages = _compile_messages([
            HumanEntry(content="Hello"),
            ModelEntry(content=[TextSegment(text="Hi!")]),
            HumanEntry(content="How are you?"),
        ])
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"

    def test_consecutive_user_merged(self) -> None:
        messages = _compile_messages([
            HumanEntry(content="Hello"),
            HumanEntry(content="Also this"),
        ])
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert len(messages[0]["content"]) == 2

    def test_tool_results_grouped(self) -> None:
        messages = _compile_messages([
            ModelEntry(content=[
                CallBlock(id="t1", name="a", arguments={}),
                CallBlock(id="t2", name="b", arguments={}),
            ]),
            ToolOutcomeEntry(tool_call_id="t1", tool_name="a", content=[TextSegment(text="r1")]),
            ToolOutcomeEntry(tool_call_id="t2", tool_name="b", content=[TextSegment(text="r2")]),
        ])
        assert len(messages) == 2
        assert messages[0]["role"] == "assistant"
        assert messages[1]["role"] == "user"
        assert len(messages[1]["content"]) == 2
        assert messages[1]["content"][0]["type"] == "tool_result"
        assert messages[1]["content"][1]["type"] == "tool_result"


class TestToolSchema:
    def test_conversion(self) -> None:
        async def _run(cid: str, args: Any, sig: Any = None, upd: Any = None) -> ToolOutcome:
            return ToolOutcome(content=[TextSegment(text="ok")])

        tool = ToolSpec(
            name="search",
            label="Search",
            description="Search the web",
            parameters={"type": "object", "properties": {"q": {"type": "string"}}},
            run=_run,
        )
        schema = _tool_schema(tool)
        assert schema["name"] == "search"
        assert schema["input_schema"]["type"] == "object"


class TestThinkingSection:
    def test_disabled(self) -> None:
        assert _thinking_section(ReasoningPolicy(), 16384) is None

    def test_enabled_with_budget(self) -> None:
        cfg = _thinking_section(ReasoningPolicy(enabled=True, budget_tokens=5000), 16384)
        assert cfg is not None
        assert cfg["type"] == "enabled"
        assert cfg["budget_tokens"] == 5000

    def test_enabled_default_budget(self) -> None:
        cfg = _thinking_section(ReasoningPolicy(enabled=True), 16384)
        assert cfg is not None
        assert cfg["budget_tokens"] == 16384 - 1024


class TestComposeBody:
    def test_basic(self) -> None:
        body = _compose_body(
            model="claude-sonnet-4-20250514",
            system="Be helpful.",
            messages=[HumanEntry(content="Hello")],
            tools=[],
            max_tokens=4096,
            thinking=ReasoningPolicy(),
        )
        assert body["model"] == "claude-sonnet-4-20250514"
        assert body["system"] == "Be helpful."
        assert body["stream"] is True
        assert body["max_tokens"] == 4096
        assert "thinking" not in body
        assert "tools" not in body

    def test_with_thinking(self) -> None:
        body = _compose_body(
            model="claude-sonnet-4-20250514",
            system="sys",
            messages=[],
            tools=[],
            max_tokens=8192,
            thinking=ReasoningPolicy(enabled=True, budget_tokens=4000),
        )
        assert body["thinking"]["type"] == "enabled"
        assert body["thinking"]["budget_tokens"] == 4000


# ===========================================================================
# courier.py — ApiRejection
# ===========================================================================


class TestApiRejection:
    def test_retriable_429(self) -> None:
        assert ApiRejection(429, "rate limit").retriable is True

    def test_retriable_529(self) -> None:
        assert ApiRejection(529, "overloaded").retriable is True

    def test_not_retriable_400(self) -> None:
        assert ApiRejection(400, "bad request").retriable is False

    def test_not_retriable_401(self) -> None:
        assert ApiRejection(401, "unauthorized").retriable is False

    def test_wait_hint(self) -> None:
        err = ApiRejection(429, "limit", wait_hint=3.0)
        assert err.wait_hint == 3.0
