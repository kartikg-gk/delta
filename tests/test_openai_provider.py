"""Tests for the OpenAI-compatible provider adapter and its sub-modules."""

from __future__ import annotations

from typing import Any

from delta_harness.contracts.tooling import ToolOutcome, ToolSpec
from delta_harness.contracts.transcript import (
    CallBlock,
    HumanEntry,
    ModelEntry,
    TextSegment,
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
from delta_model._oai.helpers import (
    build_reasoning_extra,
    extract_usage,
    map_halt_reason,
    parse_retry_after,
    parse_tool_arguments,
    pick_endpoint,
    try_parse_json,
)
from delta_model._oai.normalize import EventAssembler
from delta_model._oai.parsers import (
    CallArgFragment,
    CallBegin,
    CallResolved,
    ChatDecoder,
    Finished,
    Metadata,
    ParseFault,
    ReasoningChunk,
    ResponsesDecoder,
    TextChunk,
)
from delta_model._oai.payloads import build_chat_payload, build_responses_payload
from delta_model._oai.transport import HttpStreamError, MalformedPayload
from delta_model.settings import ReasoningPolicy

# ===========================================================================
# helpers.py
# ===========================================================================


class TestMapHaltReason:
    def test_stop(self) -> None:
        assert map_halt_reason("stop") == "stop"

    def test_none(self) -> None:
        assert map_halt_reason(None) == "stop"

    def test_length(self) -> None:
        assert map_halt_reason("length") == "length"

    def test_max_tokens(self) -> None:
        assert map_halt_reason("max_tokens") == "length"

    def test_tool_calls(self) -> None:
        assert map_halt_reason("tool_calls") == "toolUse"

    def test_function_call(self) -> None:
        assert map_halt_reason("function_call") == "toolUse"

    def test_unknown(self) -> None:
        assert map_halt_reason("content_filter") == "stop"


class TestExtractUsage:
    def test_empty(self) -> None:
        usage = extract_usage(None)
        assert usage.total_tokens == 0

    def test_chat_format(self) -> None:
        usage = extract_usage({
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        })
        assert usage.input == 100
        assert usage.output == 50
        assert usage.total_tokens == 150

    def test_responses_format(self) -> None:
        usage = extract_usage({
            "input_tokens": 200,
            "output_tokens": 80,
            "total_tokens": 280,
        })
        assert usage.input == 200
        assert usage.output == 80

    def test_cached_tokens(self) -> None:
        usage = extract_usage({
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "prompt_tokens_details": {"cached_tokens": 30},
        })
        assert usage.cache_read == 30

    def test_reasoning_tokens(self) -> None:
        usage = extract_usage({
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "completion_tokens_details": {"reasoning_tokens": 20},
        })
        assert usage.reasoning == 20


class TestPickEndpoint:
    def test_default_openai_gpt(self) -> None:
        assert pick_endpoint("gpt-4o", "https://api.openai.com/v1") == "chat"

    def test_default_openai_o1(self) -> None:
        assert pick_endpoint("o1-preview", "https://api.openai.com/v1") == "responses"

    def test_default_openai_o3(self) -> None:
        assert pick_endpoint("o3-mini", "https://api.openai.com/v1") == "responses"

    def test_custom_base_always_chat(self) -> None:
        assert pick_endpoint("o1-preview", "http://localhost:8000/v1") == "chat"

    def test_trailing_slash(self) -> None:
        assert pick_endpoint("o1", "https://api.openai.com/v1/") == "responses"


class TestJsonHelpers:
    def test_try_parse_valid(self) -> None:
        assert try_parse_json('{"a": 1}') == {"a": 1}

    def test_try_parse_invalid(self) -> None:
        assert try_parse_json("not json") is None

    def test_try_parse_array(self) -> None:
        assert try_parse_json("[1, 2]") is None

    def test_parse_tool_args_valid(self) -> None:
        assert parse_tool_arguments('{"x": 1}') == {"x": 1}

    def test_parse_tool_args_empty(self) -> None:
        assert parse_tool_arguments("") == {}

    def test_parse_tool_args_invalid(self) -> None:
        assert parse_tool_arguments("{broken") == {}


class TestBuildReasoningExtra:
    def test_disabled(self) -> None:
        assert build_reasoning_extra(ReasoningPolicy()) == {}

    def test_enabled_no_budget(self) -> None:
        extra = build_reasoning_extra(ReasoningPolicy(enabled=True))
        assert extra == {"reasoning": {"effort": "medium"}}

    def test_enabled_with_budget(self) -> None:
        extra = build_reasoning_extra(ReasoningPolicy(enabled=True, budget_tokens=5000))
        assert extra["reasoning"]["effort"] == "high"


class TestParseRetryAfter:
    def test_none(self) -> None:
        assert parse_retry_after(None) is None

    def test_numeric(self) -> None:
        assert parse_retry_after("2.5") == 2.5

    def test_invalid(self) -> None:
        assert parse_retry_after("not-a-number") is None


# ===========================================================================
# payloads.py
# ===========================================================================


def _dummy_tool() -> ToolSpec:
    async def _run(
        tool_call_id: str,
        arguments: Any,
        signal: Any = None,
        on_update: Any = None,
    ) -> ToolOutcome:
        return ToolOutcome(content=[TextSegment(text="ok")])

    return ToolSpec(
        name="get_weather",
        label="Weather",
        description="Get weather info",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}},
        run=_run,
    )


class TestBuildChatPayload:
    def test_basic(self) -> None:
        payload = build_chat_payload(
            model="gpt-4o",
            system="You are helpful.",
            messages=[HumanEntry(content="Hello")],
            tools=[],
        )
        assert payload["model"] == "gpt-4o"
        assert payload["stream"] is True
        assert payload["messages"][0] == {"role": "system", "content": "You are helpful."}
        assert payload["messages"][1] == {"role": "user", "content": "Hello"}
        assert "tools" not in payload

    def test_with_tools(self) -> None:
        payload = build_chat_payload(
            model="gpt-4o",
            system="sys",
            messages=[],
            tools=[_dummy_tool()],
        )
        assert len(payload["tools"]) == 1
        assert payload["tools"][0]["function"]["name"] == "get_weather"

    def test_assistant_with_tool_calls(self) -> None:
        call = CallBlock(id="c1", name="get_weather", arguments={"city": "NYC"})
        assistant = ModelEntry(content=[TextSegment(text="Let me check."), call])
        payload = build_chat_payload(
            model="gpt-4o",
            system="sys",
            messages=[assistant],
            tools=[],
        )
        msg = payload["messages"][1]
        assert msg["role"] == "assistant"
        assert msg["content"] == "Let me check."
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["id"] == "c1"

    def test_tool_result(self) -> None:
        result = ToolOutcomeEntry(
            tool_call_id="c1",
            tool_name="get_weather",
            content=[TextSegment(text="Sunny, 72F")],
        )
        payload = build_chat_payload(
            model="gpt-4o",
            system="sys",
            messages=[result],
            tools=[],
        )
        msg = payload["messages"][1]
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "c1"
        assert msg["content"] == "Sunny, 72F"

    def test_max_tokens(self) -> None:
        payload = build_chat_payload(
            model="gpt-4o",
            system="sys",
            messages=[],
            tools=[],
            max_tokens=1024,
        )
        assert payload["max_tokens"] == 1024

    def test_extra_params(self) -> None:
        payload = build_chat_payload(
            model="gpt-4o",
            system="sys",
            messages=[],
            tools=[],
            extra={"temperature": 0.5},
        )
        assert payload["temperature"] == 0.5


class TestBuildResponsesPayload:
    def test_basic(self) -> None:
        payload = build_responses_payload(
            model="o1",
            system="You are helpful.",
            messages=[HumanEntry(content="Hello")],
            tools=[],
        )
        assert payload["model"] == "o1"
        assert payload["instructions"] == "You are helpful."
        assert payload["input"][0] == {"role": "user", "content": "Hello"}

    def test_tool_result(self) -> None:
        result = ToolOutcomeEntry(
            tool_call_id="c1",
            tool_name="get_weather",
            content=[TextSegment(text="Rainy")],
        )
        payload = build_responses_payload(
            model="o1",
            system="sys",
            messages=[result],
            tools=[],
        )
        item = payload["input"][0]
        assert item["type"] == "function_call_output"
        assert item["call_id"] == "c1"

    def test_max_output_tokens(self) -> None:
        payload = build_responses_payload(
            model="o1",
            system="sys",
            messages=[],
            tools=[],
            max_tokens=2048,
        )
        assert payload["max_output_tokens"] == 2048


# ===========================================================================
# parsers.py
# ===========================================================================


class TestChatDecoder:
    def test_text_delta(self) -> None:
        decoder = ChatDecoder()
        signals = decoder.decode("", {
            "id": "chatcmpl-1",
            "model": "gpt-4o",
            "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}],
        })
        assert any(isinstance(s, Metadata) for s in signals)
        text_chunks = [s for s in signals if isinstance(s, TextChunk)]
        assert len(text_chunks) == 1
        assert text_chunks[0].text == "Hello"

    def test_finish_reason(self) -> None:
        decoder = ChatDecoder()
        signals = decoder.decode("", {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })
        finished = [s for s in signals if isinstance(s, Finished)]
        assert len(finished) == 1
        assert finished[0].finish_reason == "stop"
        assert finished[0].usage is not None

    def test_tool_call_reconstruction(self) -> None:
        decoder = ChatDecoder()

        # First chunk: tool call start
        s1 = decoder.decode("", {
            "choices": [{"index": 0, "delta": {
                "tool_calls": [{"index": 0, "id": "c1", "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"ci'}}],
            }, "finish_reason": None}],
        })
        begins = [s for s in s1 if isinstance(s, CallBegin)]
        assert len(begins) == 1
        assert begins[0].name == "get_weather"
        frags = [s for s in s1 if isinstance(s, CallArgFragment)]
        assert len(frags) == 1

        # Second chunk: argument continuation
        s2 = decoder.decode("", {
            "choices": [{"index": 0, "delta": {
                "tool_calls": [{"index": 0, "function": {"arguments": 'ty":"NYC"}'}}],
            }, "finish_reason": None}],
        })
        frags2 = [s for s in s2 if isinstance(s, CallArgFragment)]
        assert len(frags2) == 1

        # Third chunk: finish
        s3 = decoder.decode("", {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        })
        resolved = [s for s in s3 if isinstance(s, CallResolved)]
        assert len(resolved) == 1
        assert resolved[0].arguments_json == '{"city":"NYC"}'

    def test_usage_only_chunk(self) -> None:
        decoder = ChatDecoder()
        signals = decoder.decode("", {
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })
        finished = [s for s in signals if isinstance(s, Finished)]
        assert len(finished) == 1
        assert finished[0].usage is not None

    def test_error_object(self) -> None:
        decoder = ChatDecoder()
        signals = decoder.decode("", {
            "error": {"message": "Rate limit exceeded", "type": "rate_limit_error"},
        })
        faults = [s for s in signals if isinstance(s, ParseFault)]
        assert len(faults) == 1
        assert "Rate limit" in faults[0].message

    def test_reasoning_content(self) -> None:
        decoder = ChatDecoder()
        signals = decoder.decode("", {
            "choices": [{
                "index": 0,
                "delta": {"reasoning_content": "Let me think..."},
                "finish_reason": None,
            }],
        })
        reasoning = [s for s in signals if isinstance(s, ReasoningChunk)]
        assert len(reasoning) == 1
        assert reasoning[0].text == "Let me think..."


class TestResponsesDecoder:
    def test_created(self) -> None:
        decoder = ResponsesDecoder()
        signals = decoder.decode("response.created", {
            "type": "response.created",
            "response": {"id": "resp_1", "model": "o1"},
        })
        meta = [s for s in signals if isinstance(s, Metadata)]
        assert len(meta) == 1
        assert meta[0].response_id == "resp_1"
        assert meta[0].model == "o1"

    def test_text_delta(self) -> None:
        decoder = ResponsesDecoder()
        signals = decoder.decode("response.output_text.delta", {"delta": "Hello"})
        texts = [s for s in signals if isinstance(s, TextChunk)]
        assert len(texts) == 1
        assert texts[0].text == "Hello"

    def test_reasoning_delta(self) -> None:
        decoder = ResponsesDecoder()
        signals = decoder.decode(
            "response.reasoning_summary_text.delta", {"delta": "Thinking..."},
        )
        reasoning = [s for s in signals if isinstance(s, ReasoningChunk)]
        assert len(reasoning) == 1

    def test_tool_call_flow(self) -> None:
        decoder = ResponsesDecoder()

        # Item added
        s1 = decoder.decode("response.output_item.added", {
            "output_index": 1,
            "item": {
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_weather",
            },
        })
        assert any(isinstance(s, CallBegin) for s in s1)

        # Argument delta
        s2 = decoder.decode("response.function_call_arguments.delta", {
            "output_index": 1, "delta": '{"city"',
        })
        assert any(isinstance(s, CallArgFragment) for s in s2)

        # Arguments done
        s3 = decoder.decode("response.function_call_arguments.done", {
            "output_index": 1,
            "arguments": '{"city":"NYC"}',
        })
        resolved = [s for s in s3 if isinstance(s, CallResolved)]
        assert len(resolved) == 1
        assert resolved[0].arguments_json == '{"city":"NYC"}'

    def test_completed(self) -> None:
        decoder = ResponsesDecoder()
        signals = decoder.decode("response.completed", {
            "response": {
                "status": "completed",
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            },
        })
        finished = [s for s in signals if isinstance(s, Finished)]
        assert len(finished) == 1
        assert finished[0].usage is not None

    def test_error_event(self) -> None:
        decoder = ResponsesDecoder()
        signals = decoder.decode("error", {"error": {"message": "Server error"}})
        faults = [s for s in signals if isinstance(s, ParseFault)]
        assert len(faults) == 1

    def test_response_failed(self) -> None:
        decoder = ResponsesDecoder()
        signals = decoder.decode("response.failed", {
            "response": {"last_error": {"message": "Model overloaded"}},
        })
        faults = [s for s in signals if isinstance(s, ParseFault)]
        assert len(faults) == 1
        assert "overloaded" in faults[0].message.lower()


# ===========================================================================
# normalize.py
# ===========================================================================


class TestEventAssembler:
    def _assembler(self) -> EventAssembler:
        return EventAssembler(model="gpt-4o", provider="openai", api="chat")

    def test_text_streaming(self) -> None:
        asm = self._assembler()

        e1 = asm.accept(TextChunk(text="Hello"))
        assert any(isinstance(e, StreamOpenEvent) for e in e1)
        assert any(isinstance(e, ContentOpenEvent) for e in e1)
        assert any(isinstance(e, ContentChunkEvent) for e in e1)

        e2 = asm.accept(TextChunk(text=" world"))
        chunks = [e for e in e2 if isinstance(e, ContentChunkEvent)]
        assert len(chunks) == 1
        assert chunks[0].delta == " world"

    def test_reasoning_streaming(self) -> None:
        asm = self._assembler()

        events = asm.accept(ReasoningChunk(text="thinking"))
        assert any(isinstance(e, StreamOpenEvent) for e in events)
        assert any(isinstance(e, ReasoningOpenEvent) for e in events)
        assert any(isinstance(e, ReasoningChunkEvent) for e in events)

    def test_tool_call_flow(self) -> None:
        asm = self._assembler()

        asm.accept(TextChunk(text="Let me check."))
        e1 = asm.accept(CallBegin(index=0, call_id="c1", name="get_weather"))
        # Text should be closed before tool call opens
        assert any(isinstance(e, ContentCloseEvent) for e in e1)
        assert any(isinstance(e, CallOpenEvent) for e in e1)

        e2 = asm.accept(CallArgFragment(index=0, fragment='{"city"'))
        assert any(isinstance(e, CallChunkEvent) for e in e2)

        e3 = asm.accept(CallResolved(
            index=0, call_id="c1", name="get_weather",
            arguments_json='{"city":"NYC"}',
        ))
        close_events = [e for e in e3 if isinstance(e, CallCloseEvent)]
        assert len(close_events) == 1
        assert close_events[0].tool_call.name == "get_weather"
        assert close_events[0].tool_call.arguments == {"city": "NYC"}

    def test_metadata_stored(self) -> None:
        asm = self._assembler()
        asm.accept(Metadata(response_id="resp_1", model="gpt-4o-2024"))
        asm.accept(TextChunk(text="Hi"))

        events = asm.finalize()
        close = [e for e in events if isinstance(e, StreamCloseEvent)]
        assert len(close) == 1
        assert close[0].message.response_id == "resp_1"
        assert close[0].message.response_model == "gpt-4o-2024"

    def test_finalize_stop(self) -> None:
        asm = self._assembler()
        asm.accept(TextChunk(text="Done"))
        asm.accept(Finished(finish_reason="stop", usage={
            "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
        }))
        events = asm.finalize()

        close = [e for e in events if isinstance(e, StreamCloseEvent)]
        assert len(close) == 1
        assert close[0].reason == "stop"
        assert close[0].message.usage.input == 10
        assert close[0].message.usage.output == 5

    def test_finalize_tool_use(self) -> None:
        asm = self._assembler()
        asm.accept(CallBegin(index=0, call_id="c1", name="f"))
        asm.accept(CallResolved(index=0, call_id="c1", name="f", arguments_json="{}"))
        asm.accept(Finished(finish_reason="tool_calls"))
        events = asm.finalize()

        close = [e for e in events if isinstance(e, StreamCloseEvent)]
        assert close[0].reason == "toolUse"

    def test_finalize_length(self) -> None:
        asm = self._assembler()
        asm.accept(TextChunk(text="trunca"))
        asm.accept(Finished(finish_reason="length"))
        events = asm.finalize()

        close = [e for e in events if isinstance(e, StreamCloseEvent)]
        assert close[0].reason == "length"

    def test_fault_emits_error_event(self) -> None:
        asm = self._assembler()
        events = asm.accept(ParseFault(message="Bad response"))
        faults = [e for e in events if isinstance(e, StreamFaultEvent)]
        assert len(faults) == 1
        assert faults[0].reason == "error"

    def test_empty_response(self) -> None:
        asm = self._assembler()
        events = asm.finalize()
        assert any(isinstance(e, StreamOpenEvent) for e in events)
        assert any(isinstance(e, StreamCloseEvent) for e in events)

    def test_content_close_on_finalize(self) -> None:
        asm = self._assembler()
        asm.accept(TextChunk(text="Hello"))
        events = asm.finalize()
        assert any(isinstance(e, ContentCloseEvent) for e in events)
        close_content = [e for e in events if isinstance(e, ContentCloseEvent)]
        assert close_content[0].content == "Hello"

    def test_reasoning_close_on_finalize(self) -> None:
        asm = self._assembler()
        asm.accept(ReasoningChunk(text="step 1"))
        asm.accept(ReasoningChunk(text=" step 2"))
        events = asm.finalize()
        close_reasoning = [e for e in events if isinstance(e, ReasoningCloseEvent)]
        assert len(close_reasoning) == 1
        assert close_reasoning[0].content == "step 1 step 2"


# ===========================================================================
# transport.py
# ===========================================================================


class TestHttpStreamError:
    def test_retriable_429(self) -> None:
        assert HttpStreamError(429, "rate limit").retriable is True

    def test_retriable_500(self) -> None:
        assert HttpStreamError(500, "internal").retriable is True

    def test_not_retriable_400(self) -> None:
        assert HttpStreamError(400, "bad request").retriable is False

    def test_not_retriable_401(self) -> None:
        assert HttpStreamError(401, "unauthorized").retriable is False

    def test_retry_after(self) -> None:
        err = HttpStreamError(429, "limit", retry_after=2.5)
        assert err.retry_after_seconds == 2.5


class TestMalformedPayload:
    def test_stores_raw(self) -> None:
        exc = MalformedPayload("{broken")
        assert exc.raw == "{broken"
        assert "Invalid JSON" in str(exc)

    def test_non_object_json(self) -> None:
        exc = MalformedPayload("[1, 2, 3]")
        assert exc.raw == "[1, 2, 3]"

    def test_truncates_long_input(self) -> None:
        long = "x" * 500
        exc = MalformedPayload(long)
        assert len(str(exc)) < 300
