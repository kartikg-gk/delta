"""Anthropic prompt-cache marker placement."""

from __future__ import annotations

from delta_harness.contracts.tooling import ToolSpec
from delta_harness.contracts.transcript import (
    CallBlock,
    HumanEntry,
    ImageSegment,
    ModelEntry,
    TextSegment,
    ToolOutcomeEntry,
)
from delta_model.claude import _compose_body, resolve_cache_retention
from delta_model.settings import ReasoningPolicy


async def _noop(*_args, **_kwargs):  # pragma: no cover - never executed
    raise AssertionError


def _tool(name: str) -> ToolSpec:
    return ToolSpec(name=name, label=name, description=name, parameters={"type": "object"}, run=_noop)


def _body(messages, *, retention="short", tools=("read", "write"), system="sys"):
    return _compose_body(
        model="m", system=system, messages=messages,
        tools=[_tool(t) for t in tools], max_tokens=1024,
        thinking=ReasoningPolicy(), cache_retention=retention,
    )


def _markers(body) -> list:
    found = []
    system = body["system"]
    if isinstance(system, list):
        found += [b for b in system if "cache_control" in b]
    found += [t for t in body.get("tools", []) if "cache_control" in t]
    for message in body["messages"]:
        if isinstance(message["content"], list):
            found += [b for b in message["content"] if "cache_control" in b]
    return found


def _wide_turn(calls: int) -> list:
    ids = [f"t{i}" for i in range(calls)]
    return [
        HumanEntry(content="start"),
        ModelEntry(content=[TextSegment(text="ok"),
                            *[CallBlock(id=i, name="read", arguments={}) for i in ids]]),
        *[ToolOutcomeEntry(tool_call_id=i, tool_name="read", content="r") for i in ids],
    ]


def test_first_party_host_caches_by_default_gateways_do_not() -> None:
    assert resolve_cache_retention(None, "https://api.anthropic.com") == "short"
    assert resolve_cache_retention(None, "https://gateway.example.com/anthropic") == "none"
    assert resolve_cache_retention("long", "https://gateway.example.com") == "long"


def test_disabled_caching_leaves_payload_shape_untouched() -> None:
    body = _body([HumanEntry(content="hi")], retention="none")
    assert body["system"] == "sys"
    assert _markers(body) == []


def test_single_prompt_marks_tools_system_and_tail() -> None:
    body = _body([HumanEntry(content="hi")])
    assert body["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in body["tools"][0]
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert body["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert len(_markers(body)) == 3


def test_long_retention_requests_extended_ttl() -> None:
    body = _body([HumanEntry(content="hi")], retention="long")
    assert all(m["cache_control"] == {"type": "ephemeral", "ttl": "1h"} for m in _markers(body))


def test_wide_parallel_turn_marks_previous_boundary_too() -> None:
    body = _body(_wide_turn(12))
    messages = body["messages"]
    assert len(_markers(body)) == 4
    # Tail marker sits on the final tool_result, boundary marker on the prompt.
    assert messages[-1]["content"][-1]["type"] == "tool_result"
    assert "cache_control" in messages[-1]["content"][-1]
    assert "cache_control" in messages[0]["content"][-1]


def test_never_exceeds_four_markers() -> None:
    history = _wide_turn(3) + [ModelEntry(content=[TextSegment(text="done")]),
                               HumanEntry(content="more")] + _wide_turn(20)[1:]
    assert len(_markers(_body(history))) <= 4


def test_empty_blocks_are_never_marked() -> None:
    history = [
        HumanEntry(content="q"),
        ModelEntry(content=[CallBlock(id="t1", name="read", arguments={})]),
        ToolOutcomeEntry(tool_call_id="t1", tool_name="read", content=""),
    ]
    body = _body(history, system="")
    assert body["system"] == ""
    assert "cache_control" not in body["messages"][-1]["content"][-1]


def test_image_tail_is_markable() -> None:
    entry = HumanEntry(content=[TextSegment(text="see"),
                                ImageSegment(data="aGk=", mime_type="image/png")])
    body = _body([entry])
    assert body["messages"][-1]["content"][-1]["type"] == "image"
    assert "cache_control" in body["messages"][-1]["content"][-1]


def test_caller_transcript_is_not_mutated() -> None:
    history = _wide_turn(2)
    before = [e.model_dump() for e in history]
    _body(history)
    assert [e.model_dump() for e in history] == before
