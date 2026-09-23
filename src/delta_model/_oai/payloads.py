"""Request payload construction for Chat Completions and Responses APIs.

Each builder produces a plain ``dict`` suitable for JSON serialisation and
submission to the respective endpoint.  Payload assembly is deliberately
separated from networking so it can be tested and reasoned about in isolation.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from delta_harness.contracts.tooling import ToolSpec
from delta_harness.contracts.transcript import (
    CallBlock,
    ExtensionEntry,
    ForkSummaryEntry,
    HumanEntry,
    ImageSegment,
    ModelEntry,
    PruneSummaryEntry,
    ShellResultEntry,
    TextSegment,
    ToolOutcomeEntry,
    TranscriptEntry,
    entry_to_human,
)
from delta_model.correlation import wire_call_id

# ---------------------------------------------------------------------------
# Tool schema conversion
# ---------------------------------------------------------------------------


def _chat_function_schema(tool: ToolSpec) -> dict[str, Any]:
    """Convert a ``ToolSpec`` to a Chat Completions function tool definition."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": dict(tool.parameters),
        },
    }


def _responses_function_schema(tool: ToolSpec) -> dict[str, Any]:
    """Convert a ``ToolSpec`` to a Responses API function tool definition."""
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": dict(tool.parameters),
    }


# ---------------------------------------------------------------------------
# Content helpers
# ---------------------------------------------------------------------------


def _user_blocks(entry: HumanEntry) -> str | list[dict[str, Any]]:
    """Convert user content to the provider content format."""
    if isinstance(entry.content, str):
        return entry.content
    parts: list[dict[str, Any]] = []
    for seg in entry.content:
        if isinstance(seg, TextSegment):
            parts.append({"type": "text", "text": seg.text})
        elif isinstance(seg, ImageSegment):
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{seg.mime_type};base64,{seg.data}"},
            })
    return parts or ""


def _serialize_call(call: CallBlock) -> dict[str, Any]:
    """Serialise a ``CallBlock`` to the Chat Completions tool_calls format."""
    return {
        "id": wire_call_id(call.id),
        "type": "function",
        "function": {
            "name": call.name,
            "arguments": json.dumps(call.arguments),
        },
    }


# ---------------------------------------------------------------------------
# Chat Completions message conversion
# ---------------------------------------------------------------------------


def _chat_assistant_msg(entry: ModelEntry) -> dict[str, Any]:
    """Convert a ``ModelEntry`` to a Chat Completions assistant message."""
    msg: dict[str, Any] = {"role": "assistant"}
    text_parts = [b.text for b in entry.content if isinstance(b, TextSegment)]
    msg["content"] = "".join(text_parts) or None
    calls = [_serialize_call(b) for b in entry.content if isinstance(b, CallBlock)]
    if calls:
        msg["tool_calls"] = calls
    return msg


def _chat_tool_result_msg(entry: ToolOutcomeEntry) -> dict[str, Any]:
    """Convert a ``ToolOutcomeEntry`` to a Chat Completions tool message."""
    return {
        "role": "tool",
        "tool_call_id": wire_call_id(entry.tool_call_id),
        "content": entry.text or "",
    }


def _to_chat_message(entry: TranscriptEntry) -> dict[str, Any] | None:
    """Convert a single transcript entry to a Chat Completions message dict."""
    if isinstance(entry, HumanEntry):
        return {"role": "user", "content": _user_blocks(entry)}
    if isinstance(entry, ModelEntry):
        return _chat_assistant_msg(entry)
    if isinstance(entry, ToolOutcomeEntry):
        return _chat_tool_result_msg(entry)
    if isinstance(entry, (ExtensionEntry, ShellResultEntry, PruneSummaryEntry, ForkSummaryEntry)):
        return {"role": "user", "content": _user_blocks(entry_to_human(entry))}
    return None


# ---------------------------------------------------------------------------
# Responses API message conversion
# ---------------------------------------------------------------------------


def _to_responses_items(entry: TranscriptEntry) -> list[dict[str, Any]]:
    """Convert a single transcript entry to Responses API input items.

    A single entry may produce more than one item — for example a
    ``ModelEntry`` with both text and tool calls emits an assistant item
    followed by separate ``function_call`` items.
    """
    if isinstance(entry, HumanEntry):
        return [{"role": "user", "content": _user_blocks(entry)}]

    if isinstance(entry, ModelEntry):
        items: list[dict[str, Any]] = []
        text = entry.text
        if text:
            items.append({
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            })
        for call in entry.tool_calls:
            items.append({
                "type": "function_call",
                "call_id": wire_call_id(call.id),
                "name": call.name,
                "arguments": json.dumps(call.arguments),
            })
        return items

    if isinstance(entry, ToolOutcomeEntry):
        return [{
            "type": "function_call_output",
            "call_id": wire_call_id(entry.tool_call_id),
            "output": entry.text or "",
        }]

    if isinstance(entry, (ExtensionEntry, ShellResultEntry, PruneSummaryEntry, ForkSummaryEntry)):
        return [{"role": "user", "content": _user_blocks(entry_to_human(entry))}]

    return []


# ---------------------------------------------------------------------------
# Payload assembly
# ---------------------------------------------------------------------------


def build_chat_payload(
    *,
    model: str,
    system: str,
    messages: Sequence[TranscriptEntry],
    tools: Sequence[ToolSpec],
    max_tokens: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a complete Chat Completions request body."""
    chat_msgs: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for entry in messages:
        msg = _to_chat_message(entry)
        if msg is not None:
            chat_msgs.append(msg)

    body: dict[str, Any] = {
        "model": model,
        "stream": True,
        "stream_options": {"include_usage": True},
        "messages": chat_msgs,
    }
    if tools:
        body["tools"] = [_chat_function_schema(t) for t in tools]
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if extra:
        body.update(extra)
    return body


def build_responses_payload(
    *,
    model: str,
    system: str,
    messages: Sequence[TranscriptEntry],
    tools: Sequence[ToolSpec],
    max_tokens: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a complete Responses API request body."""
    items: list[dict[str, Any]] = []
    for entry in messages:
        items.extend(_to_responses_items(entry))

    body: dict[str, Any] = {
        "model": model,
        "stream": True,
        "instructions": system,
        "input": items,
    }
    if tools:
        body["tools"] = [_responses_function_schema(t) for t in tools]
    if max_tokens is not None:
        body["max_output_tokens"] = max_tokens
    if extra:
        body.update(extra)
    return body
