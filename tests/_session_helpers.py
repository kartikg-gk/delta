"""Shared fixtures for building a CodingSession over scripted provider streams."""

from __future__ import annotations

from delta_app.conversation import CodingSession
from delta_harness.contracts.transcript import CallBlock, ModelEntry, TextSegment
from delta_harness.provider.wire import (
    CallCloseEvent,
    ContentChunkEvent,
    StreamCloseEvent,
)
from delta_model.scripted import ReplayProvider


def _text_turn(chunks: list[str]) -> list:
    """A streamed text response delivered in *chunks*."""
    events = []
    seen = ""
    for chunk in chunks:
        seen += chunk
        partial = ModelEntry(content=[TextSegment(text=seen)], stop_reason="stop")
        events.append(ContentChunkEvent(content_index=0, delta=chunk, partial=partial))
    final = ModelEntry(content=[TextSegment(text=seen)], stop_reason="stop")
    events.append(StreamCloseEvent(reason="stop", message=final))
    return events


def _tool_turn(path: str) -> list:
    call = CallBlock(name="Write", id="c1", arguments={
        "file_path": path, "content": "print('hi')\n",
    })
    partial = ModelEntry(content=[call], stop_reason="toolUse")
    return [
        CallCloseEvent(content_index=0, tool_call=call, partial=partial),
        StreamCloseEvent(reason="toolUse", message=partial),
    ]


async def _make_session(tmp_path, streams) -> CodingSession:
    from delta_app.tools import build_tool_registry

    return await CodingSession.create(
        provider=ReplayProvider(streams),
        provider_name="replay",
        model="replay-model",
        system="test",
        tools=build_tool_registry(),
        sessions_dir=tmp_path / "sessions",
    )
