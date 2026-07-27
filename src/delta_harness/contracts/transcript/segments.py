"""Content segments and blocks that make up transcript entries."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from delta_harness.contracts.transcript._base import StrictModel
from delta_harness.contracts.values import JValue


class TextSegment(StrictModel):

    type: Literal["text"] = "text"
    text: str
    text_signature: str | None = None


class ImageSegment(StrictModel):

    type: Literal["image"] = "image"
    mime_type: str
    data: str


class ThoughtSegment(StrictModel):


    type: Literal["thinking"] = "thinking"
    thinking: str
    redacted: bool = False
    thinking_signature: str | None = None


class CallBlock(StrictModel):

    type: Literal["toolCall"] = "toolCall"
    name: str
    id: str
    arguments: dict[str, JValue] = Field(default_factory=dict)
    thought_signature: str | None = None


type InputContent = str | list[TextSegment | ImageSegment]
type ReplyContent = TextSegment | ThoughtSegment | CallBlock
type ResultContent = TextSegment | ImageSegment


def gather_text(content: str | list[Any]) -> str:
    """Return visible text from string or text/image content."""
    if isinstance(content, str):
        return content
    return "".join(block.text for block in content if isinstance(block, TextSegment))


def build_model_blocks(
    text: str,
    tool_calls: list[CallBlock] | tuple[CallBlock, ...] = (),
) -> list[ReplyContent]:
    """Build canonical ordered assistant blocks from parser accumulators."""
    blocks: list[ReplyContent] = [TextSegment(text=text)] if text else []
    blocks.extend(tool_calls)
    return blocks
