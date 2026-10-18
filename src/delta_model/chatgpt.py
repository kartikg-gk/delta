"""ChatGPT-plan provider: Codex models billed to a ChatGPT subscription.

The ChatGPT backend speaks the Responses streaming protocol, so this reuses
the OpenAI-compatible adapter's transport, retries and stream parsing and
only changes the request:

- the endpoint is ``<base>/codex/responses``;
- nothing is stored server-side (``store: false``), so the whole history is
  sent each turn, including the model's encrypted reasoning items, which are
  kept on thinking blocks and handed back so its reasoning carries over;
- the account id travels in a header next to the bearer token, and the
  caller names itself with an ``originator`` header.

Sign-in and token refresh live in the application layer; this adapter only
receives a credential resolver that returns a current token.
"""

from __future__ import annotations

import json
import platform
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from delta_harness.contracts.tooling import ToolSpec
from delta_harness.contracts.transcript import (
    ModelEntry,
    ThoughtSegment,
    TranscriptEntry,
)
from delta_model._oai.helpers import build_reasoning_extra
from delta_model._oai.payloads import build_responses_payload
from delta_model.oai_compatible import OpenAIProvider
from delta_model.settings import Credential, OpenAIProfile

CHATGPT_BASE_URL = "https://chatgpt.com/backend-api"
ORIGINATOR = "delta"
_FALLBACK_INSTRUCTIONS = "You are a helpful assistant."


def responses_url(base_url: str) -> str:
    """The streaming endpoint for *base_url*, whichever depth it is given at."""
    base = base_url.rstrip("/")
    if base.endswith("/codex/responses"):
        return base
    if base.endswith("/codex"):
        return f"{base}/responses"
    return f"{base}/codex/responses"


def _input_parts(content: object) -> list[dict[str, Any]]:
    """User content in the backend's input-part vocabulary."""
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]
    parts: list[dict[str, Any]] = []
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                parts.append({"type": "input_text", "text": part.get("text", "")})
            elif part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                parts.append({"type": "input_image", "detail": "auto", "image_url": url})
    return parts


def _reasoning_items(entry: ModelEntry) -> list[dict[str, Any]]:
    """Encrypted reasoning items this reply carried, in order."""
    items: list[dict[str, Any]] = []
    for segment in entry.content:
        if not isinstance(segment, ThoughtSegment) or not segment.thinking_signature:
            continue
        try:
            item = json.loads(segment.thinking_signature)
        except ValueError:
            continue
        if isinstance(item, dict) and item.get("type") == "reasoning":
            items.append(item)
    return items


def build_chatgpt_payload(
    *,
    model: str,
    system: str,
    messages: Sequence[TranscriptEntry],
    tools: Sequence[ToolSpec],
    reasoning: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Request body for one turn."""
    body = build_responses_payload(model=model, system=system, messages=[], tools=tools)
    items: list[dict[str, Any]] = []
    for entry in messages:
        converted = build_responses_payload(
            model=model, system="", messages=[entry], tools=[],
        )["input"]
        if isinstance(entry, ModelEntry):
            items.extend(_reasoning_items(entry))
        for item in converted:
            if item.get("role") == "user":
                item = {"role": "user", "content": _input_parts(item.get("content"))}
            items.append(item)
    body.update({
        "instructions": system or _FALLBACK_INSTRUCTIONS,
        "input": items,
        "store": False,
        "include": ["reasoning.encrypted_content"],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "text": {"verbosity": "low"},
    })
    if reasoning:
        body["reasoning"] = {**reasoning, "summary": "auto"}
    return body


class ChatGPTProvider(OpenAIProvider):
    """Responses streaming against the ChatGPT backend with plan credentials."""

    def __init__(self, profile: OpenAIProfile) -> None:
        # The backend keeps no history, so the conversation id is always sent
        # as the cache-routing hint.
        super().__init__(replace(profile, cache_affinity=True))

    def _request(
        self,
        *,
        model: str,
        system: str,
        messages: Sequence[TranscriptEntry],
        tools: Sequence[ToolSpec],
        base_url: str,
    ) -> tuple[str, str, dict[str, Any]]:
        extra = build_reasoning_extra(
            self._profile.reasoning, model=model, endpoint="responses",
        )
        reasoning = extra.get("reasoning") if isinstance(extra.get("reasoning"), dict) else None
        return "responses", responses_url(base_url), build_chatgpt_payload(
            model=model, system=system, messages=messages, tools=tools, reasoning=reasoning,
        )

    def _build_headers(self, cred: Credential) -> dict[str, str]:
        headers = super()._build_headers(cred)
        headers.update({
            "originator": ORIGINATOR,
            "OpenAI-Beta": "responses=experimental",
            "Accept": "text/event-stream",
            "User-Agent": (
                f"{ORIGINATOR} ({platform.system()} {platform.release()}; {platform.machine()})"
            ),
        })
        return headers


__all__ = [
    "CHATGPT_BASE_URL",
    "ORIGINATOR",
    "ChatGPTProvider",
    "build_chatgpt_payload",
    "responses_url",
]
