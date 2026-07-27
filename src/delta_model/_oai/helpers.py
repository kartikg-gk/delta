"""Shared utilities for the OpenAI-compatible provider layer.

Contains pure functions for finish-reason mapping, token-usage extraction,
endpoint selection, JSON parsing, and reasoning parameter construction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from delta_harness.contracts.transcript import UsageStats
from delta_model.settings import ReasoningPolicy


@dataclass(frozen=True, slots=True)
class ServerSentEvent:
    """A single Server-Sent Event parsed from an HTTP stream."""

    event: str
    data: str


# ---------------------------------------------------------------------------
# Finish-reason mapping
# ---------------------------------------------------------------------------


def map_halt_reason(reason: str | None) -> str:
    """Translate an OpenAI finish reason to Delta's HaltReason vocabulary.

    Returns one of ``"stop"``, ``"length"``, or ``"toolUse"``.
    """
    match reason:
        case "stop" | None:
            return "stop"
        case "length" | "max_tokens":
            return "length"
        case "tool_calls" | "function_call":
            return "toolUse"
        case _:
            return "stop"


# ---------------------------------------------------------------------------
# Usage extraction
# ---------------------------------------------------------------------------


def extract_usage(raw: dict[str, Any] | None) -> UsageStats:
    """Build a ``UsageStats`` from an OpenAI-style usage dictionary."""
    if not raw:
        return UsageStats()

    input_tok = raw.get("prompt_tokens") or raw.get("input_tokens") or 0
    output_tok = raw.get("completion_tokens") or raw.get("output_tokens") or 0
    total = raw.get("total_tokens") or (int(input_tok) + int(output_tok))

    cache_read = 0
    prompt_detail = raw.get("prompt_tokens_details")
    if isinstance(prompt_detail, dict):
        cache_read = prompt_detail.get("cached_tokens", 0) or 0

    reasoning = None
    comp_detail = raw.get("completion_tokens_details")
    if isinstance(comp_detail, dict):
        r = comp_detail.get("reasoning_tokens")
        if r:
            reasoning = int(r)

    return UsageStats(
        total_tokens=int(total),
        input=int(input_tok),
        output=int(output_tok),
        cache_read=int(cache_read),
        reasoning=reasoning,
    )


# ---------------------------------------------------------------------------
# Endpoint selection
# ---------------------------------------------------------------------------

_RESPONSES_PREFIXES = ("o1", "o3", "o4")
_DEFAULT_OPENAI_BASE = "https://api.openai.com/v1"


def pick_endpoint(model: str, base_url: str) -> str:
    """Return ``"responses"`` or ``"chat"`` for the given model and base URL.

    The Responses API is preferred for reasoning-family models when the base
    URL is the official OpenAI endpoint.  All other configurations default to
    Chat Completions for maximum compatibility with third-party hosts.
    """
    if base_url.rstrip("/") != _DEFAULT_OPENAI_BASE:
        return "chat"
    lower = model.lower()
    for prefix in _RESPONSES_PREFIXES:
        if lower.startswith(prefix):
            return "responses"
    return "chat"


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def try_parse_json(text: str) -> dict[str, Any] | None:
    """Attempt to parse *text* as a JSON object, returning ``None`` on failure."""
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


def parse_tool_arguments(text: str) -> dict[str, Any]:
    """Parse tool-call argument JSON, returning an empty dict on failure."""
    if not text or not text.strip():
        return {}
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except (json.JSONDecodeError, ValueError):
        return {}


# ---------------------------------------------------------------------------
# Reasoning parameters
# ---------------------------------------------------------------------------


def build_reasoning_extra(policy: ReasoningPolicy) -> dict[str, Any]:
    """Build provider-level reasoning parameters from a ``ReasoningPolicy``."""
    if not policy.enabled:
        return {}
    extra: dict[str, Any] = {"reasoning": {"effort": "medium"}}
    if policy.budget_tokens is not None:
        extra["reasoning"]["effort"] = "high"
    return extra


# ---------------------------------------------------------------------------
# Retry-After parsing
# ---------------------------------------------------------------------------


def parse_retry_after(header: str | None) -> float | None:
    """Extract a wait duration in seconds from a ``Retry-After`` header."""
    if header is None:
        return None
    try:
        return max(0.0, float(header))
    except (ValueError, TypeError):
        return None
