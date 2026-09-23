"""Shared utilities for the OpenAI-compatible provider layer.

Contains pure functions for finish-reason mapping, token-usage extraction,
endpoint selection, JSON parsing, and reasoning parameter construction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from delta_harness.contracts.transcript import CostBreakdown, UsageStats
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

    # Chat Completions and Responses name the prompt breakdown differently.
    cache_read = cache_write = 0
    prompt_detail = raw.get("prompt_tokens_details") or raw.get("input_tokens_details")
    if isinstance(prompt_detail, dict):
        cache_read = int(prompt_detail.get("cached_tokens", 0) or 0)
        cache_write = int(prompt_detail.get("cache_write_tokens", 0) or 0)

    reasoning = None
    comp_detail = raw.get("completion_tokens_details") or raw.get("output_tokens_details")
    if isinstance(comp_detail, dict):
        r = comp_detail.get("reasoning_tokens")
        if r:
            reasoning = int(r)

    # Gateways such as OpenRouter report the billed cost in USD with usage.
    cost = CostBreakdown()
    reported = raw.get("cost")
    if isinstance(reported, int | float) and not isinstance(reported, bool):
        details = raw.get("cost_details")
        details = details if isinstance(details, dict) else {}
        cost = CostBreakdown(
            total=float(reported),
            input=float(details.get("upstream_inference_prompt_cost") or 0.0),
            output=float(details.get("upstream_inference_completions_cost") or 0.0),
        )

    # The reported prompt count includes cached tokens; ``input`` keeps only
    # the fresh part so every provider reports usage the same way.
    return UsageStats(
        cost=cost,
        total_tokens=int(total),
        input=max(0, int(input_tok) - cache_read - cache_write),
        output=int(output_tok),
        cache_read=cache_read,
        cache_write=cache_write,
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


# Model families that accept a reasoning-effort setting. Other models reject
# the parameter outright, so it is only sent to these.
_REASONING_FAMILIES = ("o1", "o3", "o4", "gpt-5", "gpt-oss")
# The highest level the API names; stronger local levels are clamped to it.
_EFFORT_CEILING = "high"
_EFFORTS = ("minimal", "low", "medium", "high")


def reasons(model: str) -> bool:
    """Whether ``model`` takes a reasoning-effort parameter."""
    name = model.lower().rsplit("/", 1)[-1]
    return name.startswith(_REASONING_FAMILIES)


def build_reasoning_extra(
    policy: ReasoningPolicy, *, model: str, endpoint: str,
) -> dict[str, Any]:
    """Reasoning parameters for one request, shaped for its endpoint.

    Chat Completions takes a flat ``reasoning_effort``; the Responses API takes
    ``reasoning: {effort}``. Nothing is sent when reasoning is off or the model
    does not reason.
    """
    if not policy.enabled or not reasons(model):
        return {}
    effort = policy.effort or ("high" if policy.budget_tokens else "medium")
    if effort not in _EFFORTS:
        effort = _EFFORT_CEILING
    if endpoint == "responses":
        return {"reasoning": {"effort": effort}}
    return {"reasoning_effort": effort}


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


# ---------------------------------------------------------------------------
# In-stream failure classification
# ---------------------------------------------------------------------------

# Substrings (matched against the lower-cased code + message) that mark a
# failure as a passing condition worth one more attempt.
_PASSING_FAILURE_HINTS = (
    "overloaded",
    "service_unavailable",
    "temporarily_unavailable",
    "rate_limit",
    "internal_error",
    "server_error",
    "timeout",
)

# Rate-limit wording that actually means the account is out of budget; retrying
# those only burns the backoff budget before failing anyway.
_EXHAUSTED_QUOTA_HINTS = (
    "gousagelimiterror",
    "freeusagelimiterror",
    "monthly usage limit reached",
    "available balance",
    "insufficient_quota",
    "out of budget",
    "quota exceeded",
    "billing",
)


def stream_failure(event: str, data: dict[str, Any]) -> tuple[str | None, str | None] | None:
    """Return ``(code, message)`` if this SSE event reports a failure, else ``None``.

    Failures arrive in three shapes: a top-level ``error`` object (Chat
    Completions), an ``error`` event whose details may sit at the top level or
    under a nested ``error`` (Responses), and ``response.failed`` with details
    under ``response.error``. All three are read so the real reason surfaces
    instead of a generic fallback.
    """
    nested = data.get("error")
    response = data.get("response")
    is_failure = (
        event in ("error", "response.failed")
        or data.get("type") in ("error", "response.failed")
        or isinstance(nested, dict)
    )
    if not is_failure:
        return None

    sources: list[dict[str, Any]] = [data]
    if isinstance(nested, dict):
        sources.append(nested)
    if isinstance(response, dict):
        for key in ("error", "last_error"):
            found = response.get(key)
            if isinstance(found, dict):
                sources.append(found)

    code: str | None = None
    message: str | None = None
    for source in sources:
        raw_message = source.get("message")
        if message is None and isinstance(raw_message, str) and raw_message:
            message = raw_message
        raw_code = source.get("code")
        if code is None and isinstance(raw_code, str) and raw_code:
            code = raw_code
    if code is None and isinstance(nested, dict):
        nested_type = nested.get("type")
        if isinstance(nested_type, str) and nested_type:
            code = nested_type
    return code, message


def is_passing_failure(code: str | None, message: str | None) -> bool:
    """Whether an in-stream failure looks transient and safe to reissue."""
    text = " ".join(part for part in (code, message) if part).lower()
    if not text or any(hint in text for hint in _EXHAUSTED_QUOTA_HINTS):
        return False
    return any(hint in text for hint in _PASSING_FAILURE_HINTS)
