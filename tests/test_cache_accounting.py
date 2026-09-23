"""Prompt-cache routing hints and cache usage accounting."""

from __future__ import annotations

import json

import httpx

from delta_app.context.budget import estimate_transcript_tokens
from delta_app.conversation import SessionStats
from delta_harness.contracts.transcript import HumanEntry, ModelEntry, UsageStats
from delta_model._oai.helpers import extract_usage
from delta_model.oai_compatible import OpenAIProvider
from delta_model.settings import Credential, OpenAIProfile, RetryPolicy

_OK = "".join(
    f"data: {json.dumps(frame)}\n\n"
    for frame in (
        {"id": "c", "choices": [{"index": 0, "delta": {"content": "hi"}}]},
        {"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    )
) + "data: [DONE]\n\n"


async def _capture(*, base_url: str, model: str = "gpt-4o", cache_key: str | None,
                   affinity: bool | None = None) -> httpx.Request:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, text=_OK)

    provider = OpenAIProvider(OpenAIProfile(
        name="openai",
        credential=Credential(api_key="k", base_url=base_url),
        retry=RetryPolicy(max_retries=0),
        cache_affinity=affinity,
    ))
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    kwargs = {"cache_key": cache_key} if cache_key else {}
    _ = [e async for e in provider.stream_response(
        model=model, system="s", messages=[HumanEntry(content="q")], tools=[], **kwargs,
    )]
    await provider.close()
    return seen[0]


# ---------------------------------------------------------------------------
# Routing hint
# ---------------------------------------------------------------------------


async def test_first_party_chat_sends_key_in_body_only() -> None:
    request = await _capture(base_url="https://api.openai.com/v1", cache_key="sess-1")
    assert json.loads(request.content)["prompt_cache_key"] == "sess-1"
    assert "session_id" not in request.headers


async def test_key_is_clamped_to_limit() -> None:
    request = await _capture(base_url="https://api.openai.com/v1", cache_key="x" * 100)
    assert json.loads(request.content)["prompt_cache_key"] == "x" * 64


async def test_gateways_do_not_get_the_hint_unless_opted_in() -> None:
    plain = await _capture(base_url="http://localhost:8000/v1", cache_key="sess-1")
    assert "prompt_cache_key" not in json.loads(plain.content)
    opted = await _capture(base_url="http://localhost:8000/v1", cache_key="sess-1",
                           affinity=True)
    assert json.loads(opted.content)["prompt_cache_key"] == "sess-1"


async def test_no_key_means_no_hint() -> None:
    request = await _capture(base_url="https://api.openai.com/v1", cache_key=None)
    assert "prompt_cache_key" not in json.loads(request.content)


# ---------------------------------------------------------------------------
# Usage normalization
# ---------------------------------------------------------------------------


def test_chat_usage_separates_cached_tokens() -> None:
    usage = extract_usage({
        "prompt_tokens": 1000, "completion_tokens": 20,
        "prompt_tokens_details": {"cached_tokens": 800},
    })
    assert (usage.input, usage.cache_read, usage.cache_write) == (200, 800, 0)


def test_responses_usage_reads_cache_writes() -> None:
    usage = extract_usage({
        "input_tokens": 1000, "output_tokens": 5,
        "input_tokens_details": {"cached_tokens": 600, "cache_write_tokens": 300},
        "output_tokens_details": {"reasoning_tokens": 3},
    })
    assert (usage.input, usage.cache_read, usage.cache_write) == (100, 600, 300)
    assert usage.reasoning == 3


def test_context_estimate_counts_cached_prompt() -> None:
    reply = ModelEntry(usage=UsageStats(input=50, cache_read=9000, cache_write=950))
    assert estimate_transcript_tokens([HumanEntry(content="q"), reply]) == 10_000


# ---------------------------------------------------------------------------
# Hit rates
# ---------------------------------------------------------------------------


def test_hit_rates_hidden_without_cache_activity() -> None:
    stats = SessionStats(total_input_tokens=500, latest_prompt_tokens=500)
    assert stats.cache_hit_rate is None
    assert stats.latest_cache_hit_rate is None


def test_session_and_latest_rates_differ() -> None:
    stats = SessionStats(
        total_input_tokens=2000, cache_read_tokens=900, cache_write_tokens=1000,
        latest_prompt_tokens=1000, latest_cache_read_tokens=900,
    )
    assert stats.cache_hit_rate == 0.45
    assert stats.latest_cache_hit_rate == 0.9


async def test_providers_without_cache_key_still_work() -> None:
    """Older/custom adapters that don't declare ``cache_key`` must not receive it."""
    from delta_harness.contracts.transcript import TextSegment
    from delta_harness.driver import RuntimeConfig, RuntimeHarness
    from delta_harness.provider.wire import StreamCloseEvent

    class Legacy:
        async def stream_response(self, *, model, system, messages, tools, signal=None):
            reply = ModelEntry(content=[TextSegment(text="ok")], stop_reason="stop")
            yield StreamCloseEvent(reason="stop", message=reply)

    harness = RuntimeHarness(
        RuntimeConfig(provider=Legacy(), model="m", system="s", cache_key="sess"),
    )
    events = [e async for e in harness.submit("hi")]
    assert harness.transcript[-1].text == "ok"
    assert events
