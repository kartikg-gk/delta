"""Model catalog: snapshot building, layering, refresh, and session use."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from delta_app import catalog
from delta_app.catalog import (
    CatalogError,
    available_providers,
    build_snapshot,
    bundled_snapshot,
    current_snapshot,
    model_info,
    models_for,
    parse_snapshot,
    refresh_catalog,
    refreshed_path,
    user_catalog_problem,
)


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("DELTA_HOME", str(tmp_path))
    monkeypatch.delenv("DELTA_CATALOG", raising=False)
    catalog.reload()
    return tmp_path


def _model(**extra: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": "M", "tool_call": True, "reasoning": True,
        "limit": {"context": 200_000, "output": 32_000},
        "modalities": {"input": ["text", "image"], "output": ["text"]},
        "cost": {"input": 3, "output": 15, "cache_read": 0.3, "cache_write": 3.75},
    }
    row.update(extra)
    return row


def _source(**models: dict[str, Any]) -> dict[str, Any]:
    return {"anthropic": {"models": models}, "unrelated": {"models": {"x": _model()}}}


# ---------------------------------------------------------------------------
# Building and validating snapshots
# ---------------------------------------------------------------------------


def test_snapshot_keeps_only_usable_models() -> None:
    doc = build_snapshot(_source(
        good=_model(),
        no_tools=_model(tool_call=False),
        old=_model(status="deprecated"),
        images_only=_model(modalities={"input": ["image"], "output": ["text"]}),
    ), generated_at=1)
    assert list(doc["providers"]) == ["anthropic"]
    assert list(doc["providers"]["anthropic"]) == ["good"]


def test_snapshot_keeps_suggested_models_even_without_tool_calling() -> None:
    suggested = available_providers()[1].models[0]  # anthropic's first suggestion
    doc = build_snapshot(_source(**{suggested: _model(tool_call=False)}), generated_at=1)
    assert suggested in doc["providers"]["anthropic"]


def test_snapshot_records_limits_reasoning_vision_and_prices() -> None:
    doc = build_snapshot(_source(good=_model()), generated_at=1)
    info = parse_snapshot(doc)["anthropic"]["good"]
    assert (info.context_window, info.max_output, info.reasoning, info.vision) == (
        200_000, 32_000, True, True,
    )
    assert info.pricing is not None and info.pricing.output == 15


def test_snapshot_with_no_known_providers_is_rejected() -> None:
    with pytest.raises(CatalogError):
        build_snapshot({"unrelated": {"models": {"x": _model()}}}, generated_at=1)


@pytest.mark.parametrize("bad", [None, {}, {"schema": 2, "providers": {}}, {"schema": 1}])
def test_bad_snapshots_are_rejected(bad: object) -> None:
    with pytest.raises(CatalogError):
        parse_snapshot(bad)


def test_bundled_snapshot_covers_every_suggested_model() -> None:
    bundled = bundled_snapshot()
    assert bundled is not None
    parsed = parse_snapshot(bundled)
    for provider in ("openai", "anthropic", "openrouter", "huggingface"):
        for model in next(p for p in available_providers() if p.key == provider).models:
            assert model in parsed[provider], f"{provider}/{model} missing from bundled snapshot"


# ---------------------------------------------------------------------------
# Layering
# ---------------------------------------------------------------------------


def _write_refreshed(home: Path, generated_at: int, models: dict[str, Any]) -> None:
    (home / "model-catalog.json").write_text(json.dumps({
        "schema": 1, "checked_at": 0, "etag": None,
        "catalog": {"schema": 1, "generated_at": generated_at,
                    "providers": {"anthropic": models}},
    }), encoding="utf-8")


def test_refreshed_copy_wins_only_when_newer(home: Path) -> None:
    bundled_at = bundled_snapshot()["generated_at"]  # type: ignore[index]
    _write_refreshed(home, bundled_at - 1, {"older-model": {"name": "o"}})
    assert "older-model" not in current_snapshot()["providers"]["anthropic"]  # type: ignore[index]
    _write_refreshed(home, bundled_at + 1, {"newer-model": {"name": "n"}})
    assert "newer-model" in current_snapshot()["providers"]["anthropic"]  # type: ignore[index]


def test_user_catalog_adds_providers_models_and_corrections(home: Path) -> None:
    (home / "catalog.toml").write_text(
        '[providers.together]\n'
        'label = "Together"\n'
        'base_url = "https://api.together.xyz/v1"\n'
        'models = ["meta-llama/Llama-3.3-70B"]\n'
        '\n[[models]]\nprovider = "together"\nid = "meta-llama/Llama-3.3-70B"\n'
        'context_window = 131072\nreasoning = false\n'
        '\n[[models]]\nprovider = "openai"\nid = "gpt-5"\ncontext_window = 1234\n',
        encoding="utf-8",
    )
    catalog.reload()
    together = next(p for p in available_providers() if p.key == "together")
    assert together.key_env == "TOGETHER_API_KEY" and together.needs_key
    info = model_info("together", "meta-llama/Llama-3.3-70B")
    assert info is not None and info.context_window == 131072 and info.reasoning is False
    corrected = model_info("openai", "gpt-5")
    assert corrected is not None and corrected.context_window == 1234
    assert corrected.pricing is not None  # the rest of the bundled facts survive


def test_broken_user_catalog_is_reported_but_harmless(home: Path) -> None:
    (home / "catalog.toml").write_text("[[models]]\nprovider = 'x'\n", encoding="utf-8")
    catalog.reload()
    assert user_catalog_problem() is not None
    assert model_info("anthropic", available_providers()[1].models[0]) is not None


def test_suggested_models_come_first() -> None:
    names = models_for("openai")
    suggested = next(p for p in available_providers() if p.key == "openai").models
    assert names[: len(suggested)] == suggested
    assert len(names) > len(suggested)


# ---------------------------------------------------------------------------
# Refresh
# ---------------------------------------------------------------------------


def _client(handler: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_offline_refresh_never_touches_the_network(home: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("network used while offline")

    result = await refresh_catalog(force=True, client=_client(handler))
    assert not result.refreshed and result.model_count > 0


async def test_refresh_saves_and_publishes_the_new_catalog(
    home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DELTA_OFFLINE")
    bundled_at = bundled_snapshot()["generated_at"]  # type: ignore[index]
    body = _source(**{"brand-new": _model()})
    client = _client(lambda r: httpx.Response(200, json=body, headers={"etag": "v1"}))
    result = await refresh_catalog(client=client, now=bundled_at / 1000 + 10)
    assert result.refreshed and result.model_count == 1
    assert model_info("anthropic", "brand-new") is not None
    assert json.loads(refreshed_path().read_text(encoding="utf-8"))["etag"] == "v1"


async def test_recent_check_skips_the_network_unless_forced(
    home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DELTA_OFFLINE")
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, json=_source(good=_model()))

    await refresh_catalog(client=_client(handler), now=1_000.0)
    await refresh_catalog(client=_client(handler), now=1_060.0)
    assert len(calls) == 1
    await refresh_catalog(client=_client(handler), now=1_120.0, force=True)
    assert len(calls) == 2


async def test_unchanged_catalog_is_not_downloaded_again(
    home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DELTA_OFFLINE")
    sent: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request.headers.get("if-none-match"))
        if len(sent) == 1:
            return httpx.Response(200, json=_source(good=_model()), headers={"etag": "abc"})
        return httpx.Response(304)

    await refresh_catalog(client=_client(handler), now=1_000.0)
    result = await refresh_catalog(client=_client(handler), now=1_000.0, force=True)
    assert sent == [None, "abc"] and result.not_modified


@pytest.mark.parametrize("response", [
    httpx.Response(500),
    httpx.Response(200, text="not json"),
    httpx.Response(200, json={"unrelated": {"models": {}}}),
])
async def test_failed_refresh_keeps_the_existing_catalog(
    home: Path, monkeypatch: pytest.MonkeyPatch, response: httpx.Response,
) -> None:
    monkeypatch.delenv("DELTA_OFFLINE")
    before = current_snapshot()
    with pytest.raises(CatalogError):
        await refresh_catalog(force=True, client=_client(lambda r: response))
    assert not refreshed_path().exists()
    assert current_snapshot() == before


# ---------------------------------------------------------------------------
# The session uses the catalog
# ---------------------------------------------------------------------------


async def _session(tmp_path: Path, provider: str, model: str):  # type: ignore[no-untyped-def]
    from delta_app.conversation import CodingSession
    from delta_harness.contracts.transcript import ModelEntry, TextSegment, UsageStats
    from delta_harness.provider.wire import StreamCloseEvent
    from delta_model.scripted import ReplayProvider

    reply = ModelEntry(content=[TextSegment(text="ok")], stop_reason="stop",
                       usage=UsageStats(input=1_000_000, output=1_000_000, total_tokens=2_000_000))
    session = await CodingSession.create(
        provider=ReplayProvider([[StreamCloseEvent(reason="stop", message=reply)]]),
        provider_name=provider, model=model, system="s", sessions_dir=tmp_path,
    )

    async def no_title() -> str:
        return ""

    session.auto_name = no_title  # type: ignore[method-assign]
    return session


async def test_session_uses_catalog_window_and_prices_unpriced_replies(tmp_path: Path) -> None:
    model = available_providers()[1].models[0]
    info = model_info("anthropic", model)
    assert info is not None and info.context_window and info.pricing
    session = await _session(tmp_path, "anthropic", model)
    assert session._context_limit() == info.context_window
    _ = [e async for e in session.submit("hi")]
    expected = info.pricing.input + info.pricing.output  # one million tokens each
    assert session.usage.total_cost_usd == pytest.approx(expected)


def test_catalog_reasoning_overrides_the_name_guess() -> None:
    from delta_model._oai.helpers import build_reasoning_extra
    from delta_model.settings import ReasoningPolicy

    on = ReasoningPolicy(enabled=True, effort="high")
    # A name the built-in rules would not recognise, declared reasoning-capable.
    assert build_reasoning_extra(
        ReasoningPolicy(enabled=True, effort="high", supported=True),
        model="newfamily-1", endpoint="chat",
    ) == {"reasoning_effort": "high"}
    assert build_reasoning_extra(on, model="newfamily-1", endpoint="chat") == {}
    # A model the rules would guess reasons, declared not to.
    assert build_reasoning_extra(
        ReasoningPolicy(enabled=True, effort="high", supported=False),
        model="o4-mini", endpoint="chat",
    ) == {}


def test_update_models_command_reports_offline(
    home: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    import sys

    from delta_app.cli.main import main

    argv = sys.argv
    sys.argv = ["delta", "update", "--models"]
    try:
        assert main() == 0
    finally:
        sys.argv = argv
    out = capsys.readouterr()
    assert "Offline" in out.out + out.err
