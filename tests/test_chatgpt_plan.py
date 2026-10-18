"""ChatGPT-plan sign-in, token refresh, and the plan provider. No real network."""

from __future__ import annotations

import asyncio
import base64
import json
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from delta_app.accounts import (
    PlanLogin,
    SignInError,
    account_id_of,
    current_login,
    load_login,
    read_pasted_code,
    save_login,
    sign_in,
    sign_out,
)
from delta_harness.contracts.transcript import (
    HumanEntry,
    ModelEntry,
    TextSegment,
    ThoughtSegment,
)
from delta_harness.provider.wire import StreamCloseEvent
from delta_model.chatgpt import ChatGPTProvider, build_chatgpt_payload, responses_url
from delta_model.settings import Credential, OpenAIProfile, RetryPolicy


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("DELTA_HOME", str(tmp_path))
    return tmp_path


def _jwt(claims: dict[str, Any]) -> str:
    def part(value: dict[str, Any]) -> str:
        raw = json.dumps(value).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    return f"{part({'alg': 'none'})}.{part(claims)}.sig"


def _access(account: str = "acct-1", exp: float | None = None) -> str:
    claims: dict[str, Any] = {"https://api.openai.com/auth": {"chatgpt_account_id": account}}
    if exp is not None:
        claims["exp"] = exp
    return _jwt(claims)


def _token_client(handler: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ---------------------------------------------------------------------------
# Small pieces
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("pasted", "expected"), [
    ("http://localhost:1455/auth/callback?code=abc&state=xyz", ("abc", "xyz")),
    ("code=abc&state=xyz", ("abc", "xyz")),
    ("abc#xyz", ("abc", "xyz")),
    ("  abc  ", ("abc", None)),
    ("", (None, None)),
])
def test_pasted_codes_are_understood(pasted: str, expected: tuple[str | None, str | None]) -> None:
    assert read_pasted_code(pasted) == expected


def test_account_id_is_read_from_the_token() -> None:
    assert account_id_of(_access("acct-9")) == "acct-9"
    assert account_id_of("not.a.jwt") is None
    assert account_id_of(_jwt({"sub": "x"})) is None


# ---------------------------------------------------------------------------
# Sign-in
# ---------------------------------------------------------------------------


async def test_sign_in_with_a_pasted_code_saves_the_login(home: Path) -> None:
    seen: dict[str, str] = {}

    def token(request: httpx.Request) -> httpx.Response:
        seen.update(dict(httpx.QueryParams(request.content.decode())))
        return httpx.Response(200, json={
            "access_token": _access("acct-7"), "refresh_token": "r1", "expires_in": 3600,
        })

    urls: list[str] = []

    async def paste() -> str:
        return "the-code"

    login = await sign_in(
        show_url=urls.append, ask_code=paste, open_browser=False,
        client=_token_client(token),
    )
    assert login.account_id == "acct-7" and login.refresh == "r1"
    assert load_login() == login
    assert seen["grant_type"] == "authorization_code" and seen["code"] == "the-code"
    assert "originator=delta" in urls[0] and "code_challenge_method=S256" in urls[0]
    assert seen["code_verifier"]  # PKCE verifier sent with the exchange


async def test_a_code_from_another_sign_in_is_refused() -> None:
    async def paste() -> str:
        return "http://localhost:1455/auth/callback?code=c&state=someone-else"

    with pytest.raises(SignInError, match="state mismatch"):
        await sign_in(show_url=lambda _u: None, ask_code=paste, open_browser=False,
                      client=_token_client(lambda r: httpx.Response(500)))
    assert load_login() is None


async def test_refused_exchange_does_not_echo_secrets() -> None:
    async def paste() -> str:
        return "secret-code-123"

    def token(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "invalid_grant", "echo": "secret-code-123"})

    with pytest.raises(SignInError) as caught:
        await sign_in(show_url=lambda _u: None, ask_code=paste, open_browser=False,
                      client=_token_client(token))
    assert "invalid_grant" in str(caught.value)
    assert "secret-code-123" not in str(caught.value)


def test_sign_out_forgets_the_login() -> None:
    save_login(PlanLogin("a", "r", int(time.time() * 1000) + 10**7, "acct"))
    assert sign_out() is True
    assert load_login() is None and sign_out() is False


# ---------------------------------------------------------------------------
# Refresh
# ---------------------------------------------------------------------------


async def test_fresh_tokens_are_used_without_a_request() -> None:
    login = PlanLogin(_access(), "r", int(time.time() * 1000) + 3_600_000, "acct-1")
    save_login(login)

    def never(request: httpx.Request) -> httpx.Response:
        raise AssertionError("refreshed a fresh token")

    assert await current_login(client=_token_client(never)) == login


async def test_concurrent_refreshes_send_one_request() -> None:
    save_login(PlanLogin(_access(), "old-refresh", int(time.time() * 1000), "acct-1"))
    calls: list[str] = []

    async def token(request: httpx.Request) -> httpx.Response:
        calls.append(dict(httpx.QueryParams(request.content.decode()))["refresh_token"])
        await asyncio.sleep(0.05)
        return httpx.Response(200, json={
            "access_token": _access(), "refresh_token": "new-refresh", "expires_in": 3600,
        })

    client = _token_client(token)
    first, second = await asyncio.gather(current_login(client=client),
                                         current_login(client=client))
    assert calls == ["old-refresh"]
    assert first.refresh == second.refresh == "new-refresh"
    assert load_login().refresh == "new-refresh"  # type: ignore[union-attr]


async def test_not_signed_in_is_a_clear_error() -> None:
    with pytest.raises(SignInError, match="delta login chatgpt"):
        await current_login()


# ---------------------------------------------------------------------------
# The provider
# ---------------------------------------------------------------------------


def test_endpoint_is_found_at_any_base_depth() -> None:
    for base in ("https://chatgpt.com/backend-api", "https://chatgpt.com/backend-api/codex",
                 "https://chatgpt.com/backend-api/codex/responses/"):
        assert responses_url(base) == "https://chatgpt.com/backend-api/codex/responses"


def test_payload_is_stateless_and_hands_reasoning_back() -> None:
    item = {"type": "reasoning", "id": "rs_1", "encrypted_content": "opaque", "summary": []}
    reply = ModelEntry(content=[
        ThoughtSegment(thinking="", thinking_signature=json.dumps(item)),
        TextSegment(text="Answer"),
    ])
    body = build_chatgpt_payload(
        model="gpt-5.5", system="", messages=[HumanEntry(content="q"), reply], tools=[],
        reasoning={"effort": "high"},
    )
    assert body["store"] is False and body["stream"] is True
    assert body["instructions"]  # never empty
    assert body["include"] == ["reasoning.encrypted_content"]
    assert body["reasoning"] == {"effort": "high", "summary": "auto"}
    assert body["input"][0] == {"role": "user", "content": [{"type": "input_text", "text": "q"}]}
    assert body["input"][1] == item  # reasoning precedes the reply text
    assert body["input"][2]["role"] == "assistant"


_DONE_STREAM = (
    'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"hi"}\n\n'
    'event: response.output_item.done\ndata: {"type":"response.output_item.done","output_index":0,'
    '"item":{"type":"reasoning","id":"rs_9","encrypted_content":"enc","summary":[]}}\n\n'
    'event: response.completed\ndata: {"type":"response.completed","response":{"status":"completed"}}\n\n'
)


def _provider(handler: Any, retries: int = 0) -> ChatGPTProvider:
    async def resolver() -> Credential:
        return Credential(api_key="tok", base_url="https://chatgpt.com/backend-api",
                          extra_headers=(("chatgpt-account-id", "acct-1"),))

    provider = ChatGPTProvider(OpenAIProfile(
        name="chatgpt", credential=Credential(api_key=None, base_url=""),
        retry=RetryPolicy(max_retries=retries, max_delay_seconds=0.01), resolver=resolver,
    ))
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return provider


async def test_requests_carry_plan_headers_and_keep_reasoning() -> None:
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, text=_DONE_STREAM)

    events = [e async for e in _provider(handler).stream_response(
        model="gpt-5.5", system="s", messages=[HumanEntry(content="q")], tools=[],
        cache_key="session-1",
    )]
    request = sent[0]
    assert str(request.url) == "https://chatgpt.com/backend-api/codex/responses"
    assert request.headers["authorization"] == "Bearer tok"
    assert request.headers["chatgpt-account-id"] == "acct-1"
    assert request.headers["originator"] == "delta"
    assert json.loads(request.content)["prompt_cache_key"] == "session-1"
    last = events[-1]
    assert isinstance(last, StreamCloseEvent) and last.message.text == "hi"
    thoughts = [s for s in last.message.content if isinstance(s, ThoughtSegment)]
    assert thoughts and json.loads(thoughts[0].thinking_signature or "")["id"] == "rs_9"


async def test_used_up_plan_limit_is_not_retried() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(429, text='{"error":{"type":"usage_limit_reached"}}')

    events = [e async for e in _provider(handler, retries=3).stream_response(
        model="gpt-5.5", system="s", messages=[HumanEntry(content="q")], tools=[],
    )]
    assert len(calls) == 1 and events[-1].error.stop_reason == "error"  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def test_resolving_the_plan_provider_needs_a_sign_in() -> None:
    from delta_app.config.config import resolve_provider
    from delta_model.settings import ConfigError

    with pytest.raises(ConfigError, match="delta login chatgpt"):
        resolve_provider("chatgpt")
    save_login(PlanLogin("a", "r", int(time.time() * 1000) + 10**7, "acct"))
    assert isinstance(resolve_provider("chatgpt"), ChatGPTProvider)


def test_plan_models_have_limits_but_no_api_prices() -> None:
    from delta_app.catalog import model_info, reload

    reload()
    info = model_info("chatgpt", "gpt-5.5")
    assert info is not None and info.context_window and info.pricing is None


def test_logout_command(capsys: pytest.CaptureFixture[str]) -> None:
    from delta_app.auth import logout_cli

    assert logout_cli([]) == 0
    assert "Not signed in" in capsys.readouterr().err
    save_login(PlanLogin("a", "r", 1, "acct"))
    assert logout_cli(["chatgpt"]) == 0
    assert "Signed out" in capsys.readouterr().err
