"""Claude subscription protocol tests; no real accounts or paid requests."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from delta_app import claude_accounts as auth
from delta_app.claude_login import OAUTH_SYSTEM_PROMPT, build_provider
from delta_harness.contracts.transcript import HumanEntry
from delta_model.claude import _auth_headers, _compose_body
from delta_model.settings import ReasoningPolicy


@pytest.fixture(autouse=True)
def isolated_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth, "accounts_path", lambda: tmp_path / "claude.json")
    monkeypatch.setattr(auth, "_refresh_lock", asyncio.Lock())


def test_authorization_protocol() -> None:
    request = auth._new_request()
    params = parse_qs(urlparse(request.url).query)
    assert urlparse(request.url).netloc == "claude.ai"
    assert params["client_id"] == ["9d1c250a-e61b-44d9-88ed-5944d1962f5e"]
    assert params["redirect_uri"] == ["http://localhost:53692/callback"]
    assert params["state"] == [request.verifier]
    assert params["code"] == ["true"]
    assert "user:inference" in params["scope"][0]
    expected = base64.urlsafe_b64encode(hashlib.sha256(request.verifier.encode()).digest()).decode().rstrip("=")
    assert params["code_challenge"] == [expected]
    assert params["code_challenge_method"] == ["S256"]


async def test_manual_login_exchange_and_logout(monkeypatch: pytest.MonkeyPatch) -> None:
    request = auth._new_request()
    monkeypatch.setattr(auth, "_new_request", lambda: request)
    monkeypatch.setattr(auth._Callback, "start", lambda self: False)

    async def ask() -> str:
        return f"http://localhost:53692/callback?code=my-code&state={request.state}"

    def handler(req: httpx.Request) -> httpx.Response:
        assert str(req.url) == auth._TOKEN_URL
        assert req.headers["Content-Type"] == "application/json"
        assert json.loads(req.content) == {
            "grant_type": "authorization_code", "client_id": auth._CLIENT_ID,
            "code": "my-code", "state": request.state,
            "redirect_uri": auth._REDIRECT_URI, "code_verifier": request.verifier,
        }
        return httpx.Response(200, json={"access_token": "access", "refresh_token": "refresh", "expires_in": 3600})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await auth.sign_in(show_url=lambda url: None, ask_code=ask, open_browser=False, client=client)
    assert auth.load_login() == result
    assert auth.sign_out()
    assert auth.load_login() is None
    assert not auth.sign_out()


async def test_wrong_state_does_not_exchange(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth._Callback, "start", lambda self: False)

    async def ask() -> str:
        return "code#wrong-state"

    with pytest.raises(auth.SignInError, match="state mismatch"):
        await auth.sign_in(show_url=lambda url: None, ask_code=ask, open_browser=False)
    assert auth.load_login() is None


async def test_concurrent_refresh_rotates_once() -> None:
    auth.save_login(auth.PlanLogin("expired", "old-refresh", 0))
    calls = 0

    async def handler(req: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)
        assert json.loads(req.content)["refresh_token"] == "old-refresh"
        return httpx.Response(200, json={"access_token": "new-access", "refresh_token": "new-refresh", "expires_in": 3600})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        results = await asyncio.gather(*(auth.current_login(client) for _ in range(4)))
    assert calls == 1
    assert all(row.refresh == "new-refresh" for row in results)
    assert auth.load_login() == results[0]


async def test_refresh_without_rotation_and_redacted_failure() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"access_token": "new", "expires_in": 3600})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await auth._token_request({"refresh_token": "keep-me"}, client)
    assert result.refresh == "keep-me"

    def reject(req: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "invalid_grant", "error_description": "bad secret-refresh"})
    async with httpx.AsyncClient(transport=httpx.MockTransport(reject)) as client:
        with pytest.raises(auth.SignInError) as caught:
            await auth._token_request({"refresh_token": "secret-refresh"}, client)
    assert "secret-refresh" not in str(caught.value)
    assert "invalid_grant" in str(caught.value)


@pytest.mark.parametrize("seconds", [None, True, 0, -1, "3600", float("inf")])
async def test_invalid_expiry(seconds: object) -> None:
    # Infinity is represented directly by a mocked response to exercise validation.
    response = httpx.Response(200, content=json.dumps({"access_token": "a", "refresh_token": "r", "expires_in": seconds}))
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req: response)) as client:
        with pytest.raises(auth.SignInError, match="expires_in"):
            await auth._token_request({}, client)


async def test_runtime_headers_and_identity_reach_messages_api() -> None:
    auth.save_login(auth.PlanLogin("access", "refresh", int(time.time() * 1000) + 3600000))
    provider = build_provider()
    await provider.close()
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        assert str(req.url) == "https://api.anthropic.com/v1/messages"
        assert req.headers["Authorization"] == "Bearer access"
        assert "x-api-key" not in req.headers
        assert req.headers["user-agent"] == "claude-cli/delta"
        assert req.headers["anthropic-beta"] == "claude-code-20250219,oauth-2025-04-20"
        assert req.headers["x-app"] == "cli"
        body = json.loads(req.content)
        assert body["system"][0] == {"type": "text", "text": OAUTH_SYSTEM_PROMPT}
        assert body["system"][1]["text"] == "Delta context"
        assert "cache_control" not in body["system"][0]
        assert body["system"][1]["cache_control"] == {"type": "ephemeral"}
        return httpx.Response(200, text='event: message_start\ndata: {"message":{"id":"m","usage":{}}}\n\nevent: message_stop\ndata: {}\n\n')

    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        _ = [event async for event in provider.stream_response(model="claude-test", system="Delta context", messages=[HumanEntry(content="hello")], tools=[])]
    finally:
        await provider.close()
    assert len(seen) == 1


def test_api_key_provider_unchanged() -> None:
    from delta_model.settings import Credential
    assert _auth_headers(Credential(api_key="api-key", base_url="https://api.anthropic.com"))["x-api-key"] == "api-key"
    body = _compose_body(model="claude", system="Normal", messages=[], tools=[], max_tokens=100, thinking=ReasoningPolicy())
    assert body["system"] == "Normal"


async def test_browser_callback_checks_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth, "_CALLBACK_PORT", 0)
    callback = auth._Callback("expected-state")
    assert callback.start()
    assert callback._server is not None
    port = callback._server.server_port
    try:
        async with httpx.AsyncClient(trust_env=False) as client:
            wrong = await client.get(f"http://127.0.0.1:{port}/callback?code=bad&state=wrong")
            assert wrong.status_code == 400
            assert not callback.code.done()
            good = await client.get(f"http://127.0.0.1:{port}/callback?code=accepted&state=expected-state")
            assert good.status_code == 200
            assert await asyncio.wait_for(callback.code, 1) == "accepted"
    finally:
        callback.stop()


async def test_cancel_cleans_up_manual_waiter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth._Callback, "start", lambda self: False)
    stopped = []
    monkeypatch.setattr(auth._Callback, "stop", lambda self: stopped.append(True))
    cancelled = asyncio.Event()

    async def ask() -> str:
        try:
            await asyncio.Future()
        finally:
            cancelled.set()
        return ""

    with pytest.raises(auth.SignInError, match="timed out"):
        await auth.sign_in(show_url=lambda url: None, ask_code=ask, open_browser=False, timeout=0.01)
    await asyncio.wait_for(cancelled.wait(), 1)
    assert stopped


def test_cli_login_selects_claude(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from delta_app import claude_login
    from delta_app.auth import login_cli, logout_cli
    from delta_app.config.store import load_config

    monkeypatch.setenv("DELTA_CONFIG_DIR", str(tmp_path / "config"))

    async def fake_login() -> None:
        auth.save_login(auth.PlanLogin("a", "r", 9999999999999))

    monkeypatch.setattr(claude_login, "sign_in_on_console", fake_login)
    assert login_cli(["claude"]) == 0
    assert load_config().provider == "claude"
    assert logout_cli(["claude"]) == 0
    assert auth.load_login() is None
