"""Plan sign-in: use a ChatGPT subscription instead of an API key.

Flow (OAuth 2.0 authorization code with PKCE):

1. Build an authorization URL with a fresh verifier/challenge pair and a
   random ``state``; open it in the browser.
2. Wait for the code, from whichever arrives first: the local callback
   (``http://localhost:1455/auth/callback``) or the user pasting the code or
   the final redirect URL (for a browser on another machine).
3. Check ``state``, exchange the code for tokens, read the ChatGPT account id
   from the access token, and save everything to ``accounts.json`` in Delta's
   home, readable by the owner only.

Tokens are refreshed shortly before they expire. A refresh token may be
single-use, so refreshes are serialised (within the process by an asyncio
lock, across processes by a lock file) and the store is re-read inside the
lock: whoever refreshed first wins, and everyone else uses their result.
Error messages never include token values.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import secrets
import threading
import time
import webbrowser
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from delta_model.chatgpt import CHATGPT_BASE_URL, ORIGINATOR
from delta_model.settings import Credential

CHATGPT = "chatgpt"

_AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
_TOKEN_URL = "https://auth.openai.com/oauth/token"
# OpenAI's public client for Codex sign-in; apps identify themselves through
# the ``originator`` parameter.
_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
_CALLBACK_PORT = 1455
_CALLBACK_PATH = "/auth/callback"
_REDIRECT_URI = f"http://localhost:{_CALLBACK_PORT}{_CALLBACK_PATH}"
_SCOPE = "openid profile email offline_access"
_ACCOUNT_CLAIM = "https://api.openai.com/auth"
# Refresh this long before the access token actually expires.
_REFRESH_MARGIN_MS = 60_000
_LOCK_WAIT_SECONDS = 15.0
_LOCK_STALE_SECONDS = 60.0


class SignInError(RuntimeError):
    """Sign-in or refresh could not complete."""


@dataclass(frozen=True, slots=True)
class PlanLogin:
    """Stored tokens for one plan account."""

    access: str
    refresh: str
    expires_ms: int
    account_id: str

    def expiring(self, now_ms: int | None = None) -> bool:
        current = int(time.time() * 1000) if now_ms is None else now_ms
        return current >= self.expires_ms - _REFRESH_MARGIN_MS


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def accounts_path() -> Path:
    from delta_app.discovery import default_paths

    return default_paths().home / "accounts.json"


def _read_all() -> dict[str, Any]:
    try:
        data = json.loads(accounts_path().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        raise SignInError(f"{accounts_path()} is unreadable: {exc}") from exc
    return data if isinstance(data, dict) else {}


def _write_all(data: dict[str, Any]) -> None:
    path = accounts_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(data, stream)
    os.replace(temporary, path)
    os.chmod(path, 0o600)


def load_login(name: str = CHATGPT) -> PlanLogin | None:
    """The saved login for *name*, or ``None`` when signed out."""
    row = _read_all().get(name)
    if not isinstance(row, dict):
        return None
    try:
        return PlanLogin(
            access=str(row["access"]),
            refresh=str(row["refresh"]),
            expires_ms=int(row["expires_ms"]),
            account_id=str(row["account_id"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def save_login(login: PlanLogin, name: str = CHATGPT) -> None:
    data = _read_all()
    data[name] = asdict(login)
    _write_all(data)


def sign_out(name: str = CHATGPT) -> bool:
    """Forget the saved login. Returns whether there was one."""
    data = _read_all()
    if name not in data:
        return False
    del data[name]
    _write_all(data)
    return True


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _jwt_claims(token: str) -> dict[str, Any]:
    try:
        payload = token.split(".")[1]
        decoded = base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))
        claims = json.loads(decoded)
    except (IndexError, ValueError):
        return {}
    return claims if isinstance(claims, dict) else {}


def account_id_of(access_token: str) -> str | None:
    """The ChatGPT account id carried inside an access token."""
    auth = _jwt_claims(access_token).get(_ACCOUNT_CLAIM)
    account = auth.get("chatgpt_account_id") if isinstance(auth, dict) else None
    return account.strip() if isinstance(account, str) and account.strip() else None


def _expiry_ms(response: dict[str, Any], access_token: str) -> int:
    seconds = response.get("expires_in")
    if isinstance(seconds, int | float) and not isinstance(seconds, bool):
        return int(time.time() * 1000 + seconds * 1000)
    exp = _jwt_claims(access_token).get("exp")
    if isinstance(exp, int | float) and not isinstance(exp, bool):
        return int(exp * 1000)
    raise SignInError("the token response did not say when the token expires")


async def _token_request(form: dict[str, str], client: httpx.AsyncClient | None) -> PlanLogin:
    from delta_model.transport.client import build_async_client

    http = client or build_async_client(timeout=60)
    try:
        response = await http.post(_TOKEN_URL, data=form)
    except httpx.HTTPError as exc:
        raise SignInError(f"could not reach the sign-in server: {exc}") from exc
    finally:
        if client is None:
            await http.aclose()
    try:
        body = response.json()
    except ValueError:
        body = {}
    if response.status_code >= 400 or not isinstance(body, dict):
        reason = None
        if isinstance(body, dict):
            reason = body.get("error_description") or body.get("error")
        raise SignInError(
            f"sign-in server refused ({response.status_code}): {reason or 'no detail'}"
        )
    access = body.get("access_token")
    # A refresh may not rotate the refresh token; keep the old one then.
    refresh = body.get("refresh_token") or form.get("refresh_token")
    if not isinstance(access, str) or not access or not isinstance(refresh, str) or not refresh:
        raise SignInError("the sign-in server returned no usable tokens")
    account = account_id_of(access)
    if account is None:
        raise SignInError("the token carries no ChatGPT account; is this a ChatGPT plan?")
    return PlanLogin(access=access, refresh=refresh, expires_ms=_expiry_ms(body, access),
                     account_id=account)


# ---------------------------------------------------------------------------
# Sign-in
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Request:
    verifier: str
    state: str
    url: str


def _new_request() -> _Request:
    verifier = _b64url(secrets.token_bytes(48))
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    state = secrets.token_hex(16)
    query = urlencode({
        "response_type": "code",
        "client_id": _CLIENT_ID,
        "redirect_uri": _REDIRECT_URI,
        "scope": _SCOPE,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
        "originator": ORIGINATOR,
    })
    return _Request(verifier=verifier, state=state, url=f"{_AUTHORIZE_URL}?{query}")


def read_pasted_code(text: str) -> tuple[str | None, str | None]:
    """``(code, state)`` from a pasted redirect URL, query string, or bare code."""
    value = text.strip()
    if not value:
        return None, None
    parsed = urlparse(value)
    query = parsed.query if parsed.scheme and parsed.netloc else value if "=" in value else ""
    if query:
        params = parse_qs(query)
        codes, states = params.get("code"), params.get("state")
        return (codes[0] if codes else None), (states[0] if states else None)
    if "#" in value:
        code, state = value.split("#", 1)
        return code or None, state or None
    return value, None


class _Callback:
    """A one-shot local HTTP listener for the browser redirect."""

    def __init__(self, state: str) -> None:
        self._state = state
        self._loop = asyncio.get_running_loop()
        self.code: asyncio.Future[str] = self._loop.create_future()
        self._server: HTTPServer | None = None

    def start(self) -> bool:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - http.server naming
                url = urlparse(self.path)
                params = parse_qs(url.query)
                code = (params.get("code") or [""])[0]
                if url.path != _CALLBACK_PATH:
                    self._reply(404, "Not found.")
                elif (params.get("state") or [""])[0] != owner._state:
                    self._reply(400, "Sign-in check failed (state mismatch). Try again.")
                elif not code:
                    self._reply(400, "No sign-in code was received. Try again.")
                else:
                    self._reply(200, "Signed in to Delta. You can close this tab.")
                    owner._loop.call_soon_threadsafe(owner._deliver, code)

            def _reply(self, status: int, message: str) -> None:
                body = f"<!doctype html><meta charset=utf-8><p>{message}</p>".encode()
                self.send_response(status)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: Any) -> None:
                return

        try:
            self._server = HTTPServer(("127.0.0.1", _CALLBACK_PORT), Handler)
        except OSError:
            return False  # port busy: the user can still paste the code
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return True

    def _deliver(self, code: str) -> None:
        if not self.code.done():
            self.code.set_result(code)

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()


async def sign_in(
    *,
    show_url: Callable[[str], None],
    ask_code: Callable[[], Awaitable[str]] | None = None,
    open_browser: bool = True,
    client: httpx.AsyncClient | None = None,
    timeout: float = 300.0,
) -> PlanLogin:
    """Run the browser sign-in for a ChatGPT plan and save the result."""
    request = _new_request()
    callback = _Callback(request.state)
    listening = callback.start()
    show_url(request.url)
    if open_browser:
        webbrowser.open(request.url)
    waiters: list[asyncio.Future[str]] = []
    if listening:
        waiters.append(callback.code)
    if ask_code is not None:
        waiters.append(asyncio.ensure_future(ask_code()))
    if not waiters:
        callback.stop()
        raise SignInError(f"port {_CALLBACK_PORT} is busy and there is no way to paste a code")
    try:
        done, pending = await asyncio.wait(
            waiters, timeout=timeout, return_when=asyncio.FIRST_COMPLETED,
        )
        for waiter in pending:
            waiter.cancel()
        if not done:
            raise SignInError("sign-in timed out")
        raw = next(iter(done)).result()
    finally:
        callback.stop()

    code, state = read_pasted_code(raw)
    if state is not None and state != request.state:
        raise SignInError("sign-in check failed (state mismatch)")
    if not code:
        raise SignInError("no sign-in code was given")
    login = await _token_request({
        "grant_type": "authorization_code",
        "client_id": _CLIENT_ID,
        "code": code,
        "code_verifier": request.verifier,
        "redirect_uri": _REDIRECT_URI,
    }, client)
    save_login(login)
    return login


# ---------------------------------------------------------------------------
# Refresh, serialised
# ---------------------------------------------------------------------------

_refresh_lock = asyncio.Lock()


class _FileLock:
    """Cross-process lock: an exclusively created file, broken when stale."""

    def __init__(self, path: Path) -> None:
        self._path = path

    async def __aenter__(self) -> None:
        deadline = time.monotonic() + _LOCK_WAIT_SECONDS
        self._path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                os.close(os.open(self._path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
                return
            except FileExistsError:
                try:
                    if time.time() - self._path.stat().st_mtime > _LOCK_STALE_SECONDS:
                        self._path.unlink(missing_ok=True)
                        continue
                except FileNotFoundError:
                    continue
                if time.monotonic() > deadline:
                    raise SignInError(
                        "another Delta is refreshing the sign-in; try again"
                    ) from None
                await asyncio.sleep(0.1)

    async def __aexit__(self, *exc: object) -> None:
        self._path.unlink(missing_ok=True)


async def current_login(client: httpx.AsyncClient | None = None) -> PlanLogin:
    """The saved ChatGPT login, refreshed first if it is about to expire."""
    login = load_login()
    if login is None:
        raise SignInError("not signed in to a ChatGPT plan. Run `delta login chatgpt`.")
    if not login.expiring():
        return login
    async with _refresh_lock, _FileLock(accounts_path().with_suffix(".lock")):
        login = load_login()  # someone may have refreshed while we waited
        if login is None:
            raise SignInError("signed out while refreshing. Run `delta login chatgpt`.")
        if not login.expiring():
            return login
        fresh = await _token_request({
            "grant_type": "refresh_token",
            "client_id": _CLIENT_ID,
            "refresh_token": login.refresh,
        }, client)
        save_login(fresh)
        return fresh


async def chatgpt_credential() -> Credential:
    """Credential resolver for the ChatGPT provider."""
    login = await current_login()
    return Credential(
        api_key=login.access,
        base_url=CHATGPT_BASE_URL,
        extra_headers=(("chatgpt-account-id", login.account_id),),
    )


__all__ = [
    "CHATGPT",
    "PlanLogin",
    "SignInError",
    "account_id_of",
    "accounts_path",
    "chatgpt_credential",
    "current_login",
    "load_login",
    "read_pasted_code",
    "save_login",
    "sign_in",
    "sign_out",
]
