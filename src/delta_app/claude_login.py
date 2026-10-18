"""Claude subscription login entry points for the CLI and TUI."""

from __future__ import annotations

import sys

from delta_app.claude_accounts import (
    CLAUDE_BASE_URL,
    claude_credential,
    load_login,
    sign_in,
)
from delta_model.claude import AnthropicProvider
from delta_model.settings import AnthropicProfile, ConfigError, Credential

# The subscription transport sends this identity block before the app prompt.
OAUTH_SYSTEM_PROMPT = "You are Claude Code, Anthropic's official CLI for Claude."


def build_provider() -> AnthropicProvider:
    if load_login() is None:
        raise ConfigError("Not signed in to Claude. Run 'delta login claude'.")
    return AnthropicProvider(AnthropicProfile(
        name="claude",
        credential=Credential(api_key=None, base_url=CLAUDE_BASE_URL),
        resolver=claude_credential,
        oauth_system_prompt=OAUTH_SYSTEM_PROMPT,
        cache_retention="short",
    ))


async def sign_in_on_console() -> None:
    from delta_app.auth import _read_line_in_background

    def show(url: str) -> None:
        print(f"Opening your browser to sign in to Claude.\n{url}\n", file=sys.stderr)

    async def ask() -> str:
        return await _read_line_in_background("Paste the redirect URL or authorization code: ")

    await sign_in(show_url=show, ask_code=ask if sys.stdin.isatty() else None)
