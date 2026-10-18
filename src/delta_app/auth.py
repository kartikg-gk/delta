"""``delta login`` / ``delta logout``: connect or disconnect a provider.

API-key providers go through ``delta config``. A ChatGPT plan signs in with
the browser instead (see :mod:`delta_app.accounts`).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import threading

from delta_app.accounts import CHATGPT, SignInError, sign_in, sign_out


def _read_line_in_background(prompt: str) -> asyncio.Future[str]:
    """Read one line without blocking exit: the reader thread is a daemon,
    so a sign-in finished by the browser does not wait for Enter."""
    loop = asyncio.get_running_loop()
    future: asyncio.Future[str] = loop.create_future()

    def read() -> None:
        try:
            line = input(prompt)
        except (EOFError, OSError):
            return
        if not future.done():
            loop.call_soon_threadsafe(
                lambda: None if future.done() else future.set_result(line)
            )

    threading.Thread(target=read, daemon=True).start()
    return future


async def sign_in_on_console() -> None:
    """Browser sign-in for a ChatGPT plan, driven from the terminal."""

    def show(url: str) -> None:
        print("Opening your browser to sign in to ChatGPT.", file=sys.stderr)
        print(f"If it does not open, visit:\n  {url}\n", file=sys.stderr)

    async def ask() -> str:
        return await _read_line_in_background(
            "Or paste the final redirect URL (or code) here: "
        )

    await sign_in(show_url=show, ask_code=ask if sys.stdin.isatty() else None)


def _remember_choice(provider: str) -> None:
    from delta_app.config.store import load_config, provider_info, save_config

    config = load_config()
    config.provider = provider
    config.model = provider_info(provider).models[0]
    save_config(config)


def login_cli(args: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="delta login")
    parser.add_argument("provider", nargs="?")
    ns = parser.parse_args(args)
    if ns.provider == CHATGPT:
        try:
            asyncio.run(sign_in_on_console())
        except SignInError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            print("\nSign-in cancelled.", file=sys.stderr)
            return 130
        _remember_choice(CHATGPT)
        print("Signed in. Delta will use your ChatGPT plan.", file=sys.stderr)
        return 0
    from delta_app.cli.config_cmd import handle_config

    return handle_config(["--provider", ns.provider] if ns.provider else [])


def logout_cli(args: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="delta logout")
    parser.add_argument("provider", nargs="?", default=CHATGPT)
    ns = parser.parse_args(args)
    if ns.provider != CHATGPT:
        print(
            f"{ns.provider} uses an API key; remove it with 'delta config' or unset its "
            "environment variable.",
            file=sys.stderr,
        )
        return 1
    print("Signed out." if sign_out() else "Not signed in.", file=sys.stderr)
    return 0
