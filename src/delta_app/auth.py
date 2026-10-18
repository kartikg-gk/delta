"""API-key setup and subscription sign-in through the official CLIs."""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import subprocess
from collections.abc import Callable


def executable(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise ValueError(f"{name} CLI is not installed. Install the official CLI, then retry login.")
    return path


def subscription_command(provider: str) -> list[str]:
    if provider == "codex":
        return [executable("codex"), "login", "--device-auth"]
    if provider == "claude-code":
        return [executable("claude"), "auth", "login"]
    raise ValueError(f"Unknown subscription provider: {provider}")


async def subscription_login(provider: str, notify: Callable[[str], None]) -> None:
    command = subscription_command(provider)
    process = await asyncio.create_subprocess_exec(
        *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    assert process.stdout is not None
    try:
        async with asyncio.timeout(300):
            while line := await process.stdout.readline():
                notify(line.decode("utf-8", errors="replace").rstrip())
            if await process.wait() != 0:
                raise ValueError(f"{provider} login failed. Retry with delta login {provider}.")
    finally:
        if process.returncode is None:
            process.terminate()
            await process.wait()


def login_cli(args: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="delta login")
    parser.add_argument("provider", nargs="?")
    ns = parser.parse_args(args)
    if ns.provider in {"codex", "claude-code"}:
        try:
            code = subprocess.call(subscription_command(ns.provider))
            if code == 0:
                from delta_app.config.store import load_config, provider_info, save_config

                config = load_config()
                config.provider = ns.provider
                config.model = provider_info(ns.provider).models[0]
                save_config(config)
            return code
        except ValueError as exc:
            parser.error(str(exc))
    from delta_app.cli.config_cmd import handle_config

    return handle_config(["--provider", ns.provider] if ns.provider else [])
