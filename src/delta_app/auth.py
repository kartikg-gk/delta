"""``delta login``: connect a provider."""

from __future__ import annotations

import argparse


def login_cli(args: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="delta login")
    parser.add_argument("provider", nargs="?")
    ns = parser.parse_args(args)
    from delta_app.cli.config_cmd import handle_config

    return handle_config(["--provider", ns.provider] if ns.provider else [])
