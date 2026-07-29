"""``delta config`` — interactive provider configuration."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

from delta_app.config import prompts
from delta_app.config.store import (
    PROVIDERS,
    config_path,
    load_config,
    mask_key,
    provider_info,
    save_config,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="delta config")
    p.add_argument("--provider", default=None, help="Provider key (skips the menu).")
    p.add_argument("--model", default=None, help="Model identifier (skips the menu).")
    p.add_argument(
        "--api-key", default=None,
        help="API key (skips the prompt). Also read from DELTA_API_KEY.",
    )
    p.add_argument(
        "--visible", action="store_true",
        help="Type the API key visibly, for terminals without hidden input.",
    )
    sub = p.add_subparsers(dest="action")
    sub.add_parser("show", help="Show the saved configuration (keys masked).")
    sub.add_parser("path", help="Print the configuration file path.")
    sub.add_parser("doctor", help="Report this terminal's hidden-input support.")
    return p


def _doctor() -> int:
    """Print terminal capability diagnostics."""
    from delta_app.config.secure_input import describe_terminal

    prompts.info("Terminal hidden-input diagnostics:\n")
    for line in describe_terminal():
        prompts.info(f"  {line}")
    prompts.info(
        "\nIf hidden input is unavailable, use one of:\n"
        "  delta config --visible\n"
        "  delta config --provider openai --model gpt-5 --api-key sk-...\n"
        "  $env:DELTA_API_KEY='sk-...'; delta config --provider openai"
    )
    return 0


def _save_noninteractive(
    provider: str, model: str | None, api_key: str | None,
) -> int:
    """Save configuration without prompting."""
    try:
        info = provider_info(provider.strip().lower())
    except KeyError:
        valid = ", ".join(p.key for p in PROVIDERS)
        prompts.info(f"Unknown provider {provider!r}. Choose one of: {valid}")
        return 1

    cfg = load_config()
    key = (api_key or "").strip()
    if not key and info.needs_key and info.key not in cfg.api_keys:
        prompts.info(f"{info.label} requires an API key (--api-key or DELTA_API_KEY).")
        return 1

    cfg.provider = info.key
    cfg.model = model or cfg.model or info.models[0]
    if key:
        cfg.api_keys[info.key] = key
    path = save_config(cfg)

    prompts.info(f"Configuration saved to {path}")
    prompts.info(f"Provider: {info.label}")
    prompts.info(f"Model:    {cfg.model}")
    return 0


def _show() -> int:
    """Print the saved configuration with API keys masked."""
    cfg = load_config()
    path = config_path()
    if not path.exists():
        prompts.info(f"No configuration at {path}. Run 'delta config' to create one.")
        return 1

    prompts.info(f"Config: {path}")
    prompts.info(f"Provider: {cfg.provider or '(unset)'}")
    prompts.info(f"Model:    {cfg.model or '(unset)'}")
    for name in sorted(cfg.api_keys):
        prompts.info(f"  {name}: {mask_key(cfg.api_keys[name])}")
    return 0


def _configure(*, visible: bool = False) -> int:
    """Run the interactive setup flow."""
    cfg = load_config()

    labels = [p.label for p in PROVIDERS]
    default = next(
        (i for i, p in enumerate(PROVIDERS) if p.key == cfg.provider), 0,
    )
    info = PROVIDERS[prompts.choose("Select provider", labels, default=default)]

    prompts.info(f"\n{info.label}")

    existing = cfg.api_keys.get(info.key)
    key = prompts.ask_api_key(
        info.label,
        required=info.needs_key,
        has_existing=bool(existing),
        visible=visible,
    )
    if key is None:
        key = existing or ""
    if info.needs_key and not key.strip():
        prompts.info("API key must not be empty. Nothing saved.")
        return 1

    current_model = cfg.model if cfg.provider == info.key else None
    model = prompts.choose_model(list(info.models), current=current_model)

    if not prompts.confirm("Save as default?"):
        prompts.info("Nothing saved.")
        return 1

    cfg.provider = info.key
    cfg.model = model
    if key.strip():
        cfg.api_keys[info.key] = key.strip()
    path = save_config(cfg)

    prompts.info(f"\nConfiguration saved to {path}")
    prompts.info(f"Provider: {info.label}")
    prompts.info(f"Model:    {model}")
    return 0


def handle_config(argv: Sequence[str]) -> int:
    """Entry point for the ``config`` subcommand."""
    ns = _build_parser().parse_args(list(argv))
    if ns.action == "show":
        return _show()
    if ns.action == "path":
        prompts.info(str(config_path()))
        return 0
    if ns.action == "doctor":
        return _doctor()

    api_key = ns.api_key or os.environ.get("DELTA_API_KEY")
    if ns.provider and (api_key or ns.model):
        return _save_noninteractive(ns.provider, ns.model, api_key)

    try:
        return _configure(visible=ns.visible)
    except prompts.PromptAborted:
        prompts.info("\nCancelled. Nothing saved.")
        return 130


__all__ = ["handle_config"]
