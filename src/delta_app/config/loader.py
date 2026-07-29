"""Resolve provider, model, and credentials from environment then saved config.

Environment variables always win; the config file is the fallback.  The
existing ``delta_model.settings`` loaders read credentials from the
environment, so :func:`apply_config` exports resolved values into
``os.environ`` for this process only when they are not already set — which is
exactly the precedence rule, and keeps the provider layer untouched.
"""

from __future__ import annotations

import os

from delta_app.config.store import DeltaConfig, load_config, provider_info

_DEFAULT_PROVIDER = "anthropic"


def load_provider(config: DeltaConfig | None = None) -> str:
    """Active provider: ``DELTA_PROVIDER``, then config, then detection."""
    env = os.environ.get("DELTA_PROVIDER")
    if env:
        return env.strip().lower()

    cfg = load_config() if config is None else config
    if cfg.provider:
        return cfg.provider.strip().lower()

    # No explicit choice — infer from whichever key is present.
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return _DEFAULT_PROVIDER


def load_model(provider: str, config: DeltaConfig | None = None) -> str | None:
    """Model for *provider*: ``DELTA_MODEL``, then config, then ``None``."""
    env = os.environ.get("DELTA_MODEL")
    if env:
        return env
    cfg = load_config() if config is None else config
    if cfg.model and cfg.provider == provider:
        return cfg.model
    return None


def load_api_key(provider: str, config: DeltaConfig | None = None) -> str | None:
    """Key for *provider*: its env var, then config, then ``None``."""
    try:
        info = provider_info(provider)
    except KeyError:
        return None
    env = os.environ.get(info.key_env)
    if env:
        return env
    cfg = load_config() if config is None else config
    return cfg.api_keys.get(provider)


def apply_config(provider: str, config: DeltaConfig | None = None) -> None:
    """Export saved credentials for *provider* into this process's environment.

    Only fills variables that are unset, so real environment variables keep
    precedence.  OpenAI-compatible backends (OpenRouter, Ollama) are mapped
    onto the ``OPENAI_*`` variables the existing loader already reads.
    """
    try:
        info = provider_info(provider)
    except KeyError:
        return

    cfg = load_config() if config is None else config
    key = cfg.api_keys.get(provider)
    base_url = cfg.base_urls.get(provider) or info.base_url

    if provider == "anthropic":
        key_var, url_var = "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"
    else:
        key_var, url_var = "OPENAI_API_KEY", "OPENAI_BASE_URL"

    # A real env var for the provider's own key name wins over the config file.
    native = os.environ.get(info.key_env)
    if native:
        key = native

    if key and not os.environ.get(key_var):
        os.environ[key_var] = key
    if not os.environ.get(url_var):
        os.environ[url_var] = base_url

    # Ollama needs no credential but the OpenAI loader requires a non-empty key.
    if not info.needs_key and not os.environ.get(key_var):
        os.environ[key_var] = "ollama"


__all__ = ["apply_config", "load_api_key", "load_model", "load_provider"]
