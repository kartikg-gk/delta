"""Provider resolution, catalog, and environment-based auto-detection.

This module bridges ``delta_model.settings`` (immutable configuration types)
with ``delta_harness.provider.base`` (the ``ModelProvider`` protocol) so that
the CLI can obtain a ready-to-use provider from a name or from the environment
without knowing any provider implementation details.

Responsibilities:

1. **Detection** — infer which provider to use from env vars.
2. **Resolution** — load a typed profile, instantiate the matching adapter.
3. **Catalog** — list available providers, print setup guidance.
"""

from __future__ import annotations

import os
import sys

from delta_harness.provider.base import ModelProvider
from delta_model.settings import (
    AnthropicProfile,
    ConfigError,
    OpenAIProfile,
    load_anthropic_profile,
    load_openai_profile,
)

# ---------------------------------------------------------------------------
# Known provider identifiers
# ---------------------------------------------------------------------------

_KNOWN_PROVIDERS = ("anthropic", "openai", "scripted")


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def _detect_provider_name() -> str | None:
    """Infer the provider name from environment variables.

    Checks ``DELTA_PROVIDER`` first, then falls back to whichever API key is
    present.  Returns ``None`` when nothing is detectable.
    """
    explicit = os.environ.get("DELTA_PROVIDER")
    if explicit:
        return explicit.strip().lower()

    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"

    return None


# ---------------------------------------------------------------------------
# Builder helpers (one per provider)
# ---------------------------------------------------------------------------


def _build_anthropic(profile: AnthropicProfile) -> ModelProvider:
    """Instantiate an Anthropic adapter from a loaded profile."""
    try:
        from delta_model.anthropic import AnthropicProvider  # type: ignore[import-not-found]
    except ImportError:
        raise ConfigError(
            "The 'anthropic' provider requires the httpx extra.\n"
            "  Install it with: pip install delta[providers]"
        ) from None
    return AnthropicProvider(profile)  # type: ignore[return-value]


def _build_openai(profile: OpenAIProfile) -> ModelProvider:
    """Instantiate an OpenAI-compatible adapter from a loaded profile."""
    try:
        from delta_model.openai_compatible import OpenAIProvider  # type: ignore[import-not-found]
    except ImportError:
        raise ConfigError(
            "The 'openai' provider requires the httpx extra.\n"
            "  Install it with: pip install delta[providers]"
        ) from None
    return OpenAIProvider(profile)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def resolve_provider(name: str | None = None) -> ModelProvider:
    """Return a ready-to-use ``ModelProvider`` for *name*.

    When *name* is ``None`` the provider is auto-detected from the
    environment.  Raises ``ConfigError`` for missing configuration or
    unknown provider names.
    """
    resolved = name or _detect_provider_name()
    if resolved is None:
        raise ConfigError(
            "No provider configured.\n"
            "  Set DELTA_PROVIDER or export an API key env var.\n"
            "  Run 'delta provider setup' for guidance."
        )

    resolved = resolved.strip().lower()

    if resolved == "anthropic":
        profile = load_anthropic_profile()
        return _build_anthropic(profile)

    if resolved == "openai":
        profile = load_openai_profile()
        return _build_openai(profile)

    if resolved == "scripted":
        from delta_model.scripted import ReplayProvider

        return ReplayProvider([])  # type: ignore[return-value]

    raise ConfigError(
        f"Unknown provider: {resolved!r}.\n"
        f"  Known providers: {', '.join(_KNOWN_PROVIDERS)}"
    )


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


def _probe_status(provider_name: str) -> str:
    """Return a short status string for a provider: 'ready', 'no key', etc."""
    if provider_name == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return "no key  (set ANTHROPIC_API_KEY)"
        try:
            load_anthropic_profile()
            return "ready"
        except ConfigError as exc:
            return f"misconfigured  ({exc})"

    if provider_name == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            return "no key  (set OPENAI_API_KEY)"
        try:
            load_openai_profile()
            return "ready"
        except ConfigError as exc:
            return f"misconfigured  ({exc})"

    if provider_name == "scripted":
        return "available  (test only)"

    return "unknown"


def list_providers() -> None:
    """Print available providers and their configuration status to stderr."""
    active = _detect_provider_name()
    sys.stderr.write("Providers:\n")
    for name in _KNOWN_PROVIDERS:
        marker = "*" if name == active else " "
        status = _probe_status(name)
        sys.stderr.write(f"  {marker} {name:<12}  {status}\n")
    sys.stderr.flush()


def setup_provider() -> None:
    """Print environment-variable guidance for configuring each provider."""
    sys.stderr.write(
        "Provider setup\n"
        "\n"
        "Set the following environment variables for your chosen provider.\n"
        "\n"
        "Anthropic:\n"
        "  ANTHROPIC_API_KEY       (required)\n"
        "  ANTHROPIC_BASE_URL      (default: https://api.anthropic.com)\n"
        "  ANTHROPIC_TIMEOUT       (seconds, default: 120)\n"
        "  ANTHROPIC_MAX_TOKENS    (default: 16384)\n"
        "  ANTHROPIC_EXTRA_HEADERS (semicolon-separated Key: Value pairs)\n"
        "\n"
        "OpenAI-compatible:\n"
        "  OPENAI_API_KEY          (required)\n"
        "  OPENAI_BASE_URL         (default: https://api.openai.com/v1)\n"
        "  OPENAI_TIMEOUT          (seconds, default: 120)\n"
        "  OPENAI_ORG_ID           (optional)\n"
        "  OPENAI_EXTRA_HEADERS    (semicolon-separated Key: Value pairs)\n"
        "\n"
        "Shared:\n"
        "  DELTA_PROVIDER          (anthropic | openai | scripted)\n"
        "  DELTA_MODEL             (model identifier override)\n"
        "  DELTA_MAX_RETRIES       (default: 2)\n"
        "  DELTA_MAX_RETRY_DELAY   (seconds, default: 30)\n"
        "  DELTA_THINKING_ENABLED  (true/false, default: false)\n"
        "  DELTA_THINKING_BUDGET   (token count, optional)\n"
    )
    sys.stderr.flush()


def select_provider(name: str) -> None:
    """Validate a provider name for selection.

    Raises ``ConfigError`` if the name is not recognized.  The actual
    persistence of the selection is left to the caller (env var, dotfile,
    etc.).
    """
    normalized = name.strip().lower()
    if normalized not in _KNOWN_PROVIDERS:
        raise ConfigError(
            f"Unknown provider: {name!r}.\n"
            f"  Known providers: {', '.join(_KNOWN_PROVIDERS)}"
        )


__all__ = [
    "list_providers",
    "resolve_provider",
    "select_provider",
    "setup_provider",
]
