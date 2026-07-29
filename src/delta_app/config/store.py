"""Persisted user configuration: schema, file location, and TOML read/write.

The file lives outside the repository (``~/.delta/config.toml`` by default) and
is written with owner-only permissions because it holds API keys.  This module
knows nothing about provider implementations — it only stores strings.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Provider metadata (plain data — the set of backends Delta can talk to)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProviderInfo:
    """Static facts about a supported provider backend."""

    key: str
    label: str
    key_env: str
    base_url: str
    models: tuple[str, ...]
    needs_key: bool = True


PROVIDERS: tuple[ProviderInfo, ...] = (
    ProviderInfo(
        key="openai",
        label="OpenAI",
        key_env="OPENAI_API_KEY",
        base_url="https://api.openai.com/v1",
        models=("gpt-5", "gpt-5-mini", "gpt-4.1"),
    ),
    ProviderInfo(
        key="anthropic",
        label="Anthropic",
        key_env="ANTHROPIC_API_KEY",
        base_url="https://api.anthropic.com",
        models=("claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"),
    ),
    ProviderInfo(
        key="openrouter",
        label="OpenRouter",
        key_env="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        models=(
            "anthropic/claude-opus-5",
            "openai/gpt-5",
            "meta-llama/llama-3.3-70b-instruct",
        ),
    ),
    ProviderInfo(
        key="ollama",
        label="Ollama",
        key_env="OLLAMA_API_KEY",
        base_url="http://localhost:11434/v1",
        models=("llama3.3", "qwen2.5-coder", "deepseek-r1"),
        needs_key=False,
    ),
)


def provider_info(key: str) -> ProviderInfo:
    """Look up provider metadata by key, raising ``KeyError`` when unknown."""
    for info in PROVIDERS:
        if info.key == key:
            return info
    raise KeyError(key)


PROVIDER_KEYS: tuple[str, ...] = tuple(p.key for p in PROVIDERS)


# ---------------------------------------------------------------------------
# Config file schema
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DeltaConfig:
    """Contents of the user's ``config.toml``."""

    provider: str | None = None
    model: str | None = None
    api_keys: dict[str, str] = field(default_factory=dict)
    base_urls: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Location
# ---------------------------------------------------------------------------


def config_dir() -> Path:
    """Directory holding Delta's user configuration."""
    override = os.environ.get("DELTA_CONFIG_DIR")
    if override:
        return Path(override)
    return Path.home() / ".delta"


def config_path() -> Path:
    """Full path to ``config.toml``."""
    return config_dir() / "config.toml"


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


def load_config() -> DeltaConfig:
    """Read the config file, returning empty defaults when absent or malformed."""
    path = config_path()
    try:
        raw = path.read_bytes()
    except OSError:
        return DeltaConfig()

    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError):
        return DeltaConfig()

    providers = data.get("providers")
    api_keys: dict[str, str] = {}
    base_urls: dict[str, str] = {}
    if isinstance(providers, dict):
        for name, section in providers.items():
            if not isinstance(section, dict):
                continue
            key = section.get("api_key")
            if isinstance(key, str) and key:
                api_keys[name] = key
            url = section.get("base_url")
            if isinstance(url, str) and url:
                base_urls[name] = url

    provider = data.get("provider")
    model = data.get("model")
    return DeltaConfig(
        provider=provider if isinstance(provider, str) and provider else None,
        model=model if isinstance(model, str) and model else None,
        api_keys=api_keys,
        base_urls=base_urls,
    )


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def _quote(value: str) -> str:
    """Serialize a string as a TOML basic string."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def render_config(config: DeltaConfig) -> str:
    """Render *config* as TOML text."""
    lines: list[str] = []
    if config.provider:
        lines.append(f"provider = {_quote(config.provider)}")
    if config.model:
        lines.append(f"model = {_quote(config.model)}")

    names = sorted(set(config.api_keys) | set(config.base_urls))
    for name in names:
        lines.append("")
        lines.append(f"[providers.{name}]")
        if name in config.api_keys:
            lines.append(f"api_key = {_quote(config.api_keys[name])}")
        if name in config.base_urls:
            lines.append(f"base_url = {_quote(config.base_urls[name])}")
    return "\n".join(lines) + "\n"


def save_config(config: DeltaConfig) -> Path:
    """Write *config* to disk with owner-only permissions."""
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_config(config), encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass  # best effort — some filesystems (e.g. FAT) reject chmod
    return path


def mask_key(value: str) -> str:
    """Render an API key safe for display: last four characters only."""
    if not value:
        return ""
    if len(value) <= 4:
        return "*" * len(value)
    return "*" * (len(value) - 4) + value[-4:]


__all__ = [
    "DeltaConfig",
    "PROVIDERS",
    "PROVIDER_KEYS",
    "ProviderInfo",
    "config_dir",
    "config_path",
    "load_config",
    "mask_key",
    "provider_info",
    "render_config",
    "save_config",
]
