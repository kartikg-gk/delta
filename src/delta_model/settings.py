
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol


class ConfigError(Exception):
    """Raised when provider configuration is invalid or incomplete."""




def _read_str(name: str, *, default: str) -> str:
    """Read a string env var, returning *default* when unset or empty."""
    return os.environ.get(name) or default


def _require_str(name: str, *, hint: str = "") -> str:
    """Read a required string env var, raising on absence."""
    value = os.environ.get(name)
    if not value:
        msg = f"{name} is not set."
        if hint:
            msg = f"{msg} {hint}"
        raise ConfigError(msg)
    return value


def _read_int(name: str, *, default: int, minimum: int | None = None) -> int:
    """Read an integer env var with optional lower bound."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"{name} must be an integer, got {raw!r}.") from None
    if minimum is not None and value < minimum:
        raise ConfigError(f"{name} must be at least {minimum}, got {value}.")
    return value


def _read_int_or_none(name: str, *, minimum: int | None = None) -> int | None:
    """Read an optional integer env var that may be absent."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"{name} must be an integer, got {raw!r}.") from None
    if minimum is not None and value < minimum:
        raise ConfigError(f"{name} must be at least {minimum}, got {value}.")
    return value


def _read_float(name: str, *, default: float, positive: bool = False) -> float:
    """Read a float env var with optional positivity constraint."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a number, got {raw!r}.") from None
    if positive and value <= 0:
        raise ConfigError(f"{name} must be positive, got {value}.")
    return value


def _read_bool(name: str, *, default: bool) -> bool:
    """Read a boolean env var (true/false, 1/0, yes/no, on/off)."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    lower = raw.strip().lower()
    if lower in {"1", "true", "yes", "on"}:
        return True
    if lower in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} must be a boolean (true/false/1/0), got {raw!r}.")


def _parse_header_env(name: str) -> tuple[tuple[str, str], ...]:
    """Parse semicolon-delimited ``Key: Value`` pairs from an env var."""
    raw = os.environ.get(name)
    if not raw or not raw.strip():
        return ()
    pairs: list[tuple[str, str]] = []
    for segment in raw.split(";"):
        segment = segment.strip()
        if not segment:
            continue
        if ":" not in segment:
            raise ConfigError(
                f"Each entry in {name} must be 'Key: Value', got {segment!r}."
            )
        key, _, value = segment.partition(":")
        key = key.strip()
        if not key:
            raise ConfigError(f"Empty header name in {name}.")
        pairs.append((key, value.strip()))
    return tuple(pairs)




@dataclass(frozen=True, slots=True)
class Credential:
    """Static authentication material for a provider endpoint."""

    api_key: str | None
    base_url: str
    extra_headers: tuple[tuple[str, str], ...] = ()


class CredentialResolver(Protocol):
    """Async callable that produces fresh authentication material.

    Implementations may contact a secrets manager, rotate tokens, or
    perform any other async operation to return an up-to-date
    ``Credential`` snapshot.
    """

    async def __call__(self) -> Credential: ...


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Controls automatic reattempts after transient provider failures."""

    max_retries: int = 2
    max_delay_seconds: float = 30.0


@dataclass(frozen=True, slots=True)
class ReasoningPolicy:
    """Controls extended thinking / chain-of-thought budget."""

    enabled: bool = False
    budget_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class AnthropicProfile:
    """Fully resolved configuration for an Anthropic provider endpoint."""

    name: str
    credential: Credential
    timeout_seconds: float = 120.0
    max_tokens: int = 16384
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    reasoning: ReasoningPolicy = field(default_factory=ReasoningPolicy)
    resolver: CredentialResolver | None = None


@dataclass(frozen=True, slots=True)
class OpenAIProfile:
    """Fully resolved configuration for an OpenAI-compatible provider endpoint."""

    name: str
    credential: Credential
    timeout_seconds: float = 120.0
    organization: str | None = None
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    reasoning: ReasoningPolicy = field(default_factory=ReasoningPolicy)
    resolver: CredentialResolver | None = None




_ANTHROPIC_DEFAULT_URL = "https://api.anthropic.com"
_OPENAI_DEFAULT_URL = "https://api.openai.com/v1"


def _load_retry_policy() -> RetryPolicy:
    """Build a ``RetryPolicy`` from shared ``DELTA_`` env vars."""
    return RetryPolicy(
        max_retries=_read_int("DELTA_MAX_RETRIES", default=2, minimum=0),
        max_delay_seconds=_read_float(
            "DELTA_MAX_RETRY_DELAY", default=30.0, positive=True
        ),
    )


def _load_reasoning_policy() -> ReasoningPolicy:
    """Build a ``ReasoningPolicy`` from shared ``DELTA_`` env vars."""
    return ReasoningPolicy(
        enabled=_read_bool("DELTA_THINKING_ENABLED", default=False),
        budget_tokens=_read_int_or_none("DELTA_THINKING_BUDGET", minimum=1),
    )




def load_anthropic_profile() -> AnthropicProfile:
    credential = Credential(
        api_key=_require_str(
            "ANTHROPIC_API_KEY",
            hint="Get one at https://console.anthropic.com/settings/keys",
        ),
        base_url=_read_str("ANTHROPIC_BASE_URL", default=_ANTHROPIC_DEFAULT_URL),
        extra_headers=_parse_header_env("ANTHROPIC_EXTRA_HEADERS"),
    )
    return AnthropicProfile(
        name="anthropic",
        credential=credential,
        timeout_seconds=_read_float(
            "ANTHROPIC_TIMEOUT", default=120.0, positive=True
        ),
        max_tokens=_read_int("ANTHROPIC_MAX_TOKENS", default=16384, minimum=1),
        retry=_load_retry_policy(),
        reasoning=_load_reasoning_policy(),
    )


def load_openai_profile() -> OpenAIProfile:

    credential = Credential(
        api_key=_require_str(
            "OPENAI_API_KEY",
            hint="Get one at https://platform.openai.com/api-keys",
        ),
        base_url=_read_str("OPENAI_BASE_URL", default=_OPENAI_DEFAULT_URL),
        extra_headers=_parse_header_env("OPENAI_EXTRA_HEADERS"),
    )
    return OpenAIProfile(
        name="openai",
        credential=credential,
        timeout_seconds=_read_float(
            "OPENAI_TIMEOUT", default=120.0, positive=True
        ),
        organization=os.environ.get("OPENAI_ORG_ID") or None,
        retry=_load_retry_policy(),
        reasoning=_load_reasoning_policy(),
    )


__all__ = [
    "AnthropicProfile",
    "ConfigError",
    "Credential",
    "CredentialResolver",
    "OpenAIProfile",
    "ReasoningPolicy",
    "RetryPolicy",
    "load_anthropic_profile",
    "load_openai_profile",
]
