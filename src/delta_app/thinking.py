"""Provider-neutral thinking mode primitives.

Defines canonical *thinking levels* that the UI and configuration layer
expose to users, plus pure conversion helpers that translate those levels
into the parameters each model provider actually needs.

No provider classes, network calls, session state, or UI logic lives here.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

# ── canonical levels ────────────────────────────────────────────────────

ThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh"]

THINKING_LEVELS: tuple[ThinkingLevel, ...] = (
    "off",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
)

DEFAULT_THINKING_LEVEL: ThinkingLevel = "medium"

THINKING_DESCRIPTIONS: dict[ThinkingLevel, str] = {
    "off": "No reasoning",
    "minimal": "Very brief reasoning",
    "low": "Light reasoning",
    "medium": "Standard reasoning",
    "high": "Extended reasoning",
    "xhigh": "Maximum reasoning",
}

_LEVEL_SET: frozenset[str] = frozenset(THINKING_LEVELS)


# ── validation ──────────────────────────────────────────────────────────


def normalize_thinking_level(value: str | None) -> ThinkingLevel:
    """Resolve *value* to a canonical thinking level.

    Returns the default level when *value* is ``None``.  Strips whitespace,
    lower-cases, and raises a friendly ``ValueError`` for unknown values.
    """
    if value is None:
        return DEFAULT_THINKING_LEVEL

    cleaned = value.strip().lower()
    if cleaned not in _LEVEL_SET:
        available = ", ".join(THINKING_LEVELS)
        raise ValueError(
            f"Unknown thinking level {value!r}. "
            f"Available levels: {available}"
        )
    return cleaned  # type: ignore[return-value]


def normalize_thinking_levels(values: Sequence[str]) -> tuple[ThinkingLevel, ...]:
    """Normalize a sequence of thinking-level strings.

    Rejects empty input and duplicate levels.
    """
    if not values:
        raise ValueError("At least one thinking level is required")

    result: list[ThinkingLevel] = []
    seen: set[ThinkingLevel] = set()
    for v in values:
        level = normalize_thinking_level(v)
        if level in seen:
            raise ValueError(f"Duplicate thinking level: {level!r}")
        seen.add(level)
        result.append(level)
    return tuple(result)


# ── provider mappings ───────────────────────────────────────────────────

# Each helper converts a ThinkingLevel into the parameter shape expected
# by one family of model providers.  Provider names are intentionally kept
# out of public API surface — callers pick the right helper via their own
# provider resolution logic.

_EFFORT_MAP: dict[ThinkingLevel, str] = {
    "off": "none",
    "minimal": "minimal",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
}

_BUDGET_MAP: dict[ThinkingLevel, int | None] = {
    "off": None,
    "minimal": 1024,
    "low": 2048,
    "medium": 4096,
    "high": 8192,
    "xhigh": 16384,
}


def thinking_to_effort(level: ThinkingLevel) -> str:
    """Map a thinking level to an effort-style parameter value.

    Returns a string such as ``"none"``, ``"low"``, or ``"high"``
    suitable for providers that accept a reasoning-effort field.
    """
    return _EFFORT_MAP[level]


def thinking_to_budget(level: ThinkingLevel) -> int | None:
    """Map a thinking level to a token-budget parameter value.

    Returns ``None`` when thinking is off (the provider should omit
    the parameter entirely) or an integer token cap otherwise.
    """
    return _BUDGET_MAP[level]


# ── cycling helper ──────────────────────────────────────────────────────


def next_thinking_level(
    current: str,
    *,
    available: Sequence[ThinkingLevel] = THINKING_LEVELS,
) -> ThinkingLevel:
    """Return the level that follows *current* in *available*.

    Wraps around at the end.  Falls back to the first available level
    when *current* is not recognised.
    """
    if not available:
        return DEFAULT_THINKING_LEVEL

    cleaned = current.strip().lower()
    for i, lvl in enumerate(available):
        if lvl == cleaned:
            return available[(i + 1) % len(available)]
    # Unknown value — start from the beginning.
    return available[0]


__all__ = [
    "DEFAULT_THINKING_LEVEL",
    "THINKING_DESCRIPTIONS",
    "THINKING_LEVELS",
    "ThinkingLevel",
    "next_thinking_level",
    "normalize_thinking_level",
    "normalize_thinking_levels",
    "thinking_to_budget",
    "thinking_to_effort",
]
