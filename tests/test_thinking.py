"""Tests for thinking.py — thinking levels, validation, mappings, cycling."""

from __future__ import annotations

import pytest

from delta_app.reasoning import (
    DEFAULT_THINKING_LEVEL,
    THINKING_DESCRIPTIONS,
    THINKING_LEVELS,
    ThinkingLevel,
    next_thinking_level,
    normalize_thinking_level,
    normalize_thinking_levels,
    thinking_to_budget,
    thinking_to_effort,
)

# ── constants ───────────────────────────────────────────────────────────


class TestConstants:
    def test_levels_tuple(self) -> None:
        assert THINKING_LEVELS == ("off", "minimal", "low", "medium", "high", "xhigh")

    def test_default_is_medium(self) -> None:
        assert DEFAULT_THINKING_LEVEL == "medium"

    def test_all_levels_have_descriptions(self) -> None:
        for level in THINKING_LEVELS:
            assert level in THINKING_DESCRIPTIONS
            assert isinstance(THINKING_DESCRIPTIONS[level], str)
            assert len(THINKING_DESCRIPTIONS[level]) > 0

    def test_descriptions_count(self) -> None:
        assert len(THINKING_DESCRIPTIONS) == len(THINKING_LEVELS)

    def test_default_in_levels(self) -> None:
        assert DEFAULT_THINKING_LEVEL in THINKING_LEVELS


# ── normalize_thinking_level ────────────────────────────────────────────


class TestNormalizeThinkingLevel:
    def test_none_returns_default(self) -> None:
        assert normalize_thinking_level(None) == DEFAULT_THINKING_LEVEL

    def test_exact_match(self) -> None:
        for level in THINKING_LEVELS:
            assert normalize_thinking_level(level) == level

    def test_case_insensitive(self) -> None:
        assert normalize_thinking_level("HIGH") == "high"
        assert normalize_thinking_level("Off") == "off"
        assert normalize_thinking_level("MEDIUM") == "medium"

    def test_strips_whitespace(self) -> None:
        assert normalize_thinking_level("  low  ") == "low"
        assert normalize_thinking_level("\thigh\n") == "high"

    def test_combined_whitespace_and_case(self) -> None:
        assert normalize_thinking_level("  XHIGH  ") == "xhigh"

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown thinking level"):
            normalize_thinking_level("turbo")

    def test_error_includes_available(self) -> None:
        with pytest.raises(ValueError, match="off"):
            normalize_thinking_level("invalid")

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError):
            normalize_thinking_level("")


# ── normalize_thinking_levels ───────────────────────────────────────────


class TestNormalizeThinkingLevels:
    def test_single_value(self) -> None:
        assert normalize_thinking_levels(["high"]) == ("high",)

    def test_multiple_values(self) -> None:
        result = normalize_thinking_levels(["off", "low", "high"])
        assert result == ("off", "low", "high")

    def test_normalizes_each(self) -> None:
        result = normalize_thinking_levels(["  OFF ", "HIGH"])
        assert result == ("off", "high")

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one"):
            normalize_thinking_levels([])

    def test_duplicates_raise(self) -> None:
        with pytest.raises(ValueError, match="Duplicate"):
            normalize_thinking_levels(["high", "high"])

    def test_case_insensitive_duplicates(self) -> None:
        with pytest.raises(ValueError, match="Duplicate"):
            normalize_thinking_levels(["Low", "low"])

    def test_returns_tuple(self) -> None:
        result = normalize_thinking_levels(["off", "minimal"])
        assert isinstance(result, tuple)

    def test_all_levels(self) -> None:
        result = normalize_thinking_levels(list(THINKING_LEVELS))
        assert result == THINKING_LEVELS


# ── thinking_to_effort ──────────────────────────────────────────────────


class TestThinkingToEffort:
    def test_off(self) -> None:
        assert thinking_to_effort("off") == "none"

    def test_minimal(self) -> None:
        assert thinking_to_effort("minimal") == "minimal"

    def test_low(self) -> None:
        assert thinking_to_effort("low") == "low"

    def test_medium(self) -> None:
        assert thinking_to_effort("medium") == "medium"

    def test_high(self) -> None:
        assert thinking_to_effort("high") == "high"

    def test_xhigh(self) -> None:
        assert thinking_to_effort("xhigh") == "xhigh"

    def test_all_levels_mapped(self) -> None:
        for level in THINKING_LEVELS:
            result = thinking_to_effort(level)
            assert isinstance(result, str)


# ── thinking_to_budget ──────────────────────────────────────────────────


class TestThinkingToBudget:
    def test_off_is_none(self) -> None:
        assert thinking_to_budget("off") is None

    def test_minimal(self) -> None:
        assert thinking_to_budget("minimal") == 1024

    def test_low(self) -> None:
        assert thinking_to_budget("low") == 2048

    def test_medium(self) -> None:
        assert thinking_to_budget("medium") == 4096

    def test_high(self) -> None:
        assert thinking_to_budget("high") == 8192

    def test_xhigh(self) -> None:
        assert thinking_to_budget("xhigh") == 16384

    def test_budgets_increase(self) -> None:
        active = [level for level in THINKING_LEVELS if level != "off"]
        budgets = [thinking_to_budget(level) for level in active]
        assert budgets == sorted(budgets)  # type: ignore[type-var]

    def test_all_levels_mapped(self) -> None:
        for level in THINKING_LEVELS:
            thinking_to_budget(level)  # should not raise


# ── next_thinking_level ─────────────────────────────────────────────────


class TestNextThinkingLevel:
    def test_advances(self) -> None:
        assert next_thinking_level("off") == "minimal"
        assert next_thinking_level("minimal") == "low"
        assert next_thinking_level("low") == "medium"
        assert next_thinking_level("medium") == "high"
        assert next_thinking_level("high") == "xhigh"

    def test_wraps_around(self) -> None:
        assert next_thinking_level("xhigh") == "off"

    def test_unknown_falls_back(self) -> None:
        assert next_thinking_level("turbo") == "off"

    def test_case_insensitive(self) -> None:
        assert next_thinking_level("HIGH") == "xhigh"

    def test_strips_whitespace(self) -> None:
        assert next_thinking_level("  low  ") == "medium"

    def test_custom_available(self) -> None:
        subset: tuple[ThinkingLevel, ...] = ("off", "medium", "high")
        assert next_thinking_level("off", available=subset) == "medium"
        assert next_thinking_level("medium", available=subset) == "high"
        assert next_thinking_level("high", available=subset) == "off"

    def test_custom_available_unknown_fallback(self) -> None:
        subset: tuple[ThinkingLevel, ...] = ("low", "high")
        assert next_thinking_level("off", available=subset) == "low"

    def test_empty_available_returns_default(self) -> None:
        assert next_thinking_level("high", available=()) == DEFAULT_THINKING_LEVEL

    def test_full_cycle(self) -> None:
        current = "off"
        visited: list[str] = [current]
        for _ in range(len(THINKING_LEVELS)):
            current = next_thinking_level(current)
            visited.append(current)
        assert visited[0] == visited[-1]
        assert len(set(visited)) == len(THINKING_LEVELS)
