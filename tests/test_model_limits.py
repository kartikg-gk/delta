"""Tests for delta_model.limits."""

from __future__ import annotations

import pytest

from delta_model.limits import LimitsError, LimitsProbe, ModelLimits


class TestModelLimits:
    def test_usable_context(self):
        assert ModelLimits(100_000, 16_000).usable_context == 84_000

    def test_frozen(self):
        with pytest.raises(AttributeError):
            ModelLimits(100_000, 8_000).context_window = 1  # type: ignore[misc]

    @pytest.mark.parametrize("window,output", [
        (0, 100), (-1, 100), (100, 0), (100, -1), (100, 100), (100, 200),
    ])
    def test_rejects_invalid(self, window, output):
        with pytest.raises(LimitsError):
            ModelLimits(window, output)


class TestLimitsProbe:
    def test_protocol_detection(self):
        class P:
            async def query_limits(self, model: str) -> ModelLimits: ...

        assert isinstance(P(), LimitsProbe)
        assert not isinstance(object(), LimitsProbe)
