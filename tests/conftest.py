"""Shared test fixtures."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from delta_app.trust import reset_trust_cache


@pytest.fixture(autouse=True)
def _fresh_trust_state() -> Iterator[None]:
    """Project-trust resolutions are per process; keep them per test."""
    reset_trust_cache()
    yield
    reset_trust_cache()


@pytest.fixture(autouse=True)
def _offline_model_catalog(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """No test may reach models.dev; each starts from the bundled catalog."""
    from delta_app import catalog

    monkeypatch.setenv("DELTA_OFFLINE", "1")
    catalog.reload()
    yield
    catalog.reload()
