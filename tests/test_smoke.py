"""Smoke test: the three packages and the core contracts import."""

import delta_app
import delta_harness
import delta_model
from delta_harness import AgentEvent, CallBlock, HumanEntry, ToolSpec


def test_packages_import() -> None:
    assert delta_harness.__doc__
    assert delta_model.__doc__
    assert delta_app.__doc__


def test_contracts_importable() -> None:
    assert AgentEvent is not None
    assert ToolSpec is not None
    assert CallBlock is not None
    assert HumanEntry is not None
