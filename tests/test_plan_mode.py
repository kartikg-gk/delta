"""Tests for plan mode — restricted tools, prompt block, and the /plan command."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from delta_app.conversation import CodingSession
from delta_app.planning import (
    PLAN_MODE_INSTRUCTIONS,
    plan_system_prompt,
    restrict_tools,
)
from delta_harness.contracts.tooling import ToolOutcome, ToolSpec
from delta_harness.contracts.transcript import ModelEntry, TextSegment
from delta_harness.contracts.transcript.diagnostics import CostBreakdown, UsageStats
from delta_harness.contracts.values import JValue
from delta_harness.provider.wire import StreamCloseEvent
from delta_model.scripted import ReplayProvider

# ── helpers ────────────────────────────────────────────────────────────────


def _make_provider() -> ReplayProvider:
    reply = ModelEntry(
        model="test-model",
        content=[TextSegment(text="ok")],
        stop_reason="stop",
        usage=UsageStats(
            total_tokens=1, input=1, output=0, cost=CostBreakdown(total=0.0)
        ),
    )
    return ReplayProvider([[StreamCloseEvent(reason="stop", message=reply)]])


def _tool(name: str, calls: list[str] | None = None) -> ToolSpec:
    """Build a ToolSpec that records the command it was called with."""

    async def run(
        tool_call_id: str,
        arguments: Mapping[str, JValue],
        signal: object = None,
        on_update: object = None,
    ) -> ToolOutcome:
        if calls is not None:
            calls.append(str(arguments.get("command", "")))
        return ToolOutcome(content="ran")

    return ToolSpec(
        name=name,
        label=name,
        description=name,
        parameters={"type": "object", "properties": {}},
        run=run,
    )


def _all_tools(calls: list[str] | None = None) -> list[ToolSpec]:
    return [_tool("Read"), _tool("Write"), _tool("Edit"), _tool("Bash", calls)]


async def _session(tools: list[ToolSpec] | None = None) -> CodingSession:
    return await CodingSession.create(
        provider=_make_provider(),
        provider_name="test",
        model="test-model",
        system="BASE PROMPT",
        tools=tools if tools is not None else _all_tools(),
    )


# ── restrict_tools ─────────────────────────────────────────────────────────


def test_restrict_drops_mutating_tools() -> None:
    names = [t.name for t in restrict_tools(_all_tools())]
    assert names == ["Read", "Bash"]


def test_restrict_leaves_original_list_untouched() -> None:
    tools = _all_tools()
    restrict_tools(tools)
    assert [t.name for t in tools] == ["Read", "Write", "Edit", "Bash"]


@pytest.mark.asyncio
async def test_guarded_shell_allows_read_only_command() -> None:
    calls: list[str] = []
    bash = next(t for t in restrict_tools(_all_tools(calls)) if t.name == "Bash")

    result = await bash.execute("1", {"command": "git status"})

    assert calls == ["git status"]
    assert result.text == "ran"


@pytest.mark.asyncio
async def test_guarded_shell_refuses_mutating_command() -> None:
    calls: list[str] = []
    bash = next(t for t in restrict_tools(_all_tools(calls)) if t.name == "Bash")

    result = await bash.execute("1", {"command": "rm -f notes.txt"})

    assert calls == []
    assert "Plan mode is active" in result.text
    assert result.details == {"error": "plan_mode"}


def test_plan_system_prompt_appends_block() -> None:
    prompt = plan_system_prompt("BASE")
    assert prompt.startswith("BASE")
    assert PLAN_MODE_INSTRUCTIONS in prompt


# ── session integration ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_plan_mode_off_by_default() -> None:
    session = await _session()
    assert session.plan_mode is False
    assert [t.name for t in session.tools] == ["Read", "Write", "Edit", "Bash"]
    await session.shutdown()


@pytest.mark.asyncio
async def test_enabling_plan_mode_restricts_tools_and_prompt() -> None:
    session = await _session()

    session.set_plan_mode(True)

    assert session.plan_mode is True
    assert [t.name for t in session.tools] == ["Read", "Bash"]
    assert PLAN_MODE_INSTRUCTIONS in session._harness.settings.system
    await session.shutdown()


@pytest.mark.asyncio
async def test_disabling_plan_mode_restores_tools_and_prompt() -> None:
    session = await _session()

    session.set_plan_mode(True)
    session.set_plan_mode(False)

    assert [t.name for t in session.tools] == ["Read", "Write", "Edit", "Bash"]
    assert session._harness.settings.system == "BASE PROMPT"
    await session.shutdown()


@pytest.mark.asyncio
async def test_reload_keeps_plan_mode_applied() -> None:
    session = await _session()
    session._tools_loader = _all_tools

    session.set_plan_mode(True)
    await session.reload()

    assert [t.name for t in session.tools] == ["Read", "Bash"]
    await session.shutdown()


@pytest.mark.asyncio
async def test_new_session_inherits_unrestricted_tools() -> None:
    session = await _session()
    session.set_plan_mode(True)

    fresh = await session.new_session()

    assert fresh.plan_mode is False
    assert [t.name for t in fresh.tools] == ["Read", "Write", "Edit", "Bash"]
    await fresh.shutdown()


# ── /plan command ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_command_toggles() -> None:
    session = await _session()

    assert "on" in (await session.handle_command("/plan")).lower()
    assert session.plan_mode is True
    assert "off" in (await session.handle_command("/plan")).lower()
    assert session.plan_mode is False
    await session.shutdown()


@pytest.mark.asyncio
async def test_command_explicit_on_off() -> None:
    session = await _session()

    await session.handle_command("/plan on")
    assert session.plan_mode is True
    await session.handle_command("/plan on")  # idempotent
    assert session.plan_mode is True
    await session.handle_command("/plan off")
    assert session.plan_mode is False
    await session.shutdown()


@pytest.mark.asyncio
async def test_command_rejects_unknown_argument() -> None:
    session = await _session()

    reply = await session.handle_command("/plan sideways")

    assert "Unknown argument" in reply
    assert session.plan_mode is False
    await session.shutdown()
