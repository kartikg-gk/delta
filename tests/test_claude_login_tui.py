"""Headless TUI coverage of the Claude sign-in branch."""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from textual.widgets import Input

from delta_app import claude_accounts
from delta_app.tui.app import DeltaApp
from delta_app.tui.claude_login_screen import ClaudeLoginScreen
from delta_app.tui.model_screen import ModelChoice


@pytest.mark.parametrize("manual", [True, False])
async def test_claude_login_dialog_and_switch(monkeypatch: pytest.MonkeyPatch, manual: bool) -> None:
    app = DeltaApp()
    applied = AsyncMock()
    monkeypatch.setattr(app, "_apply_model_choice", applied)
    callback = asyncio.Event()

    async def fake_sign_in(*, show_url: Any, ask_code: Any) -> None:
        show_url("https://claude.ai/oauth/authorize?state=test")
        task = asyncio.create_task(ask_code())
        try:
            if manual:
                assert await task == "my-code"
            else:
                await callback.wait()
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    monkeypatch.setattr(claude_accounts, "sign_in", fake_sign_in)
    async with app.run_test() as pilot:
        # Install only the bridge marker: provider switching is captured above.
        app._bridge = Mock()
        worker = app.run_worker(app._open_login("claude"))
        await pilot.pause()
        assert isinstance(app.screen, ClaudeLoginScreen)
        if manual:
            app.screen.query_one(Input).value = "my-code"
            await pilot.press("enter")
        else:
            callback.set()
        await worker.wait()
        await pilot.pause()
        assert not isinstance(app.screen, ClaudeLoginScreen)
        applied.assert_awaited_once()
        assert isinstance(applied.call_args.args[0], ModelChoice)
        assert applied.call_args.args[0].provider == "claude"
        app._bridge = None
