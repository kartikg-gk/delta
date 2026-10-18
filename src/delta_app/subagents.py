"""User-launched child sessions, deliberately absent from the model's tools."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from uuid import uuid4
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from delta_app.conversation import CodingSession

from delta_harness.contracts.stream import MessageEndEvent, ToolRunStartEvent
from delta_harness.contracts.transcript import ModelEntry


@dataclass(slots=True)
class SubagentRun:
    id: str
    task: str
    status: str = "starting"
    progress: str = ""
    result: str = ""
    session_id: str = ""
    session: CodingSession | None = None
    runner: asyncio.Task[None] | None = None


class SubagentManager:
    def __init__(self, parent: CodingSession) -> None:
        self.parent = parent
        self.runs: dict[str, SubagentRun] = {}

    def launch(self, task: str) -> SubagentRun:
        if not task.strip():
            raise ValueError("Use /agent <task> to start a subagent")
        if sum(r.status in {"starting", "running"} for r in self.runs.values()) >= 4:
            raise ValueError("Four subagents are already running. Stop one before starting another.")
        run = SubagentRun(uuid4().hex[:8], task)
        self.runs[run.id] = run
        run.runner = asyncio.create_task(self._run(run))
        return run

    async def _run(self, run: SubagentRun) -> None:
        from delta_app.conversation import CodingSession
        from delta_app.runtime import attach_logger

        child = None
        try:
            parent = self.parent
            child = await CodingSession.create(
                provider=parent._provider, provider_name=parent.provider_name, model=parent.model,
                system=parent._system, tools=list(parent.tools), cwd=parent.cwd,
                sessions_dir=parent._vault.path.parent if parent._vault else None,
                skills_loader=parent._skills_loader, templates_loader=parent._templates_loader,
            )
            child.harness.settings.max_turns = parent.harness.settings.max_turns or 30
            child.harness.settings.before_tool_call = parent.harness.settings.before_tool_call
            child.harness.settings.after_tool_call = parent.harness.settings.after_tool_call
            if parent.thinking_level:
                await child.set_thinking(parent.thinking_level)
            await child.handle_command("/name Agent: " + run.task[:60])
            attach_logger(child)
            run.session, run.session_id, run.status = child, child.session_id, "running"
            async for event in child.submit(run.task):
                if isinstance(event, ToolRunStartEvent):
                    run.progress = event.tool_name
                elif isinstance(event, MessageEndEvent) and isinstance(event.message, ModelEntry):
                    run.result = event.message.text
                    if event.message.error_message:
                        raise ValueError(event.message.error_message)
            run.status = "completed"
        except asyncio.CancelledError:
            run.status = "cancelled"
        except Exception as exc:
            run.status, run.result = "failed", str(exc)
        finally:
            if child is not None:
                child.abort()
                await child.shutdown()
            run.session = None

    async def stop(self, run_id: str) -> None:
        run = self.runs.get(run_id)
        if run is None:
            raise ValueError(f"Unknown subagent: {run_id}")
        if run.runner is not None and not run.runner.done():
            run.runner.cancel()
            await asyncio.gather(run.runner, return_exceptions=True)
            run.status = "cancelled"

    async def shutdown(self) -> None:
        await asyncio.gather(*(self.stop(run.id) for run in self.runs.values()))

    async def command(self, argument: str) -> str:
        parts = argument.split(maxsplit=1)
        if not parts or parts[0] == "list":
            return "\n".join(
                f"{run.id} [{run.status}] {run.task} — {run.progress}"
                for run in self.runs.values()
            ) or "No subagents. Use /agent <task> to start one."
        if parts[0] in {"stop", "result", "view"}:
            if len(parts) != 2 or parts[1] not in self.runs:
                raise ValueError("Use /agent stop|result|view <id> with an ID from /agents")
            run = self.runs[parts[1]]
            if parts[0] == "stop":
                await self.stop(run.id)
            return f"{run.id} [{run.status}] session {run.session_id}\n{run.result}"
        run = self.launch(argument)
        return f"Started subagent {run.id}. Use /agents, /agent result {run.id}, or /agent stop {run.id}."
