"""Subscription-backed model calls via officially authenticated command-line clients.

The CLI returns a structured proposed response; Delta executes the selected tools.
Credentials remain in each CLI's own auth store. These adapters buffer one response.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from uuid import uuid4
from typing import Any, cast

from delta_harness.contracts.tooling import CancelToken, ToolSpec
from delta_harness.contracts.transcript import CallBlock, ModelEntry, TextSegment, TranscriptEntry
from delta_harness.provider.wire import StreamCloseEvent, StreamFaultEvent, StreamOpenEvent, WireEvent


def response_schema() -> dict[str, Any]:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["text", "tool_calls"],
        "properties": {
            "text": {"type": "string"},
            "tool_calls": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["name", "arguments"],
                "properties": {"name": {"type": "string"}, "arguments": {"type": "string"}},
            }},
        },
    }


class SubscriptionProvider:
    def __init__(self, name: str) -> None:
        self.name = name

    async def stream_response(
        self, *, model: str, system: str, messages: Sequence[TranscriptEntry],
        tools: Sequence[ToolSpec], signal: CancelToken | None = None,
        cache_key: str | None = None,
    ) -> AsyncIterator[WireEvent]:
        partial = ModelEntry(model=model, provider=self.name)
        yield StreamOpenEvent(partial=partial)
        process = None
        try:
            cli = "codex" if self.name == "codex" else "claude"
            binary = shutil.which(cli)
            if binary is None:
                raise ValueError(f"Install the official {cli} CLI and run delta login {self.name}.")
            prompt = system + "\n\n" + (
                "You are the model inside Delta's tool loop. Do not use native CLI tools. "
                "Return JSON with text and tool_calls; arguments is a JSON-encoded object. "
                "Request only the listed Delta tools; Delta executes them and sends results.\n"
            ) + json.dumps({
                "tools": [{"name": t.name, "description": t.description,
                           "parameters": dict(t.parameters)} for t in tools],
                "messages": [m.model_dump(mode="json") for m in messages],
            })
            with tempfile.TemporaryDirectory(prefix="delta-model-") as directory:
                schema = Path(directory) / "schema.json"
                schema.write_text(json.dumps(response_schema()), encoding="utf-8")
                if cli == "codex":
                    command = [binary, "exec", "--json", "--ephemeral", "--skip-git-repo-check",
                               "--sandbox", "read-only", "--model", model,
                               "--output-schema", str(schema), "-"]
                else:
                    command = [binary, "-p", "--output-format", "json", "--model", model,
                               "--tools", "", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
                               "--disable-slash-commands", "--no-session-persistence",
                               "--json-schema", json.dumps(response_schema())]
                env = dict(os.environ)
                # API credentials must not silently turn subscription calls into API billing.
                for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
                    env.pop(key, None)
                process = await asyncio.create_subprocess_exec(
                    *command, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE, env=env, cwd=directory,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                task = asyncio.create_task(process.communicate(prompt.encode("utf-8")))
                try:
                    async with asyncio.timeout(300):
                        while not task.done():
                            if signal is not None and signal.is_cancelled():
                                raise asyncio.CancelledError
                            await asyncio.wait({task}, timeout=0.1)
                        stdout, stderr = task.result()
                finally:
                    if process.returncode is None:
                        process.terminate()
                        await process.wait()
                    if not task.done():
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                if process.returncode:
                    raise ValueError(stderr.decode("utf-8", errors="replace")[-2000:])
                payload = self._parse(stdout.decode("utf-8"))
                content = [TextSegment(text=payload["text"])] if payload.get("text") else []
                calls = [CallBlock(id=uuid4().hex, name=call["name"],
                                   arguments=json.loads(call["arguments"]))
                         for call in payload.get("tool_calls", [])]
                message = ModelEntry(
                    content=[*content, *calls], model=model, provider=self.name,
                    stop_reason="toolUse" if calls else "stop",
                )
                yield StreamCloseEvent(reason="toolUse" if calls else "stop", message=message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            yield StreamFaultEvent(
                reason="error", error=ModelEntry(
                    model=model, provider=self.name, stop_reason="error", error_message=str(exc),
                ),
            )

    def _parse(self, output: str) -> dict[str, Any]:
        if self.name == "claude-code":
            response = json.loads(output)
            if response.get("is_error"):
                raise ValueError(response.get("result", "Claude subscription request failed"))
            return cast(dict[str, Any], response.get("structured_output") or json.loads(response["result"]))
        result = None
        for line in output.splitlines():
            event = json.loads(line)
            if event.get("type") in {"error", "turn.failed"}:
                raise ValueError(str(event.get("message") or event.get("error")))
            item = event.get("item", {})
            if event.get("type") == "item.completed" and item.get("type") == "agent_message":
                result = json.loads(item["text"])
        if result is None:
            raise ValueError("Codex returned no structured response")
        return cast(dict[str, Any], result)
