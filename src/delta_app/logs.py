"""Run logging: trajectory, raw events, and stdout captured under ``.logs/``.

Written per session so external tooling (e.g. Harbor) can read a run without
touching Delta's internals.  Three artefacts:

- ``trajectory.json`` — ordered, human-readable steps: prompts, replies,
  tool calls and their results, plus final usage totals.
- ``events.jsonl`` — every ``AgentEvent``, one JSON object per line.
- ``stdout.log`` — plain-text log lines with timestamps.

Every value is scrubbed of anything resembling a credential before it is
written.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from delta_harness.contracts.stream import (
    AgentEvent,
    MessageEndEvent,
    RunEndEvent,
    RunStartEvent,
    ToolRunEndEvent,
    ToolRunStartEvent,
)
from delta_harness.contracts.transcript import HumanEntry, ModelEntry

from delta_app.refresh import _scrub

_DIR_ENV = "DELTA_LOG_DIR"
_DISABLE_ENV = "DELTA_NO_LOGS"
_DEFAULT_DIR = ".logs"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def logs_enabled() -> bool:
    """Whether run logging is switched on."""
    return os.environ.get(_DISABLE_ENV, "").strip().lower() not in {"1", "true", "yes"}


def log_root(cwd: str | Path | None = None) -> Path:
    """Directory holding run logs — ``DELTA_LOG_DIR`` or ``<cwd>/.logs``."""
    override = os.environ.get(_DIR_ENV)
    if override:
        return Path(override)
    return Path(cwd or os.getcwd()) / _DEFAULT_DIR


def _clean(value: Any) -> Any:
    """Recursively scrub strings inside a JSON-compatible value."""
    if isinstance(value, str):
        return _scrub(value)
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


@dataclass(slots=True)
class RunLogger:
    """Writes one session's logs into ``<root>/<session_id>/``."""

    session_id: str
    root: Path
    provider: str = ""
    model: str = ""
    steps: list[dict[str, Any]] = field(default_factory=list)
    #: Optional zero-arg callable returning the session's usage stats.
    stats_source: Any = None

    # -- construction -------------------------------------------------------

    @classmethod
    def create(
        cls,
        session_id: str,
        *,
        provider: str = "",
        model: str = "",
        cwd: str | Path | None = None,
        root: Path | None = None,
    ) -> RunLogger | None:
        """Prepare the log directory, or return ``None`` when disabled."""
        if not logs_enabled():
            return None
        base = (root or log_root(cwd)) / session_id
        try:
            base.mkdir(parents=True, exist_ok=True)
        except OSError:
            return None
        logger = cls(session_id=session_id, root=base, provider=provider, model=model)
        logger.write_line(f"session {session_id} started ({provider} · {model})")
        return logger

    # -- paths --------------------------------------------------------------

    @property
    def trajectory_path(self) -> Path:
        return self.root / "trajectory.json"

    @property
    def events_path(self) -> Path:
        return self.root / "events.jsonl"

    @property
    def stdout_path(self) -> Path:
        return self.root / "stdout.log"

    # -- writing ------------------------------------------------------------

    def write_line(self, text: str) -> None:
        """Append a timestamped line to ``stdout.log``."""
        self._append(self.stdout_path, f"{_now()}  {_scrub(text)}\n")

    def record_event(self, event: AgentEvent) -> None:
        """Append one event to ``events.jsonl`` and update the trajectory."""
        try:
            payload = _clean(event.model_dump(mode="json", by_alias=True))
        except Exception:  # noqa: BLE001 - logging must never break a run
            return
        self._append(self.events_path, json.dumps(payload, default=str) + "\n")
        self._track(event)

    def finish(self, stats: Any = None) -> None:
        """Write ``trajectory.json``.

        Rewritten after every run rather than only at shutdown, so a crashed
        or killed session still leaves a complete file behind.
        """
        if stats is None and self.stats_source is not None:
            try:
                stats = self.stats_source()
            except Exception:  # noqa: BLE001
                stats = None
        doc: dict[str, Any] = {
            "session_id": self.session_id,
            "provider": self.provider,
            "model": self.model,
            "finished_at": _now(),
            "steps": self.steps,
        }
        if stats is not None:
            doc["usage"] = {
                "turns": getattr(stats, "turn_count", 0),
                "messages": getattr(stats, "message_count", 0),
                "input_tokens": getattr(stats, "total_input_tokens", 0),
                "output_tokens": getattr(stats, "total_output_tokens", 0),
                "cost_usd": getattr(stats, "total_cost_usd", 0.0),
                "context_tokens": getattr(stats, "estimated_context_tokens", 0),
            }
        # Scrub the whole document, not just the steps built here — any
        # caller-supplied content must be covered too.
        self._write(
            self.trajectory_path, json.dumps(_clean(doc), indent=2, default=str),
        )
        self.write_line("session finished")

    # -- internals ----------------------------------------------------------

    def _track(self, event: AgentEvent) -> None:
        """Fold an event into the human-readable trajectory."""
        if isinstance(event, RunStartEvent):
            self.write_line("run started")
        elif isinstance(event, ToolRunStartEvent):
            self.steps.append({
                "at": _now(), "kind": "tool_call",
                "tool": event.tool_name,
                "args": _clean(dict(event.args)),
            })
            self.write_line(f"tool {event.tool_name} started")
        elif isinstance(event, ToolRunEndEvent):
            self.steps.append({
                "at": _now(), "kind": "tool_result",
                "tool": event.tool_name,
                "is_error": event.is_error,
                "output": _scrub(event.result.text)[:4000],
            })
            self.write_line(
                f"tool {event.tool_name} {'failed' if event.is_error else 'ok'}"
            )
        elif isinstance(event, MessageEndEvent):
            message = event.message
            if isinstance(message, ModelEntry) and message.text:
                self.steps.append({
                    "at": _now(), "kind": "assistant",
                    "text": _scrub(message.text),
                    "stop_reason": message.stop_reason,
                })
            elif isinstance(message, HumanEntry) and message.text:
                self.steps.append({
                    "at": _now(), "kind": "prompt", "text": _scrub(message.text),
                })
        elif isinstance(event, RunEndEvent):
            self.write_line("run ended")
            self.finish()

    def _append(self, path: Path, text: str) -> None:
        try:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(text)
        except OSError:
            pass  # logging is best effort — never break the run

    def _write(self, path: Path, text: str) -> None:
        try:
            path.write_text(text, encoding="utf-8")
        except OSError:
            pass


__all__ = ["RunLogger", "log_root", "logs_enabled"]
