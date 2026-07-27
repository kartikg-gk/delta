"""Reload diagnostics: structured JSONL logging for hot-reload operations.

Provides the types and utilities the reload manager needs to record what
happened during a reload — expected failures, unexpected exceptions, and
the metadata that ties log lines together — without performing any reload
logic itself.

All logged data is scrubbed of secrets before it touches disk.  Log
entries never contain resource content, prompt text, API keys, request
payloads, or credentials.

Usage::

    ctx = ReloadDiagnosticContext.create()
    logger = ReloadDiagnosticLogger(log_dir)

    try:
        load_skill(path)
    except SkillLoadError as exc:
        logger.log_failure(ctx, "skills", "bad-skill", str(exc))
    except Exception as exc:
        logger.log_exception(ctx, "skills", "bad-skill", exc)
"""

from __future__ import annotations

import json
import re
import traceback
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ── constants ──────────────────────────────────────────────────────────────

_LOG_FILENAME = "reload_diag.jsonl"
_MAX_ERROR_LENGTH = 300
_MAX_TRACEBACK_LINES = 20

# Patterns that look like secrets — replaced with [REDACTED] before logging.
_SECRET_RE = re.compile(
    r"(sk-[A-Za-z0-9]{8,})"          # Anthropic / OpenAI API keys
    r"|(key-[A-Za-z0-9]{8,})"        # generic key- prefixed tokens
    r"|(Bearer\s+\S+)"               # Authorization headers
    r"|(token[=:]\s*\S+)"            # token= or token: values
    r"|(password[=:]\s*\S+)"         # password= or password: values
    r"|(secret[=:]\s*\S+)",          # secret= or secret: values
    re.IGNORECASE,
)


# ── scrubbing ──────────────────────────────────────────────────────────────


def _scrub(text: str) -> str:
    """Replace substrings that look like secrets with ``[REDACTED]``."""
    return _SECRET_RE.sub("[REDACTED]", text)


def _scrub_error(message: str) -> str:
    """Scrub and truncate an error message for safe logging."""
    scrubbed = _scrub(message)
    if len(scrubbed) > _MAX_ERROR_LENGTH:
        scrubbed = scrubbed[:_MAX_ERROR_LENGTH] + "…"
    return scrubbed


def _scrub_traceback(exc: BaseException) -> list[str]:
    """Format an exception traceback and scrub each line of secrets.

    Flattens the output of ``traceback.format_exception`` into individual
    lines, scrubs each one, and caps the result at ``_MAX_TRACEBACK_LINES``.
    """
    raw = traceback.format_exception(type(exc), exc, exc.__traceback__)
    flat: list[str] = []
    for chunk in raw:
        flat.extend(chunk.rstrip().splitlines())

    scrubbed = [_scrub(line) for line in flat]
    if len(scrubbed) > _MAX_TRACEBACK_LINES:
        overflow = len(flat) - _MAX_TRACEBACK_LINES
        scrubbed = scrubbed[:_MAX_TRACEBACK_LINES] + [f"… ({overflow} more lines)"]
    return scrubbed


# ── timestamps & run IDs ──────────────────────────────────────────────────


def _utc_now() -> datetime:
    """Return the current UTC time (timezone-aware)."""
    return datetime.now(UTC)


def _iso_utc(dt: datetime) -> str:
    """Format *dt* as a compact ISO-8601 UTC string."""
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def new_run_id() -> str:
    """Generate a unique 12-character hex identifier for a reload run."""
    return uuid.uuid4().hex[:12]


# ── ReloadDiagnosticContext ───────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ReloadDiagnosticContext:
    """Immutable metadata stamp for one reload operation.

    Every diagnostic log line emitted during the same reload shares this
    context's ``run_id`` and ``started_at``, making it trivial to
    correlate entries.  Carries no mutable state.
    """

    run_id: str
    started_at: str

    @staticmethod
    def create() -> ReloadDiagnosticContext:
        """Build a fresh context with a new run ID and current UTC time."""
        return ReloadDiagnosticContext(
            run_id=new_run_id(),
            started_at=_iso_utc(_utc_now()),
        )


# ── log entry helpers ─────────────────────────────────────────────────────


def build_base_entry(ctx: ReloadDiagnosticContext) -> dict[str, Any]:
    """Return the fields common to every diagnostic log line.

    Contains only the run correlation ID and a per-entry UTC timestamp.
    """
    return {
        "run_id": ctx.run_id,
        "ts": _iso_utc(_utc_now()),
    }


def build_failure_entry(
    ctx: ReloadDiagnosticContext,
    category: str,
    name: str,
    error: str,
    *,
    path: str = "",
) -> dict[str, Any]:
    """Build a log entry for an expected reload failure.

    The *error* message is scrubbed before inclusion.
    """
    entry = build_base_entry(ctx)
    entry.update(level="failure", category=category, name=name,
                 error=_scrub_error(error))
    if path:
        entry["path"] = path
    return entry


def build_exception_entry(
    ctx: ReloadDiagnosticContext,
    category: str,
    name: str,
    exc: BaseException,
    *,
    path: str = "",
) -> dict[str, Any]:
    """Build a log entry for an unexpected exception.

    Both the error message and the traceback are scrubbed of secrets.
    """
    entry = build_base_entry(ctx)
    entry.update(level="error", category=category, name=name,
                 error=_scrub_error(str(exc)),
                 traceback=_scrub_traceback(exc))
    if path:
        entry["path"] = path
    return entry


# ── ReloadDiagnosticLogger ────────────────────────────────────────────────


class ReloadDiagnosticLogger:
    """Appends structured JSONL diagnostic entries for reload operations.

    Each call to ``log_failure`` or ``log_exception`` writes one compact
    JSON line.  The log directory is created automatically on first write.

    The logger only writes — it never performs reload logic, builds
    summaries, or tracks state.  All error messages and tracebacks are
    scrubbed of secrets before being written.
    """

    def __init__(self, log_dir: Path, *, filename: str = _LOG_FILENAME) -> None:
        self._log_dir = log_dir
        self._filename = filename

    @property
    def log_path(self) -> Path:
        """Full path to the diagnostic log file."""
        return self._log_dir / self._filename

    # ── writing ───────────────────────────────────────────────────────

    def _append(self, entry: dict[str, Any]) -> None:
        """Serialize *entry* as compact JSON and append it as one line."""
        self._log_dir.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, separators=(",", ":")) + "\n")

    # ── public API ────────────────────────────────────────────────────

    def log_failure(
        self,
        ctx: ReloadDiagnosticContext,
        category: str,
        name: str,
        error: str,
        *,
        path: str = "",
    ) -> None:
        """Log an expected reload failure (bad file, missing field, etc.)."""
        self._append(build_failure_entry(ctx, category, name, error, path=path))

    def log_exception(
        self,
        ctx: ReloadDiagnosticContext,
        category: str,
        name: str,
        exc: BaseException,
        *,
        path: str = "",
    ) -> None:
        """Log an unexpected exception with a scrubbed traceback."""
        self._append(build_exception_entry(ctx, category, name, exc, path=path))


__all__ = [
    "ReloadDiagnosticContext",
    "ReloadDiagnosticLogger",
    "build_base_entry",
    "build_exception_entry",
    "build_failure_entry",
    "new_run_id",
]
