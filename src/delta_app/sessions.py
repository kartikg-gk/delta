"""Session lifecycle management: create, list, resume, and index.

``SessionManager`` is the application-level coordinator between
``CodingSession`` (runtime) and ``SessionCatalog`` (persistent index).
It provides the high-level API that the CLI and TUI call to manage
sessions without touching persistence details directly.

Responsibilities:

* **Create** — mint a new session, persist vault + index entry.
* **Resume** — reload an existing session, touch the index timestamp.
* **List** — enumerate sessions from the catalog (fast) or fall back
  to scanning vault files when no index exists.
* **Latest** — find the most recently used session for a working directory.
* **Export** — delegate transcript export to ``CodingSession``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from delta_app.conversation import CodingSession
from delta_harness.contracts.tooling import ToolSpec
from delta_harness.provider.base import ModelProvider
from delta_harness.session.index import SessionCatalog, SessionMeta
from delta_harness.session.records import SessionRecord, TranscriptRecord
from delta_harness.session.store import JsonlVault, jsonl_lines

# ---------------------------------------------------------------------------
# Summary returned by list_sessions
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SessionSummary:
    """Lightweight listing entry for a session."""

    session_id: str
    title: str | None
    model: str | None
    provider: str | None
    cwd: str | None
    updated_at: float | None
    message_count: int | None = None


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class SessionManager:
    """Application-level coordinator for session lifecycle and indexing.

    Wraps ``SessionCatalog`` for metadata persistence and delegates runtime
    construction to ``CodingSession.create`` / ``CodingSession.resume``.
    The manager itself is stateless beyond the sessions directory path.
    """

    def __init__(self, sessions_dir: str | Path) -> None:
        self._dir = Path(sessions_dir)
        self._catalog = SessionCatalog(self._dir)

    # ── read-only API ─────────────────────────────────────────────────

    @property
    def sessions_dir(self) -> Path:
        """Root directory containing vault files and the index."""
        return self._dir

    @property
    def catalog(self) -> SessionCatalog:
        """The underlying session catalog."""
        return self._catalog

    # ── create ────────────────────────────────────────────────────────

    async def create(
        self,
        *,
        provider: ModelProvider,
        provider_name: str,
        model: str,
        system: str,
        tools: list[ToolSpec] | None = None,
        session_id: str | None = None,
        cwd: str | None = None,
        tools_loader: Callable[[], list[ToolSpec]] | None = None,
    ) -> CodingSession:
        """Create a new persistent coding session and index it.

        Delegates to ``CodingSession.create`` (which writes the vault and
        catalog entry) and returns the ready-to-use session.
        """
        return await CodingSession.create(
            provider=provider,
            provider_name=provider_name,
            model=model,
            system=system,
            tools=tools,
            sessions_dir=self._dir,
            session_id=session_id,
            cwd=cwd,
            tools_loader=tools_loader,
        )

    # ── resume ────────────────────────────────────────────────────────

    async def resume(
        self,
        session_id: str,
        *,
        provider: ModelProvider,
        provider_name: str,
        model: str,
        system: str,
        tools: list[ToolSpec] | None = None,
        tools_loader: Callable[[], list[ToolSpec]] | None = None,
    ) -> CodingSession:
        """Resume an existing session and touch its index timestamp.

        Delegates to ``CodingSession.resume`` (which reloads the vault and
        touches the catalog) and returns the restored session.
        """
        return await CodingSession.resume(
            session_id,
            provider=provider,
            provider_name=provider_name,
            model=model,
            system=system,
            tools=tools,
            sessions_dir=self._dir,
            tools_loader=tools_loader,
        )

    # ── resume latest ─────────────────────────────────────────────────

    async def resume_latest(
        self,
        cwd: str,
        *,
        provider: ModelProvider,
        provider_name: str,
        model: str,
        system: str,
        tools: list[ToolSpec] | None = None,
        tools_loader: Callable[[], list[ToolSpec]] | None = None,
    ) -> CodingSession | None:
        """Resume the most recently used session for *cwd*, or ``None``.

        Returns ``None`` if no session has been indexed for that directory.
        """
        meta = self._catalog.latest_for_cwd(cwd)
        if meta is None:
            return None
        return await self.resume(
            meta.session_id,
            provider=provider,
            provider_name=provider_name,
            model=model,
            system=system,
            tools=tools,
            tools_loader=tools_loader,
        )

    # ── list ──────────────────────────────────────────────────────────

    def list_sessions(self) -> list[SessionSummary]:
        """Return all indexed sessions, newest first.

        Uses the catalog when available.  Falls back to scanning vault
        files so pre-index sessions are still discoverable.
        """
        indexed = self._catalog.list_all()
        if indexed:
            return [
                SessionSummary(
                    session_id=m.session_id,
                    title=m.title,
                    model=m.model,
                    provider=m.provider,
                    cwd=m.cwd,
                    updated_at=m.updated_at,
                )
                for m in indexed
            ]
        return self._scan_vault_files()

    def _scan_vault_files(self) -> list[SessionSummary]:
        """Fallback: scan vault .jsonl files when no index exists."""
        if not self._dir.exists():
            return []
        files = sorted(
            self._dir.glob("*.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        files = [f for f in files if f.name != "index.jsonl"]
        results: list[SessionSummary] = []
        for path in files:
            title: str | None = None
            n_messages = 0
            try:
                for line in jsonl_lines(path.read_text(encoding="utf-8")):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        raw = json.loads(stripped)
                    except Exception:  # noqa: BLE001
                        continue
                    rtype = raw.get("type")
                    if rtype == "label":
                        title = raw.get("label", title)
                    elif rtype == "session_info" and raw.get("title"):
                        title = raw["title"]
                    elif rtype == "message":
                        n_messages += 1
            except Exception:  # noqa: BLE001
                pass
            results.append(
                SessionSummary(
                    session_id=path.stem,
                    title=title,
                    model=None,
                    provider=None,
                    cwd=None,
                    updated_at=path.stat().st_mtime if path.exists() else None,
                    message_count=n_messages,
                )
            )
        return results

    # ── lookup ────────────────────────────────────────────────────────

    def get(self, session_id: str) -> SessionMeta | None:
        """Look up indexed metadata for *session_id*."""
        return self._catalog.get(session_id)

    def latest_for_cwd(self, cwd: str) -> SessionMeta | None:
        """Return the most recently used session metadata for *cwd*."""
        return self._catalog.latest_for_cwd(cwd)

    # ── export ────────────────────────────────────────────────────────

    async def export(self, session_id: str, fmt: str = "jsonl") -> str:
        """Export a session's vault contents in the given format.

        Supported formats: ``jsonl``, ``json``, ``text``.
        """
        vault_path = self._dir / f"{session_id}.jsonl"
        if not vault_path.exists():
            raise FileNotFoundError(f"Session vault not found: {session_id}")

        vault = JsonlVault(vault_path)
        records = await vault.read_all()

        if fmt == "jsonl":
            from delta_harness.session.store import serialize_record

            return "".join(serialize_record(r) for r in records)

        if fmt == "json":
            from pydantic import TypeAdapter

            adapter: TypeAdapter[list[SessionRecord]] = TypeAdapter(
                list[SessionRecord]
            )
            return adapter.dump_json(records, indent=2).decode()

        # Default: text
        from delta_harness.contracts.transcript import surface_text

        lines: list[str] = []
        for r in records:
            if isinstance(r, TranscriptRecord):
                lines.append(f"[{r.message.role}] {surface_text(r.message)}")
        return "\n\n".join(lines)


__all__ = [
    "SessionManager",
    "SessionSummary",
]
