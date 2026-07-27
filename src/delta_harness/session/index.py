"""Lightweight session metadata index backed by an append-only JSONL file.

The ``SessionCatalog`` manages an ``index.jsonl`` that lives alongside the
per-session vault files.  It stores one ``SessionMeta`` record per line and
supports deduplication by session id — when duplicate entries exist, the one
with the newest ``updated_at`` timestamp wins.

This module knows nothing about ``RuntimeHarness`` or transcript data.  It
only tracks the metadata needed to list, find, and resume sessions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import time

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Serialization model
# ---------------------------------------------------------------------------


class SessionMetaWire(BaseModel):
    """Pydantic model for (de)serializing a single index entry."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    vault_path: str
    cwd: str
    model: str
    provider: str
    title: str | None = None
    created_at: float = Field(default_factory=time)
    updated_at: float = Field(default_factory=time)


# ---------------------------------------------------------------------------
# Runtime representation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SessionMeta:
    """Immutable snapshot of session metadata for runtime use."""

    session_id: str
    vault_path: str
    cwd: str
    model: str
    provider: str
    title: str | None
    created_at: float
    updated_at: float

    def to_wire(self) -> SessionMetaWire:
        """Convert to a serializable wire model."""
        return SessionMetaWire(
            session_id=self.session_id,
            vault_path=self.vault_path,
            cwd=self.cwd,
            model=self.model,
            provider=self.provider,
            title=self.title,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )


def _wire_to_meta(wire: SessionMetaWire) -> SessionMeta:
    """Convert a wire model to the frozen runtime representation."""
    return SessionMeta(
        session_id=wire.session_id,
        vault_path=wire.vault_path,
        cwd=wire.cwd,
        model=wire.model,
        provider=wire.provider,
        title=wire.title,
        created_at=wire.created_at,
        updated_at=wire.updated_at,
    )


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

_INDEX_FILENAME = "index.jsonl"


class SessionCatalog:
    """Persistent index of session metadata stored as append-only JSONL.

    Each call to :meth:`upsert` appends a line to ``index.jsonl``.  Reads
    deduplicate by ``session_id``, keeping the entry with the newest
    ``updated_at`` timestamp.  Malformed lines are silently skipped so a
    single bad entry never prevents other sessions from loading.
    """

    def __init__(self, sessions_dir: str | Path) -> None:
        self._dir = Path(sessions_dir)
        self._path = self._dir / _INDEX_FILENAME

    @property
    def path(self) -> Path:
        """Absolute path to the backing ``index.jsonl``."""
        return self._path

    # ── factory helpers ───────────────────────────────────────────────

    @staticmethod
    def prepare(
        *,
        session_id: str,
        vault_path: str | Path,
        cwd: str,
        model: str,
        provider: str,
        title: str | None = None,
    ) -> SessionMeta:
        """Build a ``SessionMeta`` without writing it to any index.

        Useful when you need to construct metadata before the catalog is
        available or when batching multiple writes.
        """
        now = time()
        return SessionMeta(
            session_id=session_id,
            vault_path=str(vault_path),
            cwd=cwd,
            model=model,
            provider=provider,
            title=title,
            created_at=now,
            updated_at=now,
        )

    # ── writes ────────────────────────────────────────────────────────

    def upsert(self, meta: SessionMeta) -> None:
        """Append a metadata entry to the index.

        If an entry with the same ``session_id`` already exists on disk the
        duplicate is resolved at read time — the entry with the newest
        ``updated_at`` wins.
        """
        self._dir.mkdir(parents=True, exist_ok=True)
        line = meta.to_wire().model_dump_json(exclude_none=True) + "\n"
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(line)

    def touch(self, session_id: str) -> SessionMeta | None:
        """Update the ``updated_at`` timestamp for *session_id*.

        Returns the updated metadata, or ``None`` if the session is not
        found in the index.
        """
        existing = self.get(session_id)
        if existing is None:
            return None
        refreshed = SessionMeta(
            session_id=existing.session_id,
            vault_path=existing.vault_path,
            cwd=existing.cwd,
            model=existing.model,
            provider=existing.provider,
            title=existing.title,
            created_at=existing.created_at,
            updated_at=time(),
        )
        self.upsert(refreshed)
        return refreshed

    # ── reads ─────────────────────────────────────────────────────────

    def list_all(self) -> list[SessionMeta]:
        """Return all sessions ordered by ``updated_at`` descending.

        Duplicates are resolved by keeping the newest ``updated_at`` per
        ``session_id``.
        """
        entries = self._load_deduped()
        return sorted(entries.values(), key=lambda m: m.updated_at, reverse=True)

    def get(self, session_id: str) -> SessionMeta | None:
        """Look up a single session by its id, or ``None`` if absent."""
        entries = self._load_deduped()
        return entries.get(session_id)

    def latest_for_cwd(self, cwd: str) -> SessionMeta | None:
        """Return the most recently used session for *cwd*, or ``None``."""
        candidates = [
            m for m in self._load_deduped().values() if m.cwd == cwd
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda m: m.updated_at)

    # ── internals ─────────────────────────────────────────────────────

    def _load_deduped(self) -> dict[str, SessionMeta]:
        """Read every line, parse, and deduplicate by session_id."""
        if not self._path.exists():
            return {}

        result: dict[str, SessionMeta] = {}
        for line in self._path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
                wire = SessionMetaWire.model_validate(raw)
                meta = _wire_to_meta(wire)
            except (json.JSONDecodeError, Exception):  # noqa: BLE001
                continue
            existing = result.get(meta.session_id)
            if existing is None or meta.updated_at > existing.updated_at:
                result[meta.session_id] = meta
        return result
