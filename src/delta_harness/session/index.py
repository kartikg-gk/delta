"""Lightweight session metadata index backed by a JSONL file.

The ``SessionCatalog`` manages an ``index.jsonl`` that lives alongside the
per-session vault files.  It stores one ``SessionMeta`` record per session,
rewritten in place on update.  Reads still deduplicate by session id — newest
``updated_at`` wins — so indexes written by earlier append-only builds load
without migration.

This module knows nothing about ``RuntimeHarness`` or transcript data.  It
only tracks the metadata needed to list, find, and resume sessions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from time import time

from pydantic import BaseModel, ConfigDict, Field

from delta_harness.session.store import jsonl_lines

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
    inference_provider: str | None = None
    inference_provider_mode: str | None = None


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
    # Backend pin for routed providers, and whether it was learned or chosen.
    inference_provider: str | None = None
    inference_provider_mode: str | None = None

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
            inference_provider=self.inference_provider,
            inference_provider_mode=self.inference_provider_mode,
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
        inference_provider=wire.inference_provider,
        inference_provider_mode=wire.inference_provider_mode,
    )


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

_INDEX_FILENAME = "index.jsonl"


class SessionCatalog:
    """Persistent index of session metadata stored as JSONL.

    :meth:`upsert` rewrites the file so it carries one line per session.
    Reads still deduplicate by ``session_id`` — keeping the newest
    ``updated_at`` — so an index left over from an older append-only build
    still loads correctly.  Malformed lines are skipped, so one bad entry
    never hides the remaining sessions.
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
        """Insert or replace the entry for ``meta.session_id``.

        The index holds exactly one line per session: it is read, updated in
        memory, and rewritten.  Appending instead would let the file grow
        without bound, since every resume and every rename writes an entry,
        and each read parses the whole file.

        The rewrite goes through a temporary file and an atomic replace, so an
        interrupted write cannot truncate an index that is still needed to find
        existing sessions.

        A write carrying an older ``updated_at`` than the stored entry is
        dropped: every caller stamps the current time, so a stale timestamp
        means another process already recorded something newer.
        """
        self._dir.mkdir(parents=True, exist_ok=True)
        entries = self._load_deduped()
        current = entries.get(meta.session_id)
        if current is not None and meta.updated_at < current.updated_at:
            return
        entries[meta.session_id] = meta

        payload = "".join(
            entry.to_wire().model_dump_json(exclude_none=True) + "\n"
            for entry in sorted(entries.values(), key=lambda m: m.updated_at)
        )
        staged = self._path.with_name(f"{self._path.name}.tmp")
        staged.write_text(payload, encoding="utf-8")
        staged.replace(self._path)

    def touch(self, session_id: str) -> SessionMeta | None:
        """Update the ``updated_at`` timestamp for *session_id*.

        Returns the updated metadata, or ``None`` if the session is not
        found in the index.
        """
        existing = self.get(session_id)
        if existing is None:
            return None
        refreshed = replace(existing, updated_at=time())
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
        for line in jsonl_lines(self._path.read_text(encoding="utf-8")):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
                wire = SessionMetaWire.model_validate(raw)
                meta = _wire_to_meta(wire)
            except Exception:  # noqa: BLE001 - one bad line must not hide the rest
                continue
            existing = result.get(meta.session_id)
            # The file is append-ordered, so on an equal timestamp the later
            # line is the newer write. Coarse clocks (Windows ~15ms) tie often
            # enough that a strict `>` silently discards fresh updates.
            if existing is None or meta.updated_at >= existing.updated_at:
                result[meta.session_id] = meta
        return result
