"""Project trust: whether a folder's own instructions and resources may load.

A repository can ship files that steer the agent: ``AGENTS.md`` instructions,
skills, prompt templates, themes, and plugins that run as code. Opening an
unfamiliar checkout should not silently hand those files control, so every
such *project input* waits for an explicit decision about the working folder.
User-level resources under the Delta home are always loaded; they are the
user's own.

Decisions are resolved once per canonical folder, in this order:

1. an explicit ``--approve`` / ``--no-approve`` for this invocation;
2. no project inputs present, so there is nothing to protect;
3. the nearest saved decision for the folder or one of its ancestors;
4. the user-wide default (``always`` / ``never``; ``ask`` when unset);
5. an interactive choice, when a frontend can ask;
6. otherwise, decline.

This controls which inputs load. It is not a sandbox: tools still run with
the user's permissions whatever the decision.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal

Verdict = Literal["trusted", "untrusted"]
DefaultPolicy = Literal["ask", "always", "never"]

_STORE_NAME = "trust.json"
_STORE_VERSION = 1
_VERDICTS: frozenset[str] = frozenset({"trusted", "untrusted"})


class TrustStoreError(Exception):
    """The saved-decision file is unreadable or not in the expected shape."""


class StartupCancelled(Exception):
    """The user dismissed the trust question instead of answering it."""


class Choice(Enum):
    """Answers a frontend may give to a trust question."""

    TRUST_FOLDER = "trust-folder"
    TRUST_PARENT = "trust-parent"
    TRUST_ONCE = "trust-once"
    DISTRUST_FOLDER = "distrust-folder"
    DISTRUST_ONCE = "distrust-once"


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def canonical_folder(cwd: str | os.PathLike[str]) -> Path:
    """Absolute, symlink-free form of an existing directory."""
    folder = Path(cwd).expanduser().resolve(strict=True)
    if not folder.is_dir():
        raise NotADirectoryError(str(folder))
    return folder


def _key(folder: Path) -> str:
    # Case-insensitive filesystems compare paths case-insensitively.
    return os.path.normcase(str(folder))


# ---------------------------------------------------------------------------
# Detection (names and file types only; contents are never read)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProjectInputs:
    """Protected inputs present in a folder, counted by kind."""

    folder: Path
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return not any(self.counts.values())

    def describe(self) -> str:
        """Short human summary, e.g. ``2 instruction files, 1 skill``."""
        parts = [
            f"{n} {kind if n == 1 else kind + 's'}"
            for kind, n in self.counts.items() if n
        ]
        return ", ".join(parts) or "nothing"


def _entries(directory: Path) -> list[Path]:
    try:
        return list(directory.iterdir())
    except OSError:
        return []


def scan_project_inputs(folder: Path) -> ProjectInputs:
    """Count protected inputs by name and type alone.

    An entry that exists but cannot be inspected still counts: failing to
    look is not evidence that nothing is there.
    """
    counts = {
        "instruction file": sum(
            1 for p in (folder / "AGENTS.md", folder / ".delta" / "AGENTS.md",
                        folder / ".agents" / "AGENTS.md")
            if p.is_symlink() or p.exists()
        ),
        "skill": sum(
            1
            for base in (folder / ".delta" / "skills", folder / ".agents" / "skills")
            for entry in _entries(base)
            if (entry / "SKILL.md").is_symlink() or (entry / "SKILL.md").exists()
        ),
        "prompt template": sum(
            1
            for base in (folder / ".delta" / "prompts", folder / ".agents" / "prompts")
            for entry in _entries(base)
            if entry.suffix == ".md" and not entry.name.startswith(".")
        ),
        "theme": sum(1 for e in _entries(folder / ".delta" / "themes") if e.suffix == ".json"),
        "plugin": sum(
            1
            for entry in _entries(folder / ".delta" / "extensions")
            if not entry.name.startswith(("_", "."))
            and (entry.suffix == ".py" or (entry / "extension.py").exists())
        ),
    }
    return ProjectInputs(folder=folder, counts=counts)


# ---------------------------------------------------------------------------
# Saved decisions
# ---------------------------------------------------------------------------


class TrustStore:
    """Saved per-folder decisions in ``<delta home>/trust.json``.

    The file is strict: unknown versions or fields, relative or duplicated
    paths, and unknown verdicts make it unusable rather than partly read,
    and an unusable file is never repaired or reset behind the user's back.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def default(cls) -> TrustStore:
        from delta_app.discovery import default_paths

        return cls(default_paths().home / _STORE_NAME)

    def decisions(self) -> dict[Path, Verdict]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TrustStoreError(f"{self.path} is unreadable: {exc}") from exc
        if not isinstance(data, dict) or set(data) != {"version", "decisions"}:
            raise TrustStoreError(f"{self.path} does not have the expected fields")
        if data["version"] != _STORE_VERSION:
            raise TrustStoreError(f"{self.path} has unsupported version {data['version']!r}")
        if not isinstance(data["decisions"], list):
            raise TrustStoreError(f"{self.path}: 'decisions' must be a list")
        found: dict[Path, Verdict] = {}
        seen: set[str] = set()
        for item in data["decisions"]:
            if not isinstance(item, dict) or set(item) != {"path", "decision"}:
                raise TrustStoreError(f"{self.path}: malformed decision {item!r}")
            raw, verdict = item["path"], item["decision"]
            if not isinstance(raw, str) or not os.path.isabs(raw):
                raise TrustStoreError(f"{self.path}: path must be absolute: {raw!r}")
            if os.path.normpath(raw) != raw:
                raise TrustStoreError(f"{self.path}: path is not normalized: {raw!r}")
            if verdict not in _VERDICTS:
                raise TrustStoreError(f"{self.path}: unknown decision {verdict!r}")
            if _key(Path(raw)) in seen:
                raise TrustStoreError(f"{self.path}: duplicate path {raw!r}")
            seen.add(_key(Path(raw)))
            found[Path(raw)] = verdict
        return found

    def nearest(self, folder: Path) -> tuple[Path, Verdict] | None:
        """The saved decision for ``folder`` or its closest ancestor."""
        saved = {_key(p): (p, v) for p, v in self.decisions().items()}
        for candidate in (folder, *folder.parents):
            hit = saved.get(_key(candidate))
            if hit is not None:
                return hit
        return None

    def save(self, folder: Path, verdict: Verdict, *, replacing: Path | None = None) -> None:
        """Record a decision atomically, optionally dropping one other entry.

        The new file is fully written and flushed beside the old one, then
        swapped in with a single replace, so a crash leaves either the old
        decisions or the new ones — never a truncated file.
        """
        current = self.decisions()  # refuses to build on an unusable file
        updated = {_key(p): (p, v) for p, v in current.items()}
        if replacing is not None:
            updated.pop(_key(replacing), None)
        updated[_key(folder)] = (folder, verdict)
        body = {
            "version": _STORE_VERSION,
            "decisions": [
                {"path": str(p), "decision": v}
                for p, v in sorted(updated.values(), key=lambda pair: str(pair[0]))
            ],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, staged = tempfile.mkstemp(dir=self.path.parent, prefix=".trust-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(body, handle, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.chmod(staged, 0o600)
            except OSError:
                pass
            os.replace(staged, self.path)
        except BaseException:
            Path(staged).unlink(missing_ok=True)
            raise


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrustRequest:
    """What a frontend shows when asking: the folder and what it would load."""

    folder: Path
    inputs: ProjectInputs
    store_problem: str | None = None

    @property
    def parent(self) -> Path | None:
        parent = self.folder.parent
        return None if parent == self.folder else parent


@dataclass(frozen=True, slots=True)
class TrustOutcome:
    """The resolved decision for one folder in this run."""

    folder: Path | None
    trusted: bool
    reason: str
    inputs: ProjectInputs | None = None
    notice: str | None = None


Asker = Callable[[TrustRequest], Awaitable[Choice | None]]

# Completed resolutions for this process, keyed by canonical folder.
_resolved: dict[str, TrustOutcome] = {}
_engaged = False


def reset_trust_cache() -> None:
    """Forget every resolution made in this process (tests, reload)."""
    global _engaged
    _resolved.clear()
    _engaged = False


async def resolve_project_trust(
    cwd: str | os.PathLike[str],
    *,
    override: bool | None = None,
    default: DefaultPolicy = "ask",
    ask: Asker | None = None,
    store: TrustStore | None = None,
) -> TrustOutcome:
    """Decide whether ``cwd``'s project inputs may load; see module docs."""
    global _engaged
    _engaged = True
    try:
        folder = canonical_folder(cwd)
    except OSError as exc:
        return TrustOutcome(None, False, f"working folder could not be resolved ({exc})")

    cached = _resolved.get(_key(folder))
    if cached is not None:
        return cached

    inputs = scan_project_inputs(folder)
    if override is not None:
        flag = "--approve" if override else "--no-approve"
        outcome = TrustOutcome(folder, override, f"{flag} for this run", inputs)
    elif inputs.empty:
        # Nothing to protect. Not cached: files added later must still ask.
        return TrustOutcome(folder, True, "no project inputs", inputs)
    else:
        outcome = await _decide(folder, inputs, default, ask, store or TrustStore.default())

    _resolved[_key(folder)] = outcome
    return outcome


async def _decide(
    folder: Path,
    inputs: ProjectInputs,
    default: DefaultPolicy,
    ask: Asker | None,
    store: TrustStore,
) -> TrustOutcome:
    problem: str | None = None
    try:
        saved = store.nearest(folder)
    except TrustStoreError as exc:
        saved, problem = None, str(exc)

    if saved is not None:
        where, verdict = saved
        scope = "this folder" if where == folder else f"parent {where}"
        return TrustOutcome(folder, verdict == "trusted", f"saved decision for {scope}", inputs)
    # An unusable store can grant nothing implicitly; only a run-only
    # answer or --approve can load project inputs until it is fixed.
    if problem is None and default == "always":
        return TrustOutcome(folder, True, "user default: always trust", inputs)
    if problem is None and default == "never":
        return TrustOutcome(folder, False, "user default: never trust", inputs)
    if ask is None:
        reason = problem or "no saved decision and no way to ask"
        return TrustOutcome(folder, False, reason, inputs)

    choice = await ask(TrustRequest(folder, inputs, problem))
    if choice is None:
        raise StartupCancelled(f"No trust decision for {folder}")
    return _apply(choice, folder, inputs, store)


def _apply(choice: Choice, folder: Path, inputs: ProjectInputs, store: TrustStore) -> TrustOutcome:
    if choice is Choice.TRUST_ONCE:
        return TrustOutcome(folder, True, "trusted for this run", inputs)
    if choice is Choice.DISTRUST_ONCE:
        return TrustOutcome(folder, False, "not trusted for this run", inputs)

    target, verdict, replacing = folder, "trusted", None
    if choice is Choice.TRUST_PARENT and folder.parent != folder:
        target, replacing = folder.parent, folder
    elif choice is Choice.DISTRUST_FOLDER:
        verdict = "untrusted"
    try:
        store.save(target, verdict, replacing=replacing)  # type: ignore[arg-type]
    except (OSError, TrustStoreError) as exc:
        # A remembered grant only counts once it is actually remembered.
        notice = f"Could not save trust decision ({exc}); project inputs stay off."
        return TrustOutcome(folder, False, "decision not saved", inputs, notice)
    trusted = verdict == "trusted"
    scope = "this folder" if target == folder else f"parent {target}"
    return TrustOutcome(folder, trusted, f"saved: {verdict} ({scope})", inputs)


def project_inputs_allowed(project: str | os.PathLike[str]) -> bool:
    """Whether loaders may read ``project``'s own inputs right now.

    Code that never engages trust (a library caller, most tests) keeps the
    unrestricted behaviour. Once a frontend resolves trust, a folder without
    a completed decision loads project inputs only if it has none.
    """
    if not _engaged:
        return True
    try:
        folder = canonical_folder(project)
    except OSError:
        return False
    outcome = _resolved.get(_key(folder))
    if outcome is not None:
        return outcome.trusted
    return scan_project_inputs(folder).empty


__all__ = [
    "Choice",
    "DefaultPolicy",
    "ProjectInputs",
    "StartupCancelled",
    "TrustOutcome",
    "TrustRequest",
    "TrustStore",
    "TrustStoreError",
    "canonical_folder",
    "project_inputs_allowed",
    "reset_trust_cache",
    "resolve_project_trust",
    "scan_project_inputs",
]
