"""Canonical filesystem layout and automatic resource discovery.

Two layers, kept independent:

1. **Path utility** — ``DeltaPaths`` computes canonical filesystem locations
   for Delta's home directory, session storage, logs, and per-resource-type
   subdirectories.  It never touches the filesystem; it only returns ``Path``
   objects.

2. **Resource search paths** — ``skill_search_paths``, ``prompt_search_paths``,
   ``theme_search_paths``, and the general ``resource_search_paths`` assemble
   precedence-ordered directory lists from a ``DeltaPaths`` instance.

3. **Markdown resource helpers** — ``MarkdownResource`` and ``parse_markdown``
   provide shared frontmatter parsing, metadata extraction, and description
   derivation for any ``.md``-based resource (skills, prompts, etc.).

Directory layout
----------------

User home (durable, cross-project)::

    ~/.delta/
        sessions/
        logs/
        skills/
        prompts/
        themes/

    ~/.agents/
        skills/
        prompts/

Project-local (checked into repo, project-specific)::

    <project>/.delta/
        skills/
        prompts/
        themes/

    <project>/.agents/
        skills/
        prompts/

Precedence (lowest → highest)::

    ~/.delta/<type>  →  ~/.agents/<type>  →  <project>/.delta/<type>  →  <project>/.agents/<type>

Later locations override earlier ones when resources share the same name.
Themes are Delta-specific and skip the ``.agents`` directories.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from delta_harness.contracts.values import JValue

# ---------------------------------------------------------------------------
# Environment variable names
# ---------------------------------------------------------------------------

DELTA_HOME_ENV = "DELTA_HOME"
DELTA_SESSIONS_DIR_ENV = "DELTA_SESSIONS_DIR"

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_FRONT_MATTER_RE = re.compile(r"\A---[ \t]*\n(.*?\n)---[ \t]*\n", re.DOTALL)


def _resolve_home() -> Path:
    """Resolve Delta's home directory from ``$DELTA_HOME`` or ``~/.delta``."""
    env = os.environ.get(DELTA_HOME_ENV)
    return Path(env) if env else Path.home() / ".delta"


def _resolve_sessions(home: Path) -> Path:
    """Resolve the sessions directory from ``$DELTA_SESSIONS_DIR`` or ``<home>/sessions``."""
    env = os.environ.get(DELTA_SESSIONS_DIR_ENV)
    return Path(env) if env else home / "sessions"


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    """Remove duplicate paths while preserving order."""
    seen: set[Path] = set()
    result: list[Path] = []
    for p in paths:
        resolved = p.resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(p)
    return result


# ---------------------------------------------------------------------------
# Path utility
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DeltaPaths:
    """Canonical filesystem locations for Delta.

    Computes paths only — never reads from or writes to the filesystem.
    All directories are derived from two roots: the Delta home directory
    (``~/.delta`` by default) and the project working directory.

    Override the roots via constructor arguments or the ``DELTA_HOME`` /
    ``DELTA_SESSIONS_DIR`` environment variables.
    """

    home: Path = field(default_factory=_resolve_home)
    project: Path = field(default_factory=Path.cwd)

    # ── user home locations ───────────────────────────────────────────

    @property
    def sessions(self) -> Path:
        """Session transcript and index storage (``~/.delta/sessions``)."""
        return _resolve_sessions(self.home)

    @property
    def logs(self) -> Path:
        """Log file directory (``~/.delta/logs``)."""
        return self.home / "logs"

    @property
    def user_skills(self) -> Path:
        """User-level skill definitions (``~/.delta/skills``)."""
        return self.home / "skills"

    @property
    def user_prompts(self) -> Path:
        """User-level prompt templates (``~/.delta/prompts``)."""
        return self.home / "prompts"

    @property
    def user_themes(self) -> Path:
        """User-level theme files (``~/.delta/themes``)."""
        return self.home / "themes"

    # ── user .agents locations ────────────────────────────────────────

    @property
    def agents_home(self) -> Path:
        """Cross-tool agent resource directory (``~/.agents``)."""
        return self.home.parent / ".agents"

    @property
    def agents_skills(self) -> Path:
        """Cross-tool skill definitions (``~/.agents/skills``)."""
        return self.agents_home / "skills"

    @property
    def agents_prompts(self) -> Path:
        """Cross-tool prompt templates (``~/.agents/prompts``)."""
        return self.agents_home / "prompts"

    # ── project-local .delta locations ────────────────────────────────

    @property
    def project_skills(self) -> Path:
        """Project-level skill definitions (``<project>/.delta/skills``)."""
        return self.project / ".delta" / "skills"

    @property
    def project_prompts(self) -> Path:
        """Project-level prompt templates (``<project>/.delta/prompts``)."""
        return self.project / ".delta" / "prompts"

    @property
    def project_themes(self) -> Path:
        """Project-level theme files (``<project>/.delta/themes``)."""
        return self.project / ".delta" / "themes"

    # ── project-local .agents locations ───────────────────────────────

    @property
    def project_agents(self) -> Path:
        """Project-level cross-tool agent directory (``<project>/.agents``)."""
        return self.project / ".agents"

    @property
    def project_agents_skills(self) -> Path:
        """Project-level cross-tool skills (``<project>/.agents/skills``)."""
        return self.project_agents / "skills"

    @property
    def project_agents_prompts(self) -> Path:
        """Project-level cross-tool prompts (``<project>/.agents/prompts``)."""
        return self.project_agents / "prompts"


# ---------------------------------------------------------------------------
# Default instance factory
# ---------------------------------------------------------------------------


def default_paths(
    *,
    home: Path | None = None,
    project: Path | None = None,
) -> DeltaPaths:
    """Build a ``DeltaPaths`` with optional overrides for home and project.

    Falls back to environment variables and ``Path.cwd()`` when not
    specified.
    """
    return DeltaPaths(
        home=home or _resolve_home(),
        project=project or Path.cwd(),
    )


# ---------------------------------------------------------------------------
# Resource search paths — precedence-ordered directory lists
# ---------------------------------------------------------------------------


def skill_search_paths(paths: DeltaPaths) -> list[Path]:
    """Return skill directories in highest-precedence-first order.

    Precedence (highest → lowest):
    ``<project>/.agents/skills`` > ``<project>/.delta/skills`` >
    ``~/.agents/skills`` > ``~/.delta/skills``

    Consumers iterate this list and keep the first occurrence of each
    named resource, so the first directory wins on name collisions.
    """
    return _dedupe_paths([
        paths.project_agents_skills,
        paths.project_skills,
        paths.agents_skills,
        paths.user_skills,
    ])


def prompt_search_paths(paths: DeltaPaths) -> list[Path]:
    """Return prompt directories in highest-precedence-first order.

    Same four-level precedence as skills.
    """
    return _dedupe_paths([
        paths.project_agents_prompts,
        paths.project_prompts,
        paths.agents_prompts,
        paths.user_prompts,
    ])


def theme_search_paths(paths: DeltaPaths) -> list[Path]:
    """Return theme directories in highest-precedence-first order.

    Themes are Delta-specific — only ``.delta`` directories are searched.
    """
    return _dedupe_paths([
        paths.project_themes,
        paths.user_themes,
    ])


def resource_search_paths(
    paths: DeltaPaths,
    resource_type: str,
    *,
    delta_only: bool = False,
) -> list[Path]:
    """Return search directories for an arbitrary resource type.

    Parameters
    ----------
    paths:
        The canonical paths instance.
    resource_type:
        Subdirectory name (e.g. ``"skills"``, ``"prompts"``, ``"themes"``).
    delta_only:
        When ``True``, skip ``.agents`` directories (used for themes).
    """
    dirs: list[Path] = []
    if not delta_only:
        dirs.append(paths.project / ".agents" / resource_type)
    dirs.append(paths.project / ".delta" / resource_type)
    if not delta_only:
        dirs.append(paths.agents_home / resource_type)
    dirs.append(paths.home / resource_type)
    return _dedupe_paths(dirs)


# ---------------------------------------------------------------------------
# Resource discovery diagnostics
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResourceDiagnostic:
    """Non-fatal diagnostic emitted during resource discovery."""

    resource_name: str
    winner: Path
    overridden: Path
    message: str


def collect_override_diagnostics(
    discovered: dict[str, Path],
    new_resources: dict[str, Path],
    source_dir: Path,
) -> list[ResourceDiagnostic]:
    """Produce diagnostics when resources in *new_resources* would be
    overridden by existing entries in *discovered*.

    Called during multi-directory scanning to explain which copy won.
    """
    diagnostics: list[ResourceDiagnostic] = []
    for name, new_path in new_resources.items():
        if name in discovered:
            diagnostics.append(ResourceDiagnostic(
                resource_name=name,
                winner=discovered[name],
                overridden=new_path,
                message=(
                    f"Resource {name!r} from {source_dir} overridden by "
                    f"higher-precedence copy at {discovered[name]}"
                ),
            ))
    return diagnostics


# ---------------------------------------------------------------------------
# Markdown resource helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MarkdownResource:
    """A parsed markdown file with optional frontmatter metadata.

    Suitable for skills, prompt templates, or any ``.md``-based resource.
    """

    path: Path
    body: str
    metadata: dict[str, str]

    @property
    def name(self) -> str:
        """The ``name`` from frontmatter, or the file/directory stem."""
        return self.metadata.get("name", self.path.stem)

    @property
    def description(self) -> str:
        """The ``description`` from frontmatter, or the first non-heading paragraph."""
        explicit = self.metadata.get("description")
        if explicit:
            return explicit
        return derive_description(self.body)

    def metadata_json(self) -> dict[str, JValue]:
        """Return metadata as a JSON-compatible dictionary.

        All frontmatter values are strings; this method preserves them
        as-is since YAML-free parsing cannot infer types.
        """
        return dict(self.metadata)


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Split optional YAML-like frontmatter from the markdown body.

    Returns ``(metadata, body)``.  Only simple ``key: value`` lines are
    recognised — no nested YAML, no dependency on a YAML library.
    Surrounding quotes on values are stripped.
    """
    match = _FRONT_MATTER_RE.match(text)
    if match is None:
        return {}, text

    raw = match.group(1)
    body = text[match.end():]
    meta: dict[str, str] = {}
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        colon = stripped.find(":")
        if colon < 1:
            continue
        key = stripped[:colon].strip()
        value = stripped[colon + 1:].strip()
        if len(value) >= 2 and value[0] in ('"', "'") and value[-1] == value[0]:
            value = value[1:-1]
        meta[key] = value
    return meta, body


def derive_description(body: str) -> str:
    """Extract the first non-heading, non-blank paragraph as a description."""
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if stripped.startswith("---"):
            continue
        return stripped
    return ""


def parse_markdown(path: Path) -> MarkdownResource:
    """Read and parse a markdown file into a ``MarkdownResource``.

    Raises ``OSError`` on I/O failures and ``UnicodeDecodeError`` on
    encoding problems — callers should handle these at the boundary.
    """
    raw = path.read_text(encoding="utf-8")
    metadata, body = parse_front_matter(raw)
    return MarkdownResource(path=path, body=body, metadata=metadata)


__all__ = [
    "DELTA_HOME_ENV",
    "DELTA_SESSIONS_DIR_ENV",
    "DeltaPaths",
    "MarkdownResource",
    "ResourceDiagnostic",
    "collect_override_diagnostics",
    "default_paths",
    "derive_description",
    "parse_front_matter",
    "parse_markdown",
    "prompt_search_paths",
    "resource_search_paths",
    "skill_search_paths",
    "theme_search_paths",
]
