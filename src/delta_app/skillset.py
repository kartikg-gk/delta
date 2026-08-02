"""Markdown-based skill system: discovery, loading, expansion, and indexing.

A *skill* is a directory containing a ``SKILL.md`` file.  The markdown may
begin with a YAML front-matter block delimited by ``---`` lines; recognised
keys are ``name`` and ``description``.  When ``name`` is absent the directory
name is used; when ``description`` is absent the first non-heading paragraph
is extracted.

Skills are loaded from one or more *resource directories* in precedence order
(highest first).  A skill that appears in a higher-precedence directory
shadows any skill with the same name in lower-precedence directories.

Public surface
--------------
Types:  ``Skill``, ``SkillInvocation``, ``SkillDiagnostic``
Load:   ``load_skills``, ``load_skills_with_diagnostics``
Use:    ``expand_command``, ``parse_invocation``, ``format_invocation``
Index:  ``build_skill_index``
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Sequence


# ── constants ──────────────────────────────────────────────────────────────

_SKILL_FILE = "SKILL.md"
_AGENTS_FILE = "AGENTS.md"
_FRONT_MATTER_FENCE = "---"
_COMMAND_PREFIX = "/skill:"
_INVOCATION_TAG = "skill-invocation"


# ── diagnostic levels ─────────────────────────────────────────────────────


class Severity(Enum):
    """Non-fatal diagnostic severity."""

    INFO = "info"
    WARNING = "warning"


# ── data models ────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Skill:
    """An immutable, loaded skill definition."""

    name: str
    description: str
    body: str
    source: Path

    @property
    def command(self) -> str:
        """The slash command that invokes this skill."""
        return f"{_COMMAND_PREFIX}{self.name}"


@dataclass(frozen=True, slots=True)
class SkillInvocation:
    """A parsed ``/skill:<name>`` expansion ready for prompt injection."""

    name: str
    arguments: str
    body: str


@dataclass(frozen=True, slots=True)
class SkillDiagnostic:
    """A non-fatal issue encountered during skill discovery."""

    severity: Severity
    path: Path
    message: str


class SkillLoadError(Exception):
    """Fatal error during skill loading (unreadable file, bad encoding)."""


# ── front-matter parsing ──────────────────────────────────────────────────

_FRONT_MATTER_RE = re.compile(
    r"\A---[ \t]*\n(.*?\n)---[ \t]*\n",
    re.DOTALL,
)


def _parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Split optional YAML front matter from the markdown body.

    Returns ``(metadata_dict, body_without_front_matter)``.  Only simple
    ``key: value`` lines are recognised — no nested YAML.
    """
    match = _FRONT_MATTER_RE.match(text)
    if match is None:
        return {}, text

    raw_yaml = match.group(1)
    body = text[match.end():]
    meta: dict[str, str] = {}
    for line in raw_yaml.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        colon = line.find(":")
        if colon < 1:
            continue
        key = line[:colon].strip()
        value = line[colon + 1:].strip()
        # Strip surrounding quotes
        if len(value) >= 2 and value[0] in ('"', "'") and value[-1] == value[0]:
            value = value[1:-1]
        meta[key] = value

    return meta, body


def _derive_description(body: str) -> str:
    """Extract the first non-heading, non-empty paragraph as a description."""
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


# ── single-skill loading ──────────────────────────────────────────────────


def _load_skill_from_dir(skill_dir: Path) -> Skill:
    """Read ``SKILL.md`` from *skill_dir* and return a ``Skill``.

    Raises ``SkillLoadError`` on I/O or encoding problems.
    """
    md_path = skill_dir / _SKILL_FILE
    try:
        raw = md_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SkillLoadError(f"Cannot read {md_path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise SkillLoadError(
            f"{md_path} is not valid UTF-8: {exc}"
        ) from exc

    meta, body = _parse_front_matter(raw)
    name = meta.get("name", skill_dir.name)
    description = meta.get("description", "") or _derive_description(body)

    return Skill(
        name=name,
        description=description,
        body=body,
        source=md_path,
    )


# ── multi-directory discovery ─────────────────────────────────────────────


def _discover_in_directory(
    root: Path,
) -> tuple[list[Skill], list[SkillDiagnostic]]:
    """Scan a single resource directory for skills.

    Returns ``(skills, diagnostics)``.  Duplicates *within the same root*
    are reported as warnings.
    """
    skills: list[Skill] = []
    diagnostics: list[SkillDiagnostic] = []
    seen_names: dict[str, Path] = {}

    if not root.is_dir():
        return skills, diagnostics

    for child in sorted(root.iterdir()):
        # Skip AGENTS.md
        if child.name == _AGENTS_FILE:
            continue

        # Bare .md files are not skills
        if child.is_file() and child.suffix.lower() == ".md":
            diagnostics.append(SkillDiagnostic(
                severity=Severity.INFO,
                path=child,
                message=(
                    f"Bare markdown file ignored. To make it a skill, "
                    f"move it to {child.stem}/{_SKILL_FILE}"
                ),
            ))
            continue

        # Only directories with SKILL.md
        if not child.is_dir():
            continue
        if not (child / _SKILL_FILE).is_file():
            continue

        try:
            skill = _load_skill_from_dir(child)
        except SkillLoadError as exc:
            diagnostics.append(SkillDiagnostic(
                severity=Severity.WARNING,
                path=child,
                message=str(exc),
            ))
            continue

        if skill.name in seen_names:
            diagnostics.append(SkillDiagnostic(
                severity=Severity.WARNING,
                path=child,
                message=(
                    f"Duplicate skill name {skill.name!r} in same directory; "
                    f"already defined at {seen_names[skill.name]}"
                ),
            ))
            continue

        seen_names[skill.name] = child
        skills.append(skill)

    return skills, diagnostics


def load_skills_with_diagnostics(
    resource_dirs: Sequence[Path],
) -> tuple[list[Skill], list[SkillDiagnostic]]:
    """Load skills from multiple resource directories with diagnostics.

    *resource_dirs* is ordered highest-precedence first.  A skill discovered
    in a higher-precedence directory shadows any skill with the same name
    from a lower-precedence directory.

    Returns ``(skills, diagnostics)`` where ``skills`` is deduplicated and
    ``diagnostics`` collects all non-fatal issues from every directory.
    """
    combined: dict[str, Skill] = {}
    all_diagnostics: list[SkillDiagnostic] = []

    for root in resource_dirs:
        skills, diags = _discover_in_directory(root)
        all_diagnostics.extend(diags)
        for skill in skills:
            if skill.name not in combined:
                combined[skill.name] = skill

    return list(combined.values()), all_diagnostics


def load_skills(resource_dirs: Sequence[Path]) -> list[Skill]:
    """Load skills from resource directories, discarding diagnostics.

    Convenience wrapper around ``load_skills_with_diagnostics``.
    """
    skills, _ = load_skills_with_diagnostics(resource_dirs)
    return skills


# ── command expansion ──────────────────────────────────────────────────────


def expand_command(
    text: str,
    skills: dict[str, Skill],
) -> str | None:
    """Expand a ``/skill:<name>`` command into an XML invocation block.

    Returns the expanded text, or ``None`` if *text* is not a skill command.
    Raises ``KeyError`` if the skill name is not found.
    """
    if not text.startswith(_COMMAND_PREFIX):
        return None

    rest = text[len(_COMMAND_PREFIX):]
    parts = rest.split(None, 1)
    name = parts[0] if parts else rest
    arguments = parts[1] if len(parts) > 1 else ""

    if name not in skills:
        raise KeyError(f"Unknown skill: {name!r}")

    skill = skills[name]
    return format_invocation(SkillInvocation(
        name=name,
        arguments=arguments,
        body=skill.body,
    ))


# ── invocation formatting / parsing ────────────────────────────────────────

_INVOCATION_RE = re.compile(
    rf"<{_INVOCATION_TAG}\s+name=\"([^\"]+)\"(?:\s+arguments=\"([^\"]*)\")?>\n"
    rf"(.*?)"
    rf"</{_INVOCATION_TAG}>",
    re.DOTALL,
)


def format_invocation(invocation: SkillInvocation) -> str:
    """Render a ``SkillInvocation`` as an XML-like block."""
    attrs = f'name="{invocation.name}"'
    if invocation.arguments:
        attrs += f' arguments="{invocation.arguments}"'
    return (
        f"<{_INVOCATION_TAG} {attrs}>\n"
        f"{invocation.body}\n"
        f"</{_INVOCATION_TAG}>"
    )


def parse_invocation(text: str) -> SkillInvocation | None:
    """Parse an XML-like skill invocation block back into a ``SkillInvocation``.

    Returns ``None`` if *text* does not contain a valid invocation block.
    """
    match = _INVOCATION_RE.search(text)
    if match is None:
        return None
    return SkillInvocation(
        name=match.group(1),
        arguments=match.group(2) or "",
        body=match.group(3).strip(),
    )


# ── skill index for system prompts ────────────────────────────────────────


def build_skill_index(skills: Sequence[Skill]) -> str:
    """Build a concise skill listing suitable for inclusion in a system prompt.

    Each skill is rendered as one line: ``/skill:<name> — <description>``
    """
    if not skills:
        return ""
    lines = [
        f"  {skill.command} — {skill.description}"
        for skill in sorted(skills, key=lambda s: s.name)
    ]
    header = "Available skills (invoke with /skill:<name>):"
    return header + "\n" + "\n".join(lines)


__all__ = [
    "Severity",
    "Skill",
    "SkillDiagnostic",
    "SkillInvocation",
    "SkillLoadError",
    "build_skill_index",
    "expand_command",
    "format_invocation",
    "load_skills",
    "load_skills_with_diagnostics",
    "parse_invocation",
]
