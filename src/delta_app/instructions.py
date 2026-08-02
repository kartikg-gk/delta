"""System-prompt assembly.

Composes the model-facing system prompt from whatever pieces are actually
available at run time: a fixed identity block, per-tool prompt guidance
(``ToolSpec.prompt_snippet`` / ``prompt_guidelines``), the skill index
(``skills.build_skill_index``), project instructions from ``AGENTS.md``
files discovered across the resource precedence hierarchy, and a
working-directory/date suffix. Sections with nothing to contribute are
omitted entirely rather than left as empty headers.

``AGENTS.md`` is searched in precedence order::

    ~/.delta/AGENTS.md  →  ~/.agents/AGENTS.md  →
    <project>/.delta/AGENTS.md  →  <project>/.agents/AGENTS.md  →
    <project>/AGENTS.md

Later files take priority.  All discovered files are concatenated.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from delta_app.skillset import Skill, build_skill_index
from delta_harness.contracts.tooling import ToolSpec

_AGENTS_FILE = "AGENTS.md"

_BASE_IDENTITY = """You are Delta, a small, readable coding-agent harness. You help the \
user read, write, and modify code in their project using the tools available to you.

Work carefully: read a file before editing it, make minimal focused changes, and verify \
your work with a test or command when one is available. Prefer clear, direct answers over \
hedging."""


def _tool_guidelines(tools: Sequence[ToolSpec]) -> str:
    """Render each tool's ``prompt_snippet``/``prompt_guidelines``, skipping tools with neither.

    A guideline repeated by several tools is printed once: duplicated
    instructions waste context and read as emphasis the author did not intend.
    """
    seen: set[str] = set()
    sections: list[str] = []
    for tool in tools:
        parts: list[str] = []
        if tool.prompt_snippet:
            parts.append(tool.prompt_snippet)
        for guideline in tool.prompt_guidelines:
            text = guideline.strip()
            if text and text not in seen:
                seen.add(text)
                parts.append(f"- {text}")
        if parts:
            sections.append(f"### {tool.name}\n" + "\n".join(parts))
    if not sections:
        return ""
    return "## Tool guidelines\n\n" + "\n\n".join(sections)


def _skill_section(skills: Sequence[Skill]) -> str:
    return build_skill_index(list(skills))


def _agents_md_search_paths(cwd: str) -> list[Path]:
    """Return ``AGENTS.md`` search paths in lowest-to-highest precedence order.

    Files later in the list take priority.  All discovered files are
    concatenated into the system prompt so that project-level context
    augments (not replaces) user-level context.
    """
    from delta_app.discovery import default_paths

    paths = default_paths(project=Path(cwd))
    return [
        paths.home / _AGENTS_FILE,                  # ~/.delta/AGENTS.md
        paths.agents_home / _AGENTS_FILE,            # ~/.agents/AGENTS.md
        paths.project / ".delta" / _AGENTS_FILE,     # <project>/.delta/AGENTS.md
        paths.project / ".agents" / _AGENTS_FILE,    # <project>/.agents/AGENTS.md
        paths.project / _AGENTS_FILE,                # <project>/AGENTS.md (highest)
    ]


def _read_agents_file(path: Path) -> str:
    """Read and return the trimmed contents of an AGENTS.md, or empty string."""
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _project_context(cwd: str) -> str:
    """Collect project instructions from ``AGENTS.md`` across the precedence hierarchy."""
    search = _agents_md_search_paths(cwd)
    sections: list[str] = []
    for path in search:
        content = _read_agents_file(path)
        if content:
            sections.append(content)
    if not sections:
        return ""
    combined = "\n\n".join(sections)
    return f"## Project instructions ({_AGENTS_FILE})\n\n{combined}"


def _environment_suffix(cwd: str, *, include_date: bool) -> str:
    lines = [f"Working directory: {cwd}"]
    if include_date:
        lines.append(f"Current date: {datetime.now(UTC):%Y-%m-%d}")
    return "\n".join(lines)


def system_prompt(
    *,
    tools: Sequence[ToolSpec] = (),
    skills: Sequence[Skill] = (),
    cwd: str | None = None,
    include_date: bool = True,
) -> str:
    """Assemble the full system prompt: identity, tool guidance, skills, project context, env."""
    resolved_cwd = cwd or os.getcwd()

    sections = [
        _BASE_IDENTITY,
        _tool_guidelines(tools),
        _skill_section(skills),
        _project_context(resolved_cwd),
        _environment_suffix(resolved_cwd, include_date=include_date),
    ]
    return "\n\n".join(section for section in sections if section)


__all__ = ["system_prompt"]
