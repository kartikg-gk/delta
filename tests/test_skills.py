"""Tests for the markdown-based skill system."""

from __future__ import annotations

from pathlib import Path

import pytest

from delta_app.skillset import (
    Severity,
    Skill,
    SkillInvocation,
    SkillLoadError,
    build_skill_index,
    expand_command,
    format_invocation,
    load_skills,
    load_skills_with_diagnostics,
    parse_invocation,
)

# ── helpers ────────────────────────────────────────────────────────────────


def _write_skill(
    root: Path,
    name: str,
    content: str,
) -> Path:
    """Create a skill directory with SKILL.md at root/name/SKILL.md."""
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
    return skill_dir


def _skill_with_front_matter(
    name: str = "my-skill",
    description: str = "A test skill.",
) -> str:
    return (
        f"---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        f"---\n"
        f"\n"
        f"# {name}\n"
        f"\n"
        f"Body content here.\n"
    )


# ── test: Skill dataclass ─────────────────────────────────────────────────


def test_skill_immutable() -> None:
    skill = Skill(
        name="test",
        description="A skill",
        body="body",
        source=Path("/fake/SKILL.md"),
    )
    assert skill.name == "test"
    assert skill.command == "/skill:test"

    with pytest.raises(AttributeError):
        skill.name = "other"  # type: ignore[misc]


def test_skill_command_property() -> None:
    skill = Skill(name="refactor", description="", body="", source=Path("."))
    assert skill.command == "/skill:refactor"


# ── test: SkillInvocation dataclass ────────────────────────────────────────


def test_skill_invocation_immutable() -> None:
    inv = SkillInvocation(name="x", arguments="arg1", body="body")
    assert inv.name == "x"
    assert inv.arguments == "arg1"
    assert inv.body == "body"

    with pytest.raises(AttributeError):
        inv.name = "y"  # type: ignore[misc]


# ── test: front-matter parsing ────────────────────────────────────────────


def test_load_skill_with_front_matter(tmp_path: Path) -> None:
    content = _skill_with_front_matter("alpha", "Does alpha things.")
    _write_skill(tmp_path, "alpha", content)

    skills = load_skills([tmp_path])
    assert len(skills) == 1
    assert skills[0].name == "alpha"
    assert skills[0].description == "Does alpha things."


def test_load_skill_without_front_matter(tmp_path: Path) -> None:
    content = "# My Tool\n\nThis tool does stuff.\n\nMore details.\n"
    _write_skill(tmp_path, "my-tool", content)

    skills = load_skills([tmp_path])
    assert len(skills) == 1
    assert skills[0].name == "my-tool"  # falls back to directory name
    assert skills[0].description == "This tool does stuff."  # first paragraph


def test_load_skill_name_from_directory(tmp_path: Path) -> None:
    content = "---\ndescription: Has desc but no name.\n---\n\nBody.\n"
    _write_skill(tmp_path, "cool-skill", content)

    skills = load_skills([tmp_path])
    assert skills[0].name == "cool-skill"


def test_load_skill_description_derived_from_body(tmp_path: Path) -> None:
    content = "---\nname: derived\n---\n\n# Heading\n\nFirst real paragraph.\n"
    _write_skill(tmp_path, "derived", content)

    skills = load_skills([tmp_path])
    assert skills[0].description == "First real paragraph."


def test_load_skill_quoted_values(tmp_path: Path) -> None:
    content = '---\nname: "quoted-name"\ndescription: \'single-quoted\'\n---\n\nBody.\n'
    _write_skill(tmp_path, "quoted", content)

    skills = load_skills([tmp_path])
    assert skills[0].name == "quoted-name"
    assert skills[0].description == "single-quoted"


def test_load_skill_empty_front_matter(tmp_path: Path) -> None:
    content = "---\n---\n\nJust body.\n"
    _write_skill(tmp_path, "empty-meta", content)

    skills = load_skills([tmp_path])
    assert skills[0].name == "empty-meta"
    assert skills[0].description == "Just body."


def test_load_skill_body_preserved(tmp_path: Path) -> None:
    body = "# Title\n\nParagraph one.\n\nParagraph two.\n"
    content = f"---\nname: body-test\n---\n\n{body}"
    _write_skill(tmp_path, "body-test", content)

    skills = load_skills([tmp_path])
    assert "Paragraph one." in skills[0].body
    assert "Paragraph two." in skills[0].body


# ── test: discovery rules ─────────────────────────────────────────────────


def test_ignores_directory_without_skill_md(tmp_path: Path) -> None:
    (tmp_path / "not-a-skill").mkdir()
    (tmp_path / "not-a-skill" / "README.md").write_text("nope")

    skills = load_skills([tmp_path])
    assert skills == []


def test_ignores_agents_md(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("# Agents\n")

    skills, diags = load_skills_with_diagnostics([tmp_path])
    assert skills == []
    # AGENTS.md should not produce any diagnostic
    assert not any(d.path.name == "AGENTS.md" for d in diags)


def test_bare_md_emits_diagnostic(tmp_path: Path) -> None:
    (tmp_path / "loose-skill.md").write_text("# A skill file\n")

    skills, diags = load_skills_with_diagnostics([tmp_path])
    assert skills == []
    assert len(diags) == 1
    assert diags[0].severity == Severity.INFO
    assert "loose-skill/SKILL.md" in diags[0].message


def test_multiple_bare_md_each_get_diagnostic(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\n")
    (tmp_path / "b.md").write_text("# B\n")

    _, diags = load_skills_with_diagnostics([tmp_path])
    assert len(diags) == 2
    names = {d.path.name for d in diags}
    assert names == {"a.md", "b.md"}


def test_non_md_files_ignored(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "script.py").write_text("pass")

    skills, diags = load_skills_with_diagnostics([tmp_path])
    assert skills == []
    assert diags == []


def test_empty_directory(tmp_path: Path) -> None:
    skills, diags = load_skills_with_diagnostics([tmp_path])
    assert skills == []
    assert diags == []


def test_nonexistent_directory() -> None:
    skills, diags = load_skills_with_diagnostics([Path("/nonexistent/path")])
    assert skills == []
    assert diags == []


# ── test: duplicate detection ─────────────────────────────────────────────


def test_duplicate_name_in_same_directory(tmp_path: Path) -> None:
    _write_skill(tmp_path, "alpha", "---\nname: shared-name\n---\nBody A.\n")
    _write_skill(tmp_path, "beta", "---\nname: shared-name\n---\nBody B.\n")

    skills, diags = load_skills_with_diagnostics([tmp_path])
    assert len(skills) == 1
    assert skills[0].name == "shared-name"

    warnings = [d for d in diags if d.severity == Severity.WARNING]
    assert len(warnings) == 1
    assert "Duplicate" in warnings[0].message


# ── test: precedence ──────────────────────────────────────────────────────


def test_higher_precedence_overrides(tmp_path: Path) -> None:
    high = tmp_path / "high"
    low = tmp_path / "low"
    high.mkdir()
    low.mkdir()

    _write_skill(high, "my-skill", "---\nname: my-skill\n---\nHigh body.\n")
    _write_skill(low, "my-skill", "---\nname: my-skill\n---\nLow body.\n")

    skills = load_skills([high, low])
    assert len(skills) == 1
    assert "High body." in skills[0].body


def test_lower_precedence_adds_unique_skills(tmp_path: Path) -> None:
    high = tmp_path / "high"
    low = tmp_path / "low"
    high.mkdir()
    low.mkdir()

    _write_skill(high, "a-skill", "---\nname: a-skill\n---\nFrom high.\n")
    _write_skill(low, "b-skill", "---\nname: b-skill\n---\nFrom low.\n")

    skills = load_skills([high, low])
    names = {s.name for s in skills}
    assert names == {"a-skill", "b-skill"}


# ── test: fatal errors ────────────────────────────────────────────────────


def test_unreadable_skill_emits_warning(tmp_path: Path) -> None:
    skill_dir = tmp_path / "broken"
    skill_dir.mkdir()
    md_path = skill_dir / "SKILL.md"
    # Write invalid UTF-8
    md_path.write_bytes(b"\x80\x81\x82")

    skills, diags = load_skills_with_diagnostics([tmp_path])
    assert skills == []
    assert len(diags) == 1
    assert diags[0].severity == Severity.WARNING
    assert "UTF-8" in diags[0].message


def test_skill_load_error_direct() -> None:
    with pytest.raises(SkillLoadError):
        from delta_app.skillset import _load_skill_from_dir
        _load_skill_from_dir(Path("/nonexistent/skill"))


# ── test: command expansion ───────────────────────────────────────────────


def test_expand_command_basic() -> None:
    skill = Skill(
        name="refactor",
        description="Refactor code.",
        body="Do the refactoring.",
        source=Path("."),
    )
    registry = {"refactor": skill}

    result = expand_command("/skill:refactor", registry)
    assert result is not None
    assert "<skill-invocation" in result
    assert 'name="refactor"' in result
    assert "Do the refactoring." in result
    assert "</skill-invocation>" in result


def test_expand_command_with_arguments() -> None:
    skill = Skill(name="test", description="", body="Body.", source=Path("."))
    registry = {"test": skill}

    result = expand_command("/skill:test arg1 arg2", registry)
    assert result is not None
    assert 'arguments="arg1 arg2"' in result


def test_expand_command_not_a_skill() -> None:
    assert expand_command("/help", {}) is None
    assert expand_command("hello", {}) is None
    assert expand_command("/model test", {}) is None


def test_expand_command_unknown_skill() -> None:
    with pytest.raises(KeyError, match="missing"):
        expand_command("/skill:missing", {})


# ── test: format / parse invocation round-trip ─────────────────────────────


def test_format_invocation_basic() -> None:
    inv = SkillInvocation(name="x", arguments="", body="Do stuff.")
    text = format_invocation(inv)
    assert '<skill-invocation name="x">' in text
    assert "Do stuff." in text
    assert "</skill-invocation>" in text


def test_format_invocation_with_arguments() -> None:
    inv = SkillInvocation(name="y", arguments="a b c", body="Body.")
    text = format_invocation(inv)
    assert 'arguments="a b c"' in text


def test_parse_invocation_basic() -> None:
    text = (
        '<skill-invocation name="alpha">\n'
        "Skill body here.\n"
        "</skill-invocation>"
    )
    inv = parse_invocation(text)
    assert inv is not None
    assert inv.name == "alpha"
    assert inv.arguments == ""
    assert inv.body == "Skill body here."


def test_parse_invocation_with_arguments() -> None:
    text = (
        '<skill-invocation name="beta" arguments="foo bar">\n'
        "Content.\n"
        "</skill-invocation>"
    )
    inv = parse_invocation(text)
    assert inv is not None
    assert inv.name == "beta"
    assert inv.arguments == "foo bar"


def test_parse_invocation_no_match() -> None:
    assert parse_invocation("just plain text") is None
    assert parse_invocation("<other-tag>stuff</other-tag>") is None


def test_format_then_parse_round_trip() -> None:
    original = SkillInvocation(
        name="round-trip",
        arguments="some args",
        body="Multi-line\nbody\ncontent.",
    )
    text = format_invocation(original)
    parsed = parse_invocation(text)

    assert parsed is not None
    assert parsed.name == original.name
    assert parsed.arguments == original.arguments
    assert parsed.body == original.body


def test_round_trip_no_arguments() -> None:
    original = SkillInvocation(name="simple", arguments="", body="Just body.")
    text = format_invocation(original)
    parsed = parse_invocation(text)

    assert parsed is not None
    assert parsed.name == "simple"
    assert parsed.arguments == ""
    assert parsed.body == "Just body."


# ── test: skill index ─────────────────────────────────────────────────────


def test_build_skill_index_empty() -> None:
    assert build_skill_index([]) == ""


def test_build_skill_index_single() -> None:
    skill = Skill(name="deploy", description="Deploy the app.", body="", source=Path("."))
    index = build_skill_index([skill])
    assert "/skill:deploy" in index
    assert "Deploy the app." in index
    assert "Available skills" in index


def test_build_skill_index_sorted() -> None:
    skills = [
        Skill(name="zebra", description="Z.", body="", source=Path(".")),
        Skill(name="alpha", description="A.", body="", source=Path(".")),
        Skill(name="mid", description="M.", body="", source=Path(".")),
    ]
    index = build_skill_index(skills)
    lines = index.strip().splitlines()
    # Skills should appear alphabetically (header + 3 skill lines)
    assert len(lines) == 4
    assert "alpha" in lines[1]
    assert "mid" in lines[2]
    assert "zebra" in lines[3]


# ── test: UTF-8 handling ──────────────────────────────────────────────────


def test_utf8_content(tmp_path: Path) -> None:
    content = "---\nname: ünïcödé\ndescription: Ångström résumé\n---\n\n日本語のスキル。\n"
    _write_skill(tmp_path, "unicode", content)

    skills = load_skills([tmp_path])
    assert len(skills) == 1
    assert skills[0].name == "ünïcödé"
    assert "Ångström" in skills[0].description
    assert "日本語" in skills[0].body


# ── test: edge cases ──────────────────────────────────────────────────────


def test_skill_body_only_headings(tmp_path: Path) -> None:
    content = "# Heading One\n## Heading Two\n"
    _write_skill(tmp_path, "headings-only", content)

    skills = load_skills([tmp_path])
    assert skills[0].name == "headings-only"
    assert skills[0].description == ""  # no paragraph to derive from


def test_skill_empty_body(tmp_path: Path) -> None:
    content = "---\nname: empty\ndescription: Empty skill.\n---\n"
    _write_skill(tmp_path, "empty", content)

    skills = load_skills([tmp_path])
    assert skills[0].name == "empty"
    assert skills[0].body.strip() == ""


def test_skill_front_matter_with_comments(tmp_path: Path) -> None:
    content = "---\n# This is a comment\nname: commented\n---\n\nBody.\n"
    _write_skill(tmp_path, "commented", content)

    skills = load_skills([tmp_path])
    assert skills[0].name == "commented"


def test_expand_command_name_only() -> None:
    skill = Skill(name="x", description="", body="B.", source=Path("."))
    result = expand_command("/skill:x", {"x": skill})
    assert result is not None
    parsed = parse_invocation(result)
    assert parsed is not None
    assert parsed.name == "x"
    assert parsed.arguments == ""


def test_description_skips_front_matter_separator(tmp_path: Path) -> None:
    content = "---\nname: sep-test\n---\n\n---\n\nActual first paragraph.\n"
    _write_skill(tmp_path, "sep-test", content)

    skills = load_skills([tmp_path])
    assert skills[0].description == "Actual first paragraph."


def test_multiple_resource_dirs_three_tiers(tmp_path: Path) -> None:
    project = tmp_path / "project"
    user = tmp_path / "user"
    system = tmp_path / "system"
    for d in (project, user, system):
        d.mkdir()

    _write_skill(project, "a", "---\nname: a\n---\nProject A.\n")
    _write_skill(user, "a", "---\nname: a\n---\nUser A.\n")
    _write_skill(user, "b", "---\nname: b\n---\nUser B.\n")
    _write_skill(system, "a", "---\nname: a\n---\nSystem A.\n")
    _write_skill(system, "b", "---\nname: b\n---\nSystem B.\n")
    _write_skill(system, "c", "---\nname: c\n---\nSystem C.\n")

    skills = load_skills([project, user, system])
    by_name = {s.name: s for s in skills}

    assert len(by_name) == 3
    assert "Project A." in by_name["a"].body  # project wins
    assert "User B." in by_name["b"].body     # user wins over system
    assert "System C." in by_name["c"].body   # only in system


def test_parse_invocation_embedded_in_larger_text() -> None:
    text = (
        "Some preamble.\n"
        '<skill-invocation name="embedded">\n'
        "Inner body.\n"
        "</skill-invocation>\n"
        "Some postamble.\n"
    )
    inv = parse_invocation(text)
    assert inv is not None
    assert inv.name == "embedded"
    assert inv.body == "Inner body."
