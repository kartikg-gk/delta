"""Tests for the markdown prompt template system."""

from __future__ import annotations

from pathlib import Path

import pytest

from delta_app.prompts import (
    PromptTemplate,
    Severity,
    TemplateLoadError,
    TemplateRenderError,
    expand_slash_command,
    load_prompt_templates,
    load_prompt_templates_with_diagnostics,
    render_template,
)

# ── helpers ────────────────────────────────────────────────────────────────


def _write_template(root: Path, name: str, content: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}.md"
    path.write_text(content, encoding="utf-8")
    return path


def _tpl(
    name: str = "test",
    content: str = "Hello {{ name }}!",
    description: str | None = None,
) -> PromptTemplate:
    return PromptTemplate(
        name=name,
        path=Path(f"/fake/{name}.md"),
        content=content,
        description=description,
    )


# ── test: PromptTemplate dataclass ────────────────────────────────────────


def test_template_is_frozen() -> None:
    t = _tpl()
    with pytest.raises(AttributeError):
        t.name = "other"  # type: ignore[misc]


def test_template_fields() -> None:
    t = _tpl("greet", "Hi {{ who }}.", "A greeting template.")
    assert t.name == "greet"
    assert t.content == "Hi {{ who }}."
    assert t.description == "A greeting template."
    assert t.path == Path("/fake/greet.md")


def test_template_description_optional() -> None:
    t = _tpl(description=None)
    assert t.description is None


# ── test: rendering ───────────────────────────────────────────────────────


def test_render_basic() -> None:
    t = _tpl(content="Hello {{ name }}!")
    assert render_template(t, {"name": "World"}) == "Hello World!"


def test_render_multiple_vars() -> None:
    t = _tpl(content="{{ a }} and {{ b }}")
    assert render_template(t, {"a": "X", "b": "Y"}) == "X and Y"


def test_render_repeated_var() -> None:
    t = _tpl(content="{{ x }} {{ x }}")
    assert render_template(t, {"x": "A"}) == "A A"


def test_render_whitespace_tolerance() -> None:
    t = _tpl(content="{{x}} {{ x }} {{  x  }}")
    assert render_template(t, {"x": "V"}) == "V V V"


def test_render_strict_raises_on_missing() -> None:
    t = _tpl(content="Hello {{ missing }}!")
    with pytest.raises(TemplateRenderError, match="missing"):
        render_template(t, {})


def test_render_strict_names_template() -> None:
    t = _tpl(name="my-tpl", content="{{ x }}")
    with pytest.raises(TemplateRenderError, match="my-tpl"):
        render_template(t, {})


def test_render_non_strict_uses_fallback() -> None:
    t = _tpl(content="Hello {{ who }}!")
    result = render_template(t, {}, strict=False, missing_value="<BLANK>")
    assert result == "Hello <BLANK>!"


def test_render_non_strict_default_empty() -> None:
    t = _tpl(content="Hello {{ who }}!")
    result = render_template(t, {}, strict=False)
    assert result == "Hello !"


def test_render_no_placeholders() -> None:
    t = _tpl(content="No variables here.")
    assert render_template(t, {}) == "No variables here."


def test_render_none_variables() -> None:
    t = _tpl(content="Static.")
    assert render_template(t, None) == "Static."


def test_render_preserves_non_placeholder_braces() -> None:
    t = _tpl(content="JSON: {key: value}")
    assert render_template(t, {}) == "JSON: {key: value}"


def test_render_ignores_invalid_placeholder_names() -> None:
    t = _tpl(content="{{ 123invalid }}")
    assert render_template(t, {}) == "{{ 123invalid }}"


def test_render_underscore_var() -> None:
    t = _tpl(content="{{ my_var }}")
    assert render_template(t, {"my_var": "OK"}) == "OK"


# ── test: front-matter parsing ────────────────────────────────────────────


def test_load_with_front_matter(tmp_path: Path) -> None:
    _write_template(
        tmp_path,
        "greet",
        "---\nname: greeting\ndescription: Say hello.\n---\n\nHi {{ who }}!\n",
    )
    templates = load_prompt_templates([tmp_path])
    assert len(templates) == 1
    assert templates[0].name == "greeting"
    assert templates[0].description == "Say hello."


def test_load_without_front_matter(tmp_path: Path) -> None:
    _write_template(tmp_path, "simple", "# Title\n\nFirst paragraph.\n")
    templates = load_prompt_templates([tmp_path])
    assert len(templates) == 1
    assert templates[0].name == "simple"
    assert templates[0].description == "First paragraph."


def test_load_name_defaults_to_stem(tmp_path: Path) -> None:
    _write_template(tmp_path, "my-file", "Some content.\n")
    templates = load_prompt_templates([tmp_path])
    assert templates[0].name == "my-file"


def test_load_description_derived_skips_headings(tmp_path: Path) -> None:
    _write_template(tmp_path, "hdg", "# Heading\n## Sub\n\nReal text.\n")
    templates = load_prompt_templates([tmp_path])
    assert templates[0].description == "Real text."


def test_load_quoted_front_matter(tmp_path: Path) -> None:
    _write_template(
        tmp_path,
        "q",
        '---\nname: "quoted"\ndescription: \'single\'\n---\n\nBody.\n',
    )
    templates = load_prompt_templates([tmp_path])
    assert templates[0].name == "quoted"
    assert templates[0].description == "single"


def test_load_body_excludes_front_matter(tmp_path: Path) -> None:
    _write_template(
        tmp_path,
        "fm",
        "---\nname: fm\n---\n\nOnly body.\n",
    )
    templates = load_prompt_templates([tmp_path])
    assert "Only body." in templates[0].content
    assert "---" not in templates[0].content


# ── test: discovery rules ─────────────────────────────────────────────────


def test_ignores_non_md_files(tmp_path: Path) -> None:
    (tmp_path / "script.py").write_text("pass")
    (tmp_path / "data.json").write_text("{}")
    templates = load_prompt_templates([tmp_path])
    assert templates == []


def test_ignores_directories(tmp_path: Path) -> None:
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "file.md").write_text("inside")
    templates = load_prompt_templates([tmp_path])
    assert templates == []


def test_empty_directory(tmp_path: Path) -> None:
    templates, diags = load_prompt_templates_with_diagnostics([tmp_path])
    assert templates == []
    assert diags == []


def test_nonexistent_directory() -> None:
    templates, diags = load_prompt_templates_with_diagnostics(
        [Path("/does/not/exist")]
    )
    assert templates == []
    assert diags == []


# ── test: diagnostics ─────────────────────────────────────────────────────


def test_duplicate_name_in_same_dir(tmp_path: Path) -> None:
    _write_template(tmp_path, "alpha", "---\nname: shared\n---\nA.\n")
    _write_template(tmp_path, "beta", "---\nname: shared\n---\nB.\n")

    templates, diags = load_prompt_templates_with_diagnostics([tmp_path])
    assert len(templates) == 1
    warnings = [d for d in diags if d.severity == Severity.WARNING]
    assert len(warnings) == 1
    assert "Duplicate" in warnings[0].message


def test_unreadable_file_emits_warning(tmp_path: Path) -> None:
    path = tmp_path / "bad.md"
    path.write_bytes(b"\x80\x81\x82")

    templates, diags = load_prompt_templates_with_diagnostics([tmp_path])
    assert templates == []
    assert len(diags) == 1
    assert diags[0].severity == Severity.WARNING
    assert "UTF-8" in diags[0].message


def test_fatal_load_error_direct() -> None:
    from delta_app.prompts import _load_template_file

    with pytest.raises(TemplateLoadError):
        _load_template_file(Path("/nonexistent/template.md"))


# ── test: precedence ──────────────────────────────────────────────────────


def test_higher_precedence_wins(tmp_path: Path) -> None:
    high = tmp_path / "high"
    low = tmp_path / "low"
    _write_template(high, "tpl", "---\nname: tpl\n---\nHigh.\n")
    _write_template(low, "tpl", "---\nname: tpl\n---\nLow.\n")

    templates = load_prompt_templates([high, low])
    assert len(templates) == 1
    assert "High." in templates[0].content


def test_lower_precedence_contributes_unique(tmp_path: Path) -> None:
    high = tmp_path / "high"
    low = tmp_path / "low"
    _write_template(high, "a", "---\nname: a\n---\nA.\n")
    _write_template(low, "b", "---\nname: b\n---\nB.\n")

    templates = load_prompt_templates([high, low])
    names = {t.name for t in templates}
    assert names == {"a", "b"}


def test_three_tier_precedence(tmp_path: Path) -> None:
    p, u, s = tmp_path / "p", tmp_path / "u", tmp_path / "s"
    _write_template(p, "x", "---\nname: x\n---\nProject.\n")
    _write_template(u, "x", "---\nname: x\n---\nUser.\n")
    _write_template(u, "y", "---\nname: y\n---\nUser Y.\n")
    _write_template(s, "x", "---\nname: x\n---\nSystem.\n")
    _write_template(s, "y", "---\nname: y\n---\nSystem Y.\n")
    _write_template(s, "z", "---\nname: z\n---\nSystem Z.\n")

    templates = load_prompt_templates([p, u, s])
    by_name = {t.name: t for t in templates}
    assert len(by_name) == 3
    assert "Project." in by_name["x"].content
    assert "User Y." in by_name["y"].content
    assert "System Z." in by_name["z"].content


# ── test: deterministic ordering ──────────────────────────────────────────


def test_loading_is_deterministic(tmp_path: Path) -> None:
    for name in ["charlie", "alpha", "bravo"]:
        _write_template(tmp_path, name, f"Content of {name}.\n")

    t1 = load_prompt_templates([tmp_path])
    t2 = load_prompt_templates([tmp_path])
    assert [t.name for t in t1] == [t.name for t in t2]


# ── test: slash-command expansion ─────────────────────────────────────────


def test_expand_basic() -> None:
    tpl = _tpl("summarize", "Summarize: {{ arguments }}")
    result = expand_slash_command("/summarize the module", {"summarize": tpl})
    assert result is not None
    assert result == "Summarize: the module"


def test_expand_args_alias() -> None:
    tpl = _tpl("run", "Run: {{ args }}")
    result = expand_slash_command("/run tests", {"run": tpl})
    assert result == "Run: tests"


def test_expand_no_arguments() -> None:
    tpl = _tpl("status", "Show status.")
    result = expand_slash_command("/status", {"status": tpl})
    assert result is not None
    assert result == "Show status."


def test_expand_appends_unreferenced_args() -> None:
    tpl = _tpl("review", "Please review the code.")
    result = expand_slash_command("/review src/main.py", {"review": tpl})
    assert result is not None
    assert "Please review the code." in result
    assert "src/main.py" in result
    assert result == "Please review the code.\n\nsrc/main.py"


def test_expand_no_append_when_referenced() -> None:
    tpl = _tpl("review", "Review {{ arguments }} carefully.")
    result = expand_slash_command("/review src/main.py", {"review": tpl})
    assert result == "Review src/main.py carefully."


def test_expand_no_append_when_args_referenced() -> None:
    tpl = _tpl("review", "Review {{ args }}.")
    result = expand_slash_command("/review file.py", {"review": tpl})
    assert result == "Review file.py."


def test_expand_non_slash_returns_none() -> None:
    assert expand_slash_command("hello", {}) is None


def test_expand_double_slash_returns_none() -> None:
    assert expand_slash_command("// comment", {}) is None


def test_expand_reserved_skill_returns_none() -> None:
    assert expand_slash_command("/skill:refactor", {}) is None


def test_expand_unknown_template_returns_none() -> None:
    result = expand_slash_command("/nonexistent", {})
    assert result is None


def test_expand_empty_slash_returns_none() -> None:
    assert expand_slash_command("/", {}) is None


def test_expand_with_extra_variables() -> None:
    tpl = _tpl("info", "Model: {{ model }}, Args: {{ arguments }}")
    result = expand_slash_command(
        "/info hello",
        {"info": tpl},
        extra_variables={"model": "gpt-4"},
    )
    assert result == "Model: gpt-4, Args: hello"


def test_expand_arguments_override_extra() -> None:
    tpl = _tpl("test", "{{ arguments }}")
    result = expand_slash_command(
        "/test real-args",
        {"test": tpl},
        extra_variables={"arguments": "should-be-overridden"},
    )
    assert result == "real-args"


def test_expand_missing_vars_use_empty() -> None:
    tpl = _tpl("partial", "Val: {{ unknown }}")
    result = expand_slash_command("/partial", {"partial": tpl})
    assert result is not None
    assert result == "Val: "


# ── test: UTF-8 ───────────────────────────────────────────────────────────


def test_utf8_template(tmp_path: Path) -> None:
    _write_template(
        tmp_path,
        "intl",
        "---\nname: intl\ndescription: Ångström résumé\n---\n\n日本語 {{ var }}。\n",
    )
    templates = load_prompt_templates([tmp_path])
    assert templates[0].description == "Ångström résumé"
    result = render_template(templates[0], {"var": "テスト"})
    assert "日本語 テスト。" in result


# ── test: edge cases ──────────────────────────────────────────────────────


def test_template_no_description_derivable(tmp_path: Path) -> None:
    _write_template(tmp_path, "empty", "# Heading\n## Sub\n")
    templates = load_prompt_templates([tmp_path])
    assert templates[0].description is None


def test_render_adjacent_placeholders() -> None:
    t = _tpl(content="{{ a }}{{ b }}")
    assert render_template(t, {"a": "X", "b": "Y"}) == "XY"


def test_render_multiline_template() -> None:
    t = _tpl(content="Line 1: {{ x }}\nLine 2: {{ y }}\n")
    result = render_template(t, {"x": "A", "y": "B"})
    assert result == "Line 1: A\nLine 2: B\n"


def test_expand_multiword_arguments() -> None:
    tpl = _tpl("ask", "Question: {{ arguments }}")
    result = expand_slash_command(
        "/ask what is the meaning of life",
        {"ask": tpl},
    )
    assert result == "Question: what is the meaning of life"


def test_expand_no_append_with_empty_args() -> None:
    tpl = _tpl("plain", "Just text, no placeholders.")
    result = expand_slash_command("/plain", {"plain": tpl})
    assert result == "Just text, no placeholders."
