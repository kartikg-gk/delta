"""Tests for autocomplete.py — completion engine for Delta's CLI/TUI prompt."""

from __future__ import annotations

from pathlib import Path

import pytest

from delta_app.directives import CommandRegistry, SlashCommand, build_default_registry
from delta_app.prompts import PromptTemplate
from delta_app.skillset import Skill
from delta_app.tui.suggest import (
    CompletionItem,
    CompletionOption,
    CompletionProvider,
    CompletionState,
    build_completion_state,
    complete_arguments,
    complete_commands,
    complete_file_refs,
    complete_prompts,
    complete_shell_paths,
    complete_skills,
    get_argument_options,
    is_ignored_dir,
)

# ── fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture()
def registry() -> CommandRegistry:
    return build_default_registry()


@pytest.fixture()
def sample_skills() -> list[Skill]:
    return [
        Skill(name="refactor", description="Refactor code", body="...", source=Path("/s/refactor/SKILL.md")),
        Skill(name="review", description="Code review", body="...", source=Path("/s/review/SKILL.md")),
        Skill(name="test-gen", description="Generate tests", body="...", source=Path("/s/test-gen/SKILL.md")),
    ]


@pytest.fixture()
def sample_templates() -> list[PromptTemplate]:
    return [
        PromptTemplate(name="summarize", path=Path("/p/summarize.md"), content="...", description="Summarize text"),
        PromptTemplate(name="simplify", path=Path("/p/simplify.md"), content="...", description="Simplify code"),
        PromptTemplate(name="explain", path=Path("/p/explain.md"), content="...", description="Explain code"),
    ]


def _handler(ctx: object) -> object:
    from delta_app.directives import CommandResult
    return CommandResult(handled=True)


def _make_registry(*names: str) -> CommandRegistry:
    reg = CommandRegistry()
    for name in names:
        reg.register(SlashCommand(name=name, description=f"{name} cmd", handler=_handler))
    return reg


# ── CompletionItem ───────────────────────────────────────────────────────


class TestCompletionItem:
    def test_frozen(self) -> None:
        item = CompletionItem(display="/help", replacement="/help", start=0, end=2)
        with pytest.raises(AttributeError):
            item.display = "changed"  # type: ignore[misc]

    def test_apply_replaces_token(self) -> None:
        item = CompletionItem(display="/model", replacement="/model", start=0, end=3)
        assert item.apply("/mo") == "/model"

    def test_apply_preserves_surrounding(self) -> None:
        item = CompletionItem(display="bar", replacement="bar", start=4, end=6)
        assert item.apply("foo ba baz") == "foo bar baz"

    def test_apply_at_end(self) -> None:
        item = CompletionItem(display="/help", replacement="/help", start=10, end=12)
        assert item.apply("some text /h") == "some text /help"

    def test_defaults(self) -> None:
        item = CompletionItem(display="x", replacement="x", start=0, end=1)
        assert item.description == ""
        assert item.category == ""


# ── CompletionState ──────────────────────────────────────────────────────


class TestCompletionState:
    def test_empty_state(self) -> None:
        state = CompletionState()
        assert not state
        assert state.selected is None

    def test_bool_true_when_items(self) -> None:
        item = CompletionItem(display="a", replacement="a", start=0, end=1)
        state = CompletionState(items=(item,))
        assert state

    def test_selected_none_initially(self) -> None:
        item = CompletionItem(display="a", replacement="a", start=0, end=1)
        state = CompletionState(items=(item,))
        assert state.selected is None

    def test_select_next(self) -> None:
        items = tuple(
            CompletionItem(display=c, replacement=c, start=0, end=1) for c in "abc"
        )
        state = CompletionState(items=items)
        state.select_next()
        assert state.selected == items[0]
        state.select_next()
        assert state.selected == items[1]
        state.select_next()
        assert state.selected == items[2]

    def test_select_next_wraps(self) -> None:
        items = tuple(
            CompletionItem(display=c, replacement=c, start=0, end=1) for c in "ab"
        )
        state = CompletionState(items=items)
        state.select_next()  # 0
        state.select_next()  # 1
        state.select_next()  # wraps to 0
        assert state.selected == items[0]

    def test_select_previous(self) -> None:
        items = tuple(
            CompletionItem(display=c, replacement=c, start=0, end=1) for c in "abc"
        )
        state = CompletionState(items=items)
        state.select_previous()  # wraps to last
        assert state.selected == items[2]

    def test_select_previous_wraps(self) -> None:
        items = tuple(
            CompletionItem(display=c, replacement=c, start=0, end=1) for c in "ab"
        )
        state = CompletionState(items=items)
        state.select_next()  # index 0
        state.select_previous()  # wraps to last
        assert state.selected == items[1]

    def test_select_next_empty(self) -> None:
        state = CompletionState()
        state.select_next()  # should not raise
        assert state.selected is None

    def test_select_previous_empty(self) -> None:
        state = CompletionState()
        state.select_previous()
        assert state.selected is None


# ── is_ignored_dir ───────────────────────────────────────────────────────


class TestIsIgnoredDir:
    def test_hidden_dirs(self) -> None:
        assert is_ignored_dir(".git")
        assert is_ignored_dir(".hidden")

    def test_known_dirs(self) -> None:
        assert is_ignored_dir("node_modules")
        assert is_ignored_dir("__pycache__")
        assert is_ignored_dir(".venv")

    def test_normal_dirs(self) -> None:
        assert not is_ignored_dir("src")
        assert not is_ignored_dir("tests")

    def test_custom_ignored(self) -> None:
        assert is_ignored_dir("vendor", frozenset({"vendor"}))
        assert not is_ignored_dir("vendor", frozenset({"other"}))


# ── complete_commands ────────────────────────────────────────────────────


class TestCompleteCommands:
    def test_prefix_match(self, registry: CommandRegistry) -> None:
        items = complete_commands("/mo", registry, start=0, end=3)
        names = [i.replacement for i in items]
        assert "/model" in names

    def test_no_match(self, registry: CommandRegistry) -> None:
        items = complete_commands("/zzz", registry, start=0, end=4)
        assert items == []

    def test_all_commands_on_slash(self, registry: CommandRegistry) -> None:
        items = complete_commands("/", registry, start=0, end=1)
        # Should return all commands
        assert len(items) == len(registry.all_commands())

    def test_category_is_command(self, registry: CommandRegistry) -> None:
        items = complete_commands("/he", registry, start=0, end=3)
        for item in items:
            assert item.category == "command"

    def test_alias_match(self) -> None:
        reg = CommandRegistry()
        reg.register(SlashCommand(
            name="model", description="Switch model", handler=_handler,
            aliases=("m",),
        ))
        items = complete_commands("/m", reg, start=0, end=2)
        assert any(i.replacement == "/model" for i in items)

    def test_search_term_match(self) -> None:
        reg = CommandRegistry()
        reg.register(SlashCommand(
            name="quit", description="Exit", handler=_handler,
            search_terms=("leave",),
        ))
        items = complete_commands("/leave", reg, start=0, end=6)
        assert any(i.replacement == "/quit" for i in items)

    def test_not_a_slash(self, registry: CommandRegistry) -> None:
        items = complete_commands("model", registry, start=0, end=5)
        assert items == []

    def test_max_results(self, registry: CommandRegistry) -> None:
        items = complete_commands("/", registry, start=0, end=1, max_results=3)
        assert len(items) <= 3

    def test_description_populated(self, registry: CommandRegistry) -> None:
        items = complete_commands("/hel", registry, start=0, end=4)
        help_items = [i for i in items if i.replacement == "/help"]
        assert help_items
        assert help_items[0].description != ""


# ── complete_arguments ───────────────────────────────────────────────────


class TestCompleteArguments:
    def test_prefix_match(self) -> None:
        opts = [
            CompletionOption("text", "Plain text"),
            CompletionOption("json", "JSON format"),
            CompletionOption("jsonl", "JSON Lines"),
        ]
        items = complete_arguments("export", "js", opts, start=8, end=10)
        values = [i.replacement for i in items]
        assert "json" in values
        assert "jsonl" in values
        assert "text" not in values

    def test_empty_prefix_returns_all(self) -> None:
        opts = [CompletionOption("a", ""), CompletionOption("b", "")]
        items = complete_arguments("cmd", "", opts, start=5, end=5)
        assert len(items) == 2

    def test_category_is_argument(self) -> None:
        opts = [CompletionOption("val", "desc")]
        items = complete_arguments("cmd", "v", opts, start=5, end=6)
        assert items[0].category == "argument"

    def test_case_insensitive(self) -> None:
        opts = [CompletionOption("Dark", "Dark theme")]
        items = complete_arguments("theme", "da", opts, start=7, end=9)
        assert len(items) == 1
        assert items[0].replacement == "Dark"


# ── complete_skills ──────────────────────────────────────────────────────


class TestCompleteSkills:
    def test_prefix_match(self, sample_skills: list[Skill]) -> None:
        items = complete_skills("/skill:re", sample_skills, start=0, end=9)
        names = [i.replacement for i in items]
        assert "/skill:refactor" in names
        assert "/skill:review" in names

    def test_no_match(self, sample_skills: list[Skill]) -> None:
        items = complete_skills("/skill:zzz", sample_skills, start=0, end=10)
        assert items == []

    def test_all_on_empty_fragment(self, sample_skills: list[Skill]) -> None:
        items = complete_skills("/skill:", sample_skills, start=0, end=7)
        assert len(items) == 3

    def test_stops_after_name(self, sample_skills: list[Skill]) -> None:
        # Once arguments begin (space after name), no more skill suggestions
        items = complete_skills("/skill:refactor some args", sample_skills, start=0, end=24)
        assert items == []

    def test_category_is_skill(self, sample_skills: list[Skill]) -> None:
        items = complete_skills("/skill:r", sample_skills, start=0, end=8)
        for item in items:
            assert item.category == "skill"

    def test_not_skill_prefix(self, sample_skills: list[Skill]) -> None:
        items = complete_skills("/model", sample_skills, start=0, end=6)
        assert items == []

    def test_description_populated(self, sample_skills: list[Skill]) -> None:
        items = complete_skills("/skill:refact", sample_skills, start=0, end=13)
        assert items[0].description == "Refactor code"


# ── complete_prompts ─────────────────────────────────────────────────────


class TestCompletePrompts:
    def test_prefix_match(self, sample_templates: list[PromptTemplate]) -> None:
        items = complete_prompts("/sum", sample_templates, start=0, end=4)
        names = [i.replacement for i in items]
        assert "/summarize" in names

    def test_no_match(self, sample_templates: list[PromptTemplate]) -> None:
        items = complete_prompts("/zzz", sample_templates, start=0, end=4)
        assert items == []

    def test_category_is_prompt(self, sample_templates: list[PromptTemplate]) -> None:
        items = complete_prompts("/sim", sample_templates, start=0, end=4)
        for item in items:
            assert item.category == "prompt"

    def test_all_on_slash(self, sample_templates: list[PromptTemplate]) -> None:
        items = complete_prompts("/", sample_templates, start=0, end=1)
        assert len(items) == 3

    def test_description(self, sample_templates: list[PromptTemplate]) -> None:
        items = complete_prompts("/expl", sample_templates, start=0, end=5)
        assert items[0].description == "Explain code"

    def test_not_slash(self, sample_templates: list[PromptTemplate]) -> None:
        items = complete_prompts("explain", sample_templates, start=0, end=7)
        assert items == []


# ── complete_file_refs ───────────────────────────────────────────────────


class TestCompleteFileRefs:
    def test_top_level(self, tmp_path: Path) -> None:
        (tmp_path / "foo.py").write_text("", encoding="utf-8")
        (tmp_path / "bar.py").write_text("", encoding="utf-8")
        items = complete_file_refs("@", str(tmp_path), start=0, end=1)
        repls = [i.replacement for i in items]
        assert "@foo.py" in repls
        assert "@bar.py" in repls

    def test_prefix_filter(self, tmp_path: Path) -> None:
        (tmp_path / "alpha.py").write_text("", encoding="utf-8")
        (tmp_path / "beta.py").write_text("", encoding="utf-8")
        items = complete_file_refs("@al", str(tmp_path), start=0, end=3)
        repls = [i.replacement for i in items]
        assert "@alpha.py" in repls
        assert "@beta.py" not in repls

    def test_directory_listing(self, tmp_path: Path) -> None:
        sub = tmp_path / "src"
        sub.mkdir()
        (sub / "main.py").write_text("", encoding="utf-8")
        items = complete_file_refs("@src/", str(tmp_path), start=0, end=5)
        repls = [i.replacement for i in items]
        assert "@src/main.py" in repls

    def test_ignores_hidden(self, tmp_path: Path) -> None:
        (tmp_path / ".hidden").mkdir()
        (tmp_path / "visible").mkdir()
        items = complete_file_refs("@", str(tmp_path), start=0, end=1)
        repls = [i.replacement for i in items]
        assert not any(".hidden" in r for r in repls)
        assert "@visible/" in repls

    def test_ignores_node_modules(self, tmp_path: Path) -> None:
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "src").mkdir()
        items = complete_file_refs("@", str(tmp_path), start=0, end=1)
        repls = [i.replacement for i in items]
        assert not any("node_modules" in r for r in repls)

    def test_category_is_file(self, tmp_path: Path) -> None:
        (tmp_path / "test.py").write_text("", encoding="utf-8")
        items = complete_file_refs("@t", str(tmp_path), start=0, end=2)
        for item in items:
            assert item.category == "file"

    def test_max_results(self, tmp_path: Path) -> None:
        for i in range(20):
            (tmp_path / f"file{i:02d}.py").write_text("", encoding="utf-8")
        items = complete_file_refs("@", str(tmp_path), start=0, end=1, max_results=5)
        assert len(items) <= 5

    def test_not_file_ref(self, tmp_path: Path) -> None:
        items = complete_file_refs("src/", str(tmp_path), start=0, end=4)
        assert items == []

    def test_file_description(self, tmp_path: Path) -> None:
        (tmp_path / "app.py").write_text("", encoding="utf-8")
        items = complete_file_refs("@app", str(tmp_path), start=0, end=4)
        assert items[0].description == "Python"


# ── complete_shell_paths ─────────────────────────────────────────────────


class TestCompleteShellPaths:
    def test_basic_path(self, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / "tests").mkdir()
        items = complete_shell_paths("! ls s", str(tmp_path))
        repls = [i.replacement for i in items]
        assert "src/" in repls

    def test_double_bang(self, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()
        items = complete_shell_paths("!! git add s", str(tmp_path))
        repls = [i.replacement for i in items]
        assert "src/" in repls

    def test_rejects_absolute(self, tmp_path: Path) -> None:
        items = complete_shell_paths("! ls /etc", str(tmp_path))
        assert items == []

    def test_rejects_home_expansion(self, tmp_path: Path) -> None:
        items = complete_shell_paths("! cat ~/", str(tmp_path))
        assert items == []

    def test_rejects_shell_variable(self, tmp_path: Path) -> None:
        items = complete_shell_paths("! echo $HOME", str(tmp_path))
        assert items == []

    def test_rejects_wildcard(self, tmp_path: Path) -> None:
        items = complete_shell_paths("! ls *.py", str(tmp_path))
        assert items == []

    def test_not_shell_input(self, tmp_path: Path) -> None:
        items = complete_shell_paths("ls src", str(tmp_path))
        assert items == []

    def test_category_is_path(self, tmp_path: Path) -> None:
        (tmp_path / "foo").mkdir()
        items = complete_shell_paths("! ls f", str(tmp_path))
        for item in items:
            assert item.category == "path"

    def test_ignores_hidden(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        (tmp_path / "src").mkdir()
        items = complete_shell_paths("! ls ", str(tmp_path))
        repls = [i.replacement for i in items]
        assert not any(".git" in r for r in repls)

    def test_subdirectory_path(self, tmp_path: Path) -> None:
        sub = tmp_path / "src"
        sub.mkdir()
        (sub / "main.py").write_text("", encoding="utf-8")
        items = complete_shell_paths("! cat src/m", str(tmp_path))
        repls = [i.replacement for i in items]
        assert "src/main.py" in repls


# ── build_completion_state (integration) ─────────────────────────────────


class TestBuildCompletionState:
    def test_empty_input(self, registry: CommandRegistry) -> None:
        state = build_completion_state("", registry=registry)
        assert not state

    def test_slash_command(self, registry: CommandRegistry) -> None:
        state = build_completion_state("/mo", registry=registry)
        assert state
        repls = [i.replacement for i in state.items]
        assert "/model" in repls

    def test_command_argument(self, registry: CommandRegistry) -> None:
        state = build_completion_state("/export js", registry=registry)
        assert state
        repls = [i.replacement for i in state.items]
        assert "json" in repls
        assert "jsonl" in repls

    def test_skill_completion(
        self, registry: CommandRegistry, sample_skills: list[Skill],
    ) -> None:
        state = build_completion_state(
            "/skill:ref", registry=registry, skills=sample_skills,
        )
        assert state
        repls = [i.replacement for i in state.items]
        assert "/skill:refactor" in repls

    def test_template_completion(
        self,
        registry: CommandRegistry,
        sample_templates: list[PromptTemplate],
    ) -> None:
        state = build_completion_state(
            "/summ", registry=registry, templates=sample_templates,
        )
        assert state
        repls = [i.replacement for i in state.items]
        assert "/summarize" in repls

    def test_commands_and_templates_merged(
        self,
        registry: CommandRegistry,
        sample_templates: list[PromptTemplate],
    ) -> None:
        # "/s" should match both commands (stats, session, shell, skills)
        # and templates (summarize, simplify)
        state = build_completion_state(
            "/s", registry=registry, templates=sample_templates,
        )
        assert state
        categories = {i.category for i in state.items}
        assert "command" in categories
        assert "prompt" in categories

    def test_file_ref(self, tmp_path: Path) -> None:
        (tmp_path / "readme.md").write_text("", encoding="utf-8")
        state = build_completion_state("@read", cwd=str(tmp_path))
        assert state
        repls = [i.replacement for i in state.items]
        assert "@readme.md" in repls

    def test_shell_path(self, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()
        state = build_completion_state("! ls s", cwd=str(tmp_path))
        assert state
        repls = [i.replacement for i in state.items]
        assert "src/" in repls

    def test_no_completion_for_normal_text(self, registry: CommandRegistry) -> None:
        state = build_completion_state("hello world", registry=registry)
        assert not state

    def test_custom_argument_options(self, registry: CommandRegistry) -> None:
        custom = {
            "model": [CompletionOption("gpt-4o", "GPT-4o")],
        }
        state = build_completion_state(
            "/model gp", registry=registry, argument_options=custom,
        )
        assert state
        repls = [i.replacement for i in state.items]
        assert "gpt-4o" in repls

    def test_cursor_in_middle(self, registry: CommandRegistry) -> None:
        # Cursor at position 3 in "/mo del" — should complete "/mo"
        state = build_completion_state("/mo del", cursor=3, registry=registry)
        assert state
        repls = [i.replacement for i in state.items]
        assert "/model" in repls

    def test_max_results_respected(self, registry: CommandRegistry) -> None:
        state = build_completion_state("/", registry=registry, max_results=3)
        assert len(state.items) <= 3


# ── get_argument_options ─────────────────────────────────────────────────


class TestGetArgumentOptions:
    def test_builtin_options(self) -> None:
        opts = get_argument_options("export")
        values = [o.value for o in opts]
        assert "text" in values
        assert "json" in values

    def test_unknown_command(self) -> None:
        opts = get_argument_options("nonexistent")
        assert opts == []

    def test_extra_overrides(self) -> None:
        extra = {"model": [CompletionOption("custom", "Custom model")]}
        opts = get_argument_options("model", extra)
        values = [o.value for o in opts]
        assert "custom" in values

    def test_extra_for_unknown(self) -> None:
        extra = {"custom": [CompletionOption("val", "desc")]}
        opts = get_argument_options("custom", extra)
        assert len(opts) == 1


# ── CompletionProvider ───────────────────────────────────────────────────


class TestCompletionProvider:
    def test_base_returns_empty(self) -> None:
        provider = CompletionProvider()
        result = provider.complete("test", 4, start=0, end=4)
        assert result == []

    def test_subclass_can_override(self) -> None:
        class Custom(CompletionProvider):
            def complete(
                self, text: str, cursor: int, *, start: int, end: int,
            ) -> list[CompletionItem]:
                return [CompletionItem(
                    display="custom", replacement="custom",
                    start=start, end=end, category="custom",
                )]

        provider = Custom()
        result = provider.complete("test", 4, start=0, end=4)
        assert len(result) == 1
        assert result[0].category == "custom"


# ── sorting and deduplication ────────────────────────────────────────────


class TestSortingAndDedup:
    def test_exact_prefix_sorted_first(self, registry: CommandRegistry) -> None:
        # /he should have /help before anything matched by search terms
        items = complete_commands("/he", registry, start=0, end=3)
        if items:
            assert items[0].replacement == "/help"

    def test_no_duplicate_replacements(
        self,
        registry: CommandRegistry,
        sample_templates: list[PromptTemplate],
    ) -> None:
        state = build_completion_state(
            "/s", registry=registry, templates=sample_templates,
        )
        repls = [i.replacement for i in state.items]
        assert len(repls) == len(set(repls))


# ── case insensitivity ───────────────────────────────────────────────────


class TestCaseInsensitivity:
    def test_command_case_insensitive(self, registry: CommandRegistry) -> None:
        items = complete_commands("/MO", registry, start=0, end=3)
        assert any(i.replacement == "/model" for i in items)

    def test_skill_case_insensitive(self, sample_skills: list[Skill]) -> None:
        items = complete_skills("/skill:RE", sample_skills, start=0, end=9)
        assert any(i.replacement == "/skill:refactor" for i in items)

    def test_file_ref_case_insensitive(self, tmp_path: Path) -> None:
        (tmp_path / "README.md").write_text("", encoding="utf-8")
        items = complete_file_refs("@read", str(tmp_path), start=0, end=5)
        assert any("README" in i.replacement for i in items)


# ── edge cases ───────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_just_slash(self, registry: CommandRegistry) -> None:
        # A bare "/" has length 1, but no chars after slash → no completion
        state = build_completion_state("/", registry=registry)
        # Actually "/" starts with "/" and len > 0 but token is just "/"
        # which is len 1, so it falls through since len(token) > 1 is False
        assert not state

    def test_just_at(self, tmp_path: Path) -> None:
        (tmp_path / "file.txt").write_text("", encoding="utf-8")
        # "@" alone — len 1, token starts with @ but not len > 1
        state = build_completion_state("@", cwd=str(tmp_path))
        assert not state

    def test_double_slash_comment(self, registry: CommandRegistry) -> None:
        state = build_completion_state("//help", registry=registry)
        # "//help" is treated as a token starting with "/" but it's "//help"
        # which starts with "/" and len > 1, so it will try command completion
        # but "//help" won't match any commands since fragment would be "/help"
        # (after stripping leading /)
        # Actually complete_commands strips the leading / to get the fragment
        # So fragment = "/help" which won't match "help" as a prefix
        # So no completions
        assert not state

    def test_shell_no_token(self, tmp_path: Path) -> None:
        items = complete_shell_paths("! ", str(tmp_path))
        assert items == []

    def test_nonexistent_dir(self) -> None:
        items = complete_file_refs("@src/", "/nonexistent/path", start=0, end=5)
        assert items == []

    def test_apply_preserves_original_casing(self) -> None:
        item = CompletionItem(display="/Model", replacement="/model", start=0, end=3)
        result = item.apply("/Mo")
        assert result == "/model"
