"""Tests for commands.py — slash-command registry, parsing, and dispatch."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from delta_app.directives import (
    CommandContext,
    CommandRegistry,
    CommandResult,
    CommandSession,
    SlashCommand,
    build_default_registry,
    parse_command,
)

# ── fake session ──────────────────────────────────────────────────────────


@dataclass
class FakeSession:
    """Minimal ``CommandSession``-compatible stub for testing."""

    session_id: str = "test-123"
    cwd: str = "/project"
    model: str = "test-model"
    provider_name: str = "test-provider"
    title: str | None = None
    thinking_level: str | None = None
    tools: tuple[object, ...] = ()
    active: bool = False


def _session(**overrides: object) -> FakeSession:
    return FakeSession(**overrides)  # type: ignore[arg-type]


# ── parse_command ─────────────────────────────────────────────────────────


class TestParseCommand:
    def test_basic(self) -> None:
        assert parse_command("/help") == ("help", "")

    def test_with_args(self) -> None:
        assert parse_command("/model claude-sonnet") == ("model", "claude-sonnet")

    def test_multi_word_args(self) -> None:
        assert parse_command("/name My Session") == ("name", "My Session")

    def test_normalizes_to_lowercase(self) -> None:
        assert parse_command("/HELP") == ("help", "")

    def test_strips_arg_whitespace(self) -> None:
        assert parse_command("/model   claude  ") == ("model", "claude")

    def test_not_a_command(self) -> None:
        assert parse_command("hello") is None

    def test_empty_string(self) -> None:
        assert parse_command("") is None

    def test_double_slash_comment(self) -> None:
        assert parse_command("// this is a comment") is None

    def test_slash_only(self) -> None:
        assert parse_command("/") is None

    def test_slash_space(self) -> None:
        assert parse_command("/ hello") is None


# ── CommandResult ─────────────────────────────────────────────────────────


class TestCommandResult:
    def test_defaults(self) -> None:
        r = CommandResult()
        assert not r.handled
        assert r.message == ""
        assert not r.exit_requested
        assert not r.new_session_requested
        assert r.extras == {}

    def test_frozen(self) -> None:
        r = CommandResult(handled=True)
        with pytest.raises(AttributeError):
            r.handled = False  # type: ignore[misc]

    def test_extras(self) -> None:
        r = CommandResult(handled=True, extras={"custom": 42})
        assert r.extras["custom"] == 42


# ── SlashCommand ──────────────────────────────────────────────────────────


class TestSlashCommand:
    def test_display_usage_explicit(self) -> None:
        cmd = SlashCommand(
            name="test", description="A test", handler=lambda c: CommandResult(),
            usage="/test <arg>",
        )
        assert cmd.display_usage == "/test <arg>"

    def test_display_usage_fallback(self) -> None:
        cmd = SlashCommand(
            name="test", description="A test", handler=lambda c: CommandResult(),
        )
        assert cmd.display_usage == "/test"


# ── CommandRegistry: registration ─────────────────────────────────────────


class TestRegistration:
    def _noop(self, ctx: CommandContext) -> CommandResult:
        return CommandResult(handled=True)

    def test_register_and_get(self) -> None:
        reg = CommandRegistry()
        cmd = SlashCommand(name="foo", description="Foo", handler=self._noop)
        reg.register(cmd)
        assert reg.get("foo") is cmd

    def test_get_by_alias(self) -> None:
        reg = CommandRegistry()
        cmd = SlashCommand(
            name="quit", description="Quit", handler=self._noop,
            aliases=("exit", "q"),
        )
        reg.register(cmd)
        assert reg.get("exit") is cmd
        assert reg.get("q") is cmd

    def test_get_missing_returns_none(self) -> None:
        reg = CommandRegistry()
        assert reg.get("nonexistent") is None

    def test_duplicate_name_raises(self) -> None:
        reg = CommandRegistry()
        cmd = SlashCommand(name="foo", description="Foo", handler=self._noop)
        reg.register(cmd)
        with pytest.raises(ValueError, match="Duplicate"):
            reg.register(cmd)

    def test_alias_collision_raises(self) -> None:
        reg = CommandRegistry()
        reg.register(SlashCommand(
            name="foo", description="Foo", handler=self._noop, aliases=("f",),
        ))
        with pytest.raises(ValueError, match="collides"):
            reg.register(SlashCommand(
                name="bar", description="Bar", handler=self._noop, aliases=("f",),
            ))

    def test_alias_collides_with_name(self) -> None:
        reg = CommandRegistry()
        reg.register(SlashCommand(name="foo", description="Foo", handler=self._noop))
        with pytest.raises(ValueError, match="collides"):
            reg.register(SlashCommand(
                name="bar", description="Bar", handler=self._noop, aliases=("foo",),
            ))

    def test_all_commands_sorted(self) -> None:
        reg = CommandRegistry()
        reg.register(SlashCommand(name="zeta", description="Z", handler=self._noop))
        reg.register(SlashCommand(name="alpha", description="A", handler=self._noop))
        result = reg.all_commands()
        assert [c.name for c in result] == ["alpha", "zeta"]

    def test_names_includes_aliases(self) -> None:
        reg = CommandRegistry()
        reg.register(SlashCommand(
            name="quit", description="Quit", handler=self._noop,
            aliases=("exit", "q"),
        ))
        assert "quit" in reg.names
        assert "exit" in reg.names
        assert "q" in reg.names


# ── CommandRegistry: dispatch ─────────────────────────────────────────────


class TestDispatch:
    def test_dispatch_known_command(self) -> None:
        reg = CommandRegistry()
        reg.register(SlashCommand(
            name="ping",
            description="Ping",
            handler=lambda ctx: CommandResult(handled=True, message="pong"),
        ))
        result = reg.dispatch(_session(), "/ping")
        assert result.handled
        assert result.message == "pong"

    def test_dispatch_via_alias(self) -> None:
        reg = CommandRegistry()
        reg.register(SlashCommand(
            name="quit",
            description="Quit",
            handler=lambda ctx: CommandResult(handled=True, exit_requested=True),
            aliases=("q",),
        ))
        result = reg.dispatch(_session(), "/q")
        assert result.handled
        assert result.exit_requested

    def test_dispatch_unknown_command(self) -> None:
        reg = CommandRegistry()
        result = reg.dispatch(_session(), "/nonexistent")
        assert not result.handled

    def test_dispatch_non_command(self) -> None:
        reg = CommandRegistry()
        result = reg.dispatch(_session(), "hello world")
        assert not result.handled

    def test_dispatch_passes_args(self) -> None:
        captured: list[str] = []

        def handler(ctx: CommandContext) -> CommandResult:
            captured.append(ctx.args)
            return CommandResult(handled=True)

        reg = CommandRegistry()
        reg.register(SlashCommand(name="test", description="Test", handler=handler))
        reg.dispatch(_session(), "/test some arguments here")
        assert captured == ["some arguments here"]

    def test_dispatch_passes_session(self) -> None:
        captured: list[object] = []

        def handler(ctx: CommandContext) -> CommandResult:
            captured.append(ctx.session)
            return CommandResult(handled=True)

        reg = CommandRegistry()
        reg.register(SlashCommand(name="test", description="Test", handler=handler))
        sess = _session(model="custom-model")
        reg.dispatch(sess, "/test")
        assert captured[0] is sess

    def test_context_contains_registry(self) -> None:
        captured: list[CommandRegistry] = []

        def handler(ctx: CommandContext) -> CommandResult:
            captured.append(ctx.registry)
            return CommandResult(handled=True)

        reg = CommandRegistry()
        reg.register(SlashCommand(name="test", description="Test", handler=handler))
        reg.dispatch(_session(), "/test")
        assert captured[0] is reg

    def test_context_contains_raw_and_name(self) -> None:
        captured: list[CommandContext] = []

        def handler(ctx: CommandContext) -> CommandResult:
            captured.append(ctx)
            return CommandResult(handled=True)

        reg = CommandRegistry()
        reg.register(SlashCommand(name="test", description="Test", handler=handler))
        reg.dispatch(_session(), "/test arg1")
        assert captured[0].raw == "/test arg1"
        assert captured[0].name == "test"
        assert captured[0].args == "arg1"


# ── FakeSession satisfies protocol ────────────────────────────────────────


class TestProtocol:
    def test_fake_session_is_command_session(self) -> None:
        assert isinstance(_session(), CommandSession)


# ── Default registry: built-in commands ──────────────────────────────────


class TestDefaultRegistry:
    @pytest.fixture()
    def reg(self) -> CommandRegistry:
        return build_default_registry()

    @pytest.fixture()
    def sess(self) -> FakeSession:
        return _session()

    def test_all_expected_commands_registered(self, reg: CommandRegistry) -> None:
        expected = {
            "help", "quit", "new", "session", "model", "provider", "tools",
            "export", "reload", "resume", "think", "compact", "stats", "name",
            "branch", "rewind", "shell", "diag", "skills", "prompts", "theme",
            "login", "logout", "config", "doctor",
        }
        registered = {c.name for c in reg.all_commands()}
        assert expected <= registered

    def test_aliases_registered(self, reg: CommandRegistry) -> None:
        assert reg.get("q") is not None
        assert reg.get("exit") is not None
        assert reg.get("h") is not None
        assert reg.get("?") is not None
        assert reg.get("m") is not None
        assert reg.get("r") is not None
        assert reg.get("thinking") is not None
        assert reg.get("title") is not None
        assert reg.get("fork") is not None
        assert reg.get("sh") is not None
        assert reg.get("!") is not None
        assert reg.get("debug") is not None
        assert reg.get("templates") is not None
        assert reg.get("settings") is not None


# ── Default registry: handler behaviors ───────────────────────────────────


class TestBuiltinHandlers:
    @pytest.fixture()
    def reg(self) -> CommandRegistry:
        return build_default_registry()

    @pytest.fixture()
    def sess(self) -> FakeSession:
        return _session()

    # /help
    def test_help(self, reg: CommandRegistry, sess: FakeSession) -> None:
        result = reg.dispatch(sess, "/help")
        assert result.handled
        assert "Commands:" in result.message
        assert "/help" in result.message

    # /quit, /exit, /q
    def test_quit(self, reg: CommandRegistry, sess: FakeSession) -> None:
        for cmd in ["/quit", "/exit", "/q"]:
            result = reg.dispatch(sess, cmd)
            assert result.handled
            assert result.exit_requested

    # /new
    def test_new(self, reg: CommandRegistry, sess: FakeSession) -> None:
        result = reg.dispatch(sess, "/new")
        assert result.handled
        assert result.new_session_requested

    # /model (no args)
    def test_model_show(self, reg: CommandRegistry, sess: FakeSession) -> None:
        result = reg.dispatch(sess, "/model")
        assert result.handled
        assert "test-model" in result.message
        assert result.model_picker_requested

    # /model with arg
    def test_model_switch(self, reg: CommandRegistry, sess: FakeSession) -> None:
        result = reg.dispatch(sess, "/model gpt-4o")
        assert result.handled
        assert result.model_value == "gpt-4o"
        assert result.model_picker_requested

    # /model via alias
    def test_model_alias(self, reg: CommandRegistry, sess: FakeSession) -> None:
        result = reg.dispatch(sess, "/m gpt-4o")
        assert result.handled
        assert result.model_value == "gpt-4o"

    # /provider (no args)
    def test_provider_show(self, reg: CommandRegistry, sess: FakeSession) -> None:
        result = reg.dispatch(sess, "/provider")
        assert result.handled
        assert "test-provider" in result.message

    # /provider with arg
    def test_provider_switch(self, reg: CommandRegistry, sess: FakeSession) -> None:
        result = reg.dispatch(sess, "/provider openai")
        assert result.handled
        assert result.provider_switch_requested
        assert result.provider_value == "openai"

    # /tools
    def test_tools_empty(self, reg: CommandRegistry, sess: FakeSession) -> None:
        result = reg.dispatch(sess, "/tools")
        assert result.handled
        assert "No tools" in result.message

    def test_tools_with_tools(self, reg: CommandRegistry) -> None:
        from types import SimpleNamespace
        sess = _session(tools=(SimpleNamespace(name="bash"), SimpleNamespace(name="read")))
        result = reg.dispatch(sess, "/tools")
        assert result.handled
        assert "bash" in result.message
        assert "read" in result.message

    # /export
    def test_export_default(self, reg: CommandRegistry, sess: FakeSession) -> None:
        result = reg.dispatch(sess, "/export")
        assert result.handled
        assert result.export_requested
        assert result.export_format == "text"

    def test_export_json(self, reg: CommandRegistry, sess: FakeSession) -> None:
        result = reg.dispatch(sess, "/export json")
        assert result.export_format == "json"

    def test_export_invalid_format(self, reg: CommandRegistry, sess: FakeSession) -> None:
        result = reg.dispatch(sess, "/export xml")
        assert result.handled
        assert "Unknown format" in result.message
        assert not result.export_requested

    # /reload
    def test_reload(self, reg: CommandRegistry, sess: FakeSession) -> None:
        result = reg.dispatch(sess, "/reload")
        assert result.handled
        assert result.reload_requested

    # /resume
    def test_resume_no_args(self, reg: CommandRegistry, sess: FakeSession) -> None:
        result = reg.dispatch(sess, "/resume")
        assert result.handled
        assert "Usage" in result.message
        assert not result.resume_requested

    def test_resume_with_id(self, reg: CommandRegistry, sess: FakeSession) -> None:
        result = reg.dispatch(sess, "/resume abc123")
        assert result.handled
        assert result.resume_requested
        assert result.resume_session_id == "abc123"

    # /session
    def test_session_info(self, reg: CommandRegistry) -> None:
        sess = _session(title="My Project")
        result = reg.dispatch(sess, "/session")
        assert result.handled
        assert "test-123" in result.message
        assert "My Project" in result.message

    # /think
    def test_think_off(self, reg: CommandRegistry, sess: FakeSession) -> None:
        result = reg.dispatch(sess, "/think off")
        assert result.handled
        assert result.thinking_requested
        assert result.thinking_value is None

    def test_think_level(self, reg: CommandRegistry, sess: FakeSession) -> None:
        result = reg.dispatch(sess, "/think high")
        assert result.handled
        assert result.thinking_requested
        assert result.thinking_value == "high"

    def test_think_no_args(self, reg: CommandRegistry, sess: FakeSession) -> None:
        result = reg.dispatch(sess, "/think")
        assert result.thinking_requested
        assert result.thinking_value is None

    def test_thinking_alias(self, reg: CommandRegistry, sess: FakeSession) -> None:
        result = reg.dispatch(sess, "/thinking high")
        assert result.thinking_value == "high"

    # /compact
    def test_compact(self, reg: CommandRegistry, sess: FakeSession) -> None:
        result = reg.dispatch(sess, "/compact")
        assert result.handled
        assert result.compact_requested

    # /stats
    def test_stats(self, reg: CommandRegistry, sess: FakeSession) -> None:
        result = reg.dispatch(sess, "/stats")
        assert result.handled
        assert result.extras.get("stats_requested")

    # /name
    def test_name_show(self, reg: CommandRegistry) -> None:
        sess = _session(title="Existing Title")
        result = reg.dispatch(sess, "/name")
        assert result.handled
        assert "Existing Title" in result.message
        assert not result.name_requested

    def test_name_set(self, reg: CommandRegistry, sess: FakeSession) -> None:
        result = reg.dispatch(sess, "/name New Title")
        assert result.handled
        assert result.name_requested
        assert result.name_value == "New Title"

    def test_title_alias(self, reg: CommandRegistry, sess: FakeSession) -> None:
        result = reg.dispatch(sess, "/title My Title")
        assert result.name_requested
        assert result.name_value == "My Title"

    # /branch
    def test_branch(self, reg: CommandRegistry, sess: FakeSession) -> None:
        result = reg.dispatch(sess, "/branch some summary")
        assert result.handled
        assert result.branch_requested
        assert result.branch_summary == "some summary"

    def test_branch_no_summary(self, reg: CommandRegistry, sess: FakeSession) -> None:
        result = reg.dispatch(sess, "/branch")
        assert result.branch_requested
        assert result.branch_summary == ""

    # /rewind
    def test_rewind_no_args(self, reg: CommandRegistry, sess: FakeSession) -> None:
        # No argument means "show me the choices", not a usage error.
        result = reg.dispatch(sess, "/rewind")
        assert result.handled
        assert result.rewind_requested
        assert result.rewind_entry_id == ""

    def test_rewind_with_id(self, reg: CommandRegistry, sess: FakeSession) -> None:
        result = reg.dispatch(sess, "/rewind abc123")
        assert result.rewind_requested
        assert result.rewind_entry_id == "abc123"

    # /shell
    def test_shell_no_args(self, reg: CommandRegistry, sess: FakeSession) -> None:
        result = reg.dispatch(sess, "/shell")
        assert result.handled
        assert "Usage" in result.message
        assert not result.shell_requested

    def test_shell_with_command(self, reg: CommandRegistry, sess: FakeSession) -> None:
        result = reg.dispatch(sess, "/shell echo hello")
        assert result.shell_requested
        assert result.shell_command == "echo hello"

    def test_shell_alias_sh(self, reg: CommandRegistry, sess: FakeSession) -> None:
        result = reg.dispatch(sess, "/sh ls -la")
        assert result.shell_requested
        assert result.shell_command == "ls -la"

    def test_shell_alias_bang(self, reg: CommandRegistry, sess: FakeSession) -> None:
        result = reg.dispatch(sess, "/! pwd")
        assert result.shell_requested
        assert result.shell_command == "pwd"

    # /diag
    def test_diag(self, reg: CommandRegistry, sess: FakeSession) -> None:
        result = reg.dispatch(sess, "/diag")
        assert result.handled
        assert result.diag_requested

    def test_debug_alias(self, reg: CommandRegistry, sess: FakeSession) -> None:
        result = reg.dispatch(sess, "/debug")
        assert result.diag_requested

    # /skills
    def test_skills(self, reg: CommandRegistry, sess: FakeSession) -> None:
        result = reg.dispatch(sess, "/skills")
        assert result.handled
        assert result.extras.get("skills_requested")

    # /prompts
    def test_prompts(self, reg: CommandRegistry, sess: FakeSession) -> None:
        result = reg.dispatch(sess, "/prompts")
        assert result.handled
        assert result.extras.get("prompts_requested")

    # /theme
    def test_theme_show(self, reg: CommandRegistry, sess: FakeSession) -> None:
        result = reg.dispatch(sess, "/theme")
        assert result.handled
        assert result.extras.get("theme_requested")

    def test_theme_set(self, reg: CommandRegistry, sess: FakeSession) -> None:
        result = reg.dispatch(sess, "/theme dark")
        assert result.extras.get("theme_value") == "dark"

    # /login
    def test_login(self, reg: CommandRegistry, sess: FakeSession) -> None:
        result = reg.dispatch(sess, "/login anthropic")
        assert result.handled
        assert result.extras.get("login_requested")
        assert result.extras.get("login_provider") == "anthropic"

    # /logout
    def test_logout(self, reg: CommandRegistry, sess: FakeSession) -> None:
        result = reg.dispatch(sess, "/logout")
        assert result.handled
        assert result.extras.get("logout_requested")

    # /config
    def test_config_show(self, reg: CommandRegistry, sess: FakeSession) -> None:
        result = reg.dispatch(sess, "/config")
        assert result.handled
        assert result.extras.get("config_requested")

    def test_config_set(self, reg: CommandRegistry, sess: FakeSession) -> None:
        result = reg.dispatch(sess, "/config key=value")
        assert result.extras.get("config_args") == "key=value"

    # /doctor
    def test_doctor(self, reg: CommandRegistry, sess: FakeSession) -> None:
        result = reg.dispatch(sess, "/doctor")
        assert result.handled
        assert result.extras.get("doctor_requested")


# ── Edge cases ────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_case_insensitive_dispatch(self) -> None:
        reg = build_default_registry()
        result = reg.dispatch(_session(), "/HELP")
        assert result.handled

    def test_normal_text_is_not_handled(self) -> None:
        reg = build_default_registry()
        result = reg.dispatch(_session(), "hello world")
        assert not result.handled

    def test_empty_string_not_handled(self) -> None:
        reg = build_default_registry()
        result = reg.dispatch(_session(), "")
        assert not result.handled

    def test_comment_not_handled(self) -> None:
        reg = build_default_registry()
        result = reg.dispatch(_session(), "// comment")
        assert not result.handled

    def test_unknown_command_not_handled(self) -> None:
        reg = build_default_registry()
        result = reg.dispatch(_session(), "/xyznonexistent")
        assert not result.handled


# ── Custom command registration ───────────────────────────────────────────


class TestCustomRegistration:
    def test_extend_default_registry(self) -> None:
        reg = build_default_registry()
        reg.register(SlashCommand(
            name="custom",
            description="A custom command",
            handler=lambda ctx: CommandResult(
                handled=True, message="custom!", extras={"custom": True},
            ),
        ))
        result = reg.dispatch(_session(), "/custom")
        assert result.handled
        assert result.message == "custom!"
        assert result.extras["custom"] is True

    def test_custom_command_in_help(self) -> None:
        reg = build_default_registry()
        reg.register(SlashCommand(
            name="custom",
            description="My custom command",
            handler=lambda ctx: CommandResult(handled=True),
        ))
        result = reg.dispatch(_session(), "/help")
        assert "custom" in result.message
        assert "My custom command" in result.message
