"""Slash-command registry: parsing, dispatch, and structured result actions.

This module is the thin command-routing layer between the CLI/TUI and the
rest of the Delta architecture.  It parses user intent from ``/command args``
input and returns structured ``CommandResult`` objects describing what
the application should do — it never executes business logic itself.

Responsibilities:

* Register slash commands with metadata (name, description, aliases).
* Parse ``/command args`` input into a ``CommandContext``.
* Dispatch to the matching handler.
* Return a ``CommandResult`` the caller can act on.

Non-responsibilities (belong to the surrounding application):

* Executing provider operations, file edits, exports, or UI actions.
* Session creation, model switching, or any async work.
* Directly manipulating the conversation transcript.

Usage::

    registry = build_default_registry()
    result = registry.dispatch(session, "/model claude-sonnet-4-20250514")
    if result.handled:
        ...  # act on result flags
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Session protocol — the only view commands need of a live session
# ---------------------------------------------------------------------------


@runtime_checkable
class CommandSession(Protocol):
    """Read-only view of session state consumed by command handlers.

    Defined as a ``Protocol`` so ``commands.py`` never imports the
    concrete ``CodingSession`` class, avoiding circular dependencies.
    """

    @property
    def session_id(self) -> str: ...

    @property
    def cwd(self) -> str: ...

    @property
    def model(self) -> str: ...

    @property
    def provider_name(self) -> str: ...

    @property
    def title(self) -> str | None: ...

    @property
    def thinking_level(self) -> str | None: ...

    @property
    def tools(self) -> tuple[object, ...]: ...

    @property
    def active(self) -> bool: ...


# ---------------------------------------------------------------------------
# CommandResult — structured action descriptor
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Describes what the application should do in response to a slash command.

    Handlers return this instead of performing side effects.  The caller
    inspects the flags and executes the corresponding business logic.

    The ``extras`` dict provides an escape hatch for future commands that
    need to communicate additional data without adding new fields.
    """

    handled: bool = False
    message: str = ""

    # ── lifecycle flags ───────────────────────────────────────────────
    exit_requested: bool = False
    new_session_requested: bool = False
    reload_requested: bool = False

    # ── navigation flags ──────────────────────────────────────────────
    resume_requested: bool = False
    resume_session_id: str = ""

    # ── picker / interactive flags ────────────────────────────────────
    model_picker_requested: bool = False
    model_value: str = ""
    tools_picker_requested: bool = False
    export_requested: bool = False
    export_format: str = ""

    # ── provider switching ────────────────────────────────────────────
    provider_switch_requested: bool = False
    provider_value: str = ""

    # ── thinking mode ─────────────────────────────────────────────────
    thinking_requested: bool = False
    thinking_value: str | None = None

    # ── session metadata ──────────────────────────────────────────────
    name_requested: bool = False
    name_value: str = ""

    # ── plan mode ─────────────────────────────────────────────────────
    plan_requested: bool = False
    plan_value: str = ""

    # ── compaction ────────────────────────────────────────────────────
    compact_requested: bool = False

    # ── branching / rewind ────────────────────────────────────────────
    branch_requested: bool = False
    branch_summary: str = ""
    rewind_requested: bool = False
    rewind_entry_id: str = ""

    # ── shell ─────────────────────────────────────────────────────────
    shell_requested: bool = False
    shell_command: str = ""

    # ── diagnostics ───────────────────────────────────────────────────
    diag_requested: bool = False

    # ── extensibility ─────────────────────────────────────────────────
    extras: dict[str, object] = field(default_factory=dict)


# Convenience shortcuts
_HANDLED = CommandResult(handled=True)
_NOT_HANDLED = CommandResult()


def _ok(message: str = "", **kwargs: object) -> CommandResult:
    """Build a handled result with an optional message and extra flags."""
    return CommandResult(handled=True, message=message, **kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# CommandContext — everything a handler receives
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CommandContext:
    """Bundle passed to every command handler.

    Provides access to the session, the registry, and the parsed input
    so handlers don't need many individual parameters.
    """

    session: CommandSession
    registry: CommandRegistry
    raw: str
    name: str
    args: str


# ---------------------------------------------------------------------------
# SlashCommand — metadata for a single registered command
# ---------------------------------------------------------------------------

#: Type alias for command handler functions.
Handler = Callable[[CommandContext], CommandResult]


@dataclass(frozen=True, slots=True)
class SlashCommand:
    """Metadata and handler for a single slash command."""

    name: str
    description: str
    handler: Handler
    usage: str = ""
    aliases: tuple[str, ...] = ()
    search_terms: tuple[str, ...] = ()

    @property
    def display_usage(self) -> str:
        """Human-readable usage string, falling back to ``/<name>``."""
        return self.usage or f"/{self.name}"


# ---------------------------------------------------------------------------
# Parsing utilities
# ---------------------------------------------------------------------------


def parse_command(text: str) -> tuple[str, str] | None:
    """Parse ``/command args`` into ``(name, args)`` or ``None``.

    Returns ``None`` when *text* is not a slash command (no leading ``/``,
    empty after the slash, or a ``//`` comment).
    """
    if not text.startswith("/"):
        return None
    if text.startswith("//"):
        return None
    body = text[1:]
    if not body or body[0].isspace():
        return None
    parts = body.split(None, 1)
    name = parts[0].lower()
    args = parts[1].strip() if len(parts) > 1 else ""
    return name, args


# ---------------------------------------------------------------------------
# CommandRegistry
# ---------------------------------------------------------------------------


class CommandRegistry:
    """Dictionary-based slash command registry with alias support.

    Commands are registered once and looked up by primary name or alias.
    Dispatch returns a ``CommandResult`` — the registry never executes
    business logic.
    """

    def __init__(self) -> None:
        self._commands: dict[str, SlashCommand] = {}
        self._aliases: dict[str, str] = {}

    # ── registration ──────────────────────────────────────────────────

    def register(self, command: SlashCommand) -> None:
        """Add a command to the registry.

        Raises ``ValueError`` if the name or any alias collides with an
        existing registration.
        """
        if command.name in self._commands or command.name in self._aliases:
            raise ValueError(f"Duplicate command name: /{command.name}")
        for alias in command.aliases:
            if alias in self._commands or alias in self._aliases:
                raise ValueError(
                    f"Alias /{alias} for /{command.name} collides with "
                    f"an existing command or alias"
                )
        self._commands[command.name] = command
        for alias in command.aliases:
            self._aliases[alias] = command.name

    # ── lookup ────────────────────────────────────────────────────────

    def get(self, name: str) -> SlashCommand | None:
        """Look up a command by primary name or alias."""
        if name in self._commands:
            return self._commands[name]
        primary = self._aliases.get(name)
        if primary is not None:
            return self._commands.get(primary)
        return None

    def all_commands(self) -> list[SlashCommand]:
        """Return all registered commands in alphabetical order."""
        return sorted(self._commands.values(), key=lambda c: c.name)

    @property
    def names(self) -> frozenset[str]:
        """All registered primary names and aliases."""
        return frozenset(self._commands) | frozenset(self._aliases)

    # ── dispatch ──────────────────────────────────────────────────────

    def dispatch(self, session: CommandSession, text: str) -> CommandResult:
        """Parse *text* and dispatch to the matching handler.

        Returns an unhandled ``CommandResult`` when *text* is not a slash
        command or when no matching command is registered.
        """
        parsed = parse_command(text)
        if parsed is None:
            return _NOT_HANDLED

        name, args = parsed
        command = self.get(name)
        if command is None:
            return _NOT_HANDLED

        ctx = CommandContext(
            session=session,
            registry=self,
            raw=text,
            name=name,
            args=args,
        )
        return command.handler(ctx)


# ---------------------------------------------------------------------------
# Built-in command handlers
# ---------------------------------------------------------------------------


def _cmd_help(ctx: CommandContext) -> CommandResult:
    """List all available commands with descriptions."""
    lines = ["Commands:"]
    for cmd in ctx.registry.all_commands():
        alias_str = ""
        if cmd.aliases:
            alias_str = " (" + " ".join(f"/{a}" for a in cmd.aliases) + ")"
        lines.append(f"  {cmd.display_usage:<22}{cmd.description}{alias_str}")
    return _ok("\n".join(lines))


def _cmd_quit(ctx: CommandContext) -> CommandResult:
    """Signal that the user wants to exit."""
    return _ok(exit_requested=True)


def _cmd_version(ctx: CommandContext) -> CommandResult:
    """Show the installed version."""
    return _ok(extras={"version_requested": True})


def _cmd_new(ctx: CommandContext) -> CommandResult:
    """Request creation of a fresh session."""
    return _ok("Starting new session.", new_session_requested=True)


def _cmd_model(ctx: CommandContext) -> CommandResult:
    """Show or switch the active model."""
    if not ctx.args:
        return _ok(f"Current model: {ctx.session.model}", model_picker_requested=True)
    return _ok(
        f"Switching to model: {ctx.args}",
        model_picker_requested=True,
        model_value=ctx.args,
    )


def _cmd_provider(ctx: CommandContext) -> CommandResult:
    """Show or switch the active provider."""
    if not ctx.args:
        return _ok(f"Current provider: {ctx.session.provider_name}")
    return _ok(
        f"Switching to provider: {ctx.args}",
        provider_switch_requested=True,
        provider_value=ctx.args,
    )


def _cmd_tools(ctx: CommandContext) -> CommandResult:
    """List registered tools or open the tools picker."""
    tools = ctx.session.tools
    if not tools:
        return _ok("No tools registered.")
    lines = [f"Registered tools ({len(tools)}):"]
    for tool in tools:
        name = getattr(tool, "name", str(tool))
        lines.append(f"  {name}")
    return _ok("\n".join(lines), tools_picker_requested=True)


def _cmd_export(ctx: CommandContext) -> CommandResult:
    """Request a transcript export."""
    fmt = ctx.args or "text"
    if fmt not in {"text", "json", "jsonl"}:
        return _ok(f"Unknown format: {fmt!r}. Use text, json, or jsonl.")
    return _ok(export_requested=True, export_format=fmt)


def _cmd_reload(ctx: CommandContext) -> CommandResult:
    """Request a hot reload of tools, extensions, and system prompt."""
    return _ok("Reloading.", reload_requested=True)


def _cmd_resume(ctx: CommandContext) -> CommandResult:
    """Resume a previous session by ID."""
    if not ctx.args:
        return _ok("Usage: /resume <session-id>")
    return _ok(
        f"Resuming session: {ctx.args}",
        resume_requested=True,
        resume_session_id=ctx.args,
    )


def _cmd_session(ctx: CommandContext) -> CommandResult:
    """Show current session info."""
    title = ctx.session.title or "(untitled)"
    lines = [
        f"Session: {ctx.session.session_id}",
        f"Title:   {title}",
        f"Model:   {ctx.session.model}",
        f"Provider:{ctx.session.provider_name}",
        f"CWD:     {ctx.session.cwd}",
    ]
    return _ok("\n".join(lines))


def _cmd_think(ctx: CommandContext) -> CommandResult:
    """Set or disable thinking mode."""
    if not ctx.args or ctx.args.lower() in {"off", "none", "disable"}:
        return _ok("Thinking disabled.", thinking_requested=True, thinking_value=None)
    level = ctx.args.split()[0]
    return _ok(
        f"Thinking set to: {level}",
        thinking_requested=True,
        thinking_value=level,
    )


def _cmd_plan(ctx: CommandContext) -> CommandResult:
    """Toggle read-only plan mode, or set it with ``on``/``off``."""
    return _ok(plan_requested=True, plan_value=ctx.args.strip().lower())


def _cmd_compact(ctx: CommandContext) -> CommandResult:
    """Request context compaction."""
    return _ok(compact_requested=True)


def _cmd_stats(ctx: CommandContext) -> CommandResult:
    """Display session statistics (delegated to caller for actual data)."""
    return _ok(extras={"stats_requested": True})


def _cmd_name(ctx: CommandContext) -> CommandResult:
    """Show or set the session title."""
    if not ctx.args:
        title = ctx.session.title or ctx.session.session_id
        return _ok(f"Session: {title}")
    return _ok(
        f"Session named: {ctx.args}",
        name_requested=True,
        name_value=ctx.args,
    )


def _cmd_branch(ctx: CommandContext) -> CommandResult:
    """Fork the conversation at the current point."""
    return _ok(branch_requested=True, branch_summary=ctx.args)


def _cmd_rewind(ctx: CommandContext) -> CommandResult:
    """Rewind to a previous entry, or list the branchable ones."""
    return _ok(rewind_requested=True, rewind_entry_id=ctx.args)


def _cmd_branches(ctx: CommandContext) -> CommandResult:
    """List recorded fork points."""
    return _ok(extras={"branches_requested": True})


def _cmd_create_skill(ctx: CommandContext) -> CommandResult:
    """Save a new project skill."""
    if not ctx.args:
        return _ok("Usage: /create-skill <name> <instructions>")
    return _ok(extras={"create_skill_requested": True, "create_skill_args": ctx.args})


def _cmd_continue(ctx: CommandContext) -> CommandResult:
    """Resume the agent loop without a new user message."""
    return _ok(extras={"continue_requested": True})


def _cmd_shell(ctx: CommandContext) -> CommandResult:
    """Request execution of a shell command."""
    if not ctx.args:
        return _ok("Usage: /shell <command>")
    return _ok(shell_requested=True, shell_command=ctx.args)


def _cmd_diag(ctx: CommandContext) -> CommandResult:
    """Request session diagnostics dump."""
    return _ok(diag_requested=True)


def _cmd_system(ctx: CommandContext) -> CommandResult:
    """Show the active system prompt with each section's source."""
    return _ok(extras={"system_requested": True})


def _cmd_skills(ctx: CommandContext) -> CommandResult:
    """List available skills."""
    return _ok(extras={"skills_requested": True})


def _cmd_prompts(ctx: CommandContext) -> CommandResult:
    """List available prompt templates."""
    return _ok(extras={"prompts_requested": True})


def _cmd_theme(ctx: CommandContext) -> CommandResult:
    """Show or switch theme."""
    if not ctx.args:
        return _ok(extras={"theme_requested": True})
    return _ok(extras={"theme_requested": True, "theme_value": ctx.args})


def _cmd_login(ctx: CommandContext) -> CommandResult:
    """Request provider authentication."""
    provider = ctx.args or ""
    return _ok(extras={"login_requested": True, "login_provider": provider})


def _cmd_logout(ctx: CommandContext) -> CommandResult:
    """Request provider deauthentication."""
    provider = ctx.args or ""
    return _ok(extras={"logout_requested": True, "logout_provider": provider})


def _cmd_config(ctx: CommandContext) -> CommandResult:
    """Show or edit configuration."""
    if not ctx.args:
        return _ok(extras={"config_requested": True})
    return _ok(extras={"config_requested": True, "config_args": ctx.args})


def _cmd_doctor(ctx: CommandContext) -> CommandResult:
    """Run environment health checks."""
    return _ok(extras={"doctor_requested": True})


# ---------------------------------------------------------------------------
# Default registry builder
# ---------------------------------------------------------------------------


def build_default_registry() -> CommandRegistry:
    """Create a ``CommandRegistry`` populated with Delta's built-in commands."""
    reg = CommandRegistry()

    _BUILTINS: list[SlashCommand] = [
        SlashCommand(
            name="help",
            description="Show available commands",
            handler=_cmd_help,
            usage="/help",
            aliases=("?", "h"),
            search_terms=("commands", "usage"),
        ),
        SlashCommand(
            name="quit",
            description="End the session",
            handler=_cmd_quit,
            usage="/quit",
            aliases=("exit", "q"),
            search_terms=("leave", "close", "stop"),
        ),
        SlashCommand(
            name="version",
            description="Show version info",
            handler=_cmd_version,
            usage="/version",
            search_terms=("build", "release"),
        ),
        SlashCommand(
            name="new",
            description="Start a new session",
            handler=_cmd_new,
            usage="/new",
            search_terms=("fresh", "reset", "clear"),
        ),
        SlashCommand(
            name="session",
            description="Show current session info",
            handler=_cmd_session,
            usage="/session",
            search_terms=("info", "status"),
        ),
        SlashCommand(
            name="model",
            description="Show or switch model",
            handler=_cmd_model,
            usage="/model [name]",
            aliases=("m",),
            search_terms=("switch", "llm"),
        ),
        SlashCommand(
            name="provider",
            description="Show or switch provider",
            handler=_cmd_provider,
            usage="/provider [name]",
            search_terms=("backend", "api"),
        ),
        SlashCommand(
            name="tools",
            description="List registered tools",
            handler=_cmd_tools,
            usage="/tools",
            search_terms=("functions", "capabilities"),
        ),
        SlashCommand(
            name="export",
            description="Export transcript",
            handler=_cmd_export,
            usage="/export [text|json|jsonl]",
            search_terms=("save", "download", "dump"),
        ),
        SlashCommand(
            name="reload",
            description="Reload tools, extensions, and prompt",
            handler=_cmd_reload,
            usage="/reload",
            search_terms=("refresh",),
        ),
        SlashCommand(
            name="resume",
            description="Resume a previous session",
            handler=_cmd_resume,
            usage="/resume <session-id>",
            aliases=("r",),
            search_terms=("restore", "continue", "load"),
        ),
        SlashCommand(
            name="think",
            description="Set thinking level (off to disable)",
            handler=_cmd_think,
            usage="/think [level|off]",
            aliases=("thinking",),
            search_terms=("reasoning", "depth"),
        ),
        SlashCommand(
            name="plan",
            description="Toggle read-only plan mode",
            handler=_cmd_plan,
            usage="/plan [on|off]",
            search_terms=("planning", "readonly", "research", "propose"),
        ),
        SlashCommand(
            name="compact",
            description="Summarise older context",
            handler=_cmd_compact,
            usage="/compact",
            search_terms=("compress", "summarize", "prune"),
        ),
        SlashCommand(
            name="stats",
            description="Show token/turn/cost statistics",
            handler=_cmd_stats,
            usage="/stats",
            search_terms=("tokens", "cost", "usage"),
        ),
        SlashCommand(
            name="name",
            description="Show or set session title",
            handler=_cmd_name,
            usage="/name [title]",
            aliases=("title",),
            search_terms=("rename", "label"),
        ),
        SlashCommand(
            name="branch",
            description="Fork the conversation here",
            handler=_cmd_branch,
            usage="/branch [summary]",
            aliases=("fork",),
            search_terms=("diverge", "split"),
        ),
        SlashCommand(
            name="rewind",
            description="Rewind to a prior entry (no argument lists them)",
            handler=_cmd_rewind,
            usage="/rewind [number]",
            search_terms=("undo", "rollback", "back"),
        ),
        SlashCommand(
            name="branches",
            description="List fork points",
            handler=_cmd_branches,
            usage="/branches",
            search_terms=("forks", "tree", "history"),
        ),
        SlashCommand(
            name="create-skill",
            description="Save a new skill",
            handler=_cmd_create_skill,
            usage="/create-skill <name> <text>",
            search_terms=("add", "author", "write"),
        ),
        SlashCommand(
            name="continue",
            description="Resume the agent loop after a stop",
            handler=_cmd_continue,
            usage="/continue",
            aliases=("cont",),
            search_terms=("carry", "on", "proceed"),
        ),
        SlashCommand(
            name="shell",
            description="Run a shell command",
            handler=_cmd_shell,
            usage="/shell <command>",
            aliases=("sh", "!"),
            search_terms=("terminal", "exec", "bash"),
        ),
        SlashCommand(
            name="diag",
            description="Dump session diagnostics",
            handler=_cmd_diag,
            usage="/diag",
            aliases=("debug",),
            search_terms=("diagnostics", "inspect"),
        ),
        SlashCommand(
            name="system",
            description="Show the system prompt and where each part came from",
            handler=_cmd_system,
            usage="/system",
            search_terms=("prompt", "instructions", "provenance", "agents.md"),
        ),
        SlashCommand(
            name="skills",
            description="List available skills",
            handler=_cmd_skills,
            usage="/skills",
            search_terms=("abilities",),
        ),
        SlashCommand(
            name="prompts",
            description="List prompt templates",
            handler=_cmd_prompts,
            usage="/prompts",
            aliases=("templates",),
            search_terms=("snippets",),
        ),
        SlashCommand(
            name="theme",
            description="Show or switch theme",
            handler=_cmd_theme,
            usage="/theme [name]",
            search_terms=("color", "style", "appearance"),
        ),
        SlashCommand(
            name="login",
            description="Authenticate with a provider",
            handler=_cmd_login,
            usage="/login [provider]",
            search_terms=("auth", "authenticate", "key"),
        ),
        SlashCommand(
            name="logout",
            description="Deauthenticate from a provider",
            handler=_cmd_logout,
            usage="/logout [provider]",
            search_terms=("deauth", "revoke"),
        ),
        SlashCommand(
            name="config",
            description="Show or edit configuration",
            handler=_cmd_config,
            usage="/config [key=value]",
            aliases=("settings",),
            search_terms=("preferences", "options"),
        ),
        SlashCommand(
            name="doctor",
            description="Run environment health checks",
            handler=_cmd_doctor,
            usage="/doctor",
            search_terms=("health", "check", "verify"),
        ),
    ]

    for cmd in _BUILTINS:
        reg.register(cmd)

    return reg


__all__ = [
    "CommandContext",
    "CommandRegistry",
    "CommandResult",
    "CommandSession",
    "Handler",
    "SlashCommand",
    "build_default_registry",
    "parse_command",
]
