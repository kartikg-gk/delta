"""Autocomplete engine for Delta's CLI/TUI prompt input.

Computes structured completion candidates for slash commands, command
arguments, file references, shell paths, prompt templates, and skills.
This module is entirely UI-independent — it returns ``CompletionState``
objects that the caller renders however it wants.

Completion contexts are detected automatically from the cursor position
and current input text.  The engine never executes commands, accesses
providers, or modifies session state.

Public surface
--------------
Types:   ``CompletionOption``, ``CompletionItem``, ``CompletionState``
Entry:   ``build_completion_state``
Helpers: ``complete_commands``, ``complete_arguments``, ``complete_skills``,
         ``complete_prompts``, ``complete_file_refs``, ``complete_shell_paths``
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from delta_app.commands import CommandRegistry, SlashCommand
from delta_app.prompts import PromptTemplate
from delta_app.skills import Skill

# ── constants ──────────────────────────────────────────────────────────────

_SKILL_PREFIX = "/skill:"
_FILE_REF_PREFIX = "@"
_SHELL_PREFIXES = ("! ", "!! ")
_MAX_RESULTS_DEFAULT = 50

_IGNORED_DIRS_DEFAULT: frozenset[str] = frozenset({
    ".git", ".hg", ".svn",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "node_modules", ".tox", ".nox",
    ".venv", "venv", "env", ".env",
    "dist", "build", ".build",
    ".eggs", "*.egg-info",
    "target",
    ".next", ".nuxt",
    "coverage", ".coverage", "htmlcov",
    ".terraform",
    ".cache",
})

_UNSAFE_PATH_RE = re.compile(r"[$~*?\[\]{}]")


# ── CompletionOption ──────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CompletionOption:
    """A possible argument value for a command that accepts discrete choices."""

    value: str
    description: str = ""


# ── CompletionItem ────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CompletionItem:
    """One selectable autocomplete suggestion.

    ``start`` and ``end`` delimit the substring of the original input that
    this item would replace.  ``replacement`` is the text to substitute.
    ``display`` is the string shown in the completion menu (may differ
    from ``replacement`` for readability).
    """

    display: str
    replacement: str
    start: int
    end: int
    description: str = ""
    category: str = ""

    def apply(self, text: str) -> str:
        """Return *text* with the active token replaced by ``replacement``."""
        return text[:self.start] + self.replacement + text[self.end:]


# ── CompletionState ───────────────────────────────────────────────────────


@dataclass(slots=True)
class CompletionState:
    """Mutable selection state over a tuple of ``CompletionItem`` values.

    The selected index wraps around in both directions so keyboard
    navigation never goes out of bounds.
    """

    items: tuple[CompletionItem, ...] = ()
    selected_index: int = -1

    @property
    def selected(self) -> CompletionItem | None:
        """The currently highlighted item, or ``None`` when nothing is selected."""
        if not self.items or self.selected_index < 0:
            return None
        return self.items[self.selected_index % len(self.items)]

    def select_next(self) -> None:
        """Advance the selection, wrapping around at the end."""
        if not self.items:
            return
        self.selected_index = (self.selected_index + 1) % len(self.items)

    def select_previous(self) -> None:
        """Move the selection backward, wrapping around at the start."""
        if not self.items:
            return
        if self.selected_index < 0:
            self.selected_index = len(self.items) - 1
        else:
            self.selected_index = (self.selected_index - 1) % len(self.items)

    def __bool__(self) -> bool:
        return len(self.items) > 0


# ── ignored-path detection ────────────────────────────────────────────────


def is_ignored_dir(name: str, ignored: frozenset[str] = _IGNORED_DIRS_DEFAULT) -> bool:
    """Return whether *name* should be skipped during filesystem traversal."""
    if name.startswith("."):
        return True
    return name in ignored or name.lower() in ignored


# ── active-token parsing ─────────────────────────────────────────────────


def _active_token(text: str, cursor: int) -> tuple[str, int, int]:
    """Extract the token under the cursor and its span.

    Returns ``(token, start, end)`` where *start* and *end* are byte
    offsets into *text*.  A "token" is the contiguous non-whitespace run
    that the cursor sits in or immediately follows.
    """
    if cursor > len(text):
        cursor = len(text)

    # Walk backward to find token start
    start = cursor
    while start > 0 and not text[start - 1].isspace():
        start -= 1

    # Walk forward to find token end
    end = cursor
    while end < len(text) and not text[end].isspace():
        end += 1

    return text[start:end], start, end


def _last_token(text: str) -> tuple[str, int, int]:
    """Convenience: active token at the very end of *text*."""
    return _active_token(text, len(text))


# ── sorting ──────────────────────────────────────────────────────────────


def _sort_items(items: list[CompletionItem], prefix: str) -> list[CompletionItem]:
    """Sort completions: exact prefix matches first, then alphabetically."""
    low = prefix.lower()

    def _key(item: CompletionItem) -> tuple[int, str]:
        repl = item.replacement.lower()
        # Exact prefix match gets priority 0, others get 1
        priority = 0 if repl.startswith(low) else 1
        return priority, repl

    return sorted(items, key=_key)


def _dedupe_items(items: list[CompletionItem]) -> list[CompletionItem]:
    """Remove duplicate completions by replacement value."""
    seen: set[str] = set()
    result: list[CompletionItem] = []
    for item in items:
        if item.replacement not in seen:
            seen.add(item.replacement)
            result.append(item)
    return result


# ── slash-command completion ─────────────────────────────────────────────


def complete_commands(
    prefix: str,
    registry: CommandRegistry,
    *,
    start: int,
    end: int,
    max_results: int = _MAX_RESULTS_DEFAULT,
) -> list[CompletionItem]:
    """Generate completions for a partial slash command.

    *prefix* should include the leading ``/`` (e.g. ``"/mo"``).
    Matches against command names, aliases, and search terms.
    """
    if not prefix.startswith("/"):
        return []

    fragment = prefix[1:].lower()
    items: list[CompletionItem] = []

    for cmd in registry.all_commands():
        if _command_matches(cmd, fragment):
            items.append(CompletionItem(
                display=f"/{cmd.name}",
                replacement=f"/{cmd.name}",
                start=start,
                end=end,
                description=cmd.description,
                category="command",
            ))
        if len(items) >= max_results:
            break

    return _sort_items(items, prefix)


def _command_matches(cmd: SlashCommand, fragment: str) -> bool:
    """Return whether *cmd* is a candidate for the typed *fragment*."""
    if cmd.name.startswith(fragment):
        return True
    for alias in cmd.aliases:
        if alias.startswith(fragment):
            return True
    for term in cmd.search_terms:
        if term.startswith(fragment):
            return True
    return False


# ── argument completion ──────────────────────────────────────────────────


def complete_arguments(
    command_name: str,
    arg_prefix: str,
    options: Sequence[CompletionOption],
    *,
    start: int,
    end: int,
    max_results: int = _MAX_RESULTS_DEFAULT,
) -> list[CompletionItem]:
    """Generate argument completions for a known command.

    *options* is the list of valid argument values for the command.
    """
    low = arg_prefix.lower()
    items: list[CompletionItem] = []

    for opt in options:
        if opt.value.lower().startswith(low):
            items.append(CompletionItem(
                display=opt.value,
                replacement=opt.value,
                start=start,
                end=end,
                description=opt.description,
                category="argument",
            ))
        if len(items) >= max_results:
            break

    return _sort_items(items, arg_prefix)


# ── skill completion ─────────────────────────────────────────────────────


def complete_skills(
    prefix: str,
    skills: Sequence[Skill],
    *,
    start: int,
    end: int,
    max_results: int = _MAX_RESULTS_DEFAULT,
) -> list[CompletionItem]:
    """Generate completions for ``/skill:<partial>``.

    *prefix* should include the ``/skill:`` leader (e.g. ``"/skill:ref"``).
    Once a valid skill name is fully typed and the user is entering arguments,
    no further skill-name suggestions are returned.
    """
    if not prefix.lower().startswith(_SKILL_PREFIX):
        return []

    fragment = prefix[len(_SKILL_PREFIX):]

    # If the fragment already contains whitespace, the user is past the
    # skill name and is typing arguments — stop suggesting skills.
    if " " in fragment:
        return []

    low = fragment.lower()
    items: list[CompletionItem] = []

    for skill in skills:
        if skill.name.lower().startswith(low):
            replacement = f"{_SKILL_PREFIX}{skill.name}"
            items.append(CompletionItem(
                display=replacement,
                replacement=replacement,
                start=start,
                end=end,
                description=skill.description,
                category="skill",
            ))
        if len(items) >= max_results:
            break

    return _sort_items(items, prefix)


# ── prompt-template completion ───────────────────────────────────────────


def complete_prompts(
    prefix: str,
    templates: Sequence[PromptTemplate],
    *,
    start: int,
    end: int,
    max_results: int = _MAX_RESULTS_DEFAULT,
) -> list[CompletionItem]:
    """Generate completions for prompt template names.

    *prefix* should include the leading ``/`` (e.g. ``"/summ"``).
    Templates are offered alongside regular commands but in their own category.
    """
    if not prefix.startswith("/"):
        return []

    fragment = prefix[1:].lower()
    items: list[CompletionItem] = []

    for tpl in templates:
        if tpl.name.lower().startswith(fragment):
            items.append(CompletionItem(
                display=f"/{tpl.name}",
                replacement=f"/{tpl.name}",
                start=start,
                end=end,
                description=tpl.description or "",
                category="prompt",
            ))
        if len(items) >= max_results:
            break

    return _sort_items(items, prefix)


# ── file-reference completion ────────────────────────────────────────────


def complete_file_refs(
    prefix: str,
    cwd: str,
    *,
    start: int,
    end: int,
    ignored: frozenset[str] = _IGNORED_DIRS_DEFAULT,
    max_results: int = _MAX_RESULTS_DEFAULT,
) -> list[CompletionItem]:
    """Generate completions for ``@path/to/file`` references.

    Searches recursively under *cwd*, ignoring hidden directories and
    common build/cache directories.
    """
    if not prefix.startswith(_FILE_REF_PREFIX):
        return []

    partial = prefix[len(_FILE_REF_PREFIX):]
    base = Path(cwd)
    items: list[CompletionItem] = []

    # Determine the directory to scan and the filename prefix to match
    if "/" in partial or "\\" in partial:
        sep_pos = max(partial.rfind("/"), partial.rfind("\\"))
        dir_part = partial[:sep_pos + 1]
        file_prefix = partial[sep_pos + 1:]
        scan_dir = base / dir_part.replace("\\", "/")
    else:
        dir_part = ""
        file_prefix = partial
        scan_dir = base

    _collect_file_refs(
        scan_dir, base, dir_part, file_prefix, items,
        ignored=ignored, max_results=max_results, start=start, end=end,
    )

    return _sort_items(items, prefix)


def _collect_file_refs(
    scan_dir: Path,
    base: Path,
    dir_part: str,
    file_prefix: str,
    items: list[CompletionItem],
    *,
    ignored: frozenset[str],
    max_results: int,
    start: int,
    end: int,
    depth: int = 0,
    max_depth: int = 4,
) -> None:
    """Recursively collect file-reference completions."""
    if len(items) >= max_results or depth > max_depth:
        return

    if not scan_dir.is_dir():
        return

    try:
        entries = sorted(scan_dir.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except OSError:
        return

    low_prefix = file_prefix.lower()

    for entry in entries:
        if len(items) >= max_results:
            return

        if is_ignored_dir(entry.name, ignored) and entry.is_dir():
            continue

        try:
            rel = entry.relative_to(base).as_posix()
        except ValueError:
            continue

        if entry.is_dir():
            dirname = entry.name + "/"
            if dirname.lower().startswith(low_prefix) or not file_prefix:
                full_ref = f"{_FILE_REF_PREFIX}{rel}/"
                items.append(CompletionItem(
                    display=full_ref,
                    replacement=full_ref,
                    start=start,
                    end=end,
                    description="directory",
                    category="file",
                ))
            # Recurse into matching directories
            if entry.name.lower().startswith(low_prefix[:len(entry.name)].lower()) if file_prefix else True:
                _collect_file_refs(
                    entry, base, dir_part + entry.name + "/", "",
                    items, ignored=ignored, max_results=max_results,
                    start=start, end=end, depth=depth + 1, max_depth=max_depth,
                )
        else:
            if entry.name.lower().startswith(low_prefix):
                full_ref = f"{_FILE_REF_PREFIX}{rel}"
                items.append(CompletionItem(
                    display=full_ref,
                    replacement=full_ref,
                    start=start,
                    end=end,
                    description=_file_description(entry),
                    category="file",
                ))


def _file_description(path: Path) -> str:
    """Derive a short description for a file path."""
    suffix = path.suffix.lower()
    descriptions: dict[str, str] = {
        ".py": "Python",
        ".js": "JavaScript",
        ".ts": "TypeScript",
        ".tsx": "TypeScript JSX",
        ".jsx": "JavaScript JSX",
        ".rs": "Rust",
        ".go": "Go",
        ".md": "Markdown",
        ".toml": "TOML",
        ".yaml": "YAML",
        ".yml": "YAML",
        ".json": "JSON",
        ".txt": "Text",
        ".sh": "Shell",
        ".css": "CSS",
        ".html": "HTML",
        ".sql": "SQL",
    }
    return descriptions.get(suffix, "file")


# ── shell-path completion ────────────────────────────────────────────────


def complete_shell_paths(
    text: str,
    cwd: str,
    *,
    ignored: frozenset[str] = _IGNORED_DIRS_DEFAULT,
    max_results: int = _MAX_RESULTS_DEFAULT,
) -> list[CompletionItem]:
    """Generate path completions for shell-mode input (``!`` or ``!!``).

    Only completes relative paths.  Rejects absolute paths, home-directory
    expansion, shell variables, and wildcard expressions.
    """
    shell_prefix, body = _parse_shell_prefix(text)
    if shell_prefix is None:
        return []

    # Extract the last token from the shell body
    token, tok_start, tok_end = _last_token(body)
    if not token:
        return []

    # Reject unsafe patterns
    if _is_unsafe_path(token):
        return []

    # Adjust offsets to account for the shell prefix
    offset = len(shell_prefix)
    abs_start = offset + tok_start
    abs_end = offset + tok_end

    return _complete_relative_path(
        token, cwd, start=abs_start, end=abs_end,
        ignored=ignored, max_results=max_results,
    )


def _parse_shell_prefix(text: str) -> tuple[str | None, str]:
    """Split shell prefix (``"! "`` or ``"!! "``) from the body.

    Returns ``(prefix, body)`` or ``(None, "")`` if *text* is not a
    shell command.
    """
    for prefix in _SHELL_PREFIXES:
        if text.startswith(prefix):
            return prefix, text[len(prefix):]
    return None, ""


def _is_unsafe_path(token: str) -> bool:
    """Return whether *token* contains unsafe shell path patterns."""
    if not token:
        return False
    # Absolute paths
    if token.startswith("/") or token.startswith("\\"):
        return True
    # Windows absolute paths (C:\...)
    if len(token) >= 2 and token[1] == ":" and token[0].isalpha():
        return True
    # Home expansion
    if token.startswith("~"):
        return True
    # Shell variables, wildcards, braces
    if _UNSAFE_PATH_RE.search(token):
        return True
    return False


def _complete_relative_path(
    partial: str,
    cwd: str,
    *,
    start: int,
    end: int,
    ignored: frozenset[str],
    max_results: int,
) -> list[CompletionItem]:
    """Generate completions for a relative filesystem path."""
    base = Path(cwd)

    if "/" in partial or "\\" in partial:
        sep_pos = max(partial.rfind("/"), partial.rfind("\\"))
        dir_part = partial[:sep_pos + 1]
        name_prefix = partial[sep_pos + 1:]
        scan_dir = base / dir_part.replace("\\", "/")
    else:
        dir_part = ""
        name_prefix = partial
        scan_dir = base

    if not scan_dir.is_dir():
        return []

    try:
        entries = sorted(scan_dir.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except OSError:
        return []

    low = name_prefix.lower()
    items: list[CompletionItem] = []

    for entry in entries:
        if len(items) >= max_results:
            break
        if is_ignored_dir(entry.name, ignored) and entry.is_dir():
            continue
        if entry.name.startswith("."):
            continue
        if not entry.name.lower().startswith(low):
            continue

        if entry.is_dir():
            replacement = dir_part + entry.name + "/"
            desc = "directory"
        else:
            replacement = dir_part + entry.name
            desc = _file_description(entry)

        items.append(CompletionItem(
            display=replacement,
            replacement=replacement,
            start=start,
            end=end,
            description=desc,
            category="path",
        ))

    return _sort_items(items, partial)


# ── context detection ────────────────────────────────────────────────────


def _detect_command_arg_context(
    text: str,
    registry: CommandRegistry,
) -> tuple[str, str, int, int] | None:
    """Detect whether *text* is a command followed by a partial argument.

    Returns ``(command_name, arg_prefix, arg_start, arg_end)`` when the
    user is typing an argument for a known command, or ``None`` otherwise.
    """
    if not text.startswith("/"):
        return None

    parts = text.split(None, 1)
    if len(parts) < 2:
        return None

    cmd_name = parts[0][1:].lower()
    cmd = registry.get(cmd_name)
    if cmd is None:
        return None

    arg_text = parts[1]
    arg_start = len(text) - len(arg_text)

    # The active token within the arg text
    token, t_start, t_end = _last_token(arg_text)
    return cmd.name, token, arg_start + t_start, arg_start + t_end


# ── argument option registry ─────────────────────────────────────────────


_ARGUMENT_OPTIONS: dict[str, list[CompletionOption]] = {
    "model": [
        CompletionOption("claude-sonnet-4-20250514", "Claude Sonnet 4"),
        CompletionOption("claude-opus-4-20250514", "Claude Opus 4"),
        CompletionOption("claude-haiku-4-5-20251001", "Claude Haiku 4.5"),
    ],
    "export": [
        CompletionOption("text", "Plain text transcript"),
        CompletionOption("json", "JSON format"),
        CompletionOption("jsonl", "JSON Lines format"),
    ],
    "think": [
        CompletionOption("off", "Disable thinking"),
        CompletionOption("low", "Low thinking budget"),
        CompletionOption("medium", "Medium thinking budget"),
        CompletionOption("high", "High thinking budget"),
    ],
    "login": [
        CompletionOption("anthropic", "Anthropic API"),
        CompletionOption("openai", "OpenAI-compatible API"),
    ],
    "logout": [
        CompletionOption("anthropic", "Anthropic API"),
        CompletionOption("openai", "OpenAI-compatible API"),
    ],
    "theme": [
        CompletionOption("dark", "Dark theme"),
        CompletionOption("light", "Light theme"),
    ],
}


def get_argument_options(
    command_name: str,
    extra: dict[str, list[CompletionOption]] | None = None,
) -> list[CompletionOption]:
    """Return the argument options for *command_name*.

    Merges built-in defaults with *extra* overrides supplied by the caller.
    """
    options = list((extra or {}).get(command_name, _ARGUMENT_OPTIONS.get(command_name, [])))
    return options


# ── main API ─────────────────────────────────────────────────────────────


def build_completion_state(
    text: str,
    *,
    cursor: int | None = None,
    registry: CommandRegistry | None = None,
    skills: Sequence[Skill] = (),
    templates: Sequence[PromptTemplate] = (),
    cwd: str | None = None,
    argument_options: dict[str, list[CompletionOption]] | None = None,
    ignored_dirs: frozenset[str] = _IGNORED_DIRS_DEFAULT,
    max_results: int = _MAX_RESULTS_DEFAULT,
) -> CompletionState:
    """Inspect *text* and return a ``CompletionState`` with matching candidates.

    This is the single entry point for the autocomplete engine.  It detects
    the completion context and delegates to the appropriate helper.

    Parameters
    ----------
    text:
        The full current input line.
    cursor:
        Cursor position within *text*.  Defaults to end of string.
    registry:
        Slash-command registry used for command/argument completion.
    skills:
        Loaded skills used for ``/skill:`` completion.
    templates:
        Loaded prompt templates used for template-name completion.
    cwd:
        Working directory for file and shell-path completion.
    argument_options:
        Per-command argument option overrides.
    ignored_dirs:
        Directory names to skip during filesystem traversal.
    max_results:
        Maximum number of suggestions to return.
    """
    if cursor is None:
        cursor = len(text)

    resolved_cwd = cwd or os.getcwd()
    items: list[CompletionItem] = []

    # ── shell paths (! or !! prefix) ─────────────────────────────────
    if _is_shell_input(text):
        items = complete_shell_paths(
            text, resolved_cwd,
            ignored=ignored_dirs, max_results=max_results,
        )
        return CompletionState(items=tuple(items))

    # ── file references (@...) ───────────────────────────────────────
    token, tok_start, tok_end = _active_token(text, cursor)
    if token.startswith(_FILE_REF_PREFIX) and len(token) > 1:
        items = complete_file_refs(
            token, resolved_cwd,
            start=tok_start, end=tok_end,
            ignored=ignored_dirs, max_results=max_results,
        )
        return CompletionState(items=tuple(items))

    # ── skill completion (/skill:...) ────────────────────────────────
    if token.lower().startswith(_SKILL_PREFIX) and skills:
        items = complete_skills(
            token, skills,
            start=tok_start, end=tok_end, max_results=max_results,
        )
        return CompletionState(items=tuple(items))

    # ── command argument completion (/cmd arg) ───────────────────────
    if registry is not None:
        ctx = _detect_command_arg_context(text, registry)
        if ctx is not None:
            cmd_name, arg_prefix, arg_start, arg_end = ctx
            opts = get_argument_options(cmd_name, argument_options)
            if opts:
                items = complete_arguments(
                    cmd_name, arg_prefix, opts,
                    start=arg_start, end=arg_end, max_results=max_results,
                )
                return CompletionState(items=tuple(items))

    # ── slash command + template completion (/...) ───────────────────
    if token.startswith("/") and len(token) > 1:
        if registry is not None:
            items.extend(complete_commands(
                token, registry,
                start=tok_start, end=tok_end, max_results=max_results,
            ))
        if templates:
            items.extend(complete_prompts(
                token, templates,
                start=tok_start, end=tok_end, max_results=max_results,
            ))
        items = _dedupe_items(_sort_items(items, token))
        if len(items) > max_results:
            items = items[:max_results]
        return CompletionState(items=tuple(items))

    return CompletionState()


def _is_shell_input(text: str) -> bool:
    """Return whether *text* starts with a shell prefix."""
    return any(text.startswith(p) for p in _SHELL_PREFIXES)


# ── registration hook ────────────────────────────────────────────────────


class CompletionProvider:
    """Hook point for registering custom completion providers.

    Subclass and override ``complete`` to inject domain-specific
    completions into the engine.
    """

    def complete(
        self,
        text: str,
        cursor: int,
        *,
        start: int,
        end: int,
    ) -> list[CompletionItem]:
        """Return completions for the given input state.

        Override in subclasses.  The default returns an empty list.
        """
        return []


__all__ = [
    "CompletionItem",
    "CompletionOption",
    "CompletionProvider",
    "CompletionState",
    "build_completion_state",
    "complete_arguments",
    "complete_commands",
    "complete_file_refs",
    "complete_prompts",
    "complete_shell_paths",
    "complete_skills",
    "get_argument_options",
    "is_ignored_dir",
]
