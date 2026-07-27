"""Markdown prompt template system: discovery, loading, rendering, and slash-command expansion.

A *prompt template* is a ``.md`` file that lives in a resource directory.
Templates support ``{{ variable }}`` placeholder syntax and are invoked via
``/<template-name> arguments…`` slash commands.

Templates are loaded from one or more *resource directories* in precedence
order (highest first).  A template in a higher-precedence directory shadows
any template with the same name from a lower-precedence directory.

Public surface
--------------
Types:   ``PromptTemplate``, ``TemplateDiagnostic``, ``Severity``
Load:    ``load_prompt_templates``, ``load_prompt_templates_with_diagnostics``
Render:  ``render_template``
Expand:  ``expand_slash_command``
Errors:  ``TemplateLoadError``, ``TemplateRenderError``
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence


# ── constants ──────────────────────────────────────────────────────────────

_MD_SUFFIX = ".md"
_FRONT_MATTER_RE = re.compile(r"\A---[ \t]*\n(.*?\n)---[ \t]*\n", re.DOTALL)
_PLACEHOLDER_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
_RESERVED_PREFIXES = ("skill:",)
_ARGUMENT_ALIASES = frozenset({"arguments", "args"})


# ── diagnostic severity ───────────────────────────────────────────────────


class Severity(Enum):
    """Non-fatal diagnostic severity level."""

    INFO = "info"
    WARNING = "warning"


# ── data models ────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """An immutable, loaded markdown prompt template."""

    name: str
    path: Path
    content: str
    description: str | None = None


@dataclass(frozen=True, slots=True)
class TemplateDiagnostic:
    """A non-fatal issue encountered during template discovery."""

    severity: Severity
    path: Path
    message: str


# ── errors ─────────────────────────────────────────────────────────────────


class TemplateLoadError(Exception):
    """Fatal error while loading a template file."""


class TemplateRenderError(Exception):
    """A required placeholder variable was not supplied."""


# ── front-matter helpers ──────────────────────────────────────────────────


def _parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Split optional YAML front matter from the markdown body.

    Returns ``(metadata, body)``.  Only simple ``key: value`` lines are
    recognised — no nested YAML.
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


def _derive_description(body: str) -> str | None:
    """Extract the first non-heading, non-blank line as a description."""
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if stripped.startswith("---"):
            continue
        return stripped
    return None


# ── single-file loading ───────────────────────────────────────────────────


def _template_name_from_path(path: Path) -> str:
    """Derive a template name from a file path (stem without extension)."""
    return path.stem


def _load_template_file(path: Path) -> PromptTemplate:
    """Read a single markdown file and return a ``PromptTemplate``.

    Raises ``TemplateLoadError`` on I/O or encoding problems.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TemplateLoadError(f"Cannot read {path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise TemplateLoadError(f"{path} is not valid UTF-8: {exc}") from exc

    meta, body = _parse_front_matter(raw)
    name = meta.get("name", _template_name_from_path(path))
    description = meta.get("description") or _derive_description(body)

    return PromptTemplate(
        name=name,
        path=path,
        content=body,
        description=description,
    )


# ── directory discovery ───────────────────────────────────────────────────


def _discover_in_directory(
    root: Path,
) -> tuple[list[PromptTemplate], list[TemplateDiagnostic]]:
    """Scan *root* for ``.md`` files and load them as templates.

    Returns ``(templates, diagnostics)``.  Duplicates within the same
    directory are reported as warnings.
    """
    templates: list[PromptTemplate] = []
    diagnostics: list[TemplateDiagnostic] = []
    seen: dict[str, Path] = {}

    if not root.is_dir():
        return templates, diagnostics

    for child in sorted(root.iterdir()):
        if child.is_dir():
            continue
        if child.suffix.lower() != _MD_SUFFIX:
            continue

        try:
            tpl = _load_template_file(child)
        except TemplateLoadError as exc:
            diagnostics.append(TemplateDiagnostic(
                severity=Severity.WARNING,
                path=child,
                message=str(exc),
            ))
            continue

        if tpl.name in seen:
            diagnostics.append(TemplateDiagnostic(
                severity=Severity.WARNING,
                path=child,
                message=(
                    f"Duplicate template name {tpl.name!r}; "
                    f"already loaded from {seen[tpl.name]}"
                ),
            ))
            continue

        seen[tpl.name] = child
        templates.append(tpl)

    return templates, diagnostics


# ── public loading API ─────────────────────────────────────────────────────


def load_prompt_templates_with_diagnostics(
    resource_dirs: Sequence[Path],
) -> tuple[list[PromptTemplate], list[TemplateDiagnostic]]:
    """Load templates from multiple resource directories with diagnostics.

    *resource_dirs* is ordered highest-precedence first.  A template in a
    higher-precedence directory shadows any template with the same name
    from a lower-precedence directory.

    Returns ``(templates, diagnostics)``.
    """
    combined: dict[str, PromptTemplate] = {}
    all_diagnostics: list[TemplateDiagnostic] = []

    for root in resource_dirs:
        templates, diags = _discover_in_directory(root)
        all_diagnostics.extend(diags)
        for tpl in templates:
            if tpl.name not in combined:
                combined[tpl.name] = tpl

    return list(combined.values()), all_diagnostics


def load_prompt_templates(resource_dirs: Sequence[Path]) -> list[PromptTemplate]:
    """Load templates from resource directories, discarding diagnostics."""
    templates, _ = load_prompt_templates_with_diagnostics(resource_dirs)
    return templates


# ── placeholder utilities ─────────────────────────────────────────────────


def _collect_placeholders(content: str) -> set[str]:
    """Return the set of placeholder names found in *content*."""
    return set(_PLACEHOLDER_RE.findall(content))


def _references_arguments(content: str) -> bool:
    """Return whether *content* contains an ``{{arguments}}`` or ``{{args}}`` placeholder."""
    return bool(_ARGUMENT_ALIASES & _collect_placeholders(content))


# ── rendering ──────────────────────────────────────────────────────────────


def render_template(
    template: PromptTemplate,
    variables: Mapping[str, str] | None = None,
    *,
    strict: bool = True,
    missing_value: str = "",
) -> str:
    """Render a template by replacing ``{{ var }}`` placeholders.

    Parameters
    ----------
    template:
        The template to render.
    variables:
        Mapping of variable names to replacement strings.
    strict:
        When ``True`` (the default), raise ``TemplateRenderError`` for any
        placeholder whose name is not in *variables*.  When ``False``,
        missing placeholders are replaced with *missing_value*.
    missing_value:
        Fallback string used for missing variables when *strict* is ``False``.
    """
    vars_ = variables or {}

    def _replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in vars_:
            return vars_[name]
        if strict:
            raise TemplateRenderError(
                f"Missing variable {name!r} in template {template.name!r}"
            )
        return missing_value

    return _PLACEHOLDER_RE.sub(_replace, template.content)


# ── slash-command parsing ──────────────────────────────────────────────────


def _is_reserved_command(name: str) -> bool:
    """Return whether *name* belongs to another command system."""
    return any(name.startswith(prefix) for prefix in _RESERVED_PREFIXES)


def _parse_slash_command(text: str) -> tuple[str, str] | None:
    """Parse ``/<name> args…`` into ``(name, args)`` or return ``None``.

    Returns ``None`` for non-slash input, ``//`` comments, and commands
    reserved by other systems (e.g. ``/skill:…``).
    """
    if not text.startswith("/"):
        return None
    if text.startswith("//"):
        return None

    body = text[1:]
    parts = body.split(None, 1)
    if not parts:
        return None

    name = parts[0]
    if _is_reserved_command(name):
        return None

    arguments = parts[1] if len(parts) > 1 else ""
    return name, arguments


def _lookup_template(
    name: str,
    registry: Mapping[str, PromptTemplate],
) -> PromptTemplate | None:
    """Find a template by name, returning ``None`` on miss."""
    return registry.get(name)


def _build_variables(arguments: str) -> dict[str, str]:
    """Build the default variable mapping for a slash-command invocation."""
    return {"arguments": arguments, "args": arguments}


def _maybe_append_arguments(rendered: str, arguments: str, content: str) -> str:
    """Append *arguments* to *rendered* if the template never referenced them."""
    if not arguments:
        return rendered
    if _references_arguments(content):
        return rendered
    return rendered + "\n\n" + arguments


# ── public expansion API ───────────────────────────────────────────────────


def expand_slash_command(
    text: str,
    registry: Mapping[str, PromptTemplate],
    *,
    extra_variables: Mapping[str, str] | None = None,
) -> str | None:
    """Expand a ``/<template-name> args…`` slash command.

    Returns the rendered template string, or ``None`` if *text* is not a
    recognised template command.  Raises ``KeyError`` if the parsed command
    name does not match any loaded template.

    Parameters
    ----------
    text:
        Raw user input (e.g. ``"/summarize the login module"``).
    registry:
        Mapping of template name → ``PromptTemplate``.
    extra_variables:
        Additional variables merged into the rendering context (the built-in
        ``arguments`` / ``args`` bindings take precedence over these).
    """
    parsed = _parse_slash_command(text)
    if parsed is None:
        return None

    name, arguments = parsed
    tpl = _lookup_template(name, registry)
    if tpl is None:
        return None

    variables = dict(extra_variables) if extra_variables else {}
    variables.update(_build_variables(arguments))

    rendered = render_template(tpl, variables, strict=False, missing_value="")
    return _maybe_append_arguments(rendered, arguments, tpl.content)


__all__ = [
    "PromptTemplate",
    "Severity",
    "TemplateDiagnostic",
    "TemplateLoadError",
    "TemplateRenderError",
    "expand_slash_command",
    "load_prompt_templates",
    "load_prompt_templates_with_diagnostics",
    "render_template",
]
