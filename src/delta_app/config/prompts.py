"""Interactive prompts for ``delta config``.

Plain stdin/stdout only — no curses, no third-party TUI dependency.  All
prompts write to stderr so the flow stays usable when stdout is redirected.
API keys are read via :mod:`delta_app.config.secure_input`, which disables
terminal echo without taking over line editing.
"""

from __future__ import annotations

import sys

from delta_app.config.secure_input import read_secret


class PromptAborted(Exception):
    """Raised when the user cancels the flow (EOF or Ctrl-C)."""


def _write(text: str) -> None:
    sys.stderr.write(text)
    sys.stderr.flush()


def _readline(prompt: str) -> str:
    _write(prompt)
    try:
        line = sys.stdin.readline()
    except (EOFError, KeyboardInterrupt):
        raise PromptAborted from None
    if not line:
        raise PromptAborted
    return line.strip()


def choose(title: str, options: list[str], *, default: int = 0) -> int:
    """Show a numbered menu and return the chosen zero-based index."""
    _write(f"\n{title}\n\n")
    for i, option in enumerate(options):
        marker = ">" if i == default else " "
        _write(f"  {marker} {i + 1}. {option}\n")

    while True:
        raw = _readline(f"\nSelect [1-{len(options)}] ({default + 1}): ")
        if not raw:
            return default
        try:
            index = int(raw) - 1
        except ValueError:
            _write("Enter a number.\n")
            continue
        if 0 <= index < len(options):
            return index
        _write(f"Enter a number between 1 and {len(options)}.\n")


def choose_model(models: list[str], *, current: str | None = None) -> str:
    """Pick a model from *models*, or enter a custom identifier."""
    options = [*models, "Other (enter manually)"]
    default = models.index(current) if current in models else 0
    index = choose("Default model", options, default=default)
    if index < len(models):
        return models[index]

    while True:
        value = _readline("\nModel identifier: ")
        if value:
            return value
        _write("Model identifier must not be empty.\n")


def _read_secret(*, visible: bool = False) -> str:
    """Read a line without echoing it, treating EOF/Ctrl+C as cancellation."""
    try:
        return read_secret(_write, visible=visible).strip()
    except (EOFError, KeyboardInterrupt):
        raise PromptAborted from None


def ask_api_key(
    label: str, *, required: bool, has_existing: bool, visible: bool = False,
) -> str | None:
    """Read an API key without echoing it. Returns ``None`` to keep the existing one."""
    suffix = " (press Enter to keep the saved key)" if has_existing else ""
    while True:
        _write(f"\nAPI Key{suffix}:\n")
        if not visible:
            # Hidden input shows nothing at all — say so, or a working prompt
            # is indistinguishable from a frozen one.
            _write(
                "  (input is hidden: type or paste, then press Enter.\n"
                "   Nothing will appear on screen. Ctrl+C cancels.)\n"
            )
        value = _read_secret(visible=visible)

        if value:
            return value
        if has_existing:
            return None
        if not required:
            return ""
        _write(f"{label} requires an API key.\n")


def confirm(question: str, *, default: bool = True) -> bool:
    """Ask a yes/no question."""
    hint = "Y/n" if default else "y/N"
    while True:
        raw = _readline(f"\n{question} ({hint}) ").lower()
        if not raw:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False


def info(text: str) -> None:
    """Print a status line."""
    _write(f"{text}\n")


__all__ = [
    "PromptAborted",
    "ask_api_key",
    "choose",
    "choose_model",
    "confirm",
    "info",
]
