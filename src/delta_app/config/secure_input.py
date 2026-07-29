"""Hidden (non-echoing) line input that works across Windows and POSIX terminals.

``getpass.getpass`` is unreliable on Windows: it reads character-by-character
via ``msvcrt.getwch()``, which bypasses stdin and the console line editor.
Under ConPTY-based terminals (VS Code, Windows Terminal) input arrives through
a pipe that ``getwch()`` never reads, so the prompt hangs and paste does
nothing.

This module instead keeps the terminal's own *line* input — which already
handles typing, paste, backspace, and Ctrl+C — and only turns echo off around
a normal ``readline()``.  Strategies are tried in order and the first one that
applies wins:

1. Windows console — clear ``ENABLE_ECHO_INPUT`` via ``SetConsoleMode``.
2. POSIX terminal — clear ``ECHO`` via ``termios``.
3. Non-interactive stdin — plain read (piped/scripted input).
4. Visible input, after warning the user that the value will be shown.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

# Windows console mode flags (winbase.h)
_ENABLE_PROCESSED_INPUT = 0x0001
_ENABLE_LINE_INPUT = 0x0002
_ENABLE_ECHO_INPUT = 0x0004
_STD_INPUT_HANDLE = -10


class SecureInputUnavailable(Exception):
    """Raised by a strategy that cannot apply on this terminal."""


def _read_line() -> str:
    """Read one line from stdin, raising ``EOFError`` at end of input."""
    line = sys.stdin.readline()
    if not line:
        raise EOFError
    return line.rstrip("\r\n")


# ---------------------------------------------------------------------------
# Strategy 1 — Windows console
# ---------------------------------------------------------------------------


def _read_windows(write: Callable[[str], None]) -> str:
    """Read with echo disabled via the Windows console API."""
    if sys.platform != "win32":
        raise SecureInputUnavailable("not Windows")

    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetStdHandle.restype = wintypes.HANDLE
    kernel32.GetStdHandle.argtypes = [wintypes.DWORD]
    kernel32.GetConsoleMode.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.SetConsoleMode.argtypes = [wintypes.HANDLE, wintypes.DWORD]

    handle = kernel32.GetStdHandle(_STD_INPUT_HANDLE)
    if not handle or handle == wintypes.HANDLE(-1).value:
        raise SecureInputUnavailable("no stdin console handle")

    original = wintypes.DWORD()
    # Authoritative probe: this fails when stdin is a pipe or ConPTY stream,
    # even though isatty() may still report True.
    if not kernel32.GetConsoleMode(handle, ctypes.byref(original)):
        raise SecureInputUnavailable("stdin is not a console")

    # Keep LINE_INPUT (paste, backspace, arrow keys) and PROCESSED_INPUT
    # (Ctrl+C -> SIGINT). Only echo goes away.
    quiet = (
        original.value | _ENABLE_LINE_INPUT | _ENABLE_PROCESSED_INPUT
    ) & ~_ENABLE_ECHO_INPUT

    if not kernel32.SetConsoleMode(handle, quiet):
        raise SecureInputUnavailable("could not disable console echo")
    try:
        value = _read_line()
    finally:
        kernel32.SetConsoleMode(handle, original.value)
    write("\n")  # the user's Enter was not echoed
    return value


# ---------------------------------------------------------------------------
# Strategy 2 — POSIX terminal
# ---------------------------------------------------------------------------


def _read_posix(write: Callable[[str], None]) -> str:
    """Read with echo disabled via termios."""
    if sys.platform == "win32":
        raise SecureInputUnavailable("Windows uses the console strategy")
    try:
        import termios
    except ImportError as exc:
        raise SecureInputUnavailable("termios unavailable") from exc

    try:
        fd = sys.stdin.fileno()
        original = termios.tcgetattr(fd)
    except (OSError, ValueError, termios.error) as exc:
        raise SecureInputUnavailable("stdin is not a terminal") from exc

    quiet = list(original)
    quiet[3] &= ~termios.ECHO  # lflags
    try:
        termios.tcsetattr(fd, termios.TCSADRAIN, quiet)
        value = _read_line()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, original)
    write("\n")
    return value


# ---------------------------------------------------------------------------
# Strategy 3 — non-interactive stdin
# ---------------------------------------------------------------------------


def _read_piped(write: Callable[[str], None]) -> str:
    """Read plainly when stdin is not a terminal (piped or scripted input)."""
    if sys.stdin.isatty():
        raise SecureInputUnavailable("stdin is interactive")
    return _read_line()


# ---------------------------------------------------------------------------
# Strategy 4 — visible fallback
# ---------------------------------------------------------------------------


def _read_visible(write: Callable[[str], None]) -> str:
    """Last resort: read with echo on, after warning the user."""
    write(
        "\nWarning: this terminal does not support hidden input.\n"
        "The value you type will be VISIBLE on screen.\n"
    )
    return _read_line()


_STRATEGIES: tuple[Callable[[Callable[[str], None]], str], ...] = (
    _read_piped,
    _read_windows,
    _read_posix,
    _read_visible,
)

_STRATEGY_NAMES = {
    "_read_piped": "piped stdin (not a terminal)",
    "_read_windows": "Windows console, echo disabled",
    "_read_posix": "POSIX terminal, echo disabled",
    "_read_visible": "visible input (no hidden-input support)",
}


def read_secret(
    write: Callable[[str], None] | None = None, *, visible: bool = False,
) -> str:
    """Read one line without echoing it, falling back until a strategy works.

    ``write`` receives user-facing notices (defaults to stderr).  Set
    *visible* to skip hidden input entirely.  Raises ``EOFError`` at end of
    input and ``KeyboardInterrupt`` on Ctrl+C — both propagate so the caller
    can treat them as cancellation.
    """
    emit = write if write is not None else _default_write
    if visible:
        return _read_line()
    for strategy in _STRATEGIES:
        try:
            return strategy(emit)
        except SecureInputUnavailable:
            continue
    raise EOFError  # unreachable: _read_visible never signals unavailable


def describe_terminal() -> list[str]:
    """Report how this terminal handles hidden input (diagnostics only).

    Probes each strategy's availability without consuming any input.
    """
    lines = [
        f"platform      : {sys.platform}",
        f"stdin.isatty(): {sys.stdin.isatty()}",
    ]
    if sys.platform == "win32":
        lines.append(f"console mode  : {_probe_windows_console()}")
    lines.append(f"selected      : {_select_strategy_name()}")
    return lines


def _probe_windows_console() -> str:
    """Describe whether stdin is a real Windows console."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetStdHandle.restype = wintypes.HANDLE
    kernel32.GetStdHandle.argtypes = [wintypes.DWORD]
    kernel32.GetConsoleMode.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD),
    ]
    handle = kernel32.GetStdHandle(_STD_INPUT_HANDLE)
    mode = wintypes.DWORD()
    if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
        return f"available (mode={hex(mode.value)})"
    return f"UNAVAILABLE (error {ctypes.get_last_error()}) - hidden input not possible"


def _select_strategy_name() -> str:
    """Name the strategy that would be used, without reading input."""
    if not sys.stdin.isatty():
        return _STRATEGY_NAMES["_read_piped"]
    if sys.platform == "win32":
        if "available (" in _probe_windows_console():
            return _STRATEGY_NAMES["_read_windows"]
        return _STRATEGY_NAMES["_read_visible"]
    try:
        import termios

        termios.tcgetattr(sys.stdin.fileno())
    except Exception:
        return _STRATEGY_NAMES["_read_visible"]
    return _STRATEGY_NAMES["_read_posix"]


def _default_write(text: str) -> None:
    sys.stderr.write(text)
    sys.stderr.flush()


__all__ = ["SecureInputUnavailable", "describe_terminal", "read_secret"]
