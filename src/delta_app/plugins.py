"""Local extensions: file-based discovery, dynamic import, and hot reload.

Extensions are plain Python files the user (or the agent) drops into an
extensions directory — no packaging or reinstall step. Each file exposes a
callable named ``register`` which is invoked once at load time.

Discovery order (later entries win on name collision):

1. ``<delta_home>/extensions``       — user-level, e.g. ``~/.delta/extensions``
2. ``<project>/.delta/extensions``   — project-level
3. any explicit ``extra`` paths

Within a directory, an extension is either:

- a loose ``*.py`` file (``my_tool.py`` -> name ``my_tool``), or
- a subdirectory containing ``extension.py`` (``my_pkg/extension.py`` -> ``my_pkg``)

Files and directories whose names start with ``_`` or ``.`` are skipped.

**Why file-based rather than entry points:** ``importlib.metadata.entry_points``
resolves what was registered at *install* time, so a newly written file can never
be picked up. File discovery + :func:`unload_extensions` is what makes runtime
reload — and therefore agent-authored extensions — possible at all.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from importlib import invalidate_caches
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

DELTA_HOME_ENV = "DELTA_HOME"
EXTENSIONS_DIR_ENV = "DELTA_EXTENSIONS_DIR"

#: Attribute an extension module must expose: a zero-arg callable.
ENTRY_ATTRIBUTE = "register"

#: Prefix for synthetic module names, so extensions never collide with real packages.
_MODULE_PREFIX = "delta_ext_"

#: Module names this process has imported as extensions (used by unload).
_imported: set[str] = set()


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LoadedExtension:
    """A successfully imported and registered extension."""

    name: str
    path: Path
    module: ModuleType
    value: object | None = None


@dataclass(frozen=True, slots=True)
class ExtensionFailure:
    """An extension that could not be imported or registered."""

    name: str
    path: Path
    error: str


@dataclass(frozen=True, slots=True)
class LoadResult:
    """Outcome of one discovery + load pass."""

    loaded: tuple[LoadedExtension, ...] = ()
    failures: tuple[ExtensionFailure, ...] = ()
    searched: tuple[Path, ...] = ()

    @property
    def count(self) -> int:
        """Number of extensions successfully loaded."""
        return len(self.loaded)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def delta_home() -> Path:
    """Return the Delta home directory (``$DELTA_HOME`` or ``~/.delta``).

    Delegates to ``resources.default_paths().home`` for canonical resolution.
    """
    from delta_app.discovery import default_paths

    return default_paths().home


def extension_dirs(
    *,
    project_dir: Path | str | None = None,
    include_project: bool = True,
    extra: tuple[Path | str, ...] = (),
) -> tuple[Path, ...]:
    """Return the extension search directories, in precedence order.

    ``$DELTA_EXTENSIONS_DIR`` overrides the user-level directory when set.
    """
    dirs: list[Path] = []

    override = os.environ.get(EXTENSIONS_DIR_ENV)
    dirs.append(Path(override) if override else delta_home() / "extensions")

    if include_project:
        root = Path(project_dir) if project_dir is not None else Path.cwd()
        dirs.append(root / ".delta" / "extensions")

    dirs.extend(Path(path) for path in extra)
    return tuple(dirs)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _is_hidden(name: str) -> bool:
    return name.startswith("_") or name.startswith(".")


def discover_extension_files(dirs: tuple[Path, ...]) -> tuple[tuple[str, Path], ...]:
    """Find ``(name, path)`` pairs for every extension in *dirs*.

    Later directories override earlier ones when names collide.
    """
    found: dict[str, Path] = {}

    for directory in dirs:
        if not directory.is_dir():
            continue
        for entry in sorted(directory.iterdir()):
            if _is_hidden(entry.name):
                continue
            if entry.is_file() and entry.suffix == ".py":
                found[entry.stem] = entry
            elif entry.is_dir():
                candidate = entry / "extension.py"
                if candidate.is_file():
                    found[entry.name] = candidate

    return tuple(found.items())


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _import_module(name: str, path: Path) -> ModuleType:
    """Import *path* as a module under a synthetic, collision-free name.

    The source is compiled directly rather than run through the spec's loader,
    which deliberately bypasses Python's bytecode cache. ``SourceFileLoader``
    validates a cached ``.pyc`` on *(mtime, size)* alone, so an edit that keeps
    the file the same length within the same second is treated as unchanged and
    the stale bytecode runs instead — a reload that silently does nothing. Agent
    -authored extensions hit that case easily, so always execute fresh source.
    """
    module_name = f"{_MODULE_PREFIX}{name}"

    # A package-style extension (dir/extension.py) needs its directory on the
    # search path so sibling relative imports resolve.
    search_locations = [str(path.parent)] if path.name == "extension.py" else None

    spec = spec_from_file_location(
        module_name, path, submodule_search_locations=search_locations
    )
    if spec is None:
        raise ImportError(f"cannot build a module spec for {path}")

    module = module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        code = compile(path.read_text(encoding="utf-8"), str(path), "exec")
        exec(code, module.__dict__)  # noqa: S102 - extensions are code by design
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    _imported.add(module_name)
    return module


def load_extensions(
    *,
    project_dir: Path | str | None = None,
    include_project: bool = True,
    extra: tuple[Path | str, ...] = (),
) -> LoadResult:
    """Discover, import, and register every extension found on disk.

    A failing extension is collected in :attr:`LoadResult.failures` and never
    propagates — one bad file must not stop the session from starting.
    """
    dirs = extension_dirs(
        project_dir=project_dir, include_project=include_project, extra=extra
    )
    loaded: list[LoadedExtension] = []
    failures: list[ExtensionFailure] = []

    for name, path in discover_extension_files(dirs):
        try:
            module = _import_module(name, path)
        except BaseException as exc:  # noqa: BLE001 - extensions are untrusted
            failures.append(ExtensionFailure(name=name, path=path, error=str(exc)))
            continue

        entry = getattr(module, ENTRY_ATTRIBUTE, None)
        if entry is None:
            failures.append(
                ExtensionFailure(
                    name=name,
                    path=path,
                    error=f"no {ENTRY_ATTRIBUTE}() found in {path.name}",
                )
            )
            continue
        if not callable(entry):
            failures.append(
                ExtensionFailure(
                    name=name, path=path, error=f"{ENTRY_ATTRIBUTE} is not callable"
                )
            )
            continue

        try:
            value = entry()
        except BaseException as exc:  # noqa: BLE001 - extensions are untrusted
            failures.append(
                ExtensionFailure(
                    name=name, path=path, error=f"{ENTRY_ATTRIBUTE}() raised: {exc}"
                )
            )
            continue

        loaded.append(
            LoadedExtension(name=name, path=path, module=module, value=value)
        )

    return LoadResult(
        loaded=tuple(loaded), failures=tuple(failures), searched=dirs
    )


# ---------------------------------------------------------------------------
# Reload
# ---------------------------------------------------------------------------


def unload_extensions() -> int:
    """Drop every previously imported extension from ``sys.modules``.

    Without this, re-importing a changed file silently yields the cached old
    module and a reload appears to do nothing. Returns how many were removed.
    """
    removed = 0
    for module_name in tuple(_imported):
        if sys.modules.pop(module_name, None) is not None:
            removed += 1
        _imported.discard(module_name)
    invalidate_caches()
    return removed


def reload_extensions(
    *,
    project_dir: Path | str | None = None,
    include_project: bool = True,
    extra: tuple[Path | str, ...] = (),
) -> LoadResult:
    """Unload, then re-discover and re-import — picks up newly written files."""
    unload_extensions()
    return load_extensions(
        project_dir=project_dir, include_project=include_project, extra=extra
    )


__all__ = [
    "DELTA_HOME_ENV",
    "ENTRY_ATTRIBUTE",
    "EXTENSIONS_DIR_ENV",
    "ExtensionFailure",
    "LoadResult",
    "LoadedExtension",
    "delta_home",
    "discover_extension_files",
    "extension_dirs",
    "load_extensions",
    "reload_extensions",
    "unload_extensions",
]
