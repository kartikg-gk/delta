"""Tests for file-based extension discovery, loading, and hot reload."""

from __future__ import annotations

from pathlib import Path

import pytest

from delta_app.extensions import (
    EXTENSIONS_DIR_ENV,
    discover_extension_files,
    extension_dirs,
    load_extensions,
    reload_extensions,
    unload_extensions,
)


@pytest.fixture
def ext_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An isolated extensions directory wired up via the env override."""
    directory = tmp_path / "extensions"
    directory.mkdir()
    monkeypatch.setenv(EXTENSIONS_DIR_ENV, str(directory))
    yield directory
    unload_extensions()


def _load(**kwargs):
    return load_extensions(include_project=False, **kwargs)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


class TestDiscovery:
    def test_finds_loose_py_file(self, ext_dir: Path) -> None:
        (ext_dir / "alpha.py").write_text("def register():\n    return 1\n")
        found = discover_extension_files((ext_dir,))
        assert [name for name, _ in found] == ["alpha"]

    def test_finds_package_style_extension(self, ext_dir: Path) -> None:
        pkg = ext_dir / "beta"
        pkg.mkdir()
        (pkg / "extension.py").write_text("def register():\n    return 2\n")
        found = discover_extension_files((ext_dir,))
        assert [name for name, _ in found] == ["beta"]

    def test_skips_underscore_and_dot_files(self, ext_dir: Path) -> None:
        (ext_dir / "_private.py").write_text("def register():\n    return 0\n")
        (ext_dir / ".hidden.py").write_text("def register():\n    return 0\n")
        (ext_dir / "real.py").write_text("def register():\n    return 1\n")
        assert [name for name, _ in discover_extension_files((ext_dir,))] == ["real"]

    def test_missing_directory_is_not_an_error(self, tmp_path: Path) -> None:
        assert discover_extension_files((tmp_path / "nope",)) == ()

    def test_later_directory_wins_on_name_collision(self, tmp_path: Path) -> None:
        first, second = tmp_path / "a", tmp_path / "b"
        first.mkdir()
        second.mkdir()
        (first / "dup.py").write_text("def register():\n    return 'first'\n")
        (second / "dup.py").write_text("def register():\n    return 'second'\n")
        found = dict(discover_extension_files((first, second)))
        assert found["dup"].parent == second

    def test_env_override_replaces_user_dir(self, ext_dir: Path) -> None:
        assert extension_dirs(include_project=False) == (ext_dir,)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


class TestLoading:
    def test_register_is_called_and_value_captured(self, ext_dir: Path) -> None:
        (ext_dir / "alpha.py").write_text("def register():\n    return 'ran'\n")
        result = _load()
        assert result.count == 1
        assert result.loaded[0].name == "alpha"
        assert result.loaded[0].value == "ran"

    def test_missing_register_is_a_failure(self, ext_dir: Path) -> None:
        (ext_dir / "noentry.py").write_text("x = 1\n")
        result = _load()
        assert result.count == 0
        assert "no register()" in result.failures[0].error

    def test_non_callable_register_is_a_failure(self, ext_dir: Path) -> None:
        (ext_dir / "bad.py").write_text("register = 42\n")
        result = _load()
        assert "not callable" in result.failures[0].error

    def test_raising_register_is_isolated(self, ext_dir: Path) -> None:
        (ext_dir / "boom.py").write_text(
            "def register():\n    raise RuntimeError('boom')\n"
        )
        (ext_dir / "fine.py").write_text("def register():\n    return 'ok'\n")
        result = _load()
        # the good one still loads — one bad file must not stop the session
        assert [e.name for e in result.loaded] == ["fine"]
        assert "boom" in result.failures[0].error

    def test_import_error_is_isolated(self, ext_dir: Path) -> None:
        (ext_dir / "syntax.py").write_text("def register(:\n")
        result = _load()
        assert result.count == 0
        assert result.failures[0].name == "syntax"


# ---------------------------------------------------------------------------
# Hot reload
# ---------------------------------------------------------------------------


class TestHotReload:
    def test_same_size_edit_is_picked_up(self, ext_dir: Path) -> None:
        """Regression: CPython validates cached bytecode on (mtime, size) only.

        A same-length edit within the same second looks unchanged, so the loader
        would serve the stale .pyc and the reload would silently do nothing.
        """
        path = ext_dir / "tool.py"
        path.write_text('MARK = "v1"\n\ndef register():\n    return MARK\n')
        first = _load().loaded[0].value

        path.write_text('MARK = "v2"\n\ndef register():\n    return MARK\n')
        second = reload_extensions(include_project=False).loaded[0].value

        assert (first, second) == ("v1", "v2")

    def test_newly_written_file_is_discovered(self, ext_dir: Path) -> None:
        """The self-extension case: a file that did not exist at startup."""
        (ext_dir / "existing.py").write_text("def register():\n    return 1\n")
        assert _load().count == 1

        (ext_dir / "authored.py").write_text("def register():\n    return 2\n")
        names = sorted(e.name for e in reload_extensions(include_project=False).loaded)
        assert names == ["authored", "existing"]

    def test_deleted_file_disappears_after_reload(self, ext_dir: Path) -> None:
        path = ext_dir / "temp.py"
        path.write_text("def register():\n    return 1\n")
        assert _load().count == 1

        path.unlink()
        assert reload_extensions(include_project=False).count == 0

    def test_unload_reports_removed_count(self, ext_dir: Path) -> None:
        (ext_dir / "a.py").write_text("def register():\n    return 1\n")
        (ext_dir / "b.py").write_text("def register():\n    return 2\n")
        _load()
        assert unload_extensions() == 2
        assert unload_extensions() == 0
