"""Project trust: precedence, persistence, fail-closed paths, and loader gating."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from delta_app.discovery import default_paths, prompt_search_paths, skill_search_paths
from delta_app.instructions import system_prompt
from delta_app.plugins import extension_dirs
from delta_app.trust import (
    Choice,
    StartupCancelled,
    TrustRequest,
    TrustStore,
    TrustStoreError,
    canonical_folder,
    project_inputs_allowed,
    resolve_project_trust,
    scan_project_inputs,
)


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "home"
    root.mkdir()
    monkeypatch.setenv("DELTA_HOME", str(root))
    return root


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A folder carrying every kind of protected input."""
    proj = tmp_path / "work" / "repo"
    (proj / ".delta" / "skills" / "deploy").mkdir(parents=True)
    (proj / ".delta" / "skills" / "deploy" / "SKILL.md").write_text(
        "---\nname: deploy\ndescription: ship it\n---\nsteps", encoding="utf-8")
    (proj / ".delta" / "prompts").mkdir()
    (proj / ".delta" / "prompts" / "review.md").write_text("review", encoding="utf-8")
    (proj / ".delta" / "extensions").mkdir()
    (proj / ".delta" / "extensions" / "hook.py").write_text("x = 1", encoding="utf-8")
    (proj / "AGENTS.md").write_text("OBEY THE REPO", encoding="utf-8")
    return proj


def _answer(choice: Choice | None):
    asked: list[TrustRequest] = []

    async def ask(request: TrustRequest) -> Choice | None:
        asked.append(request)
        return choice

    return ask, asked


def _store(home: Path) -> TrustStore:
    return TrustStore(home / "trust.json")


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def test_detection_counts_inputs_without_reading_them(project: Path) -> None:
    inputs = scan_project_inputs(project)
    assert inputs.counts["instruction file"] == 1
    assert inputs.counts["skill"] == 1
    assert inputs.counts["prompt template"] == 1
    assert inputs.counts["plugin"] == 1


def test_empty_config_folders_are_not_inputs(tmp_path: Path) -> None:
    (tmp_path / ".delta" / "skills").mkdir(parents=True)
    (tmp_path / ".agents").mkdir()
    assert scan_project_inputs(tmp_path).empty


# ---------------------------------------------------------------------------
# Precedence
# ---------------------------------------------------------------------------


async def test_no_inputs_needs_no_decision(tmp_path: Path, home: Path) -> None:
    ask, asked = _answer(Choice.DISTRUST_FOLDER)
    outcome = await resolve_project_trust(tmp_path, ask=ask)
    assert outcome.trusted and not asked
    assert not _store(home).path.exists()


async def test_cli_override_wins_and_is_not_saved(project: Path, home: Path) -> None:
    _store(home).save(canonical_folder(project), "untrusted")
    outcome = await resolve_project_trust(project, override=True)
    assert outcome.trusted and "--approve" in outcome.reason
    assert _store(home).nearest(canonical_folder(project))[1] == "untrusted"


async def test_saved_ancestor_decision_applies(project: Path, home: Path) -> None:
    _store(home).save(canonical_folder(project.parent), "trusted")
    ask, asked = _answer(None)
    outcome = await resolve_project_trust(project, ask=ask)
    assert outcome.trusted and not asked


async def test_child_decline_beats_trusted_parent(project: Path, home: Path) -> None:
    store = _store(home)
    store.save(canonical_folder(project.parent), "trusted")
    store.save(canonical_folder(project), "untrusted")
    assert not (await resolve_project_trust(project)).trusted


async def test_sibling_prefix_is_not_an_ancestor(tmp_path: Path, home: Path) -> None:
    app = tmp_path / "app"
    app2 = tmp_path / "app2"
    for folder in (app, app2):
        folder.mkdir()
        (folder / "AGENTS.md").write_text("x", encoding="utf-8")
    _store(home).save(canonical_folder(app), "trusted")
    assert not (await resolve_project_trust(app2)).trusted


@pytest.mark.parametrize(("default", "expected"), [("always", True), ("never", False)])
async def test_user_default_applies_without_asking(
    project: Path, home: Path, default: str, expected: bool,
) -> None:
    ask, asked = _answer(Choice.TRUST_ONCE)
    outcome = await resolve_project_trust(project, default=default, ask=ask)  # type: ignore[arg-type]
    assert outcome.trusted is expected and not asked


async def test_headless_ask_declines(project: Path, home: Path) -> None:
    outcome = await resolve_project_trust(project, default="ask", ask=None)
    assert not outcome.trusted


async def test_decision_is_cached_per_folder(project: Path, home: Path) -> None:
    ask, asked = _answer(Choice.TRUST_ONCE)
    await resolve_project_trust(project, ask=ask)
    await resolve_project_trust(project, ask=ask)
    assert len(asked) == 1


# ---------------------------------------------------------------------------
# Interactive choices
# ---------------------------------------------------------------------------


async def test_trust_folder_is_remembered(project: Path, home: Path) -> None:
    ask, _ = _answer(Choice.TRUST_FOLDER)
    assert (await resolve_project_trust(project, ask=ask)).trusted
    assert _store(home).nearest(canonical_folder(project)) == (canonical_folder(project), "trusted")


async def test_trust_parent_saves_parent_scope(project: Path, home: Path) -> None:
    store = _store(home)
    store.save(canonical_folder(project), "untrusted")
    ask, asked = _answer(Choice.TRUST_PARENT)
    other = project.parent / "other"
    other.mkdir()
    (other / "AGENTS.md").write_text("x", encoding="utf-8")
    assert (await resolve_project_trust(other, ask=ask)).trusted
    assert asked[0].parent == canonical_folder(project.parent)
    saved = store.decisions()
    assert saved[canonical_folder(project.parent)] == "trusted"
    # A sibling's own decline is untouched and still wins for that sibling.
    assert saved[canonical_folder(project)] == "untrusted"


async def test_run_only_choices_write_nothing(project: Path, home: Path) -> None:
    ask, _ = _answer(Choice.TRUST_ONCE)
    assert (await resolve_project_trust(project, ask=ask)).trusted
    assert not _store(home).path.exists()


async def test_distrust_folder_is_remembered(project: Path, home: Path) -> None:
    ask, _ = _answer(Choice.DISTRUST_FOLDER)
    assert not (await resolve_project_trust(project, ask=ask)).trusted
    assert _store(home).nearest(canonical_folder(project))[1] == "untrusted"


async def test_dismissing_the_question_cancels_startup(project: Path, home: Path) -> None:
    ask, _ = _answer(None)
    with pytest.raises(StartupCancelled):
        await resolve_project_trust(project, ask=ask)


# ---------------------------------------------------------------------------
# Failing closed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("body", [
    "{not json",
    json.dumps({"version": 2, "decisions": []}),
    json.dumps({"version": 1, "decisions": [], "extra": 1}),
    json.dumps({"version": 1, "decisions": [{"path": "relative", "decision": "trusted"}]}),
    json.dumps({"version": 1, "decisions": [{"path": "/x", "decision": "maybe"}]}),
])
def test_malformed_store_is_rejected(home: Path, body: str) -> None:
    _store(home).path.write_text(body, encoding="utf-8")
    with pytest.raises(TrustStoreError):
        _store(home).decisions()


async def test_malformed_store_grants_nothing_implicitly(project: Path, home: Path) -> None:
    _store(home).path.write_text("{broken", encoding="utf-8")
    assert not (await resolve_project_trust(project, default="always")).trusted


async def test_malformed_store_still_allows_run_only_approval(project: Path, home: Path) -> None:
    _store(home).path.write_text("{broken", encoding="utf-8")
    ask, asked = _answer(Choice.TRUST_ONCE)
    assert (await resolve_project_trust(project, ask=ask)).trusted
    assert asked[0].store_problem
    assert _store(home).path.read_text(encoding="utf-8") == "{broken"  # never reset


async def test_unsaved_grant_does_not_grant(project: Path, home: Path) -> None:
    _store(home).path.write_text("{broken", encoding="utf-8")
    ask, _ = _answer(Choice.TRUST_FOLDER)
    outcome = await resolve_project_trust(project, ask=ask)
    assert not outcome.trusted
    assert outcome.notice


# ---------------------------------------------------------------------------
# Loader gating
# ---------------------------------------------------------------------------


async def test_untrusted_folder_loads_only_user_resources(project: Path, home: Path) -> None:
    (home / "AGENTS.md").write_text("user rules", encoding="utf-8")
    await resolve_project_trust(project, override=False)

    paths = default_paths(project=project)
    assert not paths.project_enabled
    assert all(project not in d.parents for d in skill_search_paths(paths))
    assert all(project not in d.parents for d in prompt_search_paths(paths))
    assert all(project not in d.parents for d in extension_dirs(project_dir=project))
    prompt = system_prompt(cwd=str(project))
    assert "user rules" in prompt and "OBEY THE REPO" not in prompt


async def test_trusted_folder_loads_project_resources(project: Path, home: Path) -> None:
    await resolve_project_trust(project, override=True)
    assert project / ".delta" / "skills" in skill_search_paths(default_paths(project=project))
    assert "OBEY THE REPO" in system_prompt(cwd=str(project))
    assert project / ".delta" / "extensions" in extension_dirs(project_dir=project)


async def test_unresolved_folder_with_inputs_is_off_once_trust_is_engaged(
    project: Path, tmp_path: Path, home: Path,
) -> None:
    await resolve_project_trust(tmp_path / "work", override=True)  # engage, other folder
    assert not project_inputs_allowed(project)


def test_trust_is_inert_until_a_frontend_engages_it(project: Path) -> None:
    assert project_inputs_allowed(project)


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------


async def test_dialog_returns_the_selected_choice(project: Path) -> None:
    from delta_app.tui.trust_dialog import TrustDialog

    request = TrustRequest(canonical_folder(project), scan_project_inputs(project))
    app = TrustDialog(request)
    async with app.run_test() as pilot:
        await pilot.press("down", "down", "enter")  # third option: trust for this run
    assert app.return_value is Choice.TRUST_ONCE


async def test_dialog_escape_answers_nothing(project: Path) -> None:
    from delta_app.tui.trust_dialog import TrustDialog

    request = TrustRequest(canonical_folder(project), scan_project_inputs(project))
    app = TrustDialog(request)
    async with app.run_test() as pilot:
        await pilot.press("escape")
    assert app.return_value is None


def test_approve_flags_are_mutually_exclusive() -> None:
    from delta_app.cli.main import _build_run_parser

    parser = _build_run_parser()
    assert parser.parse_args(["-a"]).approve
    assert parser.parse_args(["-na"]).no_approve
    with pytest.raises(SystemExit):
        parser.parse_args(["--approve", "--no-approve"])


async def test_headless_startup_skips_inputs_and_says_so(
    project: Path, home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    import argparse

    from delta_app.runtime import settle_project_trust

    monkeypatch.chdir(project)
    ns = argparse.Namespace(print_mode=True, prompt="hi", approve=False, no_approve=False)
    outcome = await settle_project_trust(ns)
    assert not outcome.trusted
    err = capsys.readouterr().err
    assert "Project inputs not loaded" in err and "--approve" in err


def test_parent_save_drops_the_exact_child_entry(project: Path, home: Path) -> None:
    store = _store(home)
    child, parent = canonical_folder(project), canonical_folder(project.parent)
    store.save(child, "untrusted")
    store.save(parent, "trusted", replacing=child)
    assert store.decisions() == {parent: "trusted"}
