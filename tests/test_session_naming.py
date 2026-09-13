"""Session titles: provisional naming, refinement, and no raw ids."""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, str(__file__.rsplit("test_session_naming.py", 1)[0]))

from _session_helpers import _make_session, _text_turn  # noqa: E402

from delta_app.conversation import summarize_prompt  # noqa: E402
from delta_app.tui.adapter import SessionBridge, SessionEntry  # noqa: E402


class TestSummarizePrompt:
    @pytest.mark.parametrize("raw,expected", [
        ("create fibonacci.py", "create fibonacci.py"),
        ("  create   fibonacci.py  ", "create fibonacci.py"),
        ("multi\nline\nprompt", "multi line prompt"),
        ("/compact", "compact"),
        ("", ""),
        ("   ", ""),
    ])
    def test_basic_forms(self, raw, expected):
        assert summarize_prompt(raw) == expected

    def test_long_prompt_clipped_on_word_boundary(self):
        raw = "refactor the entire provider transport layer and add retries everywhere"
        out = summarize_prompt(raw)
        assert len(out) <= 49
        assert out.endswith("…")
        assert not out.rstrip("…").endswith(" ")

    def test_clip_does_not_split_a_word(self):
        out = summarize_prompt("alpha beta gamma delta epsilon zeta eta theta iota")
        assert all(w in "alpha beta gamma delta epsilon zeta eta theta iota"
                   for w in out.rstrip("…").split())


class TestProvisionalTitle:
    async def test_title_set_on_first_submit(self, tmp_path):
        session = await _make_session(tmp_path, [_text_turn(["ok"])])
        assert session.title is None
        async for _ in session.submit("create fibonacci.py"):
            pass
        assert session.title
        assert session.session_id not in session.title
        await session.shutdown()

    async def test_title_lands_in_catalog(self, tmp_path):
        session = await _make_session(tmp_path, [_text_turn(["ok"]), _text_turn(["t"])])
        async for _ in session.submit("build the parser"):
            pass
        meta = session.catalog.get(session.session_id)
        await session.shutdown()
        assert meta is not None
        assert meta.title
        assert meta.title != session.session_id

    async def test_never_titles_with_raw_session_id(self, tmp_path):
        # Second turn is the naming call; an empty reply must not fall back
        # to the session id.
        session = await _make_session(
            tmp_path, [_text_turn(["done"]), _text_turn([""])],
        )
        async for _ in session.submit("write a haiku"):
            pass
        title = session.title
        await session.shutdown()
        assert title
        assert title != session.session_id
        assert "write a haiku" in title or title == "write a haiku"

    async def test_second_prompt_does_not_overwrite_title(self, tmp_path):
        session = await _make_session(tmp_path, [_text_turn(["a"])] * 6)
        async for _ in session.submit("first topic"):
            pass
        first = session.title
        async for _ in session.submit("second topic"):
            pass
        after = session.title
        await session.shutdown()
        assert first is not None
        assert "second topic" not in (after or "")


class TestSidebarLabels:
    def test_untitled_label_has_no_hex_id(self):
        entry = SessionEntry(
            session_id="220b6faf1234", title=None,
            updated_at=0.0, is_active=False,
        )
        assert "220b6faf" not in entry.label
        assert entry.label == "Untitled session"

    def test_titled_label_uses_title(self):
        entry = SessionEntry(
            session_id="220b6faf1234", title="Harness review",
            updated_at=0.0, is_active=True,
        )
        assert entry.label == "Harness review"

    async def test_bridge_lists_named_session(self, tmp_path):
        session = await _make_session(tmp_path, [_text_turn(["ok"])] * 4)
        bridge = SessionBridge(session)
        async for _ in session.submit("investigate the loop"):
            pass
        labels = [e.label for e in bridge.sessions()]
        await bridge.shutdown()
        assert labels
        assert all(session.session_id[:8] not in label for label in labels)
