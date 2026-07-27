"""Tests for delta_app.tools.files (Read, Write, Edit tools)."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from delta_app.tools.files import file_tools, make_edit_tool, make_read_tool, make_write_tool
from delta_harness.contracts.transcript import ImageSegment, TextSegment


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_file(tmp_path: Path) -> Path:
    p = tmp_path / "sample.txt"
    p.write_text("line one\nline two\nline three\nline four\nline five\n", encoding="utf-8")
    return p


@pytest.fixture
def image_file(tmp_path: Path) -> Path:
    p = tmp_path / "icon.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
    return p


@pytest.fixture
def read_tool():
    return make_read_tool()


@pytest.fixture
def write_tool():
    return make_write_tool()


@pytest.fixture
def edit_tool():
    return make_edit_tool()


# ---------------------------------------------------------------------------
# file_tools() registry
# ---------------------------------------------------------------------------

def test_file_tools_returns_three():
    tools = file_tools()
    assert len(tools) == 3
    names = {t.name for t in tools}
    assert names == {"Read", "Write", "Edit"}


def test_read_is_parallel():
    tool = make_read_tool()
    assert tool.parallelism == "parallel"


def test_write_is_sequential():
    tool = make_write_tool()
    assert tool.parallelism == "sequential"


def test_edit_is_sequential():
    tool = make_edit_tool()
    assert tool.parallelism == "sequential"


# ═══════════════════════════════════════════════════════════════════════════
# Read tool
# ═══════════════════════════════════════════════════════════════════════════

class TestReadTool:
    @pytest.mark.asyncio
    async def test_read_text_file(self, read_tool, sample_file):
        result = await read_tool.execute("c1", {"file_path": str(sample_file)})
        text = result.text
        assert "line one" in text
        assert "line five" in text
        # Line numbers present
        assert "1\t" in text

    @pytest.mark.asyncio
    async def test_read_with_line_numbers(self, read_tool, sample_file):
        result = await read_tool.execute("c1", {"file_path": str(sample_file)})
        text = result.text
        lines = text.strip().split("\n")
        # First line should start with "1"
        assert lines[0].lstrip().startswith("1")

    @pytest.mark.asyncio
    async def test_read_with_offset(self, read_tool, sample_file):
        result = await read_tool.execute("c1", {
            "file_path": str(sample_file),
            "offset": 3,
        })
        text = result.text
        assert "line three" in text
        assert "line one" not in text

    @pytest.mark.asyncio
    async def test_read_with_limit(self, read_tool, sample_file):
        result = await read_tool.execute("c1", {
            "file_path": str(sample_file),
            "limit": 2,
        })
        text = result.text
        assert "line one" in text
        assert "line two" in text
        assert "line three" not in text

    @pytest.mark.asyncio
    async def test_read_with_offset_and_limit(self, read_tool, sample_file):
        result = await read_tool.execute("c1", {
            "file_path": str(sample_file),
            "offset": 2,
            "limit": 2,
        })
        text = result.text
        assert "line two" in text
        assert "line three" in text
        assert "line one" not in text
        assert "line four" not in text

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self, read_tool, tmp_path):
        result = await read_tool.execute("c1", {
            "file_path": str(tmp_path / "nope.txt"),
        })
        assert "not found" in result.text.lower()

    @pytest.mark.asyncio
    async def test_read_directory_rejected(self, read_tool, tmp_path):
        result = await read_tool.execute("c1", {
            "file_path": str(tmp_path),
        })
        assert "not a regular file" in result.text.lower()

    @pytest.mark.asyncio
    async def test_read_relative_path_rejected(self, read_tool):
        result = await read_tool.execute("c1", {
            "file_path": "relative/path.txt",
        })
        assert "absolute" in result.text.lower()

    @pytest.mark.asyncio
    async def test_read_empty_file(self, read_tool, tmp_path):
        empty = tmp_path / "empty.txt"
        empty.write_text("", encoding="utf-8")
        result = await read_tool.execute("c1", {"file_path": str(empty)})
        assert "empty" in result.text.lower()

    @pytest.mark.asyncio
    async def test_read_image_returns_image_segment(self, read_tool, image_file):
        result = await read_tool.execute("c1", {"file_path": str(image_file)})
        assert len(result.content) == 1
        seg = result.content[0]
        assert isinstance(seg, ImageSegment)
        assert seg.mime_type == "image/png"
        # Verify base64 round-trips
        decoded = base64.b64decode(seg.data)
        assert decoded[:4] == b"\x89PNG"

    @pytest.mark.asyncio
    async def test_read_binary_rejected(self, read_tool, tmp_path):
        binary = tmp_path / "data.zip"
        binary.write_bytes(b"\x00" * 10)
        result = await read_tool.execute("c1", {"file_path": str(binary)})
        assert "binary" in result.text.lower()

    @pytest.mark.asyncio
    async def test_read_missing_file_path_arg(self, read_tool):
        result = await read_tool.execute("c1", {})
        assert "file_path" in result.text

    def test_format_read_call(self, read_tool):
        rendered = read_tool.format_call({"file_path": "/foo/bar.py"})
        assert "/foo/bar.py" in rendered

    def test_format_read_call_with_offset(self, read_tool):
        rendered = read_tool.format_call({
            "file_path": "/foo/bar.py",
            "offset": 10,
        })
        assert "line 10" in rendered


# ═══════════════════════════════════════════════════════════════════════════
# Write tool
# ═══════════════════════════════════════════════════════════════════════════

class TestWriteTool:
    @pytest.mark.asyncio
    async def test_write_new_file(self, write_tool, tmp_path):
        target = tmp_path / "output.txt"
        result = await write_tool.execute("c1", {
            "file_path": str(target),
            "content": "hello\nworld\n",
        })
        assert "created" in result.text.lower()
        assert target.read_text(encoding="utf-8") == "hello\nworld\n"

    @pytest.mark.asyncio
    async def test_write_overwrites_existing(self, write_tool, sample_file):
        result = await write_tool.execute("c1", {
            "file_path": str(sample_file),
            "content": "replaced",
        })
        assert "updated" in result.text.lower()
        assert sample_file.read_text(encoding="utf-8") == "replaced"

    @pytest.mark.asyncio
    async def test_write_creates_parent_dirs(self, write_tool, tmp_path):
        target = tmp_path / "a" / "b" / "c" / "deep.txt"
        result = await write_tool.execute("c1", {
            "file_path": str(target),
            "content": "deep content",
        })
        assert "created" in result.text.lower()
        assert target.exists()
        assert target.read_text(encoding="utf-8") == "deep content"

    @pytest.mark.asyncio
    async def test_write_relative_path_rejected(self, write_tool):
        result = await write_tool.execute("c1", {
            "file_path": "relative.txt",
            "content": "nope",
        })
        assert "absolute" in result.text.lower()

    @pytest.mark.asyncio
    async def test_write_empty_path_rejected(self, write_tool):
        result = await write_tool.execute("c1", {
            "file_path": "   ",
            "content": "nope",
        })
        assert "empty" in result.text.lower()

    @pytest.mark.asyncio
    async def test_write_reports_line_count(self, write_tool, tmp_path):
        target = tmp_path / "counted.txt"
        result = await write_tool.execute("c1", {
            "file_path": str(target),
            "content": "a\nb\nc",
        })
        assert "3 lines" in result.text

    def test_format_write_call(self, write_tool):
        rendered = write_tool.format_call({
            "file_path": "/foo/bar.py",
            "content": "a\nb\nc",
        })
        assert "/foo/bar.py" in rendered
        assert "3 lines" in rendered


# ═══════════════════════════════════════════════════════════════════════════
# Edit tool
# ═══════════════════════════════════════════════════════════════════════════

class TestEditTool:
    @pytest.mark.asyncio
    async def test_edit_single_match(self, edit_tool, sample_file):
        result = await edit_tool.execute("c1", {
            "file_path": str(sample_file),
            "old_string": "line two",
            "new_string": "line TWO",
        })
        assert "1 replacement" in result.text
        content = sample_file.read_text(encoding="utf-8")
        assert "line TWO" in content
        assert "line two" not in content

    @pytest.mark.asyncio
    async def test_edit_multiline_match(self, edit_tool, sample_file):
        result = await edit_tool.execute("c1", {
            "file_path": str(sample_file),
            "old_string": "line two\nline three",
            "new_string": "REPLACED",
        })
        assert "1 replacement" in result.text
        content = sample_file.read_text(encoding="utf-8")
        assert "REPLACED" in content
        assert "line two" not in content

    @pytest.mark.asyncio
    async def test_edit_rejects_ambiguous_match(self, edit_tool, tmp_path):
        f = tmp_path / "dup.txt"
        f.write_text("foo bar\nfoo baz\n", encoding="utf-8")
        result = await edit_tool.execute("c1", {
            "file_path": str(f),
            "old_string": "foo",
            "new_string": "qux",
        })
        assert "2 locations" in result.text

    @pytest.mark.asyncio
    async def test_edit_replace_all(self, edit_tool, tmp_path):
        f = tmp_path / "dup.txt"
        f.write_text("foo bar\nfoo baz\n", encoding="utf-8")
        result = await edit_tool.execute("c1", {
            "file_path": str(f),
            "old_string": "foo",
            "new_string": "qux",
            "replace_all": True,
        })
        assert "2 replacement" in result.text
        content = f.read_text(encoding="utf-8")
        assert content == "qux bar\nqux baz\n"

    @pytest.mark.asyncio
    async def test_edit_no_match_gives_hint(self, edit_tool, sample_file):
        result = await edit_tool.execute("c1", {
            "file_path": str(sample_file),
            "old_string": "nonexistent text",
            "new_string": "whatever",
        })
        assert "not found" in result.text.lower()

    @pytest.mark.asyncio
    async def test_edit_no_match_partial_hint(self, edit_tool, sample_file):
        result = await edit_tool.execute("c1", {
            "file_path": str(sample_file),
            "old_string": "line two\nwrong continuation",
            "new_string": "whatever",
        })
        text = result.text.lower()
        assert "not found" in text
        # Should show a partial match hint for "line two"
        assert "partial" in text or "line two" in text

    @pytest.mark.asyncio
    async def test_edit_identical_strings_rejected(self, edit_tool, sample_file):
        result = await edit_tool.execute("c1", {
            "file_path": str(sample_file),
            "old_string": "line one",
            "new_string": "line one",
        })
        assert "identical" in result.text.lower()

    @pytest.mark.asyncio
    async def test_edit_nonexistent_file(self, edit_tool, tmp_path):
        result = await edit_tool.execute("c1", {
            "file_path": str(tmp_path / "ghost.txt"),
            "old_string": "a",
            "new_string": "b",
        })
        assert "not found" in result.text.lower()

    @pytest.mark.asyncio
    async def test_edit_relative_path_rejected(self, edit_tool):
        result = await edit_tool.execute("c1", {
            "file_path": "relative.txt",
            "old_string": "a",
            "new_string": "b",
        })
        assert "absolute" in result.text.lower()

    @pytest.mark.asyncio
    async def test_edit_preserves_surrounding_content(self, edit_tool, sample_file):
        await edit_tool.execute("c1", {
            "file_path": str(sample_file),
            "old_string": "line three",
            "new_string": "LINE_THREE",
        })
        content = sample_file.read_text(encoding="utf-8")
        assert content == "line one\nline two\nLINE_THREE\nline four\nline five\n"

    @pytest.mark.asyncio
    async def test_edit_insert_at_beginning(self, edit_tool, tmp_path):
        f = tmp_path / "ins.txt"
        f.write_text("hello world", encoding="utf-8")
        await edit_tool.execute("c1", {
            "file_path": str(f),
            "old_string": "hello",
            "new_string": "goodbye cruel",
        })
        assert f.read_text(encoding="utf-8") == "goodbye cruel world"

    @pytest.mark.asyncio
    async def test_edit_delete_text(self, edit_tool, tmp_path):
        f = tmp_path / "del.txt"
        f.write_text("keep this remove this keep that", encoding="utf-8")
        await edit_tool.execute("c1", {
            "file_path": str(f),
            "old_string": " remove this",
            "new_string": "",
        })
        assert f.read_text(encoding="utf-8") == "keep this keep that"

    def test_format_edit_call(self, edit_tool):
        rendered = edit_tool.format_call({"file_path": "/foo/bar.py"})
        assert "/foo/bar.py" in rendered


# ═══════════════════════════════════════════════════════════════════════════
# Shared helpers (via _shared.py)
# ═══════════════════════════════════════════════════════════════════════════

class TestSharedHelpers:
    def test_truncate_output_short(self):
        from delta_app.tools._shared import truncate_output
        text = "a\nb\nc"
        assert truncate_output(text, max_lines=10) == text

    def test_truncate_output_long(self):
        from delta_app.tools._shared import truncate_output
        text = "\n".join(f"line {i}" for i in range(100))
        result = truncate_output(text, max_lines=5)
        assert "line 0" in result
        assert "line 4" in result
        assert "line 5" not in result
        assert "truncated" in result.lower()

    def test_format_with_line_numbers(self):
        from delta_app.tools._shared import format_with_line_numbers
        text = "alpha\nbeta\ngamma"
        result = format_with_line_numbers(text, start=1)
        lines = result.split("\n")
        assert lines[0].strip().startswith("1")
        assert "alpha" in lines[0]
        assert lines[2].strip().startswith("3")
        assert "gamma" in lines[2]

    def test_format_with_line_numbers_offset(self):
        from delta_app.tools._shared import format_with_line_numbers
        result = format_with_line_numbers("a\nb", start=42)
        assert "42" in result
        assert "43" in result

    def test_validate_file_path_absolute(self, tmp_path):
        from delta_app.tools._shared import validate_file_path
        p = validate_file_path(str(tmp_path / "test.txt"))
        assert p.is_absolute()

    def test_validate_file_path_relative_rejected(self):
        from delta_app.tools._shared import validate_file_path
        with pytest.raises(ValueError, match="absolute"):
            validate_file_path("relative/path.txt")

    def test_validate_file_path_empty_rejected(self):
        from delta_app.tools._shared import validate_file_path
        with pytest.raises(ValueError, match="empty"):
            validate_file_path("  ")

    def test_require_str_present(self):
        from delta_app.tools._shared import require_str
        assert require_str({"key": "val"}, "key") == "val"

    def test_require_str_missing(self):
        from delta_app.tools._shared import require_str
        with pytest.raises(ValueError):
            require_str({}, "key")

    def test_optional_int(self):
        from delta_app.tools._shared import optional_int
        assert optional_int({"n": 42}, "n") == 42
        assert optional_int({"n": 3.0}, "n") == 3
        assert optional_int({}, "n") is None
        assert optional_int({"n": "nope"}, "n") is None

    def test_optional_bool(self):
        from delta_app.tools._shared import optional_bool
        assert optional_bool({"b": True}, "b") is True
        assert optional_bool({"b": False}, "b") is False
        assert optional_bool({}, "b") is None

    def test_fault_outcome(self):
        from delta_app.tools._shared import fault
        result = fault("something broke")
        assert result.text == "something broke"

    def test_text_outcome(self):
        from delta_app.tools._shared import text_outcome
        result = text_outcome("hello")
        assert result.text == "hello"
