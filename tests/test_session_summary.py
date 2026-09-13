"""Tests for session usage summaries and the ``delta session stats`` export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from delta_app.cli.main import main
from delta_harness.contracts.transcript import HumanEntry, ModelEntry, TextSegment
from delta_harness.contracts.transcript.diagnostics import CostBreakdown, UsageStats
from delta_harness.session.records import SessionRecord, TranscriptRecord
from delta_harness.session.store import serialize_record
from delta_harness.session.summary import summarize_records


def reply(
    *,
    model: str = "claude-opus-5",
    provider: str = "anthropic",
    input_tokens: int = 10,
    output_tokens: int = 5,
    cache_read: int = 2,
    cache_write: int = 1,
    cost: float = 0.25,
) -> TranscriptRecord:
    """A persisted assistant turn carrying usage."""
    return TranscriptRecord(
        message=ModelEntry(
            content=[TextSegment(text="ok")],
            model=model,
            provider=provider,
            usage=UsageStats(
                total_tokens=input_tokens + output_tokens,
                input=input_tokens,
                output=output_tokens,
                cache_read=cache_read,
                cache_write=cache_write,
                cost=CostBreakdown(total=cost),
            ),
        )
    )


def write_session(base: Path, session_id: str, records: list[SessionRecord]) -> None:
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{session_id}.jsonl"
    path.write_text("".join(serialize_record(r) for r in records), encoding="utf-8")


class TestSummarizeRecords:
    def test_sums_usage_across_assistant_turns(self) -> None:
        summary = summarize_records("s1", [reply(), reply(), TranscriptRecord(
            message=HumanEntry(content="hi")
        )])

        assert summary.turns == 2
        assert summary.input_tokens == 20
        assert summary.output_tokens == 10
        assert summary.cache_read_tokens == 4
        assert summary.cache_write_tokens == 2
        assert summary.total_tokens == 30
        assert summary.cost_usd == pytest.approx(0.5)

    def test_reports_the_model_that_finished_the_session(self) -> None:
        summary = summarize_records(
            "s1", [reply(model="a", provider="p1"), reply(model="b", provider="p2")]
        )

        assert (summary.model, summary.provider) == ("b", "p2")

    @pytest.mark.parametrize(
        "records", [[], [TranscriptRecord(message=HumanEntry(content="hi"))]]
    )
    def test_sessions_without_assistant_turns_are_zeroed(
        self, records: list[SessionRecord]
    ) -> None:
        summary = summarize_records("s1", records)

        assert (summary.turns, summary.total_tokens, summary.cost_usd) == (0, 0, 0.0)
        assert summary.model is None and summary.provider is None


class TestStatsCommand:
    def test_prints_json_for_the_named_session(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        write_session(tmp_path, "abc", [reply()])

        assert main(["session", "stats", "abc", "--session-dir", str(tmp_path)]) == 0

        stats = json.loads(capsys.readouterr().out)
        assert stats["session_id"] == "abc"
        assert stats["input_tokens"] == 10
        assert stats["cost_usd"] == pytest.approx(0.25)

    def test_defaults_to_the_most_recent_session(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        write_session(tmp_path, "old", [reply()])
        write_session(tmp_path, "new", [reply(), reply()])
        (tmp_path / "old.jsonl").touch()
        (tmp_path / "new.jsonl").touch()
        import os

        os.utime(tmp_path / "old.jsonl", (1, 1))

        assert main(["session", "stats", "--session-dir", str(tmp_path)]) == 0
        assert json.loads(capsys.readouterr().out)["session_id"] == "new"

    @pytest.mark.parametrize("argv", [["stats"], ["stats", "missing"]])
    def test_missing_session_exits_nonzero(
        self, tmp_path: Path, argv: list[str]
    ) -> None:
        with pytest.raises(SystemExit) as excinfo:
            main(["session", *argv, "--session-dir", str(tmp_path)])

        assert excinfo.value.code == 1
