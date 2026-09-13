"""Tests for context/budget.py — token accounting, thresholds, compaction planning."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from delta_app.context.budget import (
    CHARS_PER_TOKEN,
    DEFAULT_SUMMARY_SYSTEM,
    DEFAULT_WINDOW,
    TOOL_SCHEMA_BASELINE,
    CompactionPlan,
    ContextEstimate,
    ContextLimits,
    apply_compaction,
    build_summary_prompts,
    estimate_context,
    estimate_entry_tokens,
    estimate_system_tokens,
    estimate_tool_tokens,
    estimate_transcript_tokens,
    exceeds_threshold,
    plan_compaction,
    resolve_window,
    serialize_for_summary,
)
from delta_harness.contracts.transcript import (
    CallBlock,
    CostBreakdown,
    HumanEntry,
    ModelEntry,
    PruneSummaryEntry,
    ShellResultEntry,
    TextSegment,
    ToolOutcomeEntry,
    UsageStats,
)
from delta_harness.contracts.values import JValue

# ── fixtures ────────────────────────────────────────────────────────────


def _human(text: str) -> HumanEntry:
    return HumanEntry(content=text)


def _model(
    text: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    calls: list[CallBlock] | None = None,
) -> ModelEntry:
    content: list[TextSegment | CallBlock] = [TextSegment(text=text)] if text else []
    if calls:
        content.extend(calls)
    return ModelEntry(
        content=content,
        usage=UsageStats(
            input=input_tokens,
            output=output_tokens,
            cost=CostBreakdown(),
        ),
    )


def _tool_result(text: str, name: str = "bash") -> ToolOutcomeEntry:
    return ToolOutcomeEntry(
        tool_call_id="tc_1",
        tool_name=name,
        content=text,
    )


def _summary(text: str, tokens_before: int = 1000) -> PruneSummaryEntry:
    return PruneSummaryEntry(summary=text, tokens_before=tokens_before)


def _call(name: str = "bash", args: dict[str, JValue] | None = None) -> CallBlock:
    return CallBlock(name=name, id="tc_1", arguments=args or {})


def _make_tool_spec(
    name: str = "test_tool",
    desc: str = "A test tool",
    params: dict[str, JValue] | None = None,
) -> object:
    """Minimal duck-typed stand-in for ToolSpec."""

    class _FakeSpec:
        def __init__(self) -> None:
            self.name = name
            self.description = desc
            self.parameters: Mapping[str, JValue] = params or {"type": "object"}
            self.prompt_snippet: str | None = None
            self.prompt_guidelines: tuple[str, ...] = ()

    return _FakeSpec()


# ── ContextLimits ───────────────────────────────────────────────────────


class TestContextLimits:
    def test_defaults(self) -> None:
        cfg = ContextLimits()
        assert cfg.window == DEFAULT_WINDOW
        assert cfg.compaction_threshold == 0.75
        assert cfg.keep_recent == 4
        assert cfg.min_entries == 6
        assert cfg.max_summary_chars == 50_000

    def test_frozen(self) -> None:
        cfg = ContextLimits()
        with pytest.raises(AttributeError):
            cfg.window = 1  # type: ignore[misc]

    def test_custom_values(self) -> None:
        cfg = ContextLimits(window=200_000, compaction_threshold=0.8, keep_recent=6)
        assert cfg.window == 200_000
        assert cfg.compaction_threshold == 0.8
        assert cfg.keep_recent == 6

    def test_custom_summary_system(self) -> None:
        cfg = ContextLimits(summary_system="Be very brief.")
        assert cfg.summary_system == "Be very brief."


# ── ContextEstimate ─────────────────────────────────────────────────────


class TestContextEstimate:
    def test_utilization(self) -> None:
        est = ContextEstimate(used=50_000, system=10_000, messages=40_000,
                              limit=100_000, threshold=0.75)
        assert est.utilization == pytest.approx(0.5)

    def test_utilization_zero_limit(self) -> None:
        est = ContextEstimate(used=100, system=0, messages=100, limit=0, threshold=0.75)
        assert est.utilization == 0.0

    def test_headroom(self) -> None:
        est = ContextEstimate(used=80_000, system=0, messages=80_000,
                              limit=100_000, threshold=0.75)
        assert est.headroom == 20_000

    def test_headroom_over_limit(self) -> None:
        est = ContextEstimate(used=120_000, system=0, messages=120_000,
                              limit=100_000, threshold=0.75)
        assert est.headroom == 0

    def test_needs_compaction_true(self) -> None:
        est = ContextEstimate(used=80_000, system=0, messages=80_000,
                              limit=100_000, threshold=0.75)
        assert est.needs_compaction is True

    def test_needs_compaction_false(self) -> None:
        est = ContextEstimate(used=50_000, system=0, messages=50_000,
                              limit=100_000, threshold=0.75)
        assert est.needs_compaction is False

    def test_needs_compaction_at_boundary(self) -> None:
        est = ContextEstimate(used=75_000, system=0, messages=75_000,
                              limit=100_000, threshold=0.75)
        assert est.needs_compaction is False  # not strictly >

    def test_frozen(self) -> None:
        est = ContextEstimate(used=100, system=0, messages=100, limit=1000, threshold=0.75)
        with pytest.raises(AttributeError):
            est.used = 200  # type: ignore[misc]


# ── resolve_window ──────────────────────────────────────────────────────


class TestResolveWindow:
    def test_known_model(self) -> None:
        assert resolve_window("claude-opus-4-6") == 200_000

    def test_unknown_model(self) -> None:
        assert resolve_window("unknown-model-xyz") == DEFAULT_WINDOW

    def test_override(self) -> None:
        assert resolve_window("claude-opus-4-6", override=300_000) == 300_000

    def test_override_for_unknown(self) -> None:
        assert resolve_window("unknown", override=64_000) == 64_000


# ── estimate_entry_tokens ───────────────────────────────────────────────


class TestEstimateEntryTokens:
    def test_human_entry(self) -> None:
        entry = _human("Hello world")
        tokens = estimate_entry_tokens(entry)
        assert tokens == len("Hello world") // CHARS_PER_TOKEN

    def test_model_entry_with_usage(self) -> None:
        entry = _model("response text", output_tokens=150)
        assert estimate_entry_tokens(entry) == 150

    def test_model_entry_without_usage(self) -> None:
        entry = _model("some response text")
        tokens = estimate_entry_tokens(entry)
        assert tokens > 0
        assert tokens == len("some response text") // CHARS_PER_TOKEN

    def test_model_entry_with_tool_calls(self) -> None:
        call = _call("bash", {"command": "ls -la"})
        entry = _model("", calls=[call])
        tokens = estimate_entry_tokens(entry)
        assert tokens > 0

    def test_tool_result(self) -> None:
        entry = _tool_result("file1.py\nfile2.py")
        tokens = estimate_entry_tokens(entry)
        assert tokens > 0

    def test_prune_summary(self) -> None:
        entry = _summary("This was a coding session about refactoring.")
        tokens = estimate_entry_tokens(entry)
        expected = len("This was a coding session about refactoring.") // CHARS_PER_TOKEN
        assert tokens == expected

    def test_empty_human_entry(self) -> None:
        entry = _human("")
        assert estimate_entry_tokens(entry) == 0

    def test_shell_result(self) -> None:
        entry = ShellResultEntry(command="ls", output="file.py", exit_code=0)
        tokens = estimate_entry_tokens(entry)
        assert tokens > 0


# ── estimate_tool_tokens ────────────────────────────────────────────────


class TestEstimateToolTokens:
    def test_single_tool(self) -> None:
        tool = _make_tool_spec()
        tokens = estimate_tool_tokens([tool])  # type: ignore[arg-type]
        assert tokens >= TOOL_SCHEMA_BASELINE

    def test_multiple_tools(self) -> None:
        tools = [_make_tool_spec(f"tool_{i}") for i in range(5)]
        tokens = estimate_tool_tokens(tools)  # type: ignore[arg-type]
        assert tokens >= TOOL_SCHEMA_BASELINE * 5

    def test_no_tools(self) -> None:
        assert estimate_tool_tokens([]) == 0

    def test_tool_with_large_schema(self) -> None:
        big_schema: dict[str, JValue] = {
            "type": "object",
            "properties": {f"field_{i}": {"type": "string"} for i in range(50)},
        }
        tool = _make_tool_spec(params=big_schema)
        small_tool = _make_tool_spec()
        assert estimate_tool_tokens([tool]) > estimate_tool_tokens([small_tool])  # type: ignore[arg-type]


# ── estimate_system_tokens ──────────────────────────────────────────────


class TestEstimateSystemTokens:
    def test_system_only(self) -> None:
        tokens = estimate_system_tokens("You are a helpful assistant.")
        expected = len("You are a helpful assistant.") // CHARS_PER_TOKEN
        assert tokens == expected

    def test_system_with_tools(self) -> None:
        tools = [_make_tool_spec()]
        tokens = estimate_system_tokens("system", tools)  # type: ignore[arg-type]
        assert tokens > estimate_system_tokens("system")

    def test_empty_system(self) -> None:
        assert estimate_system_tokens("") == 0


# ── estimate_transcript_tokens ──────────────────────────────────────────


class TestEstimateTranscriptTokens:
    def test_simple_transcript(self) -> None:
        transcript = [_human("hello"), _model("hi")]
        tokens = estimate_transcript_tokens(transcript)
        assert tokens > 0

    def test_with_reported_usage(self) -> None:
        transcript = [
            _human("hello"),
            _model("hi", input_tokens=500, output_tokens=50),
        ]
        tokens = estimate_transcript_tokens(transcript)
        # Should use the reported input (500) as the floor.
        assert tokens == 500

    def test_with_entries_after_reported(self) -> None:
        transcript = [
            _human("hello"),
            _model("hi", input_tokens=500, output_tokens=50),
            _human("follow up question that is somewhat long"),
        ]
        tokens = estimate_transcript_tokens(transcript)
        assert tokens > 500

    def test_empty_transcript(self) -> None:
        assert estimate_transcript_tokens([]) == 0

    def test_multiple_model_entries_uses_latest(self) -> None:
        transcript = [
            _human("a"),
            _model("b", input_tokens=100, output_tokens=10),
            _human("c"),
            _model("d", input_tokens=300, output_tokens=20),
        ]
        tokens = estimate_transcript_tokens(transcript)
        assert tokens == 300  # latest input subsumes all prior

    def test_no_usage_estimates_all(self) -> None:
        transcript = [_human("hello world"), _model("hi there")]
        tokens = estimate_transcript_tokens(transcript)
        expected = sum(estimate_entry_tokens(e) for e in transcript)
        assert tokens == expected


# ── estimate_context ────────────────────────────────────────────────────


class TestEstimateContext:
    def test_basic(self) -> None:
        transcript = [_human("hello"), _model("hi")]
        est = estimate_context(transcript, system="sys")
        assert est.used > 0
        assert est.system > 0
        assert est.messages > 0
        assert est.used == est.system + est.messages

    def test_model_window(self) -> None:
        est = estimate_context([], system="", model="claude-opus-4-6")
        assert est.limit == 200_000

    def test_custom_limits(self) -> None:
        limits = ContextLimits(window=64_000, compaction_threshold=0.5)
        est = estimate_context([], system="", limits=limits)
        assert est.limit == 64_000
        assert est.threshold == 0.5

    def test_window_override(self) -> None:
        est = estimate_context([], system="", model="claude-opus-4-6",
                               window_override=300_000)
        assert est.limit == 300_000

    def test_threshold_propagated(self) -> None:
        limits = ContextLimits(compaction_threshold=0.9)
        est = estimate_context([], system="", limits=limits)
        assert est.threshold == 0.9


# ── exceeds_threshold ───────────────────────────────────────────────────


class TestExceedsThreshold:
    def test_over(self) -> None:
        est = ContextEstimate(used=80_000, system=0, messages=80_000,
                              limit=100_000, threshold=0.75)
        assert exceeds_threshold(est) is True

    def test_under(self) -> None:
        est = ContextEstimate(used=50_000, system=0, messages=50_000,
                              limit=100_000, threshold=0.75)
        assert exceeds_threshold(est) is False


# ── plan_compaction ─────────────────────────────────────────────────────


class TestPlanCompaction:
    def _transcript(self, n: int) -> list[HumanEntry | ModelEntry]:
        entries: list[HumanEntry | ModelEntry] = []
        for i in range(n):
            if i % 2 == 0:
                entries.append(_human(f"message {i}"))
            else:
                entries.append(_model(f"response {i}"))
        return entries

    def test_too_short_returns_none(self) -> None:
        transcript = self._transcript(4)
        ids = [f"id_{i}" for i in range(4)]
        assert plan_compaction(transcript, ids) is None

    def test_exactly_min_entries(self) -> None:
        transcript = self._transcript(6)
        ids = [f"id_{i}" for i in range(6)]
        plan = plan_compaction(transcript, ids)
        assert plan is not None

    def test_split_preserves_all_entries(self) -> None:
        transcript = self._transcript(10)
        ids = [f"id_{i}" for i in range(10)]
        plan = plan_compaction(transcript, ids)
        assert plan is not None
        assert len(plan.to_summarize) + len(plan.to_keep) == 10

    def test_keeps_at_least_keep_recent(self) -> None:
        limits = ContextLimits(keep_recent=3, min_entries=4)
        transcript = self._transcript(8)
        ids = [f"id_{i}" for i in range(8)]
        plan = plan_compaction(transcript, ids, limits=limits)
        assert plan is not None
        assert len(plan.to_keep) >= 3

    def test_record_ids_aligned(self) -> None:
        transcript = self._transcript(8)
        ids = [f"id_{i}" for i in range(8)]
        plan = plan_compaction(transcript, ids)
        assert plan is not None
        assert len(plan.summarize_ids) == len(plan.to_summarize)
        assert len(plan.keep_ids) == len(plan.to_keep)
        all_ids = list(plan.summarize_ids) + list(plan.keep_ids)
        assert all_ids == ids

    def test_tokens_before_positive(self) -> None:
        transcript = self._transcript(8)
        ids = [f"id_{i}" for i in range(8)]
        plan = plan_compaction(transcript, ids)
        assert plan is not None
        assert plan.tokens_before > 0

    def test_prior_summary_detected(self) -> None:
        transcript: list[PruneSummaryEntry | HumanEntry | ModelEntry] = [
            _summary("old summary"),
            *self._transcript(8),
        ]
        ids = [f"id_{i}" for i in range(9)]
        plan = plan_compaction(transcript, ids)
        assert plan is not None
        assert plan.has_prior_summary is True
        assert plan.prior_summary_text == "old summary"

    def test_no_prior_summary(self) -> None:
        transcript = self._transcript(8)
        ids = [f"id_{i}" for i in range(8)]
        plan = plan_compaction(transcript, ids)
        assert plan is not None
        assert plan.has_prior_summary is False
        assert plan.prior_summary_text == ""

    def test_custom_min_entries(self) -> None:
        limits = ContextLimits(min_entries=3)
        transcript = self._transcript(3)
        ids = [f"id_{i}" for i in range(3)]
        plan = plan_compaction(transcript, ids, limits=limits)
        assert plan is not None

    def test_ids_shorter_than_transcript(self) -> None:
        transcript = self._transcript(8)
        ids = ["a", "b"]
        plan = plan_compaction(transcript, ids)
        assert plan is not None
        all_ids = list(plan.summarize_ids) + list(plan.keep_ids)
        assert len(all_ids) == 8

    def test_frozen(self) -> None:
        transcript = self._transcript(8)
        ids = [f"id_{i}" for i in range(8)]
        plan = plan_compaction(transcript, ids)
        assert plan is not None
        with pytest.raises(AttributeError):
            plan.tokens_before = 0  # type: ignore[misc]


# ── serialize_for_summary ───────────────────────────────────────────────


class TestSerializeForSummary:
    def test_basic(self) -> None:
        entries = [_human("hello"), _model("world")]
        text = serialize_for_summary(entries)
        assert "[user] hello" in text
        assert "[assistant] world" in text

    def test_includes_tool_calls(self) -> None:
        call = _call("bash", {"command": "ls"})
        entries = [_model("running command", calls=[call])]
        text = serialize_for_summary(entries)
        assert "bash" in text
        assert "ls" in text

    def test_truncates_at_max_chars(self) -> None:
        entries = [_human("x" * 1000) for _ in range(100)]
        text = serialize_for_summary(entries, max_chars=500)
        assert len(text) <= 600  # some overhead for labels and truncation marker
        assert "[...truncated...]" in text

    def test_tool_result_label(self) -> None:
        entries = [_tool_result("output data")]
        text = serialize_for_summary(entries)
        assert "[tool]" in text

    def test_summary_label(self) -> None:
        entries = [_summary("prior summary")]
        text = serialize_for_summary(entries)
        assert "[summary]" in text

    def test_empty_entries(self) -> None:
        assert serialize_for_summary([]) == ""

    def test_large_tool_call_args_truncated(self) -> None:
        big_args: dict[str, JValue] = {"data": "x" * 1000}
        call = _call("bash", big_args)
        entries = [_model("", calls=[call])]
        text = serialize_for_summary(entries)
        assert "..." in text


# ── build_summary_prompts ───────────────────────────────────────────────


class TestBuildSummaryPrompts:
    def _simple_plan(self, *, has_prior: bool = False) -> CompactionPlan:
        entries = [_human(f"msg {i}") for i in range(4)]
        return CompactionPlan(
            to_summarize=tuple(entries[:2]),
            to_keep=tuple(entries[2:]),
            summarize_ids=("a", "b"),
            keep_ids=("c", "d"),
            tokens_before=1000,
            has_prior_summary=has_prior,
            prior_summary_text="old summary" if has_prior else "",
        )

    def test_returns_tuple(self) -> None:
        plan = self._simple_plan()
        result = build_summary_prompts(plan)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_system_prompt_default(self) -> None:
        plan = self._simple_plan()
        system, _ = build_summary_prompts(plan)
        assert system == DEFAULT_SUMMARY_SYSTEM

    def test_custom_system(self) -> None:
        plan = self._simple_plan()
        system, _ = build_summary_prompts(plan, system="Be brief.")
        assert system == "Be brief."

    def test_user_prompt_contains_entries(self) -> None:
        plan = self._simple_plan()
        _, user = build_summary_prompts(plan)
        assert "msg 0" in user
        assert "msg 1" in user

    def test_incremental_includes_prior(self) -> None:
        plan = self._simple_plan(has_prior=True)
        _, user = build_summary_prompts(plan)
        assert "old summary" in user
        assert "Previous summary" in user

    def test_non_incremental_has_summarize_header(self) -> None:
        plan = self._simple_plan()
        _, user = build_summary_prompts(plan)
        assert "Summarize this conversation" in user


# ── apply_compaction ────────────────────────────────────────────────────


class TestApplyCompaction:
    def _plan(self) -> CompactionPlan:
        entries = [_human(f"msg {i}") for i in range(6)]
        return CompactionPlan(
            to_summarize=tuple(entries[:4]),
            to_keep=tuple(entries[4:]),
            summarize_ids=("a", "b", "c", "d"),
            keep_ids=("e", "f"),
            tokens_before=2000,
            has_prior_summary=False,
            prior_summary_text="",
        )

    def test_transcript_structure(self) -> None:
        plan = self._plan()
        result = apply_compaction(plan, "Summary of conversation.")
        assert len(result.transcript) == 3  # summary + 2 kept
        assert isinstance(result.transcript[0], PruneSummaryEntry)

    def test_summary_content(self) -> None:
        plan = self._plan()
        result = apply_compaction(plan, "Summary of conversation.")
        assert result.summary_entry.summary == "Summary of conversation."
        assert result.summary_entry.tokens_before == 2000

    def test_record_ids_aligned(self) -> None:
        plan = self._plan()
        result = apply_compaction(plan, "summary", summary_record_id="prune_1")
        assert result.record_ids[0] == "prune_1"
        assert result.record_ids[1:] == ("e", "f")

    def test_tokens_after_less_than_before(self) -> None:
        plan = self._plan()
        result = apply_compaction(plan, "brief summary")
        assert result.tokens_after < result.tokens_before

    def test_kept_entries_preserved(self) -> None:
        plan = self._plan()
        result = apply_compaction(plan, "summary")
        assert result.transcript[1:] == plan.to_keep

    def test_frozen(self) -> None:
        plan = self._plan()
        result = apply_compaction(plan, "summary")
        with pytest.raises(AttributeError):
            result.tokens_after = 0  # type: ignore[misc]

    def test_default_record_id_is_empty(self) -> None:
        plan = self._plan()
        result = apply_compaction(plan, "summary")
        assert result.record_ids[0] == ""


# ── integration ─────────────────────────────────────────────────────────


class TestIntegration:
    def test_full_compaction_flow(self) -> None:
        """End-to-end: estimate → threshold → plan → prompts → apply."""
        # Use entries without reported usage so char-based estimation
        # reflects the actual compaction reduction.
        transcript: list[HumanEntry | ModelEntry] = []
        for i in range(10):
            transcript.append(_human(f"Question {i} " + "x" * 200))
            transcript.append(_model(f"Answer {i} " + "y" * 200))
        ids = [f"id_{i}" for i in range(20)]

        # 1. Estimate context
        limits = ContextLimits(window=500, compaction_threshold=0.5)
        est = estimate_context(transcript, system="sys", limits=limits)
        assert est.needs_compaction is True

        # 2. Plan
        plan = plan_compaction(transcript, ids, limits=limits)
        assert plan is not None
        assert len(plan.to_summarize) > 0
        assert len(plan.to_keep) > 0

        # 3. Build prompts
        system_prompt, user_prompt = build_summary_prompts(plan)
        assert len(system_prompt) > 0
        assert len(user_prompt) > 0

        # 4. Apply (with fake summary)
        result = apply_compaction(plan, "This was a Q&A session.", summary_record_id="prune_1")
        assert isinstance(result.transcript[0], PruneSummaryEntry)
        assert len(result.transcript) == len(plan.to_keep) + 1
        assert result.tokens_after < result.tokens_before

    def test_incremental_compaction(self) -> None:
        """Compacting a transcript that starts with a prior summary."""
        prior = _summary("Previous session summary.")
        transcript: list[PruneSummaryEntry | HumanEntry | ModelEntry] = [prior]
        for i in range(8):
            transcript.append(_human(f"Q{i}"))
            transcript.append(_model(f"A{i}"))
        ids = [f"id_{i}" for i in range(17)]

        limits = ContextLimits(min_entries=4, keep_recent=4)
        plan = plan_compaction(transcript, ids, limits=limits)
        assert plan is not None
        assert plan.has_prior_summary is True

        _, user_prompt = build_summary_prompts(plan)
        assert "Previous summary" in user_prompt

        result = apply_compaction(plan, "Merged summary.")
        assert isinstance(result.transcript[0], PruneSummaryEntry)
        assert result.transcript[0].summary == "Merged summary."

    def test_below_threshold_no_compaction(self) -> None:
        """Short transcript should not trigger compaction."""
        transcript = [_human("hi"), _model("hello")]
        limits = ContextLimits(window=200_000)
        est = estimate_context(transcript, system="sys", limits=limits)
        assert est.needs_compaction is False
        assert plan_compaction(transcript, ["a", "b"], limits=limits) is None

    def test_deterministic(self) -> None:
        """Same input always produces the same estimate."""
        transcript = [_human("hello"), _model("world", input_tokens=500, output_tokens=50)]
        est1 = estimate_context(transcript, system="system prompt")
        est2 = estimate_context(transcript, system="system prompt")
        assert est1.used == est2.used
        assert est1.messages == est2.messages
        assert est1.system == est2.system
