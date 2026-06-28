from __future__ import annotations

from coding_agent.recovery.meta_thinker import MetaThinker
from coding_agent.types import (
    AgentState,
    MetaThinkerSignal,
    Step,
    ToolCall,
    ToolResult,
)


def _make_tool_call(name: str = "read_file") -> ToolCall:
    return ToolCall(id="c1", name=name, arguments={"path": "test.py"})


def _make_tool_result(
    success: bool = True,
    output: str = "ok",
    error: str | None = None,
) -> ToolResult:
    return ToolResult(
        tool_call_id="c1",
        tool_name="read_file",
        output=output,
        success=success,
        error=error,
    )


def _make_step(
    step_number: int = 0,
    tool_name: str = "read_file",
    success: bool = True,
) -> Step:
    return Step(
        step_number=step_number,
        thinking="",
        tool_call=_make_tool_call(tool_name),
        tool_result=_make_tool_result(success=success),
    )


def _make_state(steps: list[Step] | None = None) -> AgentState:
    state = AgentState(task="test task")
    if steps:
        for s in steps:
            state.steps.append(s)
    return state


# ── CONTINUE scenarios ────────────────────────────────────


def test_continue_on_early_steps():
    thinker = MetaThinker()
    state = _make_state()
    step = _make_step(0)
    result = thinker.evaluate(step, state)
    assert result.signal == MetaThinkerSignal.CONTINUE
    assert result.confidence > 0.8


def test_continue_on_healthy_progress():
    thinker = MetaThinker(tool_diversity_min=2)
    tools = ["read_file", "edit_file", "search_code", "edit_file", "read_file", "edit_file"]
    steps = [_make_step(i, tools[i]) for i in range(6)]
    state = _make_state(steps=steps)
    step = _make_step(6, "edit_file")
    result = thinker.evaluate(step, state)
    assert result.signal == MetaThinkerSignal.CONTINUE


def test_continue_with_context_budget():
    thinker = MetaThinker()
    state = _make_state()
    step = _make_step(0)
    result = thinker.evaluate(step, state, budget_remaining=0.5)
    assert result.signal == MetaThinkerSignal.CONTINUE


# ── INTERRUPT scenarios ───────────────────────────────────


def test_interrupt_on_repeated_failures():
    thinker = MetaThinker(progress_stall_threshold=5)
    steps = [_make_step(i, "read_file", success=False) for i in range(5)]
    state = _make_state(steps=steps)
    step = _make_step(5, "read_file", success=False)
    result = thinker.evaluate(step, state)
    assert result.signal == MetaThinkerSignal.INTERRUPT


def test_interrupt_on_low_goal_progress():
    result = MetaThinker()._aggregate({"goal_progress": 0.1, "context_health": 0.9, "behavioral": 0.9})
    assert result.signal == MetaThinkerSignal.INTERRUPT


def test_interrupt_on_low_behavioral():
    result = MetaThinker()._aggregate({"goal_progress": 0.9, "context_health": 0.9, "behavioral": 0.1})
    assert result.signal == MetaThinkerSignal.INTERRUPT


def test_interrupt_on_multiple_low_dimensions():
    result = MetaThinker()._aggregate({"goal_progress": 0.4, "context_health": 0.4, "behavioral": 0.9})
    assert result.signal == MetaThinkerSignal.INTERRUPT


# ── FALLBACK scenario ─────────────────────────────────────


def test_fallback_on_low_confidence():
    result = MetaThinker()._aggregate({"goal_progress": 0.5, "context_health": 0.5, "behavioral": 0.1})
    assert result.signal == MetaThinkerSignal.INTERRUPT


def test_fallback_all_dimensions_low():
    result = MetaThinker()._aggregate({"goal_progress": 0.3, "context_health": 0.3, "behavioral": 0.3})
    assert result.signal in (MetaThinkerSignal.INTERRUPT, MetaThinkerSignal.FALLBACK)


def test_fallback_includes_suggestion():
    result = MetaThinker()._aggregate({"goal_progress": 0.2, "context_health": 0.2, "behavioral": 0.2})
    assert result.suggestion


# ── Goal progress evaluations ─────────────────────────────


def test_goal_progress_penalizes_read_only_stuck():
    thinker = MetaThinker(progress_stall_threshold=6)
    steps = [_make_step(i, "read_file") for i in range(6)]
    state = _make_state(steps=steps)
    step = _make_step(6, "read_file")
    result = thinker.evaluate(step, state)
    assert result.confidence <= 0.8


def test_goal_progress_bonus_for_edits():
    thinker = MetaThinker(progress_stall_threshold=5, tool_diversity_min=2)
    tools = ["read_file", "edit_file", "search_code", "edit_file", "edit_file"]
    steps = [_make_step(i, tools[i]) for i in range(5)]
    state = _make_state(steps=steps)
    step = _make_step(5, "edit_file")
    result = thinker.evaluate(step, state)
    assert result.signal == MetaThinkerSignal.CONTINUE


# ── Context health evaluations ────────────────────────────


def test_context_health_full_budget():
    thinker = MetaThinker()
    state = _make_state()
    score = thinker._evaluate_context_health(state, budget_remaining=1.0)
    assert score == 1.0


def test_context_health_low_budget():
    thinker = MetaThinker()
    state = _make_state()
    score = thinker._evaluate_context_health(state, budget_remaining=0.1)
    assert score < 0.6


def test_context_health_no_budget_info():
    thinker = MetaThinker()
    state = _make_state()
    score = thinker._evaluate_context_health(state, budget_remaining=None)
    assert score == 1.0


# ── Behavioral quality evaluations ────────────────────────


def test_behavioral_diverse_tools():
    thinker = MetaThinker(tool_diversity_min=3)
    steps = [
        _make_step(0, "read_file"),
        _make_step(1, "search_code"),
        _make_step(2, "edit_file"),
    ]
    state = _make_state(steps=steps)
    score = thinker._evaluate_behavioral(state)
    assert score > 0.5


def test_behavioral_same_tool_repeated():
    thinker = MetaThinker(tool_diversity_min=3)
    steps = [_make_step(i, "read_file") for i in range(5)]
    state = _make_state(steps=steps)
    score = thinker._evaluate_behavioral(state)
    assert score < 0.8


def test_behavioral_error_cycling():
    thinker = MetaThinker(tool_diversity_min=3)
    steps = []
    for i in range(5):
        s = _make_step(i, "read_file")
        s.tool_result = _make_tool_result(success=False, error="Permission denied")
        steps.append(s)
    state = _make_state(steps=steps)
    score = thinker._evaluate_behavioral(state)
    assert score < 0.8


def test_behavioral_few_steps():
    thinker = MetaThinker()
    steps = [_make_step(0, "read_file"), _make_step(1, "edit_file")]
    state = _make_state(steps=steps)
    score = thinker._evaluate_behavioral(state)
    assert score == 1.0


# ── Progress summary ──────────────────────────────────────


def test_progress_summary_format():
    summary = MetaThinker._build_progress_summary(0.85, 0.70, 0.90)
    assert "progress=85%" in summary
    assert "context=70%" in summary
    assert "behavior=90%" in summary


def test_evaluate_returns_result_object():
    thinker = MetaThinker()
    state = _make_state()
    step = _make_step(0)
    result = thinker.evaluate(step, state)
    assert hasattr(result, "signal")
    assert hasattr(result, "confidence")
    assert hasattr(result, "reason")
    assert hasattr(result, "progress_summary")
