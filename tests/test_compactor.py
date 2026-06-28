from __future__ import annotations

from coding_agent.config import CompressionConfig
from coding_agent.context.compactor import ContextCompactor
from coding_agent.types import (
    CompactionStage,
    Message,
    Role,
)


def _make_message(role: Role, content: str) -> Message:
    return Message(role=role, content=content)


def _make_messages(count: int = 6, content_len: int = 500) -> list[Message]:
    return [
        _make_message(Role.USER if i % 2 == 0 else Role.ASSISTANT, "x" * content_len)
        for i in range(count)
    ]


def test_default_config():
    config = CompressionConfig()
    assert config.stage1_budget_reduction == 0.40
    assert config.stage6_full_llm == 0.04


def test_select_stage_none():
    compactor = ContextCompactor(CompressionConfig())
    assert compactor._select_stage(1.0) == CompactionStage.NONE


def test_select_stage_budget_reduction():
    compactor = ContextCompactor(CompressionConfig())
    assert compactor._select_stage(0.35) == CompactionStage.BUDGET_REDUCTION


def test_select_stage_observation_masking():
    compactor = ContextCompactor(CompressionConfig())
    assert compactor._select_stage(0.25) == CompactionStage.OBSERVATION_MASKING


def test_select_stage_fast_pruning():
    compactor = ContextCompactor(CompressionConfig())
    assert compactor._select_stage(0.15) == CompactionStage.FAST_PRUNING


def test_select_stage_aggressive():
    compactor = ContextCompactor(CompressionConfig())
    assert compactor._select_stage(0.10) == CompactionStage.AGGRESSIVE_COMPRESSION


def test_select_stage_reversible():
    compactor = ContextCompactor(CompressionConfig())
    assert compactor._select_stage(0.06) == CompactionStage.REVERSIBLE_COLLAPSE


def test_select_stage_full_llm():
    compactor = ContextCompactor(CompressionConfig())
    assert compactor._select_stage(0.02) == CompactionStage.FULL_LLM_SUMMARIZATION


def test_compact_noop_when_plenty_budget():
    compactor = ContextCompactor(CompressionConfig())
    messages = _make_messages(4, 100)
    result = compactor.compact(messages, budget_remaining=0.5)
    assert result is messages
    assert compactor._current_stage == CompactionStage.NONE


def test_compact_advances_stage():
    compactor = ContextCompactor(CompressionConfig())
    messages = _make_messages(6, 500)
    compactor.compact(messages, budget_remaining=0.35)
    assert compactor._current_stage == CompactionStage.BUDGET_REDUCTION


def test_compact_does_not_regress_stage():
    compactor = ContextCompactor(CompressionConfig())
    messages = _make_messages(6, 500)
    compactor.compact(messages, budget_remaining=0.35)
    compactor.compact(messages, budget_remaining=0.5)
    assert compactor._current_stage == CompactionStage.BUDGET_REDUCTION


def test_compact_advances_through_stages():
    compactor = ContextCompactor(CompressionConfig())
    messages = _make_messages(6, 500)
    compactor.compact(messages, budget_remaining=0.08)
    assert compactor._current_stage == CompactionStage.AGGRESSIVE_COMPRESSION
    compactor.compact(messages, budget_remaining=0.02)
    assert compactor._current_stage == CompactionStage.FULL_LLM_SUMMARIZATION


def test_stage1_truncates_long_tool_outputs():
    compactor = ContextCompactor(CompressionConfig())
    long_content = "\n".join(f"line {i}: " + "x" * 80 for i in range(100))
    messages = [
        _make_message(Role.USER, "short query"),
        _make_message(Role.TOOL, long_content),
    ]
    compactor.compact(messages, budget_remaining=0.30)
    lines = messages[1].content.split("\n")
    assert len(lines) < 40
    assert "lines omitted" in messages[1].content


def test_stage2_masks_old_observations():
    compactor = ContextCompactor(CompressionConfig())
    messages = _make_messages(6, 300)
    compactor.compact(messages, budget_remaining=0.20)
    assert compactor._current_stage == CompactionStage.OBSERVATION_MASKING


def test_stage3_prunes_small_outputs():
    compactor = ContextCompactor(CompressionConfig(min_output_length_for_pruning=200))
    messages = [
        _make_message(Role.TOOL, "short"),
        _make_message(Role.TOOL, "x" * 500),
        _make_message(Role.TOOL, "hi"),
    ]
    compactor.compact(messages, budget_remaining=0.15)
    assert compactor._current_stage == CompactionStage.FAST_PRUNING


def test_stage4_aggressive_keeps_last_two():
    compactor = ContextCompactor(CompressionConfig(stage4_aggressive_compression=0.20))
    messages = _make_messages(8, 400)
    compactor.compact(messages, budget_remaining=0.08)
    assert compactor._current_stage == CompactionStage.AGGRESSIVE_COMPRESSION


def test_stage5_reversible_collapse(tmp_path):
    coll_path = tmp_path / "collapse"
    coll_path.mkdir()
    compactor = ContextCompactor(CompressionConfig(
        stage5_reversible_collapse=0.12,
        collapse_serialization_path=str(coll_path),
    ))
    messages = _make_messages(6, 500)
    compactor.compact(messages, budget_remaining=0.05)
    assert compactor._current_stage == CompactionStage.REVERSIBLE_COLLAPSE


def test_stage6_logs_warning():
    compactor = ContextCompactor(CompressionConfig(stage6_full_llm=0.10))
    messages = _make_messages(4, 300)
    compactor.compact(messages, budget_remaining=0.02)
    assert compactor._current_stage == CompactionStage.FULL_LLM_SUMMARIZATION


def test_prepare_and_restore_rehydration():
    compactor = ContextCompactor(CompressionConfig())
    data = compactor.prepare_rehydration(
        plan_state={"phase": "implement", "step": 3},
        modified_files=["src/main.py"],
        current_phase="implementation",
        skills_remaining=["code-review"],
        tool_states={"router": "hybrid"},
    )
    assert data.plan_state["phase"] == "implement"
    assert data.modified_files == ["src/main.py"]
    assert data.current_phase == "implementation"
    assert data.skills_remaining == ["code-review"]

    messages = _make_messages(2, 100)
    restored = compactor.restore_rehydration(messages, data)
    assert len(restored) >= len(messages)


def test_rehydration_data_default():
    compactor = ContextCompactor(CompressionConfig())
    data = compactor.prepare_rehydration(
        plan_state={},
        modified_files=[],
        current_phase="",
    )
    assert data.plan_state == {}
    assert data.modified_files == []
