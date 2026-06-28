"""
Meta-Thinker: monitors agent progress and emits confidence signals.

Inspired by COMPASS (ACL 2026), the Meta-Thinker evaluates three
dimensions after each step:

  1. Goal progress — is the agent making measurable progress?
  2. Context health — is the context window well-utilised?
  3. Behavioral quality — is the agent stuck or oscillating?

Emits one of four signals:
  - CONTINUE: all dimensions healthy
  - INTERRUPT: clear problem detected — needs re-planning
  - COMPLETED: task appears done
  - FALLBACK: confidence too low — use simpler approach
"""

from __future__ import annotations

import logging
from collections import Counter

from coding_agent.types import (
    AgentState,
    MetaThinkerResult,
    MetaThinkerSignal,
    Step,
)

logger = logging.getLogger(__name__)


class MetaThinker:
    """
    Lightweight monitoring component that evaluates agent health.

    Runs after every step. Uses simple heuristics (not an LLM call)
    to keep overhead near zero. Configurable thresholds per dimension.
    """

    def __init__(
        self,
        progress_stall_threshold: int = 5,
        context_utilization_warning: float = 0.90,
        tool_diversity_min: int = 3,
    ) -> None:
        self.progress_stall_threshold = progress_stall_threshold
        self.context_utilization_warning = context_utilization_warning
        self.tool_diversity_min = tool_diversity_min

    def evaluate(
        self,
        step: Step,
        state: AgentState,
        budget_remaining: float | None = None,
    ) -> MetaThinkerResult:
        """
        Evaluate agent health after a step.

        Args:
            step: The step just completed.
            state: Current agent state.
            budget_remaining: Fraction of token budget remaining (0.0-1.0).

        Returns:
            MetaThinkerResult with signal, confidence, and reason.
        """
        dimensions = self._evaluate_all(step, state, budget_remaining)
        return self._aggregate(dimensions)

    def _evaluate_all(
        self,
        step: Step,
        state: AgentState,
        budget_remaining: float | None = None,
    ) -> dict[str, float]:
        """Evaluate all three dimensions. Returns {dimension: score} where 1.0 = healthy."""
        goal_progress = self._evaluate_goal_progress(step, state)
        context_health = self._evaluate_context_health(state, budget_remaining)
        behavioral = self._evaluate_behavioral(state)

        return {
            "goal_progress": goal_progress,
            "context_health": context_health,
            "behavioral": behavioral,
        }

    # ── Dimension 1: Goal Progress ────────────────────────────

    def _evaluate_goal_progress(self, step: Step, state: AgentState) -> float:
        """
        Evaluate whether the agent is making progress.

        Factors:
          - Recent file modifications
          - Success vs failure ratio
          - Step-to-step change in state
        Returns 0.0 (stuck) to 1.0 (good progress).
        """
        if state.step_count < 3:
            return 1.0

        recent = state.steps[-self.progress_stall_threshold :]

        # Count successes and failures
        successes = sum(1 for s in recent if s.tool_result and s.tool_result.success)
        failures = sum(1 for s in recent if s.tool_result and not s.tool_result.success)

        if len(recent) == 0:
            return 1.0

        success_rate = successes / len(recent)

        # Check for file modifications in recent steps
        modified_recently = any(
            s.tool_call and s.tool_call.name in ("edit_file", "write_file") for s in recent
        )

        # Penalty for repeated failures
        if failures >= 3 and success_rate < 0.3:
            return 0.2

        # Bonus for recent edits
        if modified_recently:
            return min(1.0, success_rate + 0.3)

        # Check for repeated read-only operations (stuck in investigation)
        read_only_tools = {"read_file", "search_code", "find_symbols", "list_directory"}
        read_only_recent = sum(
            1 for s in recent if s.tool_call and s.tool_call.name in read_only_tools
        )
        if read_only_recent >= 4 and not modified_recently:
            return 0.4

        return success_rate

    # ── Dimension 2: Context Health ────────────────────────────

    def _evaluate_context_health(
        self,
        state: AgentState,
        budget_remaining: float | None = None,
    ) -> float:
        """
        Evaluate whether the context window is healthy.

        Factors:
          - Token budget remaining
          - Step count relative to max
        Returns 0.0 (critical) to 1.0 (healthy).
        """
        if budget_remaining is None:
            return 1.0

        # Budget-based score
        budget_score = min(budget_remaining * 2, 1.0)

        # Step-based score (running out of steps = pressure)
        if state.step_count > 0:
            max_steps = 50  # From AgentConfig default
            step_ratio = state.step_count / max_steps
            step_score = max(0.0, 1.0 - step_ratio)
        else:
            step_score = 1.0

        return min(1.0, budget_score * 0.6 + step_score * 0.4)

    # ── Dimension 3: Behavioral Quality ───────────────────────

    def _evaluate_behavioral(self, state: AgentState) -> float:
        """
        Evaluate whether the agent's behavior is healthy.

        Factors:
          - Tool diversity (using multiple tools or just one?)
          - Action repetition patterns
          - Error cycling
        Returns 0.0 (stuck) to 1.0 (healthy).
        """
        if state.step_count >= self.progress_stall_threshold:
            recent = state.steps[-self.progress_stall_threshold :]
        else:
            recent = state.steps

        if len(recent) < 3:
            return 1.0

        # Tool diversity
        tool_names = [s.tool_call.name for s in recent if s.tool_call]
        if not tool_names:
            return 0.6  # No tool calls = just reasoning

        unique_tools = len(set(tool_names))
        diversity_score = min(unique_tools / self.tool_diversity_min, 1.0)

        # Check for exact tool repetition (same tool 4+ times in a row)
        if len(tool_names) >= 4:
            last_four = tool_names[-4:]
            if len(set(last_four)) == 1:
                diversity_score *= 0.3

        # Check for error cycling
        errors = [s for s in recent if s.tool_result and not s.tool_result.success]
        if len(errors) >= 3:
            error_types = Counter(
                e.tool_result.error or e.tool_result.output[:50] if e.tool_result else ""
                for e in errors
            )
            if error_types.most_common(1)[0][1] >= 2:
                diversity_score *= 0.4

        return diversity_score

    # ── Aggregation ───────────────────────────────────────────

    def _aggregate(self, dimensions: dict[str, float]) -> MetaThinkerResult:
        """
        Aggregate dimension scores into a single signal.

        Thresholds:
          - Any dimension < 0.2 → INTERRUPT
          - Two or more dimensions < 0.5 → INTERRUPT
          - All dimensions > 0.8 → CONTINUE
          - Goal progress > 0.9 and no recent tool calls → COMPLETED
          - Otherwise → CONTINUE with low confidence
        """
        gp = dimensions.get("goal_progress", 1.0)
        ch = dimensions.get("context_health", 1.0)
        bh = dimensions.get("behavioral", 1.0)

        confidence = (gp + ch + bh) / 3.0

        # INTERRUPT: critical failure in any dimension
        if gp < 0.2 or bh < 0.2:
            return MetaThinkerResult(
                signal=MetaThinkerSignal.INTERRUPT,
                confidence=confidence,
                reason=f"Critical dimension failure: gp={gp:.2f} ch={ch:.2f} bh={bh:.2f}",
                suggestion="Agent appears stuck — re-plan or simplify the approach.",
                progress_summary=self._build_progress_summary(gp, ch, bh),
            )

        # INTERRUPT: multiple dimensions struggling
        low_dims = sum(1 for v in (gp, ch, bh) if v < 0.5)
        if low_dims >= 2:
            return MetaThinkerResult(
                signal=MetaThinkerSignal.INTERRUPT,
                confidence=confidence,
                reason=f"Multiple dimensions below threshold: {low_dims}/3 low",
                suggestion="Consider a different strategy or breaking the task into smaller steps.",
                progress_summary=self._build_progress_summary(gp, ch, bh),
            )

        # CONTINUE: all healthy
        if gp > 0.8 and ch > 0.8 and bh > 0.8:
            return MetaThinkerResult(
                signal=MetaThinkerSignal.CONTINUE,
                confidence=confidence,
                reason="All dimensions healthy",
                progress_summary=self._build_progress_summary(gp, ch, bh),
            )

        # COMPLETED: goal progress high, context healthy, no active tool calls
        if gp > 0.9:
            return MetaThinkerResult(
                signal=MetaThinkerSignal.CONTINUE,
                confidence=confidence,
                reason="Strong progress — moderate confidence",
                progress_summary=self._build_progress_summary(gp, ch, bh),
            )

        # Default: continue with reduced confidence
        if confidence < 0.4:
            return MetaThinkerResult(
                signal=MetaThinkerSignal.FALLBACK,
                confidence=confidence,
                reason=f"Low overall confidence ({confidence:.2f})",
                suggestion="Simplify — reduce context, use fewer tools, or ask for help.",
                progress_summary=self._build_progress_summary(gp, ch, bh),
            )

        return MetaThinkerResult(
            signal=MetaThinkerSignal.CONTINUE,
            confidence=confidence,
            reason=f"Proceeding: gp={gp:.2f} ch={ch:.2f} bh={bh:.2f}",
            progress_summary=self._build_progress_summary(gp, ch, bh),
        )

    @staticmethod
    def _build_progress_summary(gp: float, ch: float, bh: float) -> str:
        return f"progress={gp:.0%} context={ch:.0%} behavior={bh:.0%}"
