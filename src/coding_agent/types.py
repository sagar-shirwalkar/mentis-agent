"""
Core types and data structures shared across the entire agent.

All dataclasses use slots=True for memory efficiency.
Python 3.12+ type syntax throughout (X | Y, type statements).
"""

from __future__ import annotations

import dataclasses
import enum
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

# ──────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────


class Role(enum.StrEnum):
    """Message role in the conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class TaskStatus(enum.StrEnum):
    """Status of a subtask or overall plan."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class RuntimeTier(enum.StrEnum):
    """Resource tier for graceful degradation."""

    LARGE = "large"
    MID = "mid"
    SMALL = "small"


class CompactionStage(enum.IntEnum):
    """Stage in the adaptive context compaction pipeline.

    Higher stages are more aggressive and expensive.
    """

    NONE = 0
    BUDGET_REDUCTION = 1
    OBSERVATION_MASKING = 2
    FAST_PRUNING = 3
    AGGRESSIVE_COMPRESSION = 4
    REVERSIBLE_COLLAPSE = 5
    FULL_LLM_SUMMARIZATION = 6


class PlanPhase(enum.StrEnum):
    """Phase in a hierarchical multi-stage plan."""

    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRY = "retry"
    ABORTED = "aborted"


class MetaThinkerSignal(enum.StrEnum):
    """Signal emitted by the Meta-Thinker monitor."""

    CONTINUE = "continue"
    INTERRUPT = "interrupt"
    COMPLETED = "completed"
    FALLBACK = "fallback"


class LoopType(enum.StrEnum):
    """Category of loop detected by the recovery system."""

    EXACT_REPETITION = "exact_repetition"
    SEMANTIC_LOOP = "semantic_loop"
    ERROR_LOOP = "error_loop"
    STALL = "stall"


class Severity(enum.StrEnum):
    """How severe a detected loop is."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class EdgeType(enum.StrEnum):
    """Type of edge in the code knowledge graph."""

    CONTAINS = "contains"
    CALLS = "calls"
    IMPORTS = "imports"
    INHERITS = "inherits"
    RESOLVES_TO = "resolves_to"


class SymbolKind(enum.StrEnum):
    """Kind of code symbol extracted by the RAG indexer."""

    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    VARIABLE = "variable"
    IMPORT = "import"
    CONSTANT = "constant"
    MODULE = "module"


class ZoneName(enum.StrEnum):
    """Named zones in the hierarchical context window."""

    IMMUTABLE = "immutable"
    TASK = "task"
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    SCRATCH = "scratch"


# ──────────────────────────────────────────────────────────────
# Messages & Conversation
# ──────────────────────────────────────────────────────────────


@dataclass(slots=True)
class Message:
    """A single message in the conversation history."""

    role: Role
    content: str
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None  # Tool name when role == TOOL
    timestamp: float = field(default_factory=time.time)

    def token_estimate(self) -> int:
        """Rough token estimate: ~4 chars per token for code, ~5 for prose."""
        chars = len(self.content)
        if chars == 0:
            return 0
        divisor = 4 if any(c in self.content for c in "{}()[];=") else 5
        return max(1, chars // divisor)


# ──────────────────────────────────────────────────────────────
# Tool Types
# ──────────────────────────────────────────────────────────────


@dataclass(slots=True)
class ToolParameter:
    """JSON-schema-style description of a single tool parameter."""

    name: str
    type: str  # "str", "int", "bool", "list[str]", etc.
    description: str
    required: bool = True
    default: Any | None = None
    enum: list[str] | None = None


@dataclass(slots=True)
class ToolSchema:
    """Full schema describing a tool the agent can invoke."""

    name: str
    description: str
    parameters: list[ToolParameter]
    # Hints for the router — which situations favour this tool
    use_when: str = ""
    token_cost_hint: str = "medium"  # "minimal" | "low" | "medium" | "high"

    def to_openai_dict(self) -> dict[str, Any]:
        """Convert to OpenAI function-calling format."""
        properties: dict[str, Any] = {}
        required: list[str] = []
        for p in self.parameters:
            prop: dict[str, Any] = {
                "type": _python_type_to_json(p.type),
                "description": p.description,
            }
            if p.enum:
                prop["enum"] = p.enum
            properties[p.name] = prop
            if p.required:
                required.append(p.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


@dataclass(slots=True)
class ToolCall:
    """A single tool invocation requested by the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class ToolResult:
    """Result returned by a tool after execution."""

    tool_call_id: str
    tool_name: str
    output: str
    success: bool = True
    error: str | None = None
    token_count: int = 0
    duration_seconds: float = 0.0


# ──────────────────────────────────────────────────────────────
# Planning
# ──────────────────────────────────────────────────────────────


@dataclass(slots=True)
class Phase:
    """A single phase in a hierarchical multi-stage plan."""

    id: int
    name: str
    description: str
    status: PlanPhase = PlanPhase.PENDING
    plan: Plan | None = None  # Tactical sub-plan for this phase


@dataclass(slots=True)
class SubTask:
    """A single decomposed sub-task within a plan."""

    id: int
    description: str
    files: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result_summary: str = ""


@dataclass(slots=True)
class Plan:
    """Strategic plan produced by the planner.

    Supports two modes:
      1. Flat (subtasks only) — backward compatible.
      2. Hierarchical (phases with sub-plans) — for multi-stage planning.
    """

    goal: str
    subtasks: list[SubTask] = field(default_factory=list)
    dependencies: dict[int, list[int]] = field(default_factory=dict)
    # Which subtask is currently active (-1 = not started)
    current_subtask_idx: int = -1
    # Hierarchical planning fields (empty for flat plans)
    phases: list[Phase] = field(default_factory=list)
    current_phase_idx: int = -1

    @property
    def current_subtask(self) -> SubTask | None:
        if 0 <= self.current_subtask_idx < len(self.subtasks):
            return self.subtasks[self.current_subtask_idx]
        return None

    @property
    def current_phase(self) -> Phase | None:
        """Get the currently active phase (hierarchical mode)."""
        if 0 <= self.current_phase_idx < len(self.phases):
            return self.phases[self.current_phase_idx]
        return None

    def advance(self) -> SubTask | None:
        """Move to the next pending subtask. Returns it or None if done."""
        # In hierarchical mode, delegate to the current phase's plan
        phase = self.current_phase
        if phase and phase.plan:
            return phase.plan.advance()

        for i, st in enumerate(self.subtasks):
            if st.status == TaskStatus.PENDING:
                # Check dependencies
                deps = self.dependencies.get(st.id, [])
                if all(
                    self.subtasks[d - 1].status == TaskStatus.COMPLETED
                    for d in deps
                    if d - 1 < len(self.subtasks)
                ):
                    st.status = TaskStatus.IN_PROGRESS
                    self.current_subtask_idx = i
                    return st
        return None

    def advance_phase(self) -> Phase | None:
        """Move to the next pending phase (hierarchical mode).

        Returns the phase or None if all phases are complete.
        """
        for i, ph in enumerate(self.phases):
            if ph.status in (PlanPhase.PENDING, PlanPhase.RETRY):
                ph.status = PlanPhase.ACTIVE
                self.current_phase_idx = i
                # If the phase has a plan, activate its first subtask
                if ph.plan and ph.plan.subtasks:
                    ph.plan.current_subtask_idx = -1
                    ph.plan.advance()
                return ph
        return None

    @property
    def is_hierarchical(self) -> bool:
        return len(self.phases) > 0


# ──────────────────────────────────────────────────────────────
# Agent State & Steps
# ──────────────────────────────────────────────────────────────


@dataclass(slots=True)
class Step:
    """Record of a single ReAct step (think → act → observe)."""

    step_number: int
    thinking: str  # Agent's chain-of-thought
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    timestamp: float = field(default_factory=time.time)

    def summary(self, max_length: int = 120) -> str:
        """One-line summary for episodic memory."""
        action = f"{self.tool_call.name}({self._arg_summary()})" if self.tool_call else "reasoning"
        result = "ok" if self.tool_result and self.tool_result.success else "error"
        text = f"Step {self.step_number}: {action} → {result}"
        return text[:max_length]

    def _arg_summary(self) -> str:
        if not self.tool_call:
            return ""
        args = self.tool_call.arguments
        # Show at most 2 key args
        items = list(args.items())[:2]
        parts = [f"{k}={v!r}" if len(repr(v)) < 30 else f"{k}=…" for k, v in items]
        return ", ".join(parts)


@dataclass(slots=True)
class AgentState:
    """Mutable state carried through the agent's lifecycle."""

    task: str
    plan: Plan | None = None
    steps: list[Step] = field(default_factory=list)
    files_modified: set[str] = field(default_factory=set)
    files_read: set[str] = field(default_factory=set)
    diagnostics_count: int = 0
    test_status: str = "unknown"
    last_error: str | None = None
    total_tokens_used: int = 0
    started_at: float = field(default_factory=time.time)

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def record_step(self, step: Step) -> None:
        self.steps.append(step)
        if step.tool_result:
            if not step.tool_result.success:
                self.last_error = step.tool_result.error or step.tool_result.output[:200]
            self.total_tokens_used += step.tool_result.token_count

    # ── JSON serialization ────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "task": self.task,
            "plan": _plan_to_dict(self.plan) if self.plan else None,
            "steps": [_step_to_dict(s) for s in self.steps],
            "files_modified": sorted(self.files_modified),
            "files_read": sorted(self.files_read),
            "diagnostics_count": self.diagnostics_count,
            "test_status": self.test_status,
            "last_error": self.last_error,
            "total_tokens_used": self.total_tokens_used,
            "started_at": self.started_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentState:
        """Deserialize from a dict produced by to_dict()."""
        return cls(
            task=data["task"],
            plan=_plan_from_dict(data["plan"]) if data.get("plan") else None,
            steps=[_step_from_dict(s) for s in data.get("steps", [])],
            files_modified=set(data.get("files_modified", [])),
            files_read=set(data.get("files_read", [])),
            diagnostics_count=data.get("diagnostics_count", 0),
            test_status=data.get("test_status", "unknown"),
            last_error=data.get("last_error"),
            total_tokens_used=data.get("total_tokens_used", 0),
            started_at=data.get("started_at", 0.0),
        )

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=_json_default)

    @classmethod
    def from_json(cls, text: str) -> AgentState:
        """Deserialize from a JSON string."""
        return cls.from_dict(json.loads(text))


# ──────────────────────────────────────────────────────────────
# Recovery
# ──────────────────────────────────────────────────────────────


@dataclass(slots=True)
class MetaThinkerResult:
    """Result from the Meta-Thinker monitor after a step."""

    signal: MetaThinkerSignal
    confidence: float
    reason: str = ""
    suggestion: str = ""
    progress_summary: str = ""


@dataclass(slots=True)
class LoopDetection:
    """Information about a detected loop."""

    loop_type: LoopType
    severity: Severity
    repeated_actions: list[Step] = field(default_factory=list)
    recurring_error: str | None = None
    message: str = ""


@dataclass(slots=True)
class RecoveryAction:
    """Action to take when a loop is detected."""

    inject_message: str | None = None
    force_think: bool = False
    suggest_tools: list[str] = field(default_factory=list)
    force_user_intervention: bool = False
    reset_working_memory: bool = False
    max_retries: int = -1  # -1 = unlimited


# ──────────────────────────────────────────────────────────────
# RAG
# ──────────────────────────────────────────────────────────────


@dataclass(slots=True)
class Symbol:
    """A code symbol extracted from a source file."""

    name: str
    kind: SymbolKind
    file_path: str
    line_start: int
    line_end: int
    signature: str  # One-liner: "def authenticate(user: str, pw: str) -> Token"
    docstring: str = ""
    body: str = ""  # Full implementation (loaded lazily)


@dataclass(slots=True)
class GraphEdge:
    """A directed edge in the code knowledge graph."""

    source_chunk_id: int
    target_name: str
    target_file: str
    edge_type: EdgeType
    line_number: int = 0


@dataclass(slots=True)
class EmbeddingResult:
    """Result of embedding a piece of text."""

    vector: list[float]
    chunk_id: int
    model: str = "numpy_random"  # Which model produced this embedding


@dataclass(slots=True)
class CodeChunk:
    """A chunk of code produced by the chunker."""

    file_path: str
    line_start: int
    line_end: int
    content: str
    symbol_name: str | None = None
    symbol_kind: SymbolKind | None = None
    token_frequencies: dict[str, int] = field(default_factory=dict)
    edges: list[GraphEdge] = field(default_factory=list)


@dataclass(slots=True)
class SearchResult:
    """A retrieval result from the RAG system."""

    content: str
    file_path: str
    line_start: int
    line_end: int
    score: float
    symbol_name: str | None = None
    source: str = "bm25"  # "bm25" | "dense" | "symbol" | "graph"


# ──────────────────────────────────────────────────────────────
# Context
# ──────────────────────────────────────────────────────────────


@dataclass(slots=True)
class ContextCompressionResult:
    """Result of a compaction operation."""

    stage: CompactionStage
    original_tokens: int
    compressed_tokens: int
    rehydration_needed: bool = False
    messages_removed: int = 0
    messages_masked: int = 0

    @property
    def savings(self) -> int:
        return self.original_tokens - self.compressed_tokens

    @property
    def compression_ratio(self) -> float:
        if self.original_tokens == 0:
            return 1.0
        return self.compressed_tokens / self.original_tokens


@dataclass(slots=True)
class RehydrationData:
    """Data preserved before a reversible collapse for post-compaction restore."""

    plan_state: dict[str, Any] = field(default_factory=dict)
    modified_files: list[str] = field(default_factory=list)
    current_phase: str = ""
    skills_remaining: list[str] = field(default_factory=list)
    tool_states: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ContextItem:
    """An item stored in a context zone."""

    content: str
    source: str  # e.g. "tool_result", "summary", "memory", "system"
    token_count: int = 0
    timestamp: float = field(default_factory=time.time)
    compressible: bool = True  # Can this be truncated/summarised?


# ──────────────────────────────────────────────────────────────
# Protocols (structural subtyping for dependency injection)
# ──────────────────────────────────────────────────────────────


class LLMProtocol(Protocol):
    """Minimal interface any LLM client must implement."""

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Message:
        ...

    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Any:
        pass

    def count_tokens(self, text: str) -> int:
        pass


class ToolExecutorProtocol(Protocol):
    """Interface for executing a tool call."""

    async def execute(self, call: ToolCall) -> ToolResult:
        pass


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────


_PY_TO_JSON_TYPE: dict[str, str] = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "list[str]": "array",
    "dict": "object",
}


def _python_type_to_json(py_type: str) -> str:
    """Map a Python type hint string to a JSON Schema type."""
    return _PY_TO_JSON_TYPE.get(py_type, "string")


# ──────────────────────────────────────────────────────────────
# JSON serialisation helpers for AgentState
# ──────────────────────────────────────────────────────────────


def _json_default(obj: Any) -> Any:
    """Default JSON encoder for types that json.dumps can't handle."""
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _plan_to_dict(plan: Plan) -> dict[str, Any]:
    return {
        "goal": plan.goal,
        "subtasks": [dataclasses.asdict(st) for st in plan.subtasks],
        "dependencies": plan.dependencies,
        "current_subtask_idx": plan.current_subtask_idx,
        "phases": [_phase_to_dict(ph) for ph in plan.phases],
        "current_phase_idx": plan.current_phase_idx,
    }


def _phase_to_dict(phase: Phase) -> dict[str, Any]:
    return {
        "id": phase.id,
        "name": phase.name,
        "description": phase.description,
        "status": phase.status.value,
        "plan": _plan_to_dict(phase.plan) if phase.plan else None,
    }


def _phase_from_dict(data: dict[str, Any]) -> Phase:
    plan_data = data.get("plan")
    return Phase(
        id=data.get("id", 0),
        name=data.get("name", ""),
        description=data.get("description", ""),
        status=PlanPhase(data.get("status", "pending")),
        plan=_plan_from_dict(plan_data) if plan_data else None,
    )


def _plan_from_dict(data: dict[str, Any]) -> Plan:
    subtasks = []
    for st_raw in data.get("subtasks", []):
        status = TaskStatus(st_raw.get("status", "pending"))
        subtasks.append(
            SubTask(
                id=st_raw.get("id", 0),
                description=st_raw.get("description", ""),
                files=st_raw.get("files", []),
                status=status,
                result_summary=st_raw.get("result_summary", ""),
            )
        )
    plan = Plan(
        goal=data.get("goal", ""),
        subtasks=subtasks,
        dependencies={int(k): v for k, v in data.get("dependencies", {}).items()},
        current_subtask_idx=data.get("current_subtask_idx", -1),
        phases=[_phase_from_dict(ph) for ph in data.get("phases", [])],
        current_phase_idx=data.get("current_phase_idx", -1),
    )
    if subtasks and plan.current_subtask_idx < 0:
        plan.current_subtask_idx = -1
    return plan


def _step_to_dict(step: Step) -> dict[str, Any]:
    return {
        "step_number": step.step_number,
        "thinking": step.thinking,
        "tool_call": dataclasses.asdict(step.tool_call) if step.tool_call else None,
        "tool_result": dataclasses.asdict(step.tool_result) if step.tool_result else None,
        "timestamp": step.timestamp,
    }


def _step_from_dict(data: dict[str, Any]) -> Step:
    tc_data = data.get("tool_call")
    tr_data = data.get("tool_result")
    return Step(
        step_number=data.get("step_number", 0),
        thinking=data.get("thinking", ""),
        tool_call=ToolCall(**tc_data) if tc_data and "id" in tc_data else None,
        tool_result=ToolResult(**tr_data) if tr_data and "tool_call_id" in tr_data else None,
        timestamp=data.get("timestamp", 0.0),
    )
