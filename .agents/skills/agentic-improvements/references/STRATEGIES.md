# STRATEGIES — Tool, Planning, and Context Patterns

> Reference file for the agentic-improvements skill.
> Detailed patterns for implementing the improvements outlined in SKILL.md.

## Table of Contents
- [1. Tool Selection Strategies](#1-tool-selection-strategies)
- [2. Planning Strategies](#2-planning-strategies)
- [3. Context Management Strategies](#3-context-management-strategies)
- [4. Graceful Degradation Chain](#4-graceful-degradation-chain)
- [5. Cross-Session Memory Pattern](#5-cross-session-memory-pattern)
- [6. (reserved)](#6-reserved)
- [7. Adaptive Context Compaction (ACC)](#7-adaptive-context-compaction-acc)
- [8. Three-Tier Hybrid RAG](#8-three-tier-hybrid-rag)
- [9. CROSSWALK.md — Session Bridge Pattern](#9-crosswalkmd--session-bridge-pattern)
- [10. Testing Patterns for Agent Improvements](#10-testing-patterns-for-agent-improvements)

---

## 1. Tool Selection Strategies

### Strategy Matrix

| Strategy | How It Works | Best For | Config Key |
|----------|-------------|----------|------------|
| `rules_only` | Hardcoded routing rules (if pattern X → use grep) | Small models (< 30B) | `tools.router.strategy` |
| `hybrid` | LLM proposes tools, rules filter/rank | Large models | `tools.router.strategy` |
| `llm_only` | Full LLM responsibility for tool choice | Frontier models | `tools.router.strategy` |

### Implementation Pattern

```python
# router.py — Tool routing with graceful degradation
class ToolRouter:
    def __init__(self, config: RouterConfig):
        self.strategy = config.strategy
        self.preferences = config.learned_preferences

    async def select(self, task: str, context: Context) -> ToolName:
        if self.strategy == "rules_only":
            return self._rule_based(task)
        elif self.strategy == "hybrid":
            candidates = await self._llm_propose(task)
            return self._filter_by_rules(candidates, context)
        else:  # llm_only
            return await self._llm_select(task)
```

### Learning Preferences

Track tool-usage patterns to improve routing over time:

```yaml
tools:
  router:
    learned_preferences: true
    preference_store: ".agent/tool_preferences.json"
```

Store per-task-type success rates and adjust routing weights accordingly.

---

## 2. Planning Strategies

### Strategy Stack

```yaml
agent:
  planner_type: tree_of_thought  # tree_of_thought | flat | hierarchical
  planner_model: null            # null = same model, or specify cheaper model
  verifier_concurrent: true      # Run verifier alongside executor
```

### Flat Planner (small models)

```
Task → Step 1 → Step 2 → ... → Step N → Done
  └── every step is a tool call with full context
```

### Tree of Thought (large models)

```
Task → Branch A → Step A1 → Step A2 → ...
     → Branch B → Step B1 → Step B2 → ...
     └── verifier evaluates branches, picks best
```

### Hierarchical Planner (new)

```
Task → Strategic Plan (3-7 phases)
        ├── Phase 1: Discover
        │     └── Tactical Plan
        │           ├── Step 1: search
        │           └── Step 2: read
        ├── Phase 2: Implement
        │     └── Tactical Plan
        │           ├── Step 1: edit
        │           └── Step 2: verify
        └── Phase 3: Verify
              └── Tactical Plan
                    └── Run tests
```

### Phase Lifecycle

```
[ACTIVE] → success → [COMPLETE]
    ↓ failure
[RETRY] → retries < max → [ACTIVE]
    ↓ retries >= max
[RECOVERY] → escalate → [ACTIVE] (with intervention)
    ↓ escalate fails
[ABORTED]
```

---

## 3. Context Management Strategies

### Zone Architecture

```
┌─────────────────────────────────────────────┐
│ IMMUTABLE (priority 0) — always present     │
│ System prompt + tool schemas + AGENTS.md    │
├─────────────────────────────────────────────┤
│ TASK (priority 1) — current objective       │
│ "Implement function X in file Y"            │
├─────────────────────────────────────────────┤
│ WORKING (priority 2) — sliding window       │
│ Recent tool calls + results (last N)        │
├─────────────────────────────────────────────┤
│ EPISODIC (priority 3) — compressed history  │
│ Summarized earlier steps                    │
├─────────────────────────────────────────────┤
│ SEMANTIC (priority 4) — project knowledge   │
│ Conventions, patterns, cross-session memory │
├─────────────────────────────────────────────┤
│ SCRATCH (priority 5) — agent notes          │
│ Meta-thinker interventions, scratchpad      │
└─────────────────────────────────────────────┘
```

### Compression Chain

```
Working zone exceeds budget
  → Compress oldest entries into episodic summary
  → If still over budget, compress two oldest episodic entries
  → If still over budget, truncate semantic zone
  → If still over budget (EMERGENCY), truncate scratch
  → If still over budget, truncate task description
```

### KV Cache Compression (TurboQuant-style)

For agents on constrained hardware, KV cache compression is the difference between 32K and 128K context on the same GPU.

```yaml
context:
  max_tokens: 128000        # Target context (was 32000)
  kv_cache:
    compression: turboquant_4bit_nc  # vLLM dtype, ~2.6-3.1× compression
    # Alternatives:
    #   fp8              → 2×, no quality loss, Hopper/Blackwell only
    #   turboquant_3bit_nc → 3.5-4×, 15-25pt quality drop on hard tasks
    #   turboquant_4bit_nc → 2.6-3.1×, ~1pt from FP8 (sweet spot)
```

**When to apply:**
- Local model profile (`local_model.yaml`): 32K → 96K effective context with `turboquant_4bit_nc`
- Mid-tier GPU (RTX 4090, 48GB): 128K+ inference feasible without FP8 attention
- Apple Silicon: MLX-INT4 + TurboQuant enables 64K+ on M-series

**Constraint:** TurboQuant compresses the KV cache during inference, not the agent's prompt context. The agent's working context window (system prompt + tool calls) is unaffected; the benefit is longer generation without OOM on long-horizon tasks.

### Adaptive Cache (AdaCache-style)

```yaml
rag:
  cache:
    strategy: adaptive
    tiers:
      l1:  # In-memory, TTL 30s
        max_entries: 20
        ttl_seconds: 30
      l2:  # Session-level, compressed
        max_entries: 50
        ttl_seconds: 300
      l3:  # Persistent RAG index
        max_entries: 1000
    confidence_threshold: 0.7  # Above this → use L1/L2 only
```

---

## 4. Graceful Degradation Chain

```
┌──────────────────────────────────────────────────┐
│ 1. Primary config (e.g., large_model.yaml)       │
│    → 200K context, ToT planner, hybrid router    │
├──────────────────────────────────────────────────┤
│ 2. Degrade to mid config                         │
│    → 64K context, flat planner, hybrid router    │
│    Trigger: LLM timeout > 3s, context > 80% full │
├──────────────────────────────────────────────────┤
│ 3. Degrade to small config                       │
│    → 32K context, rules-only router, RAG top_k=5 │
│    Trigger: repeated timeouts, context > 95% full│
├──────────────────────────────────────────────────┤
│ 4. Emergency mode                                │
│    → 8K context, single-step, no RAG, no verifier│
│    Trigger: model unreachable, budget exhausted  │
└──────────────────────────────────────────────────┘
```

### Implementation

```python
# config.py — Dynamic tier selection
class AgentConfig:
    def select_tier(self, context: RuntimeContext) -> ConfigProfile:
        if context.model_unreachable:
            return self.emergency_profile
        if context.context_usage > 0.95 or context.repeated_timeouts > 3:
            return self.small_profile
        if context.context_usage > 0.80 or context.avg_llm_latency > 3.0:
            return self.mid_profile
        return self.large_profile
```

---

## 5. Cross-Session Memory Pattern

### MEMORY.md Structure

```markdown
# MEMORY.md — Cross-Session Agent Memory

## Procedural (always loaded)
- AGENTS.md conventions
- Tool definitions and usage patterns
- Project-specific idioms

## Episodic (by timestamp)
- [2026-06-27] Session: Implemented RAG chunker
- [2026-06-26] Session: Fixed loop detection bug
- ...

## Semantic (by similarity)
- Pattern: When modifying test files, first check conftest.py for fixtures
- Pattern: API client follows httpx async pattern
- Rule: Never hardcode API keys; use env vars
```

### Checkpoint Schema (checkpoint.json)

```json
{
  "format_version": 1,
  "session_id": "abc-123-def",
  "created_at": "2026-06-27T10:30:00Z",
  "task": "Implement RAG chunker",
  "current_phase": "implementation",
  "completed_phases": ["research"],
  "context_summary": "Selected 3 chunking strategies, eliminated regex-only",
  "key_findings": [
    "AST chunking preserves structure better than line-based",
    "Tree-sitter missing Go grammar — need fallback"
  ],
  "file_states": {
    "modified": ["src/coding_agent/rag/chunker.py"],
    "created": [],
    "deleted": []
  }
}
```

### Compaction Script

```python
# scripts/compact_checkpoints.py
"""Prune and merge old checkpoints."""
import json
from pathlib import Path
from datetime import datetime, timedelta

MAX_AGE_DAYS = 14
CHECKPOINT_DIR = Path(".agent/checkpoints")

def compact():
    now = datetime.now()
    for f in sorted(CHECKPOINT_DIR.glob("*.json")):
        cp = json.loads(f.read_text())
        age = now - datetime.fromisoformat(cp["created_at"])
        if age > timedelta(days=MAX_AGE_DAYS):
            # Move to compressed archive
            archive = CHECKPOINT_DIR / "archive" / f.name
            archive.parent.mkdir(exist_ok=True)
            f.rename(archive)
```

---

## 7. Adaptive Context Compaction (ACC)

Claude Code-inspired 6-tier staged compaction pipeline, cheapest-first.
Meredith's implementation is in `context/compactor.py` with corresponding `CompactionStage` enum in `types.py`.

### Stage 1: BudgetReduction (every turn)
```
Cap each tool output to its zone budget allocation.
No history removal — just truncation of verbose outputs.
```

### Stage 2: ObservationMasking (~70% utilization)
```
Replace older tool-result messages with compact reference pointers.
"[output offloaded to scratch file]" replaces the full content.
Most recent N tool outputs retained at full fidelity.
```

### Stage 3: FastPruning (~80% utilization)
```
Drop tool outputs below MIN_LENGTH threshold (<200 chars).
Preserve outputs within PROTECTED_RECENCY window.
Cheaper than LLM-based compaction — often reclaims enough space.
```

### Stage 4: AggressiveCompression (~90% utilization)
```
Shrink retention window to only the most recent tool output.
Mask all older observations.
Trigger cache-aware path selection:
  - Hot cache → use cache_edits (delete cached blocks without rewriting)
  - Cold cache → simple truncation
```

### Stage 5: ReversibleCollapse (~95% utilization)
```
Serialize full conversation to bytes with zlib compression.
Fully reversible via deserialization — no information lost.
Middle ground between cheap truncation and expensive LLM call.
Outcome: `RehydrationData` payload that can restore exact state.
```

### Stage 6: FullLLMSummarization (~99% utilization)
```
Serialize full conversation to scratch file (non-lossy).
LLM summarization of middle portion.
Post-compaction rehydration:
  1. Restore current plan phase
  2. Restore modified files list (max 5)
  3. Re-inject skills (token-capped)
  4. Reset tool state
  5. Inject session-continuation reminder
```

### Two-Phase CoT Summarization

```python
# Phase 1: LLM writes chain-of-thought + conclusion
summary_response = await llm.generate(
    f"Compress this conversation.\n"
    f"First, reason step-by-step about what to keep.\n"
    f"Then output the structured summary.\n"
    f"{conversation}"
)
# Phase 2: Keep only the conclusion, discard reasoning
summary = parse_compact_summary(summary_response)
# The CoT is discarded — it improved quality but wastes tokens in-context
```

### Implementation Considerations

- Stage thresholds are configurable per profile (base.yaml)
- ACC should run at the start of each iteration, not as an emergency measure
- Full context serialized to scratch file before LLM summarization (non-lossy)
- Post-compaction rehydration is as important as the summary itself
- Cache-aware dual paths: same algorithm, different strategy by cache state

---

## 8. Three-Tier Hybrid RAG

BM25-only → three-tier fusion with RRF:

### Tier 1: Keyword (Fast, Always On)
```
BM25 (SQLite FTS5 or Tantivy) + ripgrep fallback
  - Handles identifier-heavy code search well
  - Low latency, zero GPU
  - Always runs first
```

### Tier 2: Dense Semantic (Medium, ONNX)
```
ONNX MiniLM → FAISS IndexFlatIP (cosine via inner product)
  - No PyTorch dependency
  - ~3ms warm search
  - Cross-encoder reranker via ONNX (optional)
  - Adaptive-k: dynamic top-k based on similarity distribution
```

### Tier 3: Structural Graph (Heavy, On-Demand)
```
AST-derived knowledge graph → BFS expansion from seed hits
  - Reuses existing tree-sitter chunker
  - Edge types: CONTAINS, CALLS, IMPORTS, INHERITS
  - Only triggered when Tier 1+2 confidence is low
  - ZoomRAG coarse-to-fine for large codebases
```

### Fusion: Reciprocal Rank Fusion

```python
def rrf(results, k=60):
    """Fuse multiple ranked result lists."""
    scores = {}
    for rank, doc in enumerate(results):
        scores[doc.id] = scores.get(doc.id, 0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])
```

### Adaptive-k (Dynamic Top-K)

```python
def adaptive_k(scores, lambda_val=0.5, min_k=1, max_k=50):
    """Select k based on similarity distribution."""
    threshold = scores.mean() + lambda_val * scores.std()
    return min(max((scores > threshold).sum(), min_k), max_k)
```

### Tier Selection

```
Query → Tier 1 (BM25) → confidence > 0.8? → return results
                          ↓ no
                       Tier 2 (Dense) → confidence > 0.8? → RRF Tier 1+2 → return
                                          ↓ no
                                       Tier 3 (Graph) → RRF all three → return
```

---

## 9. CROSSWALK.md — Session Bridge Pattern

A structured relay document that agents read at session start and write at key boundaries. Unlike MEMORY.md (cross-session knowledge index), CROSSWALK.md is a **forward-directed handoff** — every field drives the next session's reasoning, not archive the current one.

### Structure

```markdown
# CROSSWALK.md — Session Bridge

## Active Work
- **Current phase**: [phase name]
- **Files modified**: [list]
- **Uncommitted changes**: [description]

## Session History (Last 3)
- [2026-06-28] Implemented ACC pipeline — 5-stage compaction
- [2026-06-27] Built HierarchicalPlanner — phase lifecycle
- [2026-06-26] Added checkpoint system — JSON serialization

## Key Decisions
- [Decision] with rationale → [location in project docs]

## Open Questions
- [Question] blocking [component]

## Next Action
- [Single clear directive for the next session]
```

### Lifecycle

```
Session start → read CROSSWALK.md → update Active Work
     ↓
  Work...
     ↓
Key boundaries (phase complete, decision made) → update CROSSWALK.md
     ↓
Session end → write final state + Next Action
```

### Rules

- Keep under ~2,000 tokens to preserve context
- Archive older decisions to `.agent/handoff/archive/` as it grows
- Use atomic writes (temp file + rename) to prevent partial-write corruption
- Commit to git after significant milestones for audit trail
- Never duplicate content already in MEMORY.md — reference it by path

### Relationship to MEMORY.md

| File | Purpose | Loaded | Writen by |
|---|---|---|---|
| MEMORY.md | Cross-session knowledge index | Always (first 200 lines) | Agent during sessions |
| CROSSWALK.md | Session-to-session handoff | At session start | Agent at boundaries + session end |
| AGENTS.md | Static project rules | Always | Humans |

---

## 10. Testing Patterns for Agent Improvements

### What to Test

| Component | Test Type | Example |
|-----------|-----------|---------|
| Planner | Unit | `test_flat_plan_generation` |
| Router | Unit | `test_rule_based_routing` |
| Context compression | Unit | `test_zone_compaction_orders` |
| Graceful degradation | Integration | `test_fallback_chain_on_timeout` |
| Memory persistence | Integration | `test_save_and_load_episodic` |
| ACP protocol | Integration | `test_acp_tool_call_roundtrip` |

### Test Fixture Pattern

```python
# tests/conftest.py — Agent test fixtures
@pytest.fixture
def config_small():
    return AgentConfig.from_yaml("config/local_model.yaml")

@pytest.fixture
def config_large():
    return AgentConfig.from_yaml("config/large_model.yaml")

@pytest.fixture
def mock_llm_small():
    """Simulate a small model: slow, limited output."""
    ...

@pytest.fixture
def mock_llm_large():
    """Simulate a large model: fast, rich output."""
    ...
```
