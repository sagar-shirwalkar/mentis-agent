# MEMORY.md — Cross-Session Memory Architecture

## Overview

This file documents Meredith's persistent memory system. Memories are
stored in three categories — procedural, episodic, and semantic — and
persisted in `.agent/memory.db` (SQLite).

**Session bridging:** CROSSWALK.md (project root) provides forward-directed
handoff between sessions — read at session start, written at boundaries.
Keeps under ~2000 tokens; older entries archived to `.agent/handoff/archive/`.

**Reference architecture:** See `.agents/skills/agentic-improvements/references/CLAUDE_CODE_ARCH.md`
for Claude Code's 7-layer memory stack (CLAUDE.md → auto memory → session
memory → agent memory → relevant memories → auto dream) and 5-tier compaction
pipeline.

## New Implementations (2026-06-28)

These components were added in the agentic improvements session:

### Adaptive Context Compaction (ACC)

6-stage staged compaction pipeline in `context/compactor.py`:
1. **BudgetReduction** — per-zone cap on tool outputs, runs every turn
2. **ObservationMasking** — replace older tool results with compact reference pointers
3. **FastPruning** — drop low-value (<200 char) tool outputs within retention window
4. **AggressiveCompression** — shrink retention window, trigger cache-aware dual-path
5. **ReversibleCollapse** — serialized non-lossy compression (byte shrink)
6. **FullLLMSummarization** — async LLM summarization with post-compaction rehydration

Stages 1-4 handle 95%+ of compaction needs without invoking the LLM.
Stage 6 is async and currently logs a warning with pass-through.

### Three-Tier Hybrid RAG

Three-tier retrieval cascade in `rag/retriever.py`:
1. **BM25** — fast keyword, always on, no GPU
2. **Dense** (ONNX MiniLM or numpy_random) — semantic similarity, RRF fusion with BM25
3. **Structural Graph** (AST-derived) — BFS from seed hits, edge types: CALLS/IMPORTS/INHERITS/CONTAINS

Adaptive-k adjusts top-k based on similarity distribution. Cascade short-circuits
when confidence thresholds are met.

### Meta-Thinker

Heuristic loop monitor in `recovery/meta_thinker.py`: evaluates goal progress,
context health, and behavioral quality dimensions each step. Emits
CONTINUE/INTERRUPT/COMPLETED/FALLBACK signals. No LLM dependency — zero
latency overhead.

### TurboQuant Config

MLX TurboQuant configuration in `config/local_model.yaml`: kv_bits, weight_bits,
sink_tokens, layer_adaptive flags for KV cache + weight quantization on Apple
Silicon. MLX server startup in `llm/local.py` passes flags when enabled.

## Procedural Memory (Always Loaded)

Loaded into every session unconditionally:

- **AGENTS.md** — Project instructions and conventions for the agent
- **Config conventions** — Tool, router, and planner settings from config YAML
- **Tool definitions** — Available tools, schemas, and usage rules

Never evicted during a session.

## Episodic Memory (Temporal Retrieval)

Captures what happened in previous sessions. Keyed by timestamp:

- **Checkpoint summaries** — Compressed state snapshots every N steps
- **Session summaries** — End-of-session artifacts from `save_session()`
- **Error patterns** — Failures encountered and their resolutions

Compacted periodically (prune >90 days, merge consecutive entries).

### Session Memory Extraction

Claude Code pattern: session memory extraction runs in parallel (every 3-5
steps), not as an emergency-only measure. Key points:

- Each extraction captures: current goal, files modified, key decisions,
  blockers, and tool usage patterns
- Saved to `.agent/memory/sessions/<session_id>/` as structured YAML
- Parallel extraction: runs in a sub-agent or background task without
  blocking the main loop
- On session start, the most recent session memories are loaded alongside
  procedural memories
- Session memories can be "dreamed" (replayed and consolidated) during idle
  periods to move observations from episodic to semantic storage

## Semantic Memory (Similarity Retrieval)

Patterns and conventions learned over time. Retrieved by relevance:

- **Project conventions** — Code style, naming, structural patterns
- **Tool usage patterns** — Which tools work best for which task types
- **Learned patterns** — Recurring solutions and idioms

Stored with confidence scores and access counts for ranking.

## Checkpoint Format

Saved as JSON in `.agent/checkpoints/<session_id>.json`:

```json
{
  "session_id": "uuid",
  "timestamp": 1234567890.0,
  "step_number": 5,
  "task": "Description of the task",
  "plan": { "goal": "...", "subtasks": [...], "dependencies": {} },
  "steps": [...],
  "files_modified": ["src/main.py"],
  "total_tokens_used": 15000,
  "summary": "Step 5: edit_file -> ok"
}
```

## Compaction

Prunes and merges checkpoints:

1. Delete checkpoints older than 90 days (low-access only)
2. Merge consecutive checkpoints from the same session into summaries
3. Rebuild semantic index from episodic summaries

Manual: `python scripts/compact_checkpoints.py`
Automatic: at session start if last compaction >7 days ago
