# CROSSWALK.md — Session Bridge

## Active Work
- **Current phase**: Handoff — implementation complete; delivering handoff document, updated CROSSWALK/MEMORY/README, and skill file review
- **Implementation**: ACC (6-stage compactor), Three-Tier RAG (BM25+dense+graph), Meta-Thinker (heuristic monitor), TurboQuant (MLX config) — all done
- **Uncommitted changes**: Yes — 4 new components, rewrites/extensions to 10+ files, 4 reference .md files, updated skill files

## Session History (Last 3)
- [2026-06-28] Research round 1+2: COMPASS, TurboQuant, AdaCache, ACC, lightweight RAG, graph RAG, Claude Code architecture; created 4 reference .md files; updated skill files; created CROSSWALK.md
- [2026-06-28] Implementation: ACC compactor, Three-Tier RAG (embedder/graph/retriever/indexer), Meta-Thinker, TurboQuant config; all 224 tests pass, lint clean, mypy clean
- (earlier sessions: see MEMORY.md)

## Key Decisions
- **ACC stages**: cheaper-first progression with reversible collapse (stage 5) and LLM summarization (stage 6) — stages 1-4 handle 95%+ of cases without LLM call
- **Dense retriever**: numpy_random as default (zero deps, deterministic) with ONNX MiniLM upgrade path
- **Graph retriever**: BFS from seed vector hits (not full random-walk) — sufficient for project-scale code queries
- **Meta-Thinker**: heuristics only (no LLM call per step) — catches stuck/degraded states without latency
- **Three-tier cascade**: confidence thresholds after each tier — cheaper than always running all three

## Open Questions
- ACC stage 6 integration path: async summarization needs caller-side handling (currently logs warning + pass-through)
- Dense retriever needs re-index for existing DBs — document or auto-migrate?

## Next Action
- Write tests for all 4 new components and rewritten retriever
- Tune RAG cascade thresholds via benchmark queries
- Test ACC staging end-to-end with real agent loop
