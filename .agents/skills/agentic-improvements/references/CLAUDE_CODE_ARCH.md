# Claude Code — Architecture Analysis

Analysis of Claude Code's context engineering, memory, and agent loop architecture from the open-source codebase and documentation.

## Context Compaction Pipeline

Five graduated tiers, 3,960 lines of TypeScript across `src/services/compact/`:

| Tier | Name | Trigger | Behavior |
|---|---|---|---|
| 1 | Budget Reduction | Every turn | Cap individual tool outputs to budget allocation |
| 2 | Snip | Token pressure | Drop low-value tool results (verbose builds, large dumps); protect head/tail boundary |
| 3 | Microcompact | Medium pressure | Cache-aware surgical pruning — two code paths selected by cache state. Hot path uses `cache_edits` (delete cached blocks without rewriting). Cold path uses simple truncation. |
| 4 | Context Collapse | ~90% utilization | **Reversible** read-time projection. Original messages kept in storage; model sees projected view with summaries from collapse store. Collapse runs before autocompact to pre-empt it. |
| 5 | Autocompact | ~167K tokens (200K - 20K output - 13K buffer) | Fork sub-agent with `querySource: 'compact'`, full LLM summary. Post-compaction cleanup: restore plan state, modified files (max 5), skills, reset tool states. |

### Key Innovations

- **Cache-aware dual paths**: Same algorithm, different strategy based on whether the prompt cache is hot or cold
- **Two-phase CoT summarization**: LLM writes chain-of-thought scratchpad, then `formatCompactSummary()` strips the CoT and keeps only the conclusion. Reasoning discarded, conclusion preserved.
- **Compaction as transaction**: prepare → generate summary → clear stale state → restore operational context → continue in same turn
- **Post-compact cleanup**: restores critical operational context that compaction discards (modified files, plan state, skills)

## Memory Architecture

Seven layers:

| Layer | Name | Lifecycle | Storage |
|---|---|---|---|
| 1 | CLAUDE.md | Permanent | Project root, version controlled |
| 2 | Auto Memory (memdir) | Cross-session | `~/.claude/projects/<slug>/memory/` |
| 3 | Background Extract | Cross-session (async) | Same memdir, background worker |
| 4 | Session Memory | Single session | `~/.claude/projects/<slug>/<sessionId>/session-memory/summary.md` |
| 5 | Agent Memory | Cross-session | Three scopes (user/project/local) |
| 6 | Relevant Memories | On-demand per turn | Memory as Attachment |
| 7 | Auto Dream | Idle-triggered | Writes back to memdir |

### MEMORY.md Pattern

First 200 lines or 25KB loaded at session start. Acts as index — one entry per line, each linking to a topic file. Must be kept concise; detailed notes go into separate files. Entries use hooks (specific context) to help the agent decide relevance without reading the file.

### Session Memory

Automatic background system. Watches conversation, extracts important parts, saves structured summaries to disk. No user input required. `/remember` command promotes patterns from session memory → CLAUDE.local.md.

## System Reminder Pipeline

~50 reminder types re-evaluated on every API turn. Critical volatile state (plan progress, file modifications, resource pressure) remains reachable even after compaction discards messages that originally contained it. Includes `session-continuation` reminder: compressed summary of what was accomplished, injected after compaction or resume.

## Key Takeaways for Meredith

1. **Cheapest-first compaction** — progressive tiers avoid triggering expensive LLM summarization until necessary
2. **Cache awareness** — algorithm choice should depend on whether the prompt cache is hot
3. **Reversible compaction** — context collapse keeps original messages, making compaction non-destructive
4. **Post-compaction rehydration** — restoring operational context after summarization is as important as the summary itself
5. **Session memory extraction** — running memory extraction in parallel with the main session, not as an emergency measure
6. **Reminder pipeline** — ~50 injection points for volatile state that survive compaction
