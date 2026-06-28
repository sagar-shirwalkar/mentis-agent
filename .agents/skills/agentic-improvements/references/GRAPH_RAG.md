# Graph-Assisted RAG for Code ASTs

Survey of deterministic graph-based retrieval for code understanding, using AST parsing rather than LLM-based entity extraction.

## Key Insight

For code, **deterministic AST-derived graphs beat LLM-extracted graphs** on multi-hop structural questions (Reliable Graph-RAG for Codebases, arXiv 2026). Lower indexing cost, higher reliability, no hallucination risk.

## Approaches

### PyCodeKG (Closest to Meredith's Stack)

**Stack:** Python + tree-sitter + SQLite + LanceDB (or any vector DB)
**Edge types:** CONTAINS, CALLS, IMPORTS, INHERITS, RESOLVES_TO

Two-phase search:
1. **Vector phase**: query → embedding → k closest functions/classes by cosine similarity
2. **Graph expansion phase**: BFS hop expansion along typed edges from seed hits

Structure is ground truth; embeddings are an acceleration layer. When graph and vector disagree, graph wins.

```python
def search(query, k=5, hops=2):
    seeds = vector_index.search(query, k)
    results = set(seeds)
    for seed in seeds:
        results |= graph.bfs(seed, hops=hops, edge_types=["CALLS", "INHERITS", "CONTAINS"])
    return results
```

### ZoomRAG (ACL 2026 — Fast RWR)

Hierarchical random-walk zooming across multi-scale graphs:
1. **Coarse level**: global relational graph → query-initiated random walk → locate relevant documents
2. **Fine level**: zoom into selected documents → second random walk → pinpoint salient chunks

**0.019s per query** with concurrent processing (algorithm-parallel ZoomRAG). 2.2-4.9% absolute accuracy gains over SOTA RAG.

The "fast" part comes from the coarse-to-fine hierarchy, not algorithmic shortcuts — the coarse level quickly narrows the search space before the expensive fine-level walk.

### GraphRAG-MCP (MCP-Ready)

**Stack:** tree-sitter + Rust Leiden + ONNX + sqlite-vec + FTS5
Zero LLM cost ingestion. Leiden community detection clusters code into semantic neighborhoods. MCP server exposes `local_search`, `global_search`, `get_graph_topology`.

### CodeGraph (Rust, Fastest)

**Stack:** Rust + tree-sitter + FastML + SurrealDB
**Index tiers:** fast (AST nodes + core edges), balanced (+LSP symbols), full (+dataflow + architecture)

## Integration Pattern for Meredith

Since Meredith already uses tree-sitter AST chunking (`rag/chunker.py`), the graph builder would:
1. Reuse the existing tree-sitter parsing pipeline
2. Extract edges: function A calls function B, class C inherits from D, module E imports F
3. Store in SQLite (already used by `rag/indexer.py`) or a lightweight graph store
4. At query time: BM25/dense → RRF fusion → BFS expansion from top hits

```
Query → BM25 dense → top-k chunks → graph BFS expansion → enriched context
```

## Fast RWR Assessment

ZoomRAG's hierarchical random walk is the paper-proven fast approach. For code AST graphs (DAGs with bounded fan-out), BFS from vector seed hits (PyCodeKG's approach) is simpler and likely as effective for typical code queries. Full RWR over an entire code graph would be overkill — AST call graphs have at most hundreds of nodes for a typical project module. The ZoomRAG pattern would be relevant if Meredith indexed cross-repository or monorepo-scale code (10K+ files).
