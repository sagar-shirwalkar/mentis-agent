# Lightweight RAG for Constrained Hardware

Survey of production-ready RAG approaches that avoid PyTorch dependency and run on 8-24GB systems.

## Approaches

### VelociRAG (Recommended Pattern)

**Stack:** ONNX Runtime + FAISS CPU + SQLite FTS5 + TinyBERT reranker
**Footprint:** ~80MB models, <1GB RAM, no GPU, no PyTorch

Four-layer fusion: vector similarity (MiniLM ONNX) + BM25 (SQLite FTS5) + knowledge graph + metadata filtering. Fused via Reciprocal Rank Fusion. Cross-encoder reranker (TinyBERT via ONNX). MCP server included.

```yaml
rag:
  strategy: velocirag_pattern
  layers:
    - vector      # ONNX MiniLM + FAISS IndexFlatIP
    - keyword     # BM25 (SQLite FTS5)
    - graph       # AST-derived knowledge graph
    - metadata    # File type, path patterns
  fusion: rrf
  reranker: tinybert_onnx
```

### SwiftRAG (Simplest Drop-in)

**Stack:** numpy (core) + optional FAISS
**Footprint:** Zero heavy dependencies

One-call pipeline: chunk → embed → index → search. Built-in hashing embedder works offline with no API key. Optional sentence-transformers for quality.

### MiniRAG (Hybrid Balance)

**Stack:** FastText + FAISS + Tantivy (Rust BM25) + SQLite
**Footprint:** ~200MB total

Hybrid dense/sparse with tunable alpha. Tantivy provides high-performance BM25 in Rust with Python bindings. FAISS IndexFlatIP for cosine via inner product. Fully configuration-driven.

### Key Architectural Choices

| Decision | Option A (Heavy) | Option B (Light) |
|---|---|---|
| Embedding backend | sentence-transformers (PyTorch) | ONNX Runtime / FastText / hash |
| Vector index | FAISS IVF | FAISS IndexFlatIP (brute-force, fine for <100K chunks) |
| Keyword index | Elasticsearch | SQLite FTS5 / Tantivy |
| Reranker | Cross-encoder (PyTorch) | Cross-encoder (ONNX) or skip |
| Fusion | Learned weights | Reciprocal Rank Fusion (fixed k=60) |

### Adaptive-k (EMNLP 2025)

Dynamic top-k based on similarity distribution statistics. Up to 99% token reduction on factoid QA. No training needed. Simple thresholding: compute mean and std of similarity scores, set k = count of scores above `mean + λ * std`.

```python
def adaptive_k(scores, lambda_val=0.5, min_k=1, max_k=50):
    threshold = scores.mean() + lambda_val * scores.std()
    return min(max((scores > threshold).sum(), min_k), max_k)
```

## Relevance to Meredith

Meredith's current retriever is BM25-only with a comment that dense retrieval is "optional and can be added later." The lightweight RAG stack (ONNX MiniLM + FAISS + BM25) would add semantic search with:
- No PyTorch dependency
- <500MB additional footprint
- ~3ms warm search via FAISS flat index
- RRF fusion for hybrid ranking
