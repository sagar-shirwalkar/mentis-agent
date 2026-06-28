"""
RAG (Retrieval-Augmented Generation) subsystem.

Provides:
  - AST-aware chunking (RegexChunker, AstChunker)
  - Deterministic hash-based or ONNX embeddings (Embedder)
  - SQLite-backed code index with symbol + chunk storage (Indexer)
  - Code knowledge graph with BFS expansion (CodeGraph)
  - Three-tier hybrid retrieval: BM25 → Dense → Graph cascade (Retriever)
"""

from coding_agent.rag.chunker import AstChunker, Chunker, RegexChunker
from coding_agent.rag.embedder import Embedder
from coding_agent.rag.graph import CodeGraph
from coding_agent.rag.indexer import Indexer
from coding_agent.rag.retriever import (
    BM25,
    DenseRetriever,
    GraphRetriever,
    Retriever,
    rrf_fuse,
)

__all__ = [
    "AstChunker",
    "BM25",
    "Chunker",
    "CodeGraph",
    "DenseRetriever",
    "Embedder",
    "GraphRetriever",
    "Indexer",
    "RegexChunker",
    "Retriever",
    "rrf_fuse",
]
