"""
Three-tier hybrid retriever: BM25 → Dense → Graph cascade.

Tier 1 (BM25): fast keyword search, always on, no GPU.
Tier 2 (Dense): semantic search via numpy/ONNX embeddings.
Tier 3 (Graph): AST-derived knowledge graph with BFS expansion.

Fusion: Reciprocal Rank Fusion (RRF) combines results from
active tiers. Adaptive-k selects dynamic top-k based on score
distribution.
"""

from __future__ import annotations

import logging
import math
import re

import numpy as np

from coding_agent.config import AppConfig
from coding_agent.rag.graph import CodeGraph
from coding_agent.rag.indexer import Indexer
from coding_agent.types import EdgeType, SearchResult

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# BM25 implementation (unchanged from original)
# ──────────────────────────────────────────────────────────────


class BM25:
    """
    Okapi BM25 ranking for code chunks.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._doc_freqs: dict[str, int] = {}
        self._doc_lengths: dict[int, int] = {}
        self._avg_dl: float = 0.0
        self._n_docs: int = 0
        self._doc_tokens: dict[int, dict[str, int]] = {}

    def index(self, doc_tokens: dict[int, dict[str, int]]) -> None:
        self._doc_tokens = doc_tokens
        self._n_docs = len(doc_tokens)
        self._doc_freqs = {}
        total_length = 0
        for doc_id, freqs in doc_tokens.items():
            doc_length = sum(freqs.values())
            self._doc_lengths[doc_id] = doc_length
            total_length += doc_length
            for token in freqs:
                self._doc_freqs[token] = self._doc_freqs.get(token, 0) + 1
        self._avg_dl = total_length / self._n_docs if self._n_docs > 0 else 1.0

    def score(self, query_tokens: list[str], top_k: int = 10) -> list[tuple[int, float]]:
        scores: dict[int, float] = {}
        for term in query_tokens:
            if term not in self._doc_freqs:
                continue
            df = self._doc_freqs[term]
            idf = math.log((self._n_docs - df + 0.5) / (df + 0.5) + 1.0)
            for doc_id, freqs in self._doc_tokens.items():
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                dl = self._doc_lengths[doc_id]
                tf_norm = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / self._avg_dl)
                )
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf_norm
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


# ──────────────────────────────────────────────────────────────
# Dense retriever (Tier 2)
# ──────────────────────────────────────────────────────────────


class DenseRetriever:
    """
    Semantic search via embedding vectors.

    Uses numpy cosine similarity. The embedder (numpy_random or ONNX)
    is shared with the indexer.
    """

    def __init__(self, indexer: Indexer) -> None:
        self.indexer = indexer
        self._vectors: dict[int, list[float]] = {}
        self._chunk_ids: list[int] = []
        self._ready = False

    async def start(self) -> None:
        """Load embeddings into memory."""
        try:
            self._vectors, self._chunk_ids = self.indexer.get_all_embeddings()
            self._ready = len(self._vectors) > 0
            if self._ready:
                logger.info("Dense retriever ready: %d vectors", len(self._vectors))
        except Exception as exc:
            logger.warning("Dense retriever failed to load: %s", exc)
            self._ready = False

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """Search by embedding cosine similarity."""
        if not self._ready or not self._chunk_ids:
            return []

        query_vec = np.array(self.indexer.embedder.embed(query), dtype=np.float32)
        if np.linalg.norm(query_vec) == 0:
            return []

        scores: list[tuple[int, float]] = []
        for chunk_id in self._chunk_ids:
            vec = self._vectors.get(chunk_id)
            if vec is None:
                continue
            doc_vec = np.array(vec, dtype=np.float32)
            sim = float(np.dot(query_vec, doc_vec))
            scores.append((chunk_id, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ──────────────────────────────────────────────────────────────
# Graph retriever (Tier 3)
# ──────────────────────────────────────────────────────────────


class GraphRetriever:
    """
    Graph-expanded search from seed chunks.

    BFS traversal through CALLS, IMPORTS, and INHERITS edges.
    Triggered when Tier 1+2 confidence is low.
    """

    def __init__(self, indexer: Indexer) -> None:
        self.indexer = indexer
        self.graph: CodeGraph | None = None
        self._ready = False

    async def start(self) -> None:
        """Reference the graph from the indexer."""
        self.graph = self.indexer.graph
        self._ready = self.graph is not None

    def expand(
        self,
        seed_results: list[SearchResult],
        max_depth: int = 2,
        max_results: int = 10,
    ) -> list[SearchResult]:
        """BFS-expand from seed results to find related code."""
        if not self._ready or not self.graph or not seed_results:
            return []

        assert self.indexer._conn is not None

        # Find chunk IDs for seed results
        seed_ids: list[int] = []
        for sr in seed_results:
            rows = self.indexer._conn.execute(
                "SELECT id FROM chunks WHERE file_path = ? AND line_start = ?",
                (sr.file_path, sr.line_start),
            ).fetchone()
            if rows:
                seed_ids.append(rows[0])

        if not seed_ids:
            return []

        # BFS expansion
        expanded_ids = self.graph.bfs_expand(
            seed_ids,
            max_depth=max_depth,
            max_results=max_results,
            edge_types=[EdgeType.CALLS, EdgeType.IMPORTS, EdgeType.INHERITS],
        )

        # Exclude original seeds
        expanded_ids = [cid for cid in expanded_ids if cid not in set(seed_ids)]

        # Fetch chunk data
        results: list[SearchResult] = []
        for chunk_id in expanded_ids:
            row = self.indexer._conn.execute(
                "SELECT file_path, line_start, line_end, content, symbol_name "
                "FROM chunks WHERE id = ?",
                (chunk_id,),
            ).fetchone()
            if row:
                results.append(
                    SearchResult(
                        content=row[3][:500],
                        file_path=row[0],
                        line_start=row[1],
                        line_end=row[2],
                        score=0.5,  # Graph results get moderate base score
                        symbol_name=row[4],
                        source="graph",
                    )
                )

        return results


# ──────────────────────────────────────────────────────────────
# Fusion helpers
# ──────────────────────────────────────────────────────────────


def rrf_fuse(
    result_lists: list[list[tuple[SearchResult, float]]],
    k: int = 60,
    top_k: int = 10,
) -> list[SearchResult]:
    """
    Reciprocal Rank Fusion.

    Each result list has (SearchResult, weight) pairs.
    Higher weight means more influence in the fusion.
    """
    scores: dict[tuple[str, int], float] = {}

    for results in result_lists:
        for rank, (result, weight) in enumerate(results):
            key = (result.file_path, result.line_start)
            scores[key] = scores.get(key, 0.0) + weight / (k + rank)

    # Look up full result objects
    seen: dict[tuple[str, int], SearchResult] = {}
    for results in result_lists:
        for result, _weight in results:
            key = (result.file_path, result.line_start)
            if key not in seen:
                seen[key] = result

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [seen[key] for key, _score in ranked[:top_k]]


def adaptive_k(scores: list[float], lambda_val: float = 0.5, min_k: int = 1) -> int:
    """Select k based on score distribution."""
    arr = np.array(scores, dtype=np.float64)
    if len(arr) < 3:
        return len(arr)
    threshold = float(np.mean(arr) + lambda_val * float(np.std(arr)))
    return max(min_k, int(np.sum(arr > threshold)))


# ──────────────────────────────────────────────────────────────
# Three-Tier Retriever
# ──────────────────────────────────────────────────────────────


class Retriever:
    """
    Three-tier hybrid retriever.

    Cascade: BM25 (fast) → confidence? → Dense (medium) → confidence? → Graph (slow)
    At each stage, if confidence > threshold, skip deeper tiers.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.indexer = Indexer(config)
        self.bm25 = BM25()
        self.dense = DenseRetriever(self.indexer)
        self.graph_retriever = GraphRetriever(self.indexer)
        self._bm25_indexed = False

    async def start(self) -> None:
        """Start all tiers."""
        await self.indexer.start()

        chunk_count = self.indexer.get_chunk_count()
        if chunk_count == 0 or self.config.rag.reindex_on_startup:
            result = await self.indexer.index_project(force=self.config.rag.reindex_on_startup)
            logger.info("Indexed: %s", result)

        self._build_bm25_index()
        await self.dense.start()
        await self.graph_retriever.start()

    async def close(self) -> None:
        await self.indexer.close()

    def _build_bm25_index(self) -> None:
        doc_tokens = self.indexer.get_all_chunk_freqs()
        if doc_tokens:
            self.bm25.index(doc_tokens)
            self._bm25_indexed = True
            logger.info("BM25 index built: %d documents", len(doc_tokens))

    # ── Main search entry point ───────────────────────────────

    def search(
        self,
        query: str,
        top_k: int | None = None,
        search_type: str = "hybrid",
        confidence_threshold: float = 0.7,
    ) -> list[SearchResult]:
        """
        Three-tier cascade search.

        Args:
            query: Natural language or code query.
            top_k: Number of results to return.
            search_type: "bm25" | "symbol" | "dense" | "graph" | "hybrid"
            confidence_threshold: Skip deeper tiers if BM25 confidence exceeds this.

        Returns:
            Ranked list of SearchResult objects.
        """
        k = top_k or self.config.rag.retrieval.top_k

        if search_type == "symbol":
            return self._symbol_search(query, k)
        if search_type == "bm25":
            return self._bm25_search(query, k)
        if search_type == "dense":
            return self._dense_search(query, k)
        if search_type == "graph":
            return self._graph_search(query, k)

        return self._three_tier_search(query, k, confidence_threshold)

    def _three_tier_search(
        self,
        query: str,
        top_k: int,
        confidence_threshold: float,
    ) -> list[SearchResult]:
        """
        Three-tier cascade:

        Tier 1: BM25 → confidence > threshold? → return
        Tier 2: Dense → RRF with Tier 1 → confidence > threshold? → return
        Tier 3: Graph expansion → RRF all three → return
        """
        # Tier 1: BM25
        bm25_results = self._bm25_search(query, top_k)
        bm25_confidence = self._estimate_confidence(bm25_results, query)
        logger.debug(
            "Tier 1 (BM25): %d results, confidence=%.2f",
            len(bm25_results),
            bm25_confidence,
        )

        if bm25_confidence >= confidence_threshold:
            return bm25_results

        # Tier 2: Dense + RRF
        dense_results = self._dense_search(query, top_k)
        logger.debug("Tier 2 (Dense): %d results", len(dense_results))

        tier2_fused = rrf_fuse(
            [
                [(r, 0.7) for r in bm25_results],
                [(r, 0.3) for r in dense_results],
            ],
            top_k=top_k,
        )
        tier2_confidence = self._estimate_confidence(tier2_fused, query)

        if tier2_confidence >= confidence_threshold:
            return tier2_fused

        # Tier 3: Graph expansion + RRF all three
        graph_results = self._graph_search(query, top_k, seed_results=tier2_fused)
        logger.debug("Tier 3 (Graph): %d results", len(graph_results))

        return rrf_fuse(
            [
                [(r, 0.4) for r in bm25_results],
                [(r, 0.3) for r in dense_results],
                [(r, 0.3) for r in graph_results],
            ],
            top_k=top_k,
        )

    @staticmethod
    def _estimate_confidence(results: list[SearchResult], query: str) -> float:
        """
        Estimate search confidence from result quality.

        Factors: top score, score drop-off, result count.
        Returns 0.0-1.0.
        """
        if not results:
            return 0.0

        # Top score factor
        top_score = results[0].score

        drop_off = results[0].score - results[1].score if len(results) > 1 else 1.0

        # Result count factor (fewer results = less confident)
        count_factor = min(len(results) / 5, 1.0)

        confidence = top_score * 0.4 + min(drop_off, 1.0) * 0.3 + count_factor * 0.3
        return min(confidence, 1.0)

    # ── Symbol search ─────────────────────────────────────────

    def find_symbol(self, name: str) -> list[SearchResult]:
        symbols = self.indexer.search_symbols(name, limit=10)
        results: list[SearchResult] = []
        for sym in symbols:
            content = sym.signature
            if sym.docstring:
                content += f"\n  {sym.docstring[:100]}"
            results.append(
                SearchResult(
                    content=content,
                    file_path=sym.file_path,
                    line_start=sym.line_start,
                    line_end=sym.line_end,
                    score=1.0 if sym.name == name else 0.5,
                    symbol_name=sym.name,
                    source="symbol",
                )
            )
        return results

    def find_symbol_body(self, name: str, file_path: str | None = None) -> SearchResult | None:
        body = self.indexer.get_symbol_body(name, file_path)
        if body is None:
            return None
        symbols = self.indexer.search_symbols(name, limit=1)
        if not symbols:
            return None
        sym = symbols[0]
        return SearchResult(
            content=body,
            file_path=sym.file_path,
            line_start=sym.line_start,
            line_end=sym.line_end,
            score=1.0,
            symbol_name=sym.name,
            source="symbol",
        )

    def project_overview(self) -> str:
        assert self.indexer._conn is not None
        rows = self.indexer._conn.execute(
            "SELECT DISTINCT file_path FROM symbols ORDER BY file_path"
        ).fetchall()
        files = [r[0] for r in rows]
        if not files:
            return f"Project at {self.config.agent.working_directory} (not yet indexed)"
        lines: list[str] = [f"Project structure ({len(files)} files with symbols):"]
        for file_path in files[:30]:
            symbols = self.indexer.get_file_symbols(file_path)
            sym_names = [f"{s.kind.value} {s.name}" for s in symbols[:8]]
            if len(symbols) > 8:
                sym_names.append(f"... +{len(symbols) - 8} more")
            lines.append(f"  {file_path}: {', '.join(sym_names)}")
        if len(files) > 30:
            lines.append(f"  ... +{len(files) - 30} more files")
        return "\n".join(lines)

    # ── Internal search methods ───────────────────────────────

    def _symbol_search(self, query: str, top_k: int) -> list[SearchResult]:
        symbols = self.indexer.search_symbols(query, limit=top_k)
        results: list[SearchResult] = []
        for sym in symbols:
            content = sym.signature
            if sym.docstring:
                content += f"\n  {sym.docstring[:100]}"
            results.append(
                SearchResult(
                    content=content,
                    file_path=sym.file_path,
                    line_start=sym.line_start,
                    line_end=sym.line_end,
                    score=1.0 if sym.name == query else 0.7,
                    symbol_name=sym.name,
                    source="symbol",
                )
            )
        return results

    def _bm25_search(self, query: str, top_k: int) -> list[SearchResult]:
        if not self._bm25_indexed:
            return []
        query_tokens = re.findall(r"\w+", query.lower())
        if not query_tokens:
            return []
        scored = self.bm25.score(query_tokens, top_k=top_k * 2)
        assert self.indexer._conn is not None
        results: list[SearchResult] = []
        for chunk_id, score in scored[:top_k]:
            row = self.indexer._conn.execute(
                "SELECT file_path, line_start, line_end, content, symbol_name, symbol_kind "
                "FROM chunks WHERE id = ?",
                (chunk_id,),
            ).fetchone()
            if row:
                results.append(
                    SearchResult(
                        content=row[3],
                        file_path=row[0],
                        line_start=row[1],
                        line_end=row[2],
                        score=score,
                        symbol_name=row[4],
                        source="bm25",
                    )
                )
        return results

    def _dense_search(self, query: str, top_k: int) -> list[SearchResult]:
        """Dense semantic search via embeddings."""
        scored = self.dense.search(query, top_k=top_k)
        assert self.indexer._conn is not None
        results: list[SearchResult] = []
        for chunk_id, score in scored:
            row = self.indexer._conn.execute(
                "SELECT file_path, line_start, line_end, content, symbol_name "
                "FROM chunks WHERE id = ?",
                (chunk_id,),
            ).fetchone()
            if row:
                results.append(
                    SearchResult(
                        content=row[3],
                        file_path=row[0],
                        line_start=row[1],
                        line_end=row[2],
                        score=float(score),
                        symbol_name=row[4],
                        source="dense",
                    )
                )
        return results

    def _graph_search(
        self,
        query: str,
        top_k: int,
        seed_results: list[SearchResult] | None = None,
    ) -> list[SearchResult]:
        """Graph-expanded search. Uses BM25 results as seeds if none provided."""
        if not seed_results:
            seed_results = self._bm25_search(query, top_k=3)

        return self.graph_retriever.expand(
            seed_results,
            max_depth=2,
            max_results=top_k,
        )

    # ── Legacy hybrid (for backward compat) ───────────────────

    def _hybrid_search(self, query: str, top_k: int) -> list[SearchResult]:
        """Legacy BM25 + symbol hybrid."""
        bm25_weight = self.config.rag.retrieval.bm25_weight
        symbol_weight = 1.0 - bm25_weight

        symbol_results = self._symbol_search(query, top_k=top_k)
        bm25_results = self._bm25_search(query, top_k=top_k)

        if bm25_results:
            max_bm25 = max(r.score for r in bm25_results)
            if max_bm25 > 0:
                for r in bm25_results:
                    r.score = (r.score / max_bm25) * bm25_weight

        for r in symbol_results:
            r.score = r.score * symbol_weight

        seen: set[tuple[str, int]] = set()
        merged: list[SearchResult] = []
        for r in symbol_results:
            key = (r.file_path, r.line_start)
            if key not in seen:
                seen.add(key)
                merged.append(r)
        for r in bm25_results:
            key = (r.file_path, r.line_start)
            if key not in seen:
                seen.add(key)
                merged.append(r)
        merged.sort(key=lambda r: r.score, reverse=True)
        return merged[:top_k]
