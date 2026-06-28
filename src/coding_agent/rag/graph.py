"""
Code knowledge graph: builds AST-derived edges between code chunks.

Edge types:
  - CALLS: function A calls function B
  - IMPORTS: file A imports symbol from file B
  - INHERITS: class A inherits from class B
  - CONTAINS: file contains function/class

The graph is stored alongside the chunk index and queried for
graph-expanded retrieval (third tier in Three-Tier RAG).
"""

from __future__ import annotations

import logging
import re
import sqlite3
from collections import deque

from coding_agent.types import EdgeType, GraphEdge

logger = logging.getLogger(__name__)

# Regex patterns for extracting calls, imports, inheritance
_CALL_PATTERN = re.compile(r"\b(\w+)\s*\(")
_IMPORT_PATTERN_PY = re.compile(
    r"^(?:from\s+(\S+)\s+)?import\s+(\S+)(?:\s+as\s+\S+)?",
    re.MULTILINE,
)
_IMPORT_PATTERN_TS = re.compile(
    r"(?:import\s+\{?\s*(\w+)\s*\}?\s+from\s+['\"](\S+)['\"]|"
    r"import\s+(\w+)\s+from\s+['\"](\S+)['\"])",
)
_INHERIT_PATTERN = re.compile(r"class\s+\w+\s*\((.+?)\):")


class CodeGraph:
    """
    Directed graph of code relationships.

    Edges are stored in a SQLite table alongside the chunk index.
    The graph is built during indexing and queried at retrieval time.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._create_table()

    def _create_table(self) -> None:
        """Create the edges table if it does not exist."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS graph_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_chunk_id INTEGER NOT NULL,
                target_name TEXT NOT NULL,
                target_file TEXT NOT NULL DEFAULT '',
                edge_type TEXT NOT NULL,
                line_number INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (source_chunk_id) REFERENCES chunks(id)
            );

            CREATE INDEX IF NOT EXISTS idx_edges_target
                ON graph_edges(target_name);

            CREATE INDEX IF NOT EXISTS idx_edges_source
                ON graph_edges(source_chunk_id);

            CREATE INDEX IF NOT EXISTS idx_edges_type
                ON graph_edges(edge_type);
        """)
        self._conn.commit()

    def insert_edges(self, chunk_id: int, edges: list[GraphEdge]) -> None:
        """Insert edges for a single chunk."""
        for edge in edges:
            self._conn.execute(
                "INSERT INTO graph_edges "
                "(source_chunk_id, target_name, target_file, edge_type, line_number) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    chunk_id,
                    edge.target_name,
                    edge.target_file,
                    edge.edge_type.value,
                    edge.line_number,
                ),
            )
        if edges:
            self._conn.commit()

    def remove_file_edges(self, file_path: str) -> None:
        """Remove all edges associated with chunks from a file."""
        self._conn.execute(
            "DELETE FROM graph_edges WHERE source_chunk_id IN "
            "(SELECT id FROM chunks WHERE file_path = ?)",
            (file_path,),
        )
        self._conn.commit()

    # ── Graph traversal ───────────────────────────────────────

    def bfs_expand(
        self,
        seed_chunk_ids: list[int],
        max_depth: int = 2,
        max_results: int = 20,
        edge_types: list[EdgeType] | None = None,
    ) -> list[int]:
        """
        Breadth-first search from seed chunks.

        Returns chunk IDs reachable within max_depth hops.
        Filters by edge type if specified.
        """
        type_filter = edge_types or [EdgeType.CALLS, EdgeType.IMPORTS, EdgeType.INHERITS]

        visited: set[int] = set(seed_chunk_ids)
        queue: deque[tuple[int, int]] = deque((cid, 0) for cid in seed_chunk_ids)
        results: list[int] = list(seed_chunk_ids)

        while queue and len(results) < max_results:
            current_id, depth = queue.popleft()

            if depth >= max_depth:
                continue

            # Find chunks this one points to
            for edge_type in type_filter:
                rows = self._conn.execute(
                    "SELECT DISTINCT c.id FROM chunks c "
                    "INNER JOIN graph_edges e ON e.target_name = c.symbol_name "
                    "WHERE e.source_chunk_id = ? AND e.edge_type = ? "
                    "AND c.id NOT IN ({}) "
                    "LIMIT ?".format(",".join("?" * len(visited))),
                    (current_id, edge_type.value) + tuple(visited) + (max_results - len(results),),
                ).fetchall()

                for (neighbor_id,) in rows:
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        results.append(neighbor_id)
                        queue.append((neighbor_id, depth + 1))

        return results[:max_results]

    def get_chunk_ids_for_target(
        self,
        target_name: str,
        edge_type: EdgeType | None = None,
    ) -> list[int]:
        """Find chunk IDs that reference a given target."""
        if edge_type:
            rows = self._conn.execute(
                "SELECT source_chunk_id FROM graph_edges WHERE target_name = ? AND edge_type = ?",
                (target_name, edge_type.value),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT source_chunk_id FROM graph_edges WHERE target_name = ?",
                (target_name,),
            ).fetchall()
        return [r[0] for r in rows]


# ── Edge extraction helpers ─────────────────────────────────


def extract_calls(content: str) -> list[str]:
    """Extract function call targets from a code chunk."""
    matches = _CALL_PATTERN.findall(content)
    # Filter out language keywords
    keywords = {
        "if",
        "else",
        "for",
        "while",
        "return",
        "def",
        "class",
        "import",
        "from",
        "try",
        "except",
        "with",
        "as",
        "not",
        "and",
        "or",
        "in",
        "is",
        "assert",
        "raise",
        "break",
        "continue",
        "pass",
        "elif",
        "self",
        "None",
        "True",
        "False",
        "print",
        "len",
        "range",
        "type",
        "isinstance",
        "hasattr",
        "getattr",
        "setattr",
        "super",
        "map",
        "filter",
        "zip",
        "enumerate",
        "sorted",
        "open",
        "str",
        "int",
        "float",
        "list",
        "dict",
        "set",
        "tuple",
        "const",
        "let",
        "var",
        "function",
        "export",
        "await",
        "async",
        "yield",
        "throw",
        "new",
        "delete",
        "typeof",
    }
    return [m for m in matches if m not in keywords][:10]


def extract_imports(content: str, language: str = "python") -> list[tuple[str, str]]:
    """Extract (source_module, imported_name) pairs."""
    if language == "python":
        return _extract_imports_python(content)
    elif language in ("typescript", "javascript"):
        return _extract_imports_ts(content)
    return []


def _extract_imports_python(content: str) -> list[tuple[str, str]]:
    results: list[tuple[str, str]] = []
    for match in _IMPORT_PATTERN_PY.finditer(content):
        source = match.group(1) or ""
        imported = match.group(2) or ""
        if imported:
            results.append((source, imported))
    return results


def _extract_imports_ts(content: str) -> list[tuple[str, str]]:
    results: list[tuple[str, str]] = []
    for match in _IMPORT_PATTERN_TS.finditer(content):
        name = match.group(1) or match.group(3) or ""
        source = match.group(2) or match.group(4) or ""
        if name and source:
            results.append((source, name))
    return results


def extract_inheritance(content: str) -> list[str]:
    """Extract parent class names from class definitions."""
    match = _INHERIT_PATTERN.search(content)
    if not match:
        return []
    parents = match.group(1)
    return [p.strip() for p in parents.split(",") if p.strip() and p.strip() != "object"]
