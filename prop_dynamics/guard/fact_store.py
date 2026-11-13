"""Persistent trusted-fact store (SQLite) that keeps a GenealogyGraph in sync.

Design notes
------------
* SQLite is the source of truth. The in-memory GenealogyGraph is a search index
  that is rebuilt from the database on start-up and after every delete.
* Only facts added through this store are persisted. Claims that the middleware
  adds to the graph at runtime (green verified and yellow unverified atoms) are
  NOT saved, so they disappear on restart or after a delete. That keeps the
  trusted set under your control.
* Rebuilding re-embeds every fact, which is fine for hundreds of facts. For
  thousands you would want incremental removal.
"""

import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

from prop_dynamics.defense.genealogy_graph import GenealogyGraph, VerificationStatus

TRUSTED_AGENT = "trusted"


class FactStore:
    def __init__(self, db_path: Union[str, Path], graph: GenealogyGraph):
        if not isinstance(db_path, (str, Path)) or not str(db_path).strip():
            raise ValueError("db_path must be a non-empty path")
        if graph is None:
            raise TypeError("graph is required")
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            expanded_path = Path(self.db_path).expanduser()
            expanded_path.resolve().parent.mkdir(parents=True, exist_ok=True)
            self.db_path = str(expanded_path)
        self.graph = graph
        self._lock = threading.RLock()
        self._closed = False
        self._conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()
        self._graph_ids: Dict[int, str] = {}  # database id -> graph fact id
        self.reload()

    def _ensure_schema(self) -> None:
        """Create the scoped schema or migrate the original global-only table."""
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS facts ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "content TEXT NOT NULL, "
            "source TEXT NOT NULL DEFAULT 'manual', "
            "scope TEXT NOT NULL DEFAULT 'global', "
            "created_at REAL NOT NULL, "
            "UNIQUE(scope, content))"
        )
        columns = {row["name"] for row in self._conn.execute("PRAGMA table_info(facts)")}
        if "scope" in columns:
            self._conn.commit()
            return

        # SQLite cannot add a uniqueness constraint to the legacy table.
        # Rebuild it in one transaction and preserve existing row IDs.
        legacy_source = "source" if "source" in columns else "'manual'"
        with self._conn:
            self._conn.execute("ALTER TABLE facts RENAME TO facts_legacy")
            self._conn.execute(
                "CREATE TABLE facts ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "content TEXT NOT NULL, "
                "source TEXT NOT NULL DEFAULT 'manual', "
                "scope TEXT NOT NULL DEFAULT 'global', "
                "created_at REAL NOT NULL, "
                "UNIQUE(scope, content))"
            )
            self._conn.execute(
                "INSERT INTO facts (id, content, source, scope, created_at) "
                f"SELECT id, content, {legacy_source}, 'global', created_at FROM facts_legacy"
            )
            self._conn.execute("DROP TABLE facts_legacy")

    @staticmethod
    def _clean(content: str, field: str, max_length: int) -> str:
        if not isinstance(content, str):
            raise TypeError(f"{field} must be a string")
        text = " ".join(content.split())
        if not text:
            raise ValueError(f"{field} must not be empty")
        if len(text) > max_length:
            raise ValueError(f"{field} must be at most {max_length} characters")
        return text

    @classmethod
    def _clean_scope(cls, scope: str) -> str:
        return cls._clean(scope, "scope", 200)

    def reload(self) -> int:
        """Rebuild the default graph from global facts only.

        Task-scoped facts are loaded into a new graph for each check so that
        concurrent or consecutive tasks cannot influence one another.
        """
        with self._lock:
            self._ensure_open()
            self.graph.clear()
            self._graph_ids.clear()
            rows = self._conn.execute(
                "SELECT id, content FROM facts WHERE scope = 'global' ORDER BY id"
            ).fetchall()
            for row in rows:
                fact = self.graph.add(
                    content=row["content"],
                    agent_id=TRUSTED_AGENT,
                    status=VerificationStatus.VERIFIED,
                )
                self._graph_ids[row["id"]] = fact.id
            return len(rows)

    def add(self, content: str, source: str = "manual", scope: str = "global") -> Dict:
        """Add one fact. Returns the stored row plus a 'created' flag."""
        return self.add_many([content], source=source, scope=scope)[0]

    def add_many(self, contents: List[str], source: str = "manual", scope: str = "global") -> List[Dict]:
        """Atomically add a batch and keep the graph in sync with the database."""
        if isinstance(contents, (str, bytes)) or not isinstance(contents, list):
            raise TypeError("contents must be a list of strings")
        if not contents:
            raise ValueError("contents must not be empty")
        cleaned = [self._clean(content, "fact", 8000) for content in contents]
        source = self._clean(source, "source", 500)
        scope = self._clean_scope(scope)
        with self._lock:
            self._ensure_open()
            try:
                with self._conn:
                    results = []
                    for text in cleaned:
                        created_at = time.time()
                        cur = self._conn.execute(
                            "INSERT OR IGNORE INTO facts (content, source, scope, created_at) VALUES (?, ?, ?, ?)",
                            (text, source, scope, created_at),
                        )
                        row = self._conn.execute(
                            "SELECT id, content, source, scope, created_at FROM facts "
                            "WHERE scope = ? AND content = ?", (scope, text)
                        ).fetchone()
                        if cur.rowcount and scope == "global":
                            fact = self.graph.add(
                                content=text, agent_id=TRUSTED_AGENT,
                                status=VerificationStatus.VERIFIED,
                            )
                            self._graph_ids[row["id"]] = fact.id
                        results.append({**dict(row), "created": bool(cur.rowcount)})
                return results
            except Exception:
                # The SQLite transaction may have rolled back after graph.add.
                self.reload()
                raise

    def list(self, limit: int = 100, offset: int = 0, scope: Optional[str] = None) -> List[Dict]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("limit must be an integer between 1 and 1000")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        with self._lock:
            self._ensure_open()
            if scope is None:
                rows = self._conn.execute(
                    "SELECT id, content, source, scope, created_at FROM facts ORDER BY id LIMIT ? OFFSET ?",
                    (limit, offset),
                ).fetchall()
            else:
                scope = self._clean_scope(scope)
                rows = self._conn.execute(
                    "SELECT id, content, source, scope, created_at FROM facts "
                    "WHERE scope = ? ORDER BY id LIMIT ? OFFSET ?",
                    (scope, limit, offset),
                ).fetchall()
            return [dict(r) for r in rows]

    def delete(self, fact_id: int) -> bool:
        if isinstance(fact_id, bool) or not isinstance(fact_id, int) or fact_id < 1:
            raise ValueError("fact_id must be a positive integer")
        with self._lock:
            self._ensure_open()
            row = self._conn.execute("SELECT scope FROM facts WHERE id = ?", (fact_id,)).fetchone()
            if row is None:
                return False
            cur = self._conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
            self._conn.commit()
            if cur.rowcount and row["scope"] == "global":
                self.reload()
            return bool(cur.rowcount)

    def clear_scope(self, scope: str) -> int:
        """Delete facts in one scope and return the number removed."""
        scope = self._clean_scope(scope)
        with self._lock:
            self._ensure_open()
            cur = self._conn.execute("DELETE FROM facts WHERE scope = ?", (scope,))
            self._conn.commit()
            if scope == "global":
                self.reload()
            return int(cur.rowcount)

    def count(self, scope: Optional[str] = None) -> int:
        with self._lock:
            self._ensure_open()
            if scope is None:
                return self._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
            scope = self._clean_scope(scope)
            return self._conn.execute(
                "SELECT COUNT(*) FROM facts WHERE scope = ?", (scope,)
            ).fetchone()[0]

    def graph_for_scope(self, scope: str) -> GenealogyGraph:
        """Return a fresh graph containing global and requested scoped facts.

        The returned graph is intentionally request-local. Middleware can add
        lineage nodes while screening a message without contaminating a later
        message or a different task.
        """
        scope = self._clean_scope(scope)
        with self._lock:
            self._ensure_open()
            graph = GenealogyGraph(
                embedding_dim=self.graph.embedding_dim,
                similarity_threshold=self.graph.similarity_threshold,
            )
            # The template model is immutable after application startup and is
            # shared by sentence-transformers internally. Reusing it avoids a
            # model load for every request.
            graph._local_model = self.graph._local_model
            graph._embedding_func = self.graph._embedding_func
            if scope == "global":
                rows = self._conn.execute(
                    "SELECT content FROM facts WHERE scope = 'global' ORDER BY id"
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT content FROM facts WHERE scope IN ('global', ?) "
                    "ORDER BY CASE scope WHEN 'global' THEN 0 ELSE 1 END, id",
                    (scope,),
                ).fetchall()
            seen = set()
            for row in rows:
                content = row["content"]
                if content in seen:
                    continue
                seen.add(content)
                graph.add(
                    content=content,
                    agent_id=TRUSTED_AGENT,
                    status=VerificationStatus.VERIFIED,
                )
            return graph

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("fact store is closed")

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._conn.close()
                self._closed = True

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
