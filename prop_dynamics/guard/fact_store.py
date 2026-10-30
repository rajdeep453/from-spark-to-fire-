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
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS facts ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "content TEXT NOT NULL UNIQUE, "
            "source TEXT NOT NULL DEFAULT 'manual', "
            "created_at REAL NOT NULL)"
        )
        columns = {row["name"] for row in self._conn.execute("PRAGMA table_info(facts)")}
        if "source" not in columns:
            self._conn.execute(
                "ALTER TABLE facts ADD COLUMN source TEXT NOT NULL DEFAULT 'manual'"
            )
        self._conn.commit()
        self._graph_ids: Dict[int, str] = {}  # database id -> graph fact id
        self.reload()

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

    def reload(self) -> int:
        """Rebuild the in-memory graph from the database."""
        with self._lock:
            self._ensure_open()
            self.graph.clear()
            self._graph_ids.clear()
            rows = self._conn.execute("SELECT id, content FROM facts ORDER BY id").fetchall()
            for row in rows:
                fact = self.graph.add(
                    content=row["content"],
                    agent_id=TRUSTED_AGENT,
                    status=VerificationStatus.VERIFIED,
                )
                self._graph_ids[row["id"]] = fact.id
            return len(rows)

    def add(self, content: str, source: str = "manual") -> Dict:
        """Add one fact. Returns the stored row plus a 'created' flag."""
        return self.add_many([content], source=source)[0]

    def add_many(self, contents: List[str], source: str = "manual") -> List[Dict]:
        """Atomically add a batch and keep the graph in sync with the database."""
        if isinstance(contents, (str, bytes)) or not isinstance(contents, list):
            raise TypeError("contents must be a list of strings")
        if not contents:
            raise ValueError("contents must not be empty")
        cleaned = [self._clean(content, "fact", 8000) for content in contents]
        source = self._clean(source, "source", 500)
        with self._lock:
            self._ensure_open()
            try:
                with self._conn:
                    results = []
                    for text in cleaned:
                        created_at = time.time()
                        cur = self._conn.execute(
                            "INSERT OR IGNORE INTO facts (content, source, created_at) VALUES (?, ?, ?)",
                            (text, source, created_at),
                        )
                        row = self._conn.execute(
                            "SELECT id, content, source, created_at FROM facts WHERE content = ?", (text,)
                        ).fetchone()
                        if cur.rowcount:
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

    def list(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("limit must be an integer between 1 and 1000")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        with self._lock:
            self._ensure_open()
            rows = self._conn.execute(
                "SELECT id, content, source, created_at FROM facts ORDER BY id LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            return [dict(r) for r in rows]

    def delete(self, fact_id: int) -> bool:
        if isinstance(fact_id, bool) or not isinstance(fact_id, int) or fact_id < 1:
            raise ValueError("fact_id must be a positive integer")
        with self._lock:
            self._ensure_open()
            cur = self._conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
            self._conn.commit()
            if cur.rowcount == 0:
                return False
            self.reload()
            return True

    def count(self) -> int:
        with self._lock:
            self._ensure_open()
            return self._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]

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
