"""Persistent trusted-fact store (SQLite) that keeps a GenealogyGraph in sync.

Design notes
------------
* SQLite is the source of truth. The in-memory GenealogyGraph is a search index
  that is rebuilt from the database on start-up and after every delete.
* Only facts added through this store are persisted. Claims that the middleware
  adds to the graph at runtime (green atoms) are NOT saved, so they disappear on
  restart or after a delete. That keeps the trusted set under your control.
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
        self.db_path = str(db_path)
        self.graph = graph
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS facts ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "content TEXT NOT NULL UNIQUE, "
            "created_at REAL NOT NULL)"
        )
        self._conn.commit()
        self._graph_ids: Dict[int, str] = {}  # database id -> graph fact id
        self.reload()

    @staticmethod
    def _clean(content: str) -> str:
        return " ".join((content or "").split())

    def reload(self) -> int:
        """Rebuild the in-memory graph from the database."""
        with self._lock:
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

    def add(self, content: str) -> Dict:
        """Add one fact. Returns the stored row plus a 'created' flag."""
        text = self._clean(content)
        if not text:
            raise ValueError("fact must not be empty")
        with self._lock:
            row = self._conn.execute(
                "SELECT id, content, created_at FROM facts WHERE content = ?", (text,)
            ).fetchone()
            if row is not None:
                return {**dict(row), "created": False}
            created_at = time.time()
            cur = self._conn.execute(
                "INSERT INTO facts (content, created_at) VALUES (?, ?)", (text, created_at)
            )
            self._conn.commit()
            fact = self.graph.add(
                content=text, agent_id=TRUSTED_AGENT, status=VerificationStatus.VERIFIED
            )
            self._graph_ids[cur.lastrowid] = fact.id
            return {"id": cur.lastrowid, "content": text, "created_at": created_at, "created": True}

    def list(self) -> List[Dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, content, created_at FROM facts ORDER BY id"
            ).fetchall()
            return [dict(r) for r in rows]

    def delete(self, fact_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
            self._conn.commit()
            if cur.rowcount == 0:
                return False
            self.reload()
            return True

    def count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
