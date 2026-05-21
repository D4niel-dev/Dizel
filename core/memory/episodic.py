"""
core/memory/episodic.py — Disk-backed episodic memory with SQLite.

Stores past interactions, notable events, and task history.
Survives across sessions. Indexed for fast retrieval.
"""

import json
import os
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional

from .schema import MemoryItem, MemoryType
from .store import MemoryQuery, MemoryStore

_DEFAULT_DB = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "data", "memory", "episodic.db"
)


class EpisodicMemory(MemoryStore):
    """SQLite-backed store for past interactions and task history."""

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = db_path or os.path.normpath(_DEFAULT_DB)
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id          TEXT PRIMARY KEY,
                type        TEXT NOT NULL DEFAULT 'episodic',
                source      TEXT NOT NULL,
                content     TEXT NOT NULL,
                metadata    TEXT DEFAULT '{}',
                timestamp   TEXT NOT NULL,
                confidence  REAL DEFAULT 1.0,
                relevance   REAL DEFAULT 0.0,
                expiry      TEXT,
                tags        TEXT DEFAULT '[]'
            );
            CREATE INDEX IF NOT EXISTS idx_mem_source ON memories(source);
            CREATE INDEX IF NOT EXISTS idx_mem_timestamp ON memories(timestamp);
            CREATE INDEX IF NOT EXISTS idx_mem_type ON memories(type);
        """)
        self._conn.commit()

    # ── MemoryStore interface ─────────────────────────────────────────────

    def write(self, item: MemoryItem) -> str:
        item.type = MemoryType.EPISODIC
        self._conn.execute(
            """INSERT OR REPLACE INTO memories
               (id, type, source, content, metadata, timestamp, confidence, relevance, expiry, tags)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                item.id,
                item.type.value,
                item.source,
                item.content,
                json.dumps(item.metadata),
                item.timestamp.isoformat(),
                item.confidence,
                item.relevance_score,
                item.expiry.isoformat() if item.expiry else None,
                json.dumps(item.tags),
            ),
        )
        self._conn.commit()
        return item.id

    def read(self, item_id: str) -> Optional[MemoryItem]:
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id = ?", (item_id,)
        ).fetchone()
        return self._row_to_item(row) if row else None

    def query(self, q: MemoryQuery) -> List[MemoryItem]:
        clauses = []
        params: list = []

        if not q.include_expired:
            clauses.append("(expiry IS NULL OR expiry > ?)")
            params.append(datetime.utcnow().isoformat())

        if q.sources:
            placeholders = ",".join("?" for _ in q.sources)
            clauses.append(f"source IN ({placeholders})")
            params.extend(q.sources)

        if q.min_confidence > 0:
            clauses.append("confidence >= ?")
            params.append(q.min_confidence)

        if q.since:
            clauses.append("timestamp >= ?")
            params.append(q.since.isoformat())

        if q.text:
            clauses.append("content LIKE ?")
            params.append(f"%{q.text}%")

        where = " AND ".join(clauses) if clauses else "1=1"
        sql = f"SELECT * FROM memories WHERE {where} ORDER BY timestamp DESC LIMIT ?"
        params.append(q.limit)

        rows = self._conn.execute(sql, params).fetchall()
        results = [self._row_to_item(r) for r in rows]

        # Post-filter by tags (JSON list in column)
        if q.tags:
            tag_set = set(q.tags)
            results = [m for m in results if tag_set & set(m.tags)]

        # Post-filter by metadata
        for key, val in q.metadata_filters.items():
            results = [m for m in results if m.metadata.get(key) == val]

        return results

    def delete(self, item_id: str) -> bool:
        cur = self._conn.execute("DELETE FROM memories WHERE id = ?", (item_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()
        return row[0]

    def clear(self) -> int:
        n = self.count()
        self._conn.execute("DELETE FROM memories")
        self._conn.commit()
        return n

    # ── Episodic-specific helpers ─────────────────────────────────────────

    def store_session_summary(self, session_id: str, summary: str, agent: str = "dizel") -> str:
        """Store a compressed summary of a completed session."""
        item = MemoryItem(
            type=MemoryType.EPISODIC,
            source=agent,
            content=summary,
            tags=["session_summary"],
            metadata={"session_id": session_id},
            confidence=0.9,
        )
        return self.write(item)

    def get_recent_sessions(self, limit: int = 5) -> List[MemoryItem]:
        """Retrieve the most recent session summaries."""
        return self.query(MemoryQuery(tags=["session_summary"], limit=limit))
        
    def archive_session(self, session_id: str, working_memory) -> str:
        """Extract working memory items, summarize them, and store in episodic memory."""
        items = working_memory.query(MemoryQuery(limit=50))
        if not items:
            return ""
            
        from .policy import MemoryPolicy
        summary = MemoryPolicy().summarize_items(items, max_length=1500)
        return self.store_session_summary(session_id, summary)

    # ── Internal ──────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_item(row: sqlite3.Row) -> MemoryItem:
        return MemoryItem(
            id=row["id"],
            type=MemoryType(row["type"]),
            source=row["source"],
            content=row["content"],
            metadata=json.loads(row["metadata"]),
            timestamp=datetime.fromisoformat(row["timestamp"]),
            confidence=row["confidence"],
            relevance_score=row["relevance"],
            expiry=datetime.fromisoformat(row["expiry"]) if row["expiry"] else None,
            tags=json.loads(row["tags"]),
        )

    def close(self):
        self._conn.close()

    def __del__(self):
        try:
            self._conn.close()
        except Exception:
            pass
