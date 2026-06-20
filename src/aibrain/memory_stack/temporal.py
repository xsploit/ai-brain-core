from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import GraphQuery, TemporalFact


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteTemporalGraphStore:
    """Temporal fact store that preserves the GraphMemoryStore contract locally.

    This is the built-in fallback. LadybugDB can replace it through the same
    contract when the optional dependency is installed.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.RLock()
        self._init()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            conn = sqlite3.connect(
                self.path,
                check_same_thread=False,
                isolation_level=None,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA temp_store=MEMORY")
            self._conn = conn
        return self._conn

    def _init(self) -> None:
        with self._lock:
            conn = self._connect()
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS brain_temporal_facts (
                    id TEXT PRIMARY KEY,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    valid_from TEXT NOT NULL,
                    valid_until TEXT,
                    confidence REAL NOT NULL DEFAULT 0.7,
                    importance REAL NOT NULL DEFAULT 0.5,
                    source_event_id TEXT,
                    source_session_id TEXT,
                    supersedes_memory_id TEXT,
                    last_accessed_at TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_temporal_fact_spo ON brain_temporal_facts(subject, predicate, object)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_temporal_fact_active ON brain_temporal_facts(valid_until)"
            )

    async def upsert_fact(self, fact: TemporalFact) -> TemporalFact:
        now = utc_now()
        if fact.created_at is None:
            fact.created_at = now
        fact.updated_at = now
        if not fact.valid_from:
            fact.valid_from = now
        await asyncio.to_thread(self._upsert_fact_sync, fact)
        return fact

    def _upsert_fact_sync(self, fact: TemporalFact) -> None:
        with self._lock:
            self._connect().execute(
                """
                INSERT INTO brain_temporal_facts (
                    id, subject, predicate, object, valid_from, valid_until,
                    confidence, importance, source_event_id, source_session_id,
                    supersedes_memory_id, last_accessed_at, metadata_json,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    subject = excluded.subject,
                    predicate = excluded.predicate,
                    object = excluded.object,
                    valid_from = excluded.valid_from,
                    valid_until = excluded.valid_until,
                    confidence = excluded.confidence,
                    importance = excluded.importance,
                    source_event_id = excluded.source_event_id,
                    source_session_id = excluded.source_session_id,
                    supersedes_memory_id = excluded.supersedes_memory_id,
                    last_accessed_at = excluded.last_accessed_at,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                _fact_params(fact),
            )

    async def search_facts(self, query: GraphQuery) -> list[TemporalFact]:
        return await asyncio.to_thread(self._search_facts_sync, query)

    def _search_facts_sync(self, query: GraphQuery) -> list[TemporalFact]:
        where: list[str] = []
        params: list[Any] = []
        if query.subject:
            where.append("subject = ?")
            params.append(query.subject)
        if query.predicate:
            where.append("predicate = ?")
            params.append(query.predicate)
        if query.object:
            where.append("object = ?")
            params.append(query.object)
        if not query.include_expired:
            where.append("valid_until IS NULL")
        tokens = [token.lower() for token in query.text.split() if token.strip()]
        if tokens:
            like_clauses = []
            for token in tokens[:8]:
                like_clauses.append("(lower(subject) LIKE ? OR lower(predicate) LIKE ? OR lower(object) LIKE ?)")
                pattern = f"%{token}%"
                params.extend([pattern, pattern, pattern])
            where.append("(" + " OR ".join(like_clauses) + ")")
        sql = "SELECT * FROM brain_temporal_facts"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY importance * confidence DESC, updated_at DESC LIMIT ?"
        params.append(max(query.top_k, 1))
        with self._lock:
            rows = self._connect().execute(sql, params).fetchall()
            now = utc_now()
            for row in rows:
                self._connect().execute(
                    "UPDATE brain_temporal_facts SET last_accessed_at = ? WHERE id = ?",
                    (now, row["id"]),
                )
        return [_row_to_fact(row) for row in rows]

    async def invalidate_fact(
        self,
        fact_id: str,
        *,
        valid_until: str,
        reason: str,
    ) -> None:
        await asyncio.to_thread(self._invalidate_fact_sync, fact_id, valid_until, reason)

    def _invalidate_fact_sync(self, fact_id: str, valid_until: str, reason: str) -> None:
        with self._lock:
            row = self._connect().execute(
                "SELECT metadata_json FROM brain_temporal_facts WHERE id = ?",
                (fact_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown fact_id: {fact_id}")
            metadata = json.loads(row["metadata_json"] or "{}")
            metadata["invalidated_reason"] = reason
            self._connect().execute(
                """
                UPDATE brain_temporal_facts
                SET valid_until = ?, metadata_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (valid_until, json.dumps(metadata), utc_now(), fact_id),
            )

    def close(self) -> None:
        with self._lock:
            conn = self._conn
            self._conn = None
            if conn is not None:
                conn.close()


def _fact_params(fact: TemporalFact) -> tuple[Any, ...]:
    return (
        fact.id,
        fact.subject,
        fact.predicate,
        fact.object,
        fact.valid_from,
        fact.valid_until,
        fact.confidence,
        fact.importance,
        fact.source_event_id,
        fact.source_session_id,
        fact.supersedes_memory_id,
        fact.last_accessed_at,
        json.dumps(fact.metadata),
        fact.created_at,
        fact.updated_at,
    )


def _row_to_fact(row: sqlite3.Row | dict[str, Any]) -> TemporalFact:
    return TemporalFact(
        id=row["id"],
        subject=row["subject"],
        predicate=row["predicate"],
        object=row["object"],
        valid_from=row["valid_from"],
        valid_until=row["valid_until"],
        confidence=float(row["confidence"]),
        importance=float(row["importance"]),
        source_event_id=row["source_event_id"],
        source_session_id=row["source_session_id"],
        supersedes_memory_id=row["supersedes_memory_id"],
        last_accessed_at=row["last_accessed_at"],
        metadata=json.loads(row["metadata_json"] or "{}"),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
