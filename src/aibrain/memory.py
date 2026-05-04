from __future__ import annotations

import asyncio
import json
import math
import sqlite3
import threading
from pathlib import Path
from typing import Any
from uuid import uuid4

from .embeddings import EmbeddingProvider, HashEmbeddingProvider
from .thread_store import utc_now
from .types import MemoryRecord


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    dot = sum(left[i] * right[i] for i in range(size))
    left_norm = math.sqrt(sum(value * value for value in left[:size])) or 1.0
    right_norm = math.sqrt(sum(value * value for value in right[:size])) or 1.0
    return dot / (left_norm * right_norm)


class SQLiteMemoryStore:
    def __init__(
        self,
        path: str | Path,
        embedding_provider: EmbeddingProvider | None = None,
        dimensions: int = 256,
        vec_overfetch: int = 5,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_provider = embedding_provider or HashEmbeddingProvider(dimensions)
        self.dimensions = dimensions
        self.vec_overfetch = max(1, vec_overfetch)
        self.sqlite_vec_enabled = False
        self._vec_backfilled = False
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.RLock()
        self._init()

    def _connect(self) -> sqlite3.Connection:
        with self._lock:
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
                self._try_load_sqlite_vec(conn)
                self._conn = conn
            return self._conn

    def _try_load_sqlite_vec(self, conn: sqlite3.Connection) -> None:
        try:
            import sqlite_vec  # type: ignore

            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            self.sqlite_vec_enabled = True
        except Exception:
            self.sqlite_vec_enabled = False

    def _init(self) -> None:
        with self._lock:
            conn = self._connect()
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS brain_memories (
                    id TEXT PRIMARY KEY,
                    vector_rowid INTEGER UNIQUE,
                    scope TEXT NOT NULL,
                    thread_id TEXT,
                    persona_id TEXT,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    importance REAL NOT NULL DEFAULT 0.5,
                    embedding_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(brain_memories)").fetchall()
            }
            if "vector_rowid" not in columns:
                conn.execute("ALTER TABLE brain_memories ADD COLUMN vector_rowid INTEGER UNIQUE")
            if self.sqlite_vec_enabled:
                conn.execute(
                    f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS brain_memory_vec
                    USING vec0(embedding float[{self.dimensions}])
                    """
                )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_thread ON brain_memories(thread_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_persona ON brain_memories(persona_id)"
            )

    def close(self) -> None:
        with self._lock:
            conn = self._conn
            self._conn = None
            self._vec_backfilled = False
            if conn is not None:
                conn.close()

    async def remember(
        self,
        content: str,
        *,
        scope: str = "global",
        thread_id: str | None = None,
        persona_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        importance: float = 0.5,
    ) -> MemoryRecord:
        now = utc_now()
        memory_id = str(uuid4())
        vector_rowid = uuid4().int & 0x7FFFFFFFFFFFFFFF if self.sqlite_vec_enabled else None
        embedding = await self.embedding_provider.embed(content)
        await asyncio.to_thread(
            self._remember_sync,
            memory_id,
            vector_rowid,
            scope,
            thread_id,
            persona_id,
            content,
            metadata or {},
            importance,
            embedding,
            now,
        )
        return MemoryRecord(
            id=memory_id,
            scope=scope,
            thread_id=thread_id,
            persona_id=persona_id,
            content=content,
            metadata=metadata or {},
            importance=importance,
            created_at=now,
        )

    def _remember_sync(
        self,
        memory_id: str,
        vector_rowid: int | None,
        scope: str,
        thread_id: str | None,
        persona_id: str | None,
        content: str,
        metadata: dict[str, Any],
        importance: float,
        embedding: list[float],
        now: str,
    ) -> None:
        with self._lock:
            conn = self._connect()
            conn.execute(
                """
                INSERT INTO brain_memories (
                    id, vector_rowid, scope, thread_id, persona_id, content, metadata_json,
                    importance, embedding_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    vector_rowid,
                    scope,
                    thread_id,
                    persona_id,
                    content,
                    json.dumps(metadata),
                    importance,
                    json.dumps(embedding),
                    now,
                    now,
                ),
            )
            if self.sqlite_vec_enabled:
                if vector_rowid is None:
                    vector_rowid = uuid4().int & 0x7FFFFFFFFFFFFFFF
                    conn.execute(
                        "UPDATE brain_memories SET vector_rowid = ? WHERE id = ?",
                        (vector_rowid, memory_id),
                    )
                conn.execute(
                    "INSERT INTO brain_memory_vec(rowid, embedding) VALUES (?, ?)",
                    (vector_rowid, json.dumps(embedding)),
                )
            else:
                self._vec_backfilled = False

    async def search(
        self,
        query: str,
        *,
        query_embedding: list[float] | None = None,
        top_k: int = 5,
        min_score: float = 0.0,
        scope: str | list[str] | tuple[str, ...] | None = None,
        thread_id: str | None = None,
        persona_id: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[MemoryRecord]:
        query_embedding = query_embedding or await self.embed_query(query)
        where: list[str] = []
        params: list[Any] = []
        if scope is not None:
            if isinstance(scope, (list, tuple)):
                placeholders = ",".join("?" for _ in scope)
                where.append(f"scope IN ({placeholders})")
                params.extend(scope)
            else:
                where.append("scope = ?")
                params.append(scope)
        if thread_id is not None:
            where.append("(thread_id = ? OR thread_id IS NULL)")
            params.append(thread_id)
        if persona_id is not None:
            where.append("(persona_id = ? OR persona_id IS NULL)")
            params.append(persona_id)
        return await asyncio.to_thread(
            self._search_sync,
            query_embedding,
            where,
            params,
            metadata_filter,
            top_k,
            min_score,
        )

    def _search_sync(
        self,
        query_embedding: list[float],
        where: list[str],
        params: list[Any],
        metadata_filter: dict[str, Any] | None,
        top_k: int,
        min_score: float,
    ) -> list[MemoryRecord]:
        with self._lock:
            conn = self._connect()
            if self.sqlite_vec_enabled:
                self._backfill_sqlite_vec(conn)
            rows = self._search_rows_with_sqlite_vec(
                conn,
                query_embedding,
                where,
                params,
                top_k,
            )
            if rows is None:
                return self._search_fallback_sync(
                    query_embedding,
                    where,
                    params,
                    metadata_filter,
                    top_k,
                    min_score,
                )
        scored: list[MemoryRecord] = []
        for row in rows:
            metadata = json.loads(row["metadata_json"] or "{}")
            if metadata_filter and not _metadata_matches(metadata, metadata_filter):
                continue
            if "vec_score" in row.keys():
                semantic_score = row["vec_score"]
            else:
                embedding = json.loads(row["embedding_json"])
                semantic_score = cosine_similarity(query_embedding, embedding)
            score = semantic_score * (0.5 + float(row["importance"]))
            if score < min_score:
                continue
            scored.append(
                MemoryRecord(
                    id=row["id"],
                    scope=row["scope"],
                    thread_id=row["thread_id"],
                    persona_id=row["persona_id"],
                    content=row["content"],
                    metadata=metadata,
                    importance=float(row["importance"]),
                    score=score,
                    created_at=row["created_at"],
                )
            )
        scored.sort(key=lambda record: record.score, reverse=True)
        return scored[:top_k]

    async def embed_query(self, query: str) -> list[float]:
        return await self.embedding_provider.embed(query)

    def _search_fallback_sync(
        self,
        query_embedding: list[float],
        where: list[str],
        params: list[Any],
        metadata_filter: dict[str, Any] | None,
        top_k: int,
        min_score: float,
    ) -> list[MemoryRecord]:
        with self._lock:
            conn = self._connect()
            sql = "SELECT * FROM brain_memories"
            if where:
                sql += " WHERE " + " AND ".join(where)
            rows = conn.execute(sql, params).fetchall()
        scored: list[MemoryRecord] = []
        for row in rows:
            metadata = json.loads(row["metadata_json"] or "{}")
            if metadata_filter and not _metadata_matches(metadata, metadata_filter):
                continue
            embedding = json.loads(row["embedding_json"])
            semantic_score = cosine_similarity(query_embedding, embedding)
            score = semantic_score * (0.5 + float(row["importance"]))
            if score < min_score:
                continue
            scored.append(
                MemoryRecord(
                    id=row["id"],
                    scope=row["scope"],
                    thread_id=row["thread_id"],
                    persona_id=row["persona_id"],
                    content=row["content"],
                    metadata=metadata,
                    importance=float(row["importance"]),
                    score=score,
                    created_at=row["created_at"],
                )
            )
        scored.sort(key=lambda record: record.score, reverse=True)
        return scored[:top_k]

    def _search_rows_with_sqlite_vec(
        self,
        conn: sqlite3.Connection,
        query_embedding: list[float],
        where: list[str],
        params: list[Any],
        top_k: int,
    ) -> list[sqlite3.Row | dict[str, Any]] | None:
        if not self.sqlite_vec_enabled:
            return None
        try:
            vector_rows = conn.execute(
                """
                SELECT rowid, distance
                FROM brain_memory_vec
                WHERE embedding MATCH ? AND k = ?
                ORDER BY distance
                """,
                (json.dumps(query_embedding), max(top_k * self.vec_overfetch, top_k)),
            ).fetchall()
        except sqlite3.Error:
            return None
        if not vector_rows:
            return []
        scores = {row["rowid"]: 1.0 / (1.0 + float(row["distance"])) for row in vector_rows}
        placeholders = ",".join("?" for _ in scores)
        sql = f"SELECT * FROM brain_memories WHERE vector_rowid IN ({placeholders})"
        sql_params: list[Any] = list(scores)
        if where:
            sql += " AND " + " AND ".join(where)
            sql_params.extend(params)
        rows = conn.execute(sql, sql_params).fetchall()
        decorated: list[dict[str, Any]] = []
        for row in rows:
            row_dict = dict(row)
            row_dict["vec_score"] = scores.get(row["vector_rowid"], 0.0)
            decorated.append(row_dict)
        return decorated

    def _backfill_sqlite_vec(self, conn: sqlite3.Connection) -> None:
        if not self.sqlite_vec_enabled or self._vec_backfilled:
            return
        rows = conn.execute(
            """
            SELECT m.id, m.vector_rowid, m.embedding_json
            FROM brain_memories m
            LEFT JOIN brain_memory_vec v ON v.rowid = m.vector_rowid
            WHERE m.vector_rowid IS NULL OR v.rowid IS NULL
            """
        ).fetchall()
        for row in rows:
            vector_rowid = row["vector_rowid"]
            if vector_rowid is None:
                vector_rowid = uuid4().int & 0x7FFFFFFFFFFFFFFF
                conn.execute(
                    "UPDATE brain_memories SET vector_rowid = ? WHERE id = ?",
                    (vector_rowid, row["id"]),
                )
            exists = conn.execute(
                "SELECT rowid FROM brain_memory_vec WHERE rowid = ?",
                (vector_rowid,),
            ).fetchone()
            if exists:
                continue
            conn.execute(
                "INSERT OR IGNORE INTO brain_memory_vec(rowid, embedding) VALUES (?, ?)",
                (vector_rowid, row["embedding_json"]),
            )
        self._vec_backfilled = True

    def forget(self, memory_id: str) -> bool:
        with self._lock:
            conn = self._connect()
            row = conn.execute(
                "SELECT vector_rowid FROM brain_memories WHERE id = ?",
                (memory_id,),
            ).fetchone()
            cursor = conn.execute("DELETE FROM brain_memories WHERE id = ?", (memory_id,))
            if row and self.sqlite_vec_enabled:
                conn.execute("DELETE FROM brain_memory_vec WHERE rowid = ?", (row["vector_rowid"],))
            return cursor.rowcount > 0


def _metadata_matches(metadata: dict[str, Any], expected: dict[str, Any]) -> bool:
    for key, value in expected.items():
        if metadata.get(key) != value:
            return False
    return True
