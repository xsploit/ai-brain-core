from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..embeddings import EmbeddingProvider
from ..memory import cosine_similarity
from .contracts import RecallHit, RecallItem


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteVectorRecallStore:
    def __init__(
        self,
        path: str | Path,
        *,
        embedding_provider: EmbeddingProvider,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_provider = embedding_provider
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
                CREATE TABLE IF NOT EXISTS brain_recall_items (
                    id TEXT PRIMARY KEY,
                    scope TEXT NOT NULL,
                    thread_id TEXT,
                    persona_id TEXT,
                    source_event_id TEXT,
                    source_fact_id TEXT,
                    text TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    importance REAL NOT NULL DEFAULT 0.5,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_recall_scope ON brain_recall_items(scope, thread_id, persona_id)"
            )

    async def add(self, item: RecallItem) -> str:
        if item.created_at is None:
            item.created_at = utc_now()
        embedding = await self.embedding_provider.embed(item.text)
        await asyncio.to_thread(self._add_sync, item, embedding)
        return item.id

    def _add_sync(self, item: RecallItem, embedding: list[float]) -> None:
        with self._lock:
            self._connect().execute(
                """
                INSERT INTO brain_recall_items (
                    id, scope, thread_id, persona_id, source_event_id, source_fact_id,
                    text, embedding_json, importance, metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    scope = excluded.scope,
                    thread_id = excluded.thread_id,
                    persona_id = excluded.persona_id,
                    source_event_id = excluded.source_event_id,
                    source_fact_id = excluded.source_fact_id,
                    text = excluded.text,
                    embedding_json = excluded.embedding_json,
                    importance = excluded.importance,
                    metadata_json = excluded.metadata_json
                """,
                (
                    item.id,
                    item.scope,
                    item.thread_id,
                    item.persona_id,
                    item.source_event_id,
                    item.source_fact_id,
                    item.text,
                    json.dumps(embedding),
                    item.importance,
                    json.dumps(item.metadata),
                    item.created_at,
                ),
            )

    async def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RecallHit]:
        query_embedding = await self.embedding_provider.embed(query)
        return await asyncio.to_thread(
            self._search_sync,
            query_embedding,
            top_k,
            filters or {},
        )

    def _search_sync(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: dict[str, Any],
    ) -> list[RecallHit]:
        where: list[str] = []
        params: list[Any] = []
        for key in ("scope", "thread_id", "persona_id", "source_event_id", "source_fact_id"):
            value = filters.get(key)
            if value is None:
                continue
            if isinstance(value, (list, tuple, set)):
                placeholders = ", ".join("?" for _ in value)
                where.append(f"{key} IN ({placeholders})")
                params.extend(value)
            else:
                where.append(f"{key} = ?")
                params.append(value)
        sql = "SELECT * FROM brain_recall_items"
        if where:
            sql += " WHERE " + " AND ".join(where)
        with self._lock:
            rows = self._connect().execute(sql, params).fetchall()
        hits: list[RecallHit] = []
        for row in rows:
            metadata = json.loads(row["metadata_json"] or "{}")
            if not _metadata_matches(metadata, filters):
                continue
            embedding = json.loads(row["embedding_json"])
            semantic = cosine_similarity(query_embedding, embedding)
            score = semantic * (0.5 + float(row["importance"]))
            hits.append(
                RecallHit(
                    id=row["id"],
                    text=row["text"],
                    score=score,
                    scope=row["scope"],
                    thread_id=row["thread_id"],
                    persona_id=row["persona_id"],
                    source_event_id=row["source_event_id"],
                    source_fact_id=row["source_fact_id"],
                    importance=float(row["importance"]),
                    metadata=metadata,
                    created_at=row["created_at"],
                )
            )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[: max(top_k, 1)]

    def close(self) -> None:
        with self._lock:
            conn = self._conn
            self._conn = None
            if conn is not None:
                conn.close()


class TurboVecRecallStore:
    """Optional TurboVec-backed recall store.

    Metadata is stored in SQLite while TurboVec owns the compressed vectors.
    """

    def __init__(
        self,
        index_path: str | Path,
        metadata_path: str | Path,
        *,
        embedding_provider: EmbeddingProvider,
        dimensions: int,
        bit_width: int = 4,
    ):
        try:
            from turbovec import IdMapIndex  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "TurboVecRecallStore requires the optional 'turbovec' package."
            ) from exc
        self.index_path = Path(index_path)
        self.metadata = SQLiteVectorRecallStore(metadata_path, embedding_provider=embedding_provider)
        self.embedding_provider = embedding_provider
        self._id_map_index_type = IdMapIndex
        if self.index_path.exists():
            self.index = IdMapIndex.load(str(self.index_path))
        else:
            self.index = IdMapIndex(dim=dimensions, bit_width=bit_width)
        self._lock = asyncio.Lock()

    async def add(self, item: RecallItem) -> str:
        embedding = await self.embedding_provider.embed(item.text)
        numeric_id = _numeric_id(item.id)
        async with self._lock:
            if self.index.contains(numeric_id):
                self.index.remove(numeric_id)
            await asyncio.to_thread(
                self.index.add_with_ids,
                _as_vector_array([embedding]),
                _as_id_array([numeric_id]),
            )
            await asyncio.to_thread(self.index.write, str(self.index_path))
        item.metadata = {**item.metadata, "turbovec_id": numeric_id}
        await self.metadata.add(item)
        return item.id

    async def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RecallHit]:
        if filters:
            return await self.metadata.search(query, top_k=top_k, filters=filters)
        embedding = await self.embedding_provider.embed(query)
        async with self._lock:
            scores, ids = await asyncio.to_thread(
                self.index.search,
                _as_vector_array([embedding]),
                k=top_k,
            )
        metadata_hits = {
            hit.id: hit
            for hit in await self.metadata.search(query, top_k=max(top_k * 4, top_k), filters=None)
        }
        hits: list[RecallHit] = []
        by_numeric_id = {
            str(hit.metadata.get("turbovec_id")): hit
            for hit in metadata_hits.values()
        }
        first_scores = scores[0] if len(scores) else []
        first_ids = ids[0] if len(ids) else []
        for score, item_id in zip(first_scores, first_ids):
            hit = by_numeric_id.get(str(int(item_id)))
            if hit is None:
                continue
            hit.score = float(score) * (0.5 + hit.importance)
            hits.append(hit)
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]

    def close(self) -> None:
        self.metadata.close()


def _as_vector_array(vectors: list[list[float]]) -> Any:
    import numpy as np

    return np.asarray(vectors, dtype="float32")


def _as_query_vector(vector: list[float]) -> Any:
    import numpy as np

    return np.asarray(vector, dtype="float32")


def _as_id_array(ids: list[int]) -> Any:
    import numpy as np

    return np.asarray(ids, dtype="uint64")


def _numeric_id(value: str) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)


def _metadata_matches(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
    column_keys = {"scope", "thread_id", "persona_id", "source_event_id", "source_fact_id"}
    for key, expected in filters.items():
        if key in column_keys:
            continue
        actual = metadata.get(key)
        if isinstance(expected, (list, tuple, set)):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True
