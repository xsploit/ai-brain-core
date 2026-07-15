from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .contracts import RawEvent


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteRawEventStore:
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
                CREATE TABLE IF NOT EXISTS brain_raw_events (
                    id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    thread_id TEXT,
                    persona_id TEXT,
                    session_id TEXT,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_raw_events_thread ON brain_raw_events(thread_id, created_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_raw_events_session ON brain_raw_events(session_id, created_at)"
            )

    async def append(self, event: RawEvent) -> RawEvent:
        if not event.id:
            event.id = str(uuid4())
        if event.created_at is None:
            event.created_at = utc_now()
        await asyncio.to_thread(self._append_sync, event)
        return event

    def _append_sync(self, event: RawEvent) -> None:
        with self._lock:
            self._connect().execute(
                """
                INSERT INTO brain_raw_events (
                    id, event_type, actor, thread_id, persona_id, session_id,
                    content, metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    event.event_type,
                    event.actor,
                    event.thread_id,
                    event.persona_id,
                    event.session_id,
                    event.content,
                    json.dumps(event.metadata),
                    event.created_at,
                ),
            )

    async def list_thread_events(self, thread_id: str, limit: int = 100) -> list[RawEvent]:
        return await asyncio.to_thread(self._list_thread_events_sync, thread_id, limit)

    def _list_thread_events_sync(self, thread_id: str, limit: int) -> list[RawEvent]:
        with self._lock:
            rows = self._connect().execute(
                """
                SELECT * FROM brain_raw_events
                WHERE thread_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (thread_id, limit),
            ).fetchall()
        return [_row_to_event(row) for row in reversed(rows)]

    async def list_thread_events_matching(
        self,
        *,
        thread_ids: list[str] | None = None,
        thread_like_patterns: list[str] | None = None,
        persona_id: str | None = None,
        limit: int = 100,
    ) -> list[RawEvent]:
        return await asyncio.to_thread(
            self._list_thread_events_matching_sync,
            thread_ids or [],
            thread_like_patterns or [],
            persona_id,
            limit,
        )

    def _list_thread_events_matching_sync(
        self,
        thread_ids: list[str],
        thread_like_patterns: list[str],
        persona_id: str | None,
        limit: int,
    ) -> list[RawEvent]:
        predicates: list[str] = []
        params: list[Any] = []
        if thread_ids:
            placeholders = ", ".join("?" for _ in thread_ids)
            predicates.append(f"thread_id IN ({placeholders})")
            params.extend(thread_ids)
        for pattern in thread_like_patterns:
            predicates.append("thread_id LIKE ?")
            params.append(pattern)
        where = " OR ".join(predicates)
        if not where:
            where = "thread_id IS NOT NULL"
        if persona_id:
            where = f"({where}) AND persona_id = ?"
            params.append(persona_id)
        params.append(max(limit, 1))
        with self._lock:
            rows = self._connect().execute(
                f"""
                SELECT * FROM brain_raw_events
                WHERE {where}
                ORDER BY created_at DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [_row_to_event(row) for row in reversed(rows)]

    def close(self) -> None:
        with self._lock:
            conn = self._conn
            self._conn = None
            if conn is not None:
                conn.close()


def _row_to_event(row: sqlite3.Row | dict[str, Any]) -> RawEvent:
    return RawEvent(
        id=row["id"],
        event_type=row["event_type"],
        actor=row["actor"],
        thread_id=row["thread_id"],
        persona_id=row["persona_id"],
        session_id=row["session_id"],
        content=row["content"],
        metadata=json.loads(row["metadata_json"] or "{}"),
        created_at=row["created_at"],
    )
