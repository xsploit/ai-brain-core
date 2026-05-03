from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .types import ThreadState


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteThreadStore:
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
                CREATE TABLE IF NOT EXISTS brain_threads (
                    thread_id TEXT PRIMARY KEY,
                    persona_id TEXT NOT NULL,
                    openai_conversation_id TEXT,
                    last_response_id TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def close(self) -> None:
        with self._lock:
            conn = self._conn
            self._conn = None
            if conn is not None:
                conn.close()

    def create(
        self,
        persona_id: str,
        thread_id: str | None = None,
        openai_conversation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ThreadState:
        now = utc_now()
        state = ThreadState(
            thread_id=thread_id or str(uuid4()),
            persona_id=persona_id,
            openai_conversation_id=openai_conversation_id,
            metadata=metadata or {},
            created_at=now,
            updated_at=now,
        )
        self.upsert(state)
        return state

    def get(self, thread_id: str) -> ThreadState | None:
        with self._lock:
            conn = self._connect()
            row = conn.execute(
                "SELECT * FROM brain_threads WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
        if row is None:
            return None
        return ThreadState(
            thread_id=row["thread_id"],
            persona_id=row["persona_id"],
            openai_conversation_id=row["openai_conversation_id"],
            last_response_id=row["last_response_id"],
            metadata=json.loads(row["metadata_json"] or "{}"),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def upsert(self, state: ThreadState) -> None:
        now = utc_now()
        state.updated_at = now
        if state.created_at is None:
            state.created_at = now
        with self._lock:
            conn = self._connect()
            conn.execute(
                """
                INSERT INTO brain_threads (
                    thread_id, persona_id, openai_conversation_id, last_response_id,
                    metadata_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    persona_id = excluded.persona_id,
                    openai_conversation_id = excluded.openai_conversation_id,
                    last_response_id = excluded.last_response_id,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    state.thread_id,
                    state.persona_id,
                    state.openai_conversation_id,
                    state.last_response_id,
                    json.dumps(state.metadata),
                    state.created_at,
                    state.updated_at,
                ),
            )

    def update_remote_ids(
        self,
        thread_id: str,
        *,
        openai_conversation_id: str | None = None,
        last_response_id: str | None = None,
    ) -> ThreadState:
        state = self.get(thread_id)
        if state is None:
            raise KeyError(f"Unknown thread_id: {thread_id}")
        if openai_conversation_id is not None:
            state.openai_conversation_id = openai_conversation_id
        if last_response_id is not None:
            state.last_response_id = last_response_id
        self.upsert(state)
        return state

    def list(self, limit: int = 100) -> list[ThreadState]:
        with self._lock:
            conn = self._connect()
            rows = conn.execute(
                "SELECT * FROM brain_threads ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            ThreadState(
                thread_id=row["thread_id"],
                persona_id=row["persona_id"],
                openai_conversation_id=row["openai_conversation_id"],
                last_response_id=row["last_response_id"],
                metadata=json.loads(row["metadata_json"] or "{}"),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]
