from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteChatHistoryStore:
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
                CREATE TABLE IF NOT EXISTS brain_chat_messages (
                    id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chat_messages_thread ON brain_chat_messages(thread_id, created_at)"
            )

    def append(
        self,
        thread_id: str,
        role: str,
        content: str | list[dict[str, Any]],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        row = {
            "id": str(uuid4()),
            "thread_id": thread_id,
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "created_at": utc_now(),
        }
        with self._lock:
            self._connect().execute(
                """
                INSERT INTO brain_chat_messages (
                    id, thread_id, role, content_json, metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    row["id"],
                    thread_id,
                    role,
                    json.dumps(content),
                    json.dumps(row["metadata"]),
                    row["created_at"],
                ),
            )
        return row

    def list(self, thread_id: str, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connect().execute(
                """
                SELECT * FROM brain_chat_messages
                WHERE thread_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (thread_id, max(limit, 1)),
            ).fetchall()
        return [_row_to_message(row) for row in reversed(rows)]

    def close(self) -> None:
        with self._lock:
            conn = self._conn
            self._conn = None
            if conn is not None:
                conn.close()


def _row_to_message(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "thread_id": row["thread_id"],
        "role": row["role"],
        "content": json.loads(row["content_json"]),
        "metadata": json.loads(row["metadata_json"] or "{}"),
        "created_at": row["created_at"],
    }
