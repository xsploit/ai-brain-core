from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any

from .models import (
    Evidence,
    EvidenceGap,
    GrilloEpisode,
    OpinionEdge,
    TemporalFact,
    dataclass_dict,
    from_json_dict,
    from_json_list,
    to_json,
    utc_now,
)


class SQLiteGrilloV2Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.RLock()
        self._init()

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA temp_store=MEMORY")
            self._conn = conn
        return self._conn

    def _init(self) -> None:
        with self._lock:
            self._connect().executescript(
                """
                CREATE TABLE IF NOT EXISTS grillo_v2_episodes (
                    episode_id TEXT PRIMARY KEY,
                    scope_key TEXT NOT NULL,
                    source TEXT NOT NULL,
                    actor_id TEXT,
                    participant_ids_json TEXT NOT NULL DEFAULT '[]',
                    channel_id TEXT,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    occurred_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_grillo_v2_episodes_scope_time
                    ON grillo_v2_episodes(scope_key, occurred_at);
                CREATE INDEX IF NOT EXISTS idx_grillo_v2_episodes_actor
                    ON grillo_v2_episodes(scope_key, actor_id, occurred_at);

                CREATE TABLE IF NOT EXISTS grillo_v2_evidence (
                    evidence_id TEXT PRIMARY KEY,
                    scope_key TEXT NOT NULL,
                    episode_id TEXT NOT NULL,
                    quote TEXT NOT NULL,
                    extractor TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_grillo_v2_evidence_scope_episode
                    ON grillo_v2_evidence(scope_key, episode_id);

                CREATE TABLE IF NOT EXISTS grillo_v2_temporal_facts (
                    fact_id TEXT PRIMARY KEY,
                    scope_key TEXT NOT NULL,
                    subject_id TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object_value TEXT NOT NULL,
                    claim TEXT NOT NULL,
                    evidence_ids_json TEXT NOT NULL DEFAULT '[]',
                    confidence REAL NOT NULL,
                    valid_from TEXT NOT NULL,
                    valid_to TEXT,
                    contradicts_json TEXT NOT NULL DEFAULT '[]',
                    missing_evidence_json TEXT NOT NULL DEFAULT '[]',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_grillo_v2_facts_scope_subject
                    ON grillo_v2_temporal_facts(scope_key, subject_id, valid_to, confidence);
                CREATE INDEX IF NOT EXISTS idx_grillo_v2_facts_scope_predicate
                    ON grillo_v2_temporal_facts(scope_key, predicate, valid_to, confidence);

                CREATE TABLE IF NOT EXISTS grillo_v2_opinion_edges (
                    edge_id TEXT PRIMARY KEY,
                    scope_key TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    score REAL NOT NULL,
                    rationale TEXT NOT NULL,
                    evidence_ids_json TEXT NOT NULL DEFAULT '[]',
                    valid_from TEXT NOT NULL,
                    valid_to TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_grillo_v2_opinions_scope_target
                    ON grillo_v2_opinion_edges(scope_key, target_id, valid_to, relation);

                CREATE TABLE IF NOT EXISTS grillo_v2_reflection_cursors (
                    cursor_key TEXT PRIMARY KEY,
                    scope_key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    def append_episode(self, episode: GrilloEpisode) -> GrilloEpisode:
        with self._lock:
            self._connect().execute(
                """
                INSERT OR REPLACE INTO grillo_v2_episodes (
                    episode_id, scope_key, source, actor_id, participant_ids_json,
                    channel_id, content, metadata_json, occurred_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode.episode_id,
                    episode.scope_key,
                    episode.source,
                    episode.actor_id,
                    to_json(episode.participant_ids),
                    episode.channel_id,
                    episode.content,
                    to_json(episode.metadata),
                    episode.occurred_at,
                ),
            )
        return episode

    def append_evidence(self, evidence: Evidence) -> Evidence:
        with self._lock:
            self._connect().execute(
                """
                INSERT OR REPLACE INTO grillo_v2_evidence (
                    evidence_id, scope_key, episode_id, quote, extractor,
                    confidence, metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evidence.evidence_id,
                    evidence.scope_key,
                    evidence.episode_id,
                    evidence.quote,
                    evidence.extractor,
                    evidence.confidence,
                    to_json(evidence.metadata),
                    evidence.created_at,
                ),
            )
        return evidence

    def upsert_fact(self, fact: TemporalFact) -> TemporalFact:
        fact.updated_at = utc_now()
        with self._lock:
            self._connect().execute(
                """
                INSERT OR REPLACE INTO grillo_v2_temporal_facts (
                    fact_id, scope_key, subject_id, predicate, object_value, claim,
                    evidence_ids_json, confidence, valid_from, valid_to, contradicts_json,
                    missing_evidence_json, metadata_json, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fact.fact_id,
                    fact.scope_key,
                    fact.subject_id,
                    fact.predicate,
                    fact.object_value,
                    fact.claim,
                    to_json(fact.evidence_ids),
                    fact.confidence,
                    fact.valid_from,
                    fact.valid_to,
                    to_json(fact.contradicts),
                    to_json([dataclass_dict(gap) for gap in fact.missing_evidence]),
                    to_json(fact.metadata),
                    fact.updated_at,
                ),
            )
        return fact

    def invalidate_fact(self, fact_id: str, *, valid_to: str | None = None, contradicts: list[str] | None = None) -> None:
        with self._lock:
            row = self._connect().execute(
                "SELECT contradicts_json FROM grillo_v2_temporal_facts WHERE fact_id = ?",
                (fact_id,),
            ).fetchone()
            if row is None:
                return
            merged = _dedupe([*from_json_list(row["contradicts_json"]), *(contradicts or [])])
            self._connect().execute(
                """
                UPDATE grillo_v2_temporal_facts
                SET valid_to = ?, contradicts_json = ?, updated_at = ?
                WHERE fact_id = ?
                """,
                (valid_to or utc_now(), to_json(merged), utc_now(), fact_id),
            )

    def upsert_opinion_edge(self, edge: OpinionEdge) -> OpinionEdge:
        edge.updated_at = utc_now()
        with self._lock:
            self._connect().execute(
                """
                INSERT OR REPLACE INTO grillo_v2_opinion_edges (
                    edge_id, scope_key, source_id, target_id, relation, score,
                    rationale, evidence_ids_json, valid_from, valid_to, metadata_json, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    edge.edge_id,
                    edge.scope_key,
                    edge.source_id,
                    edge.target_id,
                    edge.relation,
                    edge.score,
                    edge.rationale,
                    to_json(edge.evidence_ids),
                    edge.valid_from,
                    edge.valid_to,
                    to_json(edge.metadata),
                    edge.updated_at,
                ),
            )
        return edge

    def list_recent_episodes(
        self,
        scope_key: str,
        *,
        actor_id: str | None = None,
        channel_id: str | None = None,
        limit: int = 8,
    ) -> list[GrilloEpisode]:
        where = ["scope_key = ?"]
        params: list[Any] = [scope_key]
        if actor_id is not None:
            where.append("(actor_id = ? OR participant_ids_json LIKE ?)")
            params.extend([actor_id, f'%"{actor_id}"%'])
        if channel_id is not None:
            where.append("channel_id = ?")
            params.append(channel_id)
        params.append(max(1, int(limit)))
        with self._lock:
            rows = self._connect().execute(
                f"""
                SELECT * FROM grillo_v2_episodes
                WHERE {' AND '.join(where)}
                ORDER BY occurred_at DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [_episode_from_row(row) for row in reversed(rows)]

    def list_active_facts(
        self,
        scope_key: str,
        *,
        subject_id: str | None = None,
        query: str = "",
        limit: int = 8,
    ) -> list[TemporalFact]:
        where = ["scope_key = ?", "valid_to IS NULL"]
        params: list[Any] = [scope_key]
        if subject_id is not None:
            where.append("subject_id = ?")
            params.append(subject_id)
        rows = self._fact_rows(where, params, limit=max(1, int(limit)) * 4)
        facts = [_fact_from_row(row) for row in rows]
        if query.strip():
            facts = sorted(
                facts,
                key=lambda fact: (_text_score(query, fact.claim), fact.confidence, fact.updated_at),
                reverse=True,
            )
            if subject_id is None:
                facts = [fact for fact in facts if _text_score(query, fact.claim) > 0]
        return facts[: max(1, int(limit))]

    def search_facts(self, scope_key: str, query: str, *, limit: int = 8) -> list[TemporalFact]:
        rows = self._fact_rows(["scope_key = ?", "valid_to IS NULL"], [scope_key], limit=500)
        facts = [_fact_from_row(row) for row in rows]
        scored = [
            (max(_text_score(query, fact.claim), _text_score(query, fact.object_value)), fact)
            for fact in facts
        ]
        return [fact for score, fact in sorted(scored, key=lambda item: (item[0], item[1].confidence), reverse=True) if score > 0][
            : max(1, int(limit))
        ]

    def list_opinion_edges(
        self,
        scope_key: str,
        *,
        source_id: str | None = None,
        target_id: str | None = None,
        limit: int = 8,
    ) -> list[OpinionEdge]:
        where = ["scope_key = ?", "valid_to IS NULL"]
        params: list[Any] = [scope_key]
        if source_id is not None:
            where.append("source_id = ?")
            params.append(source_id)
        if target_id is not None:
            where.append("target_id = ?")
            params.append(target_id)
        params.append(max(1, int(limit)))
        with self._lock:
            rows = self._connect().execute(
                f"""
                SELECT * FROM grillo_v2_opinion_edges
                WHERE {' AND '.join(where)}
                ORDER BY ABS(score) DESC, updated_at DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [_opinion_from_row(row) for row in rows]

    def set_cursor(self, cursor_key: str, scope_key: str, value: str) -> None:
        with self._lock:
            self._connect().execute(
                """
                INSERT INTO grillo_v2_reflection_cursors (cursor_key, scope_key, value, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(cursor_key) DO UPDATE SET
                    scope_key = excluded.scope_key,
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (cursor_key, scope_key, value, utc_now()),
            )

    def get_cursor(self, cursor_key: str) -> str | None:
        with self._lock:
            row = self._connect().execute(
                "SELECT value FROM grillo_v2_reflection_cursors WHERE cursor_key = ?",
                (cursor_key,),
            ).fetchone()
        return str(row["value"]) if row else None

    def _fact_rows(self, where: list[str], params: list[Any], *, limit: int) -> list[sqlite3.Row]:
        params = [*params, max(1, int(limit))]
        with self._lock:
            return self._connect().execute(
                f"""
                SELECT * FROM grillo_v2_temporal_facts
                WHERE {' AND '.join(where)}
                ORDER BY confidence DESC, updated_at DESC
                LIMIT ?
                """,
                params,
            ).fetchall()


def _episode_from_row(row: sqlite3.Row) -> GrilloEpisode:
    return GrilloEpisode(
        episode_id=row["episode_id"],
        scope_key=row["scope_key"],
        source=row["source"],
        content=row["content"],
        actor_id=row["actor_id"],
        participant_ids=[str(item) for item in from_json_list(row["participant_ids_json"])],
        channel_id=row["channel_id"],
        occurred_at=row["occurred_at"],
        metadata=from_json_dict(row["metadata_json"]),
    )


def _fact_from_row(row: sqlite3.Row) -> TemporalFact:
    return TemporalFact(
        fact_id=row["fact_id"],
        scope_key=row["scope_key"],
        subject_id=row["subject_id"],
        predicate=row["predicate"],
        object_value=row["object_value"],
        claim=row["claim"],
        evidence_ids=[str(item) for item in from_json_list(row["evidence_ids_json"])],
        confidence=float(row["confidence"]),
        valid_from=row["valid_from"],
        valid_to=row["valid_to"],
        contradicts=[str(item) for item in from_json_list(row["contradicts_json"])],
        missing_evidence=[
            EvidenceGap(
                question=str(item.get("question", "")),
                why=str(item.get("why", "")),
                needed=str(item.get("needed", "")),
            )
            for item in from_json_list(row["missing_evidence_json"])
            if isinstance(item, dict)
        ],
        metadata=from_json_dict(row["metadata_json"]),
        updated_at=row["updated_at"],
    )


def _opinion_from_row(row: sqlite3.Row) -> OpinionEdge:
    return OpinionEdge(
        edge_id=row["edge_id"],
        scope_key=row["scope_key"],
        source_id=row["source_id"],
        target_id=row["target_id"],
        relation=row["relation"],
        score=float(row["score"]),
        rationale=row["rationale"],
        evidence_ids=[str(item) for item in from_json_list(row["evidence_ids_json"])],
        valid_from=row["valid_from"],
        valid_to=row["valid_to"],
        metadata=from_json_dict(row["metadata_json"]),
        updated_at=row["updated_at"],
    )


def _text_score(query: str, text: str) -> int:
    query_tokens = {token for token in _tokens(query) if len(token) >= 3}
    text_tokens = set(_tokens(text))
    if not query_tokens or not text_tokens:
        return 0
    return len(query_tokens & text_tokens)


def _tokens(value: str) -> list[str]:
    return [part.casefold() for part in "".join(ch if ch.isalnum() else " " for ch in value).split()]


def _dedupe(items: list[Any]) -> list[Any]:
    seen: set[str] = set()
    result: list[Any] = []
    for item in items:
        key = str(item)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result
