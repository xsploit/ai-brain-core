from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any

from .models import (
    Evidence,
    EvidenceGap,
    GrilloEntity,
    GrilloEpisode,
    GrilloMemoryDocument,
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
                CREATE TABLE IF NOT EXISTS grillo_v2_entities (
                    entity_id TEXT PRIMARY KEY,
                    entity_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    aliases_json TEXT NOT NULL DEFAULT '[]',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_grillo_v2_entities_type
                    ON grillo_v2_entities(entity_type, name);

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

                CREATE TABLE IF NOT EXISTS grillo_v2_memory_docs (
                    memory_id TEXT PRIMARY KEY,
                    scope_key TEXT NOT NULL,
                    document_type TEXT NOT NULL,
                    subject_id TEXT,
                    title TEXT NOT NULL,
                    body TEXT NOT NULL,
                    evidence_ids_json TEXT NOT NULL DEFAULT '[]',
                    importance REAL NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_grillo_v2_memory_docs_scope_subject
                    ON grillo_v2_memory_docs(scope_key, subject_id, importance, updated_at);
                CREATE INDEX IF NOT EXISTS idx_grillo_v2_memory_docs_scope_type
                    ON grillo_v2_memory_docs(scope_key, document_type, importance, updated_at);

                CREATE TABLE IF NOT EXISTS grillo_v2_reflection_cursors (
                    cursor_key TEXT PRIMARY KEY,
                    scope_key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    def upsert_entity(self, entity: GrilloEntity) -> GrilloEntity:
        with self._lock:
            row = self._connect().execute(
                "SELECT aliases_json, metadata_json FROM grillo_v2_entities WHERE entity_id = ?",
                (entity.entity_id,),
            ).fetchone()
            aliases = entity.aliases
            metadata = dict(entity.metadata)
            if row is not None:
                aliases = _dedupe([*entity.aliases, *[str(item) for item in from_json_list(row["aliases_json"])]])
                metadata = {**from_json_dict(row["metadata_json"]), **metadata}
            self._connect().execute(
                """
                INSERT INTO grillo_v2_entities (
                    entity_id, entity_type, name, aliases_json, metadata_json, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(entity_id) DO UPDATE SET
                    entity_type = excluded.entity_type,
                    name = excluded.name,
                    aliases_json = excluded.aliases_json,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    entity.entity_id,
                    entity.entity_type,
                    entity.name,
                    to_json(aliases),
                    to_json(metadata),
                    utc_now(),
                ),
            )
            entity.aliases = aliases
            entity.metadata = metadata
        return entity

    def get_entity(self, entity_id: str | None) -> GrilloEntity | None:
        if not entity_id:
            return None
        with self._lock:
            row = self._connect().execute(
                "SELECT * FROM grillo_v2_entities WHERE entity_id = ?",
                (entity_id,),
            ).fetchone()
        return _entity_from_row(row) if row is not None else None

    def list_entities(self, *, entity_type: str | None = None, limit: int = 50) -> list[GrilloEntity]:
        where: list[str] = []
        params: list[Any] = []
        if entity_type:
            where.append("entity_type = ?")
            params.append(entity_type)
        params.append(max(1, int(limit)))
        sql_where = f"WHERE {' AND '.join(where)}" if where else ""
        with self._lock:
            rows = self._connect().execute(
                f"""
                SELECT * FROM grillo_v2_entities
                {sql_where}
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [_entity_from_row(row) for row in rows]

    def list_evidence(self, scope_key: str, *, limit: int = 50) -> list[Evidence]:
        with self._lock:
            rows = self._connect().execute(
                """
                SELECT * FROM grillo_v2_evidence
                WHERE scope_key = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (scope_key, max(1, int(limit))),
            ).fetchall()
        return [_evidence_from_row(row) for row in rows]

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

    def upsert_memory_document(self, document: GrilloMemoryDocument) -> GrilloMemoryDocument:
        document.updated_at = utc_now()
        with self._lock:
            self._connect().execute(
                """
                INSERT OR REPLACE INTO grillo_v2_memory_docs (
                    memory_id, scope_key, document_type, subject_id, title, body,
                    evidence_ids_json, importance, metadata_json, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document.memory_id,
                    document.scope_key,
                    document.document_type,
                    document.subject_id,
                    document.title,
                    document.body,
                    to_json(document.evidence_ids),
                    document.importance,
                    to_json(document.metadata),
                    document.updated_at,
                ),
            )
        return document

    def list_memory_documents(
        self,
        scope_key: str,
        *,
        subject_id: str | None = None,
        document_type: str | None = None,
        query: str = "",
        limit: int = 6,
    ) -> list[GrilloMemoryDocument]:
        where = ["scope_key = ?"]
        params: list[Any] = [scope_key]
        if subject_id is not None:
            where.append("(subject_id = ? OR subject_id IS NULL)")
            params.append(subject_id)
        if document_type is not None:
            where.append("document_type = ?")
            params.append(document_type)
        params.append(max(1, int(limit)) * 4)
        with self._lock:
            rows = self._connect().execute(
                f"""
                SELECT * FROM grillo_v2_memory_docs
                WHERE {' AND '.join(where)}
                ORDER BY importance DESC, updated_at DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        documents = [_memory_document_from_row(row) for row in rows]
        if query.strip():
            documents = sorted(
                documents,
                key=lambda document: (
                    max(_text_score(query, document.title), _text_score(query, document.body)),
                    document.importance,
                    document.updated_at,
                ),
                reverse=True,
            )
        return documents[: max(1, int(limit))]

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

    def list_episode_scopes(self, *, limit: int = 50) -> list[str]:
        with self._lock:
            rows = self._connect().execute(
                """
                SELECT scope_key, MAX(occurred_at) AS latest
                FROM grillo_v2_episodes
                GROUP BY scope_key
                ORDER BY latest ASC
                LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
        return [str(row["scope_key"]) for row in rows]

    def list_episodes_after(
        self,
        scope_key: str,
        *,
        after_occurred_at: str | None = None,
        after_episode_id: str | None = None,
        limit: int = 12,
    ) -> list[GrilloEpisode]:
        where = ["scope_key = ?"]
        params: list[Any] = [scope_key]
        if after_occurred_at is not None:
            where.append("(occurred_at > ? OR (occurred_at = ? AND episode_id > ?))")
            params.extend([after_occurred_at, after_occurred_at, after_episode_id or ""])
        params.append(max(1, int(limit)))
        with self._lock:
            rows = self._connect().execute(
                f"""
                SELECT * FROM grillo_v2_episodes
                WHERE {' AND '.join(where)}
                ORDER BY occurred_at ASC, episode_id ASC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [_episode_from_row(row) for row in rows]

    def list_active_facts(
        self,
        scope_key: str,
        *,
        subject_id: str | None = None,
        query: str = "",
        limit: int = 8,
    ) -> list[TemporalFact]:
        facts = self.list_temporal_facts(
            scope_key,
            subject_id=subject_id,
            include_expired=False,
            limit=max(1, int(limit)) * 4,
        )
        if query.strip():
            facts = sorted(
                facts,
                key=lambda fact: (_text_score(query, fact.claim), fact.confidence, fact.updated_at),
                reverse=True,
            )
            if subject_id is None:
                facts = [fact for fact in facts if _text_score(query, fact.claim) > 0]
        return facts[: max(1, int(limit))]

    def list_temporal_facts(
        self,
        scope_key: str,
        *,
        subject_id: str | None = None,
        include_expired: bool = False,
        limit: int = 50,
    ) -> list[TemporalFact]:
        where = ["scope_key = ?"]
        params: list[Any] = [scope_key]
        if not include_expired:
            where.append("valid_to IS NULL")
        if subject_id is not None:
            where.append("subject_id = ?")
            params.append(subject_id)
        rows = self._fact_rows(where, params, limit=max(1, int(limit)))
        return [_fact_from_row(row) for row in rows]

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
        include_expired: bool = False,
        limit: int = 8,
    ) -> list[OpinionEdge]:
        where = ["scope_key = ?"]
        params: list[Any] = [scope_key]
        if not include_expired:
            where.append("valid_to IS NULL")
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

    def counts(self) -> dict[str, int]:
        tables = {
            "entities": "grillo_v2_entities",
            "episodes": "grillo_v2_episodes",
            "evidence": "grillo_v2_evidence",
            "facts": "grillo_v2_temporal_facts",
            "active_facts": "grillo_v2_temporal_facts WHERE valid_to IS NULL",
            "opinion_edges": "grillo_v2_opinion_edges",
            "active_opinion_edges": "grillo_v2_opinion_edges WHERE valid_to IS NULL",
            "memory_docs": "grillo_v2_memory_docs",
        }
        out: dict[str, int] = {}
        with self._lock:
            conn = self._connect()
            for key, table in tables.items():
                row = conn.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()
                out[key] = int(row["count"] if row else 0)
        return out

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


def _entity_from_row(row: sqlite3.Row) -> GrilloEntity:
    return GrilloEntity(
        entity_id=row["entity_id"],
        entity_type=row["entity_type"],
        name=row["name"],
        aliases=[str(item) for item in from_json_list(row["aliases_json"])],
        metadata=from_json_dict(row["metadata_json"]),
    )


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


def _evidence_from_row(row: sqlite3.Row) -> Evidence:
    return Evidence(
        evidence_id=row["evidence_id"],
        scope_key=row["scope_key"],
        episode_id=row["episode_id"],
        quote=row["quote"],
        extractor=row["extractor"],
        confidence=float(row["confidence"]),
        metadata=from_json_dict(row["metadata_json"]),
        created_at=row["created_at"],
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


def _memory_document_from_row(row: sqlite3.Row) -> GrilloMemoryDocument:
    return GrilloMemoryDocument(
        memory_id=row["memory_id"],
        scope_key=row["scope_key"],
        document_type=row["document_type"],
        subject_id=row["subject_id"],
        title=row["title"],
        body=row["body"],
        evidence_ids=[str(item) for item in from_json_list(row["evidence_ids_json"])],
        importance=float(row["importance"]),
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
