from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import Evidence, GrilloEntity, GrilloEpisode, TemporalFact, from_json_dict, from_json_list
from .store import SQLiteGrilloV2Store


GUILD_USER_SCOPE_RE = re.compile(r"^discord:guild:(?P<guild_id>[^:]+):user:(?P<user_id>[^:]+):persona:(?P<persona>.+)$")
DM_SCOPE_RE = re.compile(r"^discord:dm:(?P<user_id>[^:]+):persona:(?P<persona>.+)$")


@dataclass(slots=True)
class GrilloV2BackfillResult:
    episodes: int = 0
    entities: int = 0
    evidence: int = 0
    facts: int = 0
    skipped: int = 0


def backfill_grillo_v1(
    *,
    source_path: str | Path,
    target: SQLiteGrilloV2Store,
    persona_id: str = "neuro-sama-v2",
    limit: int = 10_000,
) -> GrilloV2BackfillResult:
    path = Path(source_path)
    if not path.exists():
        return GrilloV2BackfillResult(skipped=1)
    result = GrilloV2BackfillResult()
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error:
        return GrilloV2BackfillResult(skipped=1)
    try:
        result.episodes += _backfill_turns(conn, target, persona_id=persona_id, limit=limit)
        evidence, facts = _backfill_candidates(conn, target, persona_id=persona_id, limit=limit)
        result.evidence += evidence
        result.facts += facts
        evidence, facts = _backfill_slots(conn, target, persona_id=persona_id, limit=limit)
        result.evidence += evidence
        result.facts += facts
    finally:
        conn.close()
    return result


def backfill_discord_identity(
    *,
    source_path: str | Path,
    target: SQLiteGrilloV2Store,
    persona_id: str = "neuro-sama-v2",
    limit: int = 10_000,
) -> GrilloV2BackfillResult:
    path = Path(source_path)
    if not path.exists():
        return GrilloV2BackfillResult(skipped=1)
    result = GrilloV2BackfillResult()
    try:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT *
            FROM discord_user_identities
            ORDER BY last_seen_at DESC
            LIMIT ?
            """,
            (max(1, int(limit)),),
        ).fetchall()
    except sqlite3.Error:
        return GrilloV2BackfillResult(skipped=1)
    finally:
        if conn is not None:
            conn.close()
    for row in rows:
        scope_key = f"discord:guild:{row['guild_id']}:persona:v2"
        entity_id = _discord_user_entity_id(row["user_id"])
        aliases = [str(item) for item in from_json_list(row["aliases_json"])]
        entity = target.upsert_entity(
            GrilloEntity(
                entity_id=entity_id,
                entity_type="person",
                name=str(row["display_name"] or row["username"] or row["user_id"]),
                aliases=_dedupe([*aliases, str(row["user_id"]), f"<@{row['user_id']}>"]),
                metadata={
                    "source": "discord_identity",
                    "guild_id": str(row["guild_id"]),
                    "username": row["username"],
                    "display_name": row["display_name"],
                    "global_name": row["global_name"],
                    "mention": row["mention"],
                    "is_bot": bool(row["is_bot"]),
                    "first_seen_at": row["first_seen_at"],
                    "last_seen_at": row["last_seen_at"],
                    "message_count": int(row["message_count"] or 0),
                },
            )
        )
        result.entities += 1
        for predicate, value in (
            ("username", row["username"]),
            ("display_name", row["display_name"]),
            ("global_name", row["global_name"]),
            ("mention", row["mention"]),
        ):
            if not value:
                continue
            target.upsert_fact(
                TemporalFact(
                    fact_id=f"fact:identity:{row['guild_id']}:{row['user_id']}:{predicate}",
                    scope_key=scope_key,
                    subject_id=entity.entity_id,
                    predicate=predicate,
                    object_value=str(value),
                    claim=f"{entity.name} has Discord {predicate} {value}.",
                    evidence_ids=[],
                    confidence=0.95,
                    valid_from=str(row["first_seen_at"]),
                    valid_to=None,
                    metadata={"source": "discord_identity", "persona_id": persona_id},
                )
            )
            result.facts += 1
        for alias in entity.aliases[:32]:
            target.upsert_fact(
                TemporalFact(
                    fact_id=f"fact:identity:{row['guild_id']}:{row['user_id']}:alias:{_slug(alias)}",
                    scope_key=scope_key,
                    subject_id=entity.entity_id,
                    predicate="alias",
                    object_value=alias,
                    claim=f"{entity.name} is also known as {alias}.",
                    evidence_ids=[],
                    confidence=0.9,
                    valid_from=str(row["first_seen_at"]),
                    valid_to=None,
                    metadata={"source": "discord_identity", "persona_id": persona_id},
                )
            )
            result.facts += 1
    return result


def _backfill_turns(
    conn: sqlite3.Connection,
    target: SQLiteGrilloV2Store,
    *,
    persona_id: str,
    limit: int,
) -> int:
    rows = _query_or_empty(
        conn,
        """
        SELECT *
        FROM grillo_turns
        ORDER BY created_at ASC
        LIMIT ?
        """,
        (max(1, int(limit)),),
    )
    count = 0
    for row in rows:
        metadata = from_json_dict(row["metadata_json"])
        scope_key = _v2_scope(row["scope_key"], metadata)
        actor_id = _actor_id(row, metadata, persona_id=persona_id)
        target.append_episode(
            GrilloEpisode(
                episode_id=f"episode:v1:{row['turn_id']}",
                scope_key=scope_key,
                source=f"grillo_v1:{row['source']}",
                content=row["content"],
                actor_id=actor_id,
                participant_ids=_dedupe([actor_id, _participant_entity(row["participant_key"])]),
                channel_id=str(row["channel_id"]) if row["channel_id"] else None,
                occurred_at=row["created_at"],
                metadata={
                    **metadata,
                    "source_scope_key": row["scope_key"],
                    "source_turn_id": row["turn_id"],
                    "source_role": row["role"],
                    "author_name": row["author_name"],
                },
            )
        )
        count += 1
    return count


def _backfill_candidates(
    conn: sqlite3.Connection,
    target: SQLiteGrilloV2Store,
    *,
    persona_id: str,
    limit: int,
) -> tuple[int, int]:
    rows = _query_or_empty(
        conn,
        """
        SELECT *
        FROM grillo_candidates
        ORDER BY created_at ASC
        LIMIT ?
        """,
        (max(1, int(limit)),),
    )
    evidence_count = 0
    fact_count = 0
    for row in rows:
        scope_key = _v2_scope(row["scope_key"], {})
        subject_id = _participant_entity(row["participant_key"])
        evidence = Evidence(
            evidence_id=f"evidence:v1_candidate:{row['candidate_id']}",
            scope_key=scope_key,
            episode_id="",
            quote=row["content"],
            extractor="grillo_v1_candidate",
            confidence=float(row["confidence"]),
            created_at=row["created_at"],
            metadata={
                "source_candidate_id": row["candidate_id"],
                "source_turn_ids": from_json_list(row["source_turn_ids_json"]),
                "tags": from_json_list(row["tags_json"]),
            },
        )
        target.append_evidence(evidence)
        evidence_count += 1
        target.upsert_fact(
            TemporalFact(
                fact_id=f"fact:v1_candidate:{row['candidate_id']}",
                scope_key=scope_key,
                subject_id=subject_id,
                predicate=f"candidate:{row['type']}",
                object_value=row["summary"] or row["content"],
                claim=row["summary"] or row["content"],
                evidence_ids=[evidence.evidence_id],
                confidence=float(row["confidence"]),
                valid_from=row["created_at"],
                metadata={
                    "source": "grillo_v1_candidate",
                    "persona_id": persona_id,
                    "promoted": bool(row["promoted"]),
                    "tags": from_json_list(row["tags_json"]),
                },
            )
        )
        fact_count += 1
    return evidence_count, fact_count


def _backfill_slots(
    conn: sqlite3.Connection,
    target: SQLiteGrilloV2Store,
    *,
    persona_id: str,
    limit: int,
) -> tuple[int, int]:
    rows = _query_or_empty(
        conn,
        """
        SELECT *
        FROM grillo_slots
        ORDER BY updated_at ASC
        LIMIT ?
        """,
        (max(1, int(limit)),),
    )
    evidence_count = 0
    fact_count = 0
    for row in rows:
        scope_key = _v2_scope(row["scope_key"], {})
        subject_id = _participant_entity(row["participant_key"])
        for index, item in enumerate(from_json_list(row["items_json"])):
            text = str(item)
            if not text.strip():
                continue
            evidence = Evidence(
                evidence_id=f"evidence:v1_slot:{row['slot_id']}:{index}",
                scope_key=scope_key,
                episode_id="",
                quote=text,
                extractor="grillo_v1_slot",
                confidence=0.8,
                created_at=row["updated_at"],
                metadata={
                    "source_slot_id": row["slot_id"],
                    "source_candidate_ids": from_json_list(row["source_candidate_ids_json"]),
                    "slot_name": row["slot_name"],
                },
            )
            target.append_evidence(evidence)
            evidence_count += 1
            target.upsert_fact(
                TemporalFact(
                    fact_id=f"fact:v1_slot:{row['slot_id']}:{index}",
                    scope_key=scope_key,
                    subject_id=subject_id,
                    predicate=f"slot:{row['slot_name']}",
                    object_value=text,
                    claim=text,
                    evidence_ids=[evidence.evidence_id],
                    confidence=0.8,
                    valid_from=row["updated_at"],
                    metadata={"source": "grillo_v1_slot", "persona_id": persona_id},
                )
            )
            fact_count += 1
    return evidence_count, fact_count


def _query_or_empty(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...]) -> list[sqlite3.Row]:
    try:
        return conn.execute(sql, params).fetchall()
    except sqlite3.Error:
        return []


def _v2_scope(scope_key: str, metadata: dict[str, Any]) -> str:
    guild_id = metadata.get("guild_id")
    if guild_id:
        return f"discord:guild:{guild_id}:persona:v2"
    match = GUILD_USER_SCOPE_RE.match(str(scope_key))
    if match:
        return f"discord:guild:{match.group('guild_id')}:persona:v2"
    match = DM_SCOPE_RE.match(str(scope_key))
    if match:
        return f"discord:dm:{match.group('user_id')}:persona:v2"
    return f"{scope_key}:v2"


def _actor_id(row: sqlite3.Row, metadata: dict[str, Any], *, persona_id: str) -> str:
    if str(row["role"]) == "assistant":
        return persona_id
    if metadata.get("author_id"):
        return _discord_user_entity_id(metadata["author_id"])
    return _participant_entity(row["participant_key"])


def _participant_entity(participant_key: Any) -> str:
    value = str(participant_key or "unknown")
    if value.isdigit():
        return _discord_user_entity_id(value)
    if value.startswith("discord_user:"):
        return value
    return f"participant:{value}"


def _discord_user_entity_id(user_id: Any) -> str:
    return f"discord_user:{user_id}"


def _slug(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in value)[:80].strip("-")
    return cleaned or "alias"


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
