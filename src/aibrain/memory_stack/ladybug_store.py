from __future__ import annotations

import asyncio
import ast
import json
from pathlib import Path
from typing import Any

from ..numeric import safe_float
from .contracts import GraphQuery, TemporalFact


class LadybugGraphMemoryStore:
    """Optional LadybugDB graph adapter for temporal facts."""

    def __init__(self, path: str | Path):
        try:
            import ladybug as lb  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "LadybugGraphMemoryStore requires the optional 'ladybug' package."
            ) from exc
        self.path = Path(path)
        self._lb = lb
        self.db = lb.Database(str(self.path))
        self.conn = lb.Connection(self.db)
        self._init_schema()

    def _init_schema(self) -> None:
        for statement in (
            """
            CREATE NODE TABLE IF NOT EXISTS Fact(
                id STRING PRIMARY KEY,
                subject STRING,
                predicate STRING,
                object STRING,
                valid_from STRING,
                valid_until STRING,
                confidence DOUBLE,
                importance DOUBLE,
                source_event_id STRING,
                source_session_id STRING,
                supersedes_memory_id STRING,
                last_accessed_at STRING,
                metadata_json STRING,
                created_at STRING,
                updated_at STRING
            );
            """,
            """
            CREATE NODE TABLE IF NOT EXISTS MemoryScope(
                id STRING PRIMARY KEY,
                source STRING,
                channel STRING
            );
            """,
            """
            CREATE NODE TABLE IF NOT EXISTS Persona(
                id STRING PRIMARY KEY
            );
            """,
            """
            CREATE NODE TABLE IF NOT EXISTS Participant(
                id STRING PRIMARY KEY,
                source STRING,
                channel STRING,
                login STRING
            );
            """,
            """
            CREATE NODE TABLE IF NOT EXISTS RelationshipProfile(
                id STRING PRIMARY KEY,
                scope_key STRING,
                persona_id STRING,
                relationship_stage STRING,
                mood STRING,
                trust INT64,
                attraction INT64,
                respect INT64,
                irritation INT64,
                jealousy INT64,
                guard INT64,
                turn_count INT64,
                last_seen_at STRING,
                last_diary_turn_count INT64,
                last_action_tag STRING,
                summary STRING,
                diary_entry STRING,
                facts_json STRING,
                diary_history_json STRING,
                affect_state_json STRING,
                tone_preferences_json STRING,
                interaction_style_json STRING,
                boundaries_json STRING,
                active_threads_json STRING,
                updated_at STRING
            );
            """,
            """
            CREATE NODE TABLE IF NOT EXISTS RelationshipFact(
                id STRING PRIMARY KEY,
                profile_id STRING,
                scope_key STRING,
                text STRING,
                updated_at STRING
            );
            """,
            "CREATE REL TABLE IF NOT EXISTS HAS_RELATIONSHIP(FROM MemoryScope TO RelationshipProfile);",
            "CREATE REL TABLE IF NOT EXISTS RELATIONSHIP_AS_PERSONA(FROM RelationshipProfile TO Persona);",
            "CREATE REL TABLE IF NOT EXISTS RELATIONSHIP_WITH_PARTICIPANT(FROM RelationshipProfile TO Participant);",
            "CREATE REL TABLE IF NOT EXISTS HAS_RELATIONSHIP_FACT(FROM RelationshipProfile TO RelationshipFact);",
        ):
            self.conn.execute(statement)

    async def upsert_fact(self, fact: TemporalFact) -> TemporalFact:
        await asyncio.to_thread(
            self.conn.execute,
            """
            MERGE (f:Fact {id: $id})
            SET f.subject = $subject,
                f.predicate = $predicate,
                f.object = $object,
                f.valid_from = $valid_from,
                f.valid_until = $valid_until,
                f.confidence = $confidence,
                f.importance = $importance,
                f.source_event_id = $source_event_id,
                f.source_session_id = $source_session_id,
                f.supersedes_memory_id = $supersedes_memory_id,
                f.last_accessed_at = $last_accessed_at,
                f.metadata_json = $metadata_json,
                f.created_at = $created_at,
                f.updated_at = $updated_at
            """,
            _fact_params(fact),
        )
        return fact

    async def search_facts(self, query: GraphQuery) -> list[TemporalFact]:
        return await asyncio.to_thread(self._search_facts_sync, query)

    def _search_facts_sync(self, query: GraphQuery) -> list[TemporalFact]:
        clauses = []
        params: dict[str, Any] = {"limit": max(query.top_k, 1)}
        if not query.include_expired:
            clauses.append("f.valid_until IS NULL")
        for field in ("subject", "predicate", "object"):
            value = getattr(query, field)
            if value:
                clauses.append(f"f.{field} = ${field}")
                params[field] = value
        if query.text.strip():
            params["needle"] = query.text.lower()
            clauses.append(
                "(contains(lower(f.subject), $needle) OR contains(lower(f.predicate), $needle) OR contains(lower(f.object), $needle))"
            )
        cypher = "MATCH (f:Fact)"
        if clauses:
            cypher += " WHERE " + " AND ".join(clauses)
        cypher += " RETURN f ORDER BY f.importance * f.confidence DESC LIMIT $limit"
        result = self.conn.execute(cypher, params).rows_as_dict()
        rows: list[dict[str, Any]] = []
        while result.has_next():
            row = result.get_next()
            if isinstance(row, dict):
                rows.append(row.get("f", row))
        return [_row_to_fact(row) for row in rows]

    async def invalidate_fact(
        self,
        fact_id: str,
        *,
        valid_until: str,
        reason: str,
    ) -> None:
        await asyncio.to_thread(
            self.conn.execute,
            """
            MATCH (f:Fact {id: $id})
            SET f.valid_until = $valid_until,
                f.metadata_json = $metadata_json
            """,
            {
                "id": fact_id,
                "valid_until": valid_until,
                "metadata_json": json.dumps({"invalidated_reason": reason}),
            },
        )

    async def upsert_relationship_profile(self, profile: Any) -> None:
        await asyncio.to_thread(self._upsert_relationship_profile_sync, profile)

    def _upsert_relationship_profile_sync(self, profile: Any) -> None:
        params = _relationship_profile_params(profile)
        scope = _parse_scope_key(params["scope_key"])
        self.conn.execute(
            """
            MERGE (s:MemoryScope {id: $scope_key})
            SET s.source = $scope_source,
                s.channel = $scope_channel
            """,
            {
                "scope_key": params["scope_key"],
                "scope_source": scope["source"],
                "scope_channel": scope["channel"],
            },
        )
        self.conn.execute("MERGE (:Persona {id: $persona_id})", {"persona_id": params["persona_id"]})
        for participant_key in params["participant_keys"]:
            participant = _parse_participant_key(participant_key)
            self.conn.execute(
                """
                MERGE (p:Participant {id: $participant_key})
                SET p.source = $source,
                    p.channel = $channel,
                    p.login = $login
                """,
                {
                    "participant_key": participant_key,
                    "source": participant["source"],
                    "channel": participant["channel"],
                    "login": participant["login"],
                },
            )
        self.conn.execute(
            """
            MERGE (p:RelationshipProfile {id: $profile_id})
            SET p.scope_key = $scope_key,
                p.persona_id = $persona_id,
                p.relationship_stage = $relationship_stage,
                p.mood = $mood,
                p.trust = $trust,
                p.attraction = $attraction,
                p.respect = $respect,
                p.irritation = $irritation,
                p.jealousy = $jealousy,
                p.guard = $guard,
                p.turn_count = $turn_count,
                p.last_seen_at = $last_seen_at,
                p.last_diary_turn_count = $last_diary_turn_count,
                p.last_action_tag = $last_action_tag,
                p.summary = $summary,
                p.diary_entry = $diary_entry,
                p.facts_json = $facts_json,
                p.diary_history_json = $diary_history_json,
                p.affect_state_json = $affect_state_json,
                p.tone_preferences_json = $tone_preferences_json,
                p.interaction_style_json = $interaction_style_json,
                p.boundaries_json = $boundaries_json,
                p.active_threads_json = $active_threads_json,
                p.updated_at = $updated_at
            """,
            params,
        )
        self.conn.execute(
            """
            MATCH (s:MemoryScope), (p:RelationshipProfile)
            WHERE s.id = $scope_key AND p.id = $profile_id
            MERGE (s)-[:HAS_RELATIONSHIP]->(p)
            """,
            params,
        )
        self.conn.execute(
            """
            MATCH (p:RelationshipProfile), (persona:Persona)
            WHERE p.id = $profile_id AND persona.id = $persona_id
            MERGE (p)-[:RELATIONSHIP_AS_PERSONA]->(persona)
            """,
            params,
        )
        for participant_key in params["participant_keys"]:
            self.conn.execute(
                """
                MATCH (p:RelationshipProfile), (participant:Participant)
                WHERE p.id = $profile_id AND participant.id = $participant_key
                MERGE (p)-[:RELATIONSHIP_WITH_PARTICIPANT]->(participant)
                """,
                {**params, "participant_key": participant_key},
            )
        for index, fact in enumerate(params["facts"]):
            fact_params = {
                "fact_id": f"{params['profile_id']}:fact:{index}",
                "profile_id": params["profile_id"],
                "scope_key": params["scope_key"],
                "text": fact,
                "updated_at": params["updated_at"],
            }
            self.conn.execute(
                """
                MERGE (f:RelationshipFact {id: $fact_id})
                SET f.profile_id = $profile_id,
                    f.scope_key = $scope_key,
                    f.text = $text,
                    f.updated_at = $updated_at
                """,
                fact_params,
            )
            self.conn.execute(
                """
                MATCH (p:RelationshipProfile), (f:RelationshipFact)
                WHERE p.id = $profile_id AND f.id = $fact_id
                MERGE (p)-[:HAS_RELATIONSHIP_FACT]->(f)
                """,
                fact_params,
            )

    async def get_relationship_profile_graph(self, scope_key: str) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._get_relationship_profile_graph_sync, scope_key)

    def _get_relationship_profile_graph_sync(self, scope_key: str) -> dict[str, Any] | None:
        result = self.conn.execute(
            """
            MATCH (p:RelationshipProfile)
            WHERE p.scope_key = $scope_key
            RETURN p
            LIMIT 1
            """,
            {"scope_key": scope_key},
        ).rows_as_dict()
        while result.has_next():
            row = result.get_next()
            if isinstance(row, dict):
                profile = row.get("p", row)
                return profile if isinstance(profile, dict) else dict(profile)
        return None

    def close(self) -> None:
        close = getattr(self.conn, "close", None)
        if close:
            close()


def _fact_params(fact: TemporalFact) -> dict[str, Any]:
    return {
        "id": fact.id,
        "subject": fact.subject,
        "predicate": fact.predicate,
        "object": fact.object,
        "valid_from": fact.valid_from,
        "valid_until": fact.valid_until,
        "confidence": fact.confidence,
        "importance": fact.importance,
        "source_event_id": fact.source_event_id,
        "source_session_id": fact.source_session_id,
        "supersedes_memory_id": fact.supersedes_memory_id,
        "last_accessed_at": fact.last_accessed_at,
        "metadata_json": json.dumps(fact.metadata),
        "created_at": fact.created_at,
        "updated_at": fact.updated_at,
    }


def _relationship_profile_params(profile: Any) -> dict[str, Any]:
    facts = _clean_list(getattr(profile, "facts", []))
    return {
        "profile_id": _string_value(getattr(profile, "profile_id", "")),
        "scope_key": _string_value(getattr(profile, "scope_key", "")),
        "persona_id": _string_value(getattr(profile, "persona_id", "")) or "unknown",
        "participant_keys": _clean_list(getattr(profile, "participant_keys", [])),
        "relationship_stage": _string_value(getattr(profile, "relationship_stage", "")) or "new",
        "mood": _string_value(getattr(profile, "mood", "")) or "guarded",
        "trust": _int_value(getattr(profile, "trust", 4)),
        "attraction": _int_value(getattr(profile, "attraction", 1)),
        "respect": _int_value(getattr(profile, "respect", 4)),
        "irritation": _int_value(getattr(profile, "irritation", 1)),
        "jealousy": _int_value(getattr(profile, "jealousy", 0)),
        "guard": _int_value(getattr(profile, "guard", 16)),
        "turn_count": _int_value(getattr(profile, "turn_count", 0)),
        "last_seen_at": _string_value(getattr(profile, "last_seen_at", "")),
        "last_diary_turn_count": _int_value(getattr(profile, "last_diary_turn_count", 0)),
        "last_action_tag": _string_value(getattr(profile, "last_action_tag", "")) or "none",
        "summary": _string_value(getattr(profile, "summary", "")),
        "diary_entry": _string_value(getattr(profile, "diary_entry", "")),
        "facts": facts,
        "facts_json": json.dumps(facts),
        "diary_history_json": json.dumps(_clean_list(getattr(profile, "diary_history", []))),
        "affect_state_json": json.dumps(getattr(profile, "affect_state", {}) or {}),
        "tone_preferences_json": json.dumps(_clean_list(getattr(profile, "tone_preferences", []))),
        "interaction_style_json": json.dumps(_clean_list(getattr(profile, "interaction_style", []))),
        "boundaries_json": json.dumps(_clean_list(getattr(profile, "boundaries", []))),
        "active_threads_json": json.dumps(_clean_list(getattr(profile, "active_threads", []))),
        "updated_at": _string_value(getattr(profile, "updated_at", "")),
    }


def _parse_scope_key(scope_key: str) -> dict[str, str]:
    parts = scope_key.split(":")
    persona_index = parts.index("persona") if "persona" in parts else -1
    return {
        "source": parts[0] if parts else "local",
        "channel": ":".join(parts[1:persona_index]) if persona_index > 1 else "local",
        "persona_id": ":".join(parts[persona_index + 1 :]) if persona_index >= 0 else "unknown",
    }


def _parse_participant_key(participant_key: str) -> dict[str, str]:
    parts = participant_key.split(":")
    return {
        "source": parts[0] if parts else "local",
        "channel": parts[1] if len(parts) > 1 else "local",
        "login": ":".join(parts[2:]) if len(parts) > 2 else "unknown",
    }


def _clean_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [_string_value(item) for item in values if _string_value(item)]


def _string_value(value: Any) -> str:
    return str(value or "").replace("\r", " ").replace("\n", " ").strip()[:2400]


def _int_value(value: Any) -> int:
    return int(round(safe_float(value, 0)))


def _row_to_fact(row: dict[str, Any]) -> TemporalFact:
    return TemporalFact(
        id=row["id"],
        subject=row["subject"],
        predicate=row["predicate"],
        object=row["object"],
        valid_from=row["valid_from"],
        valid_until=row.get("valid_until"),
        confidence=safe_float(row.get("confidence"), 0.7),
        importance=safe_float(row.get("importance"), 0.5),
        source_event_id=row.get("source_event_id"),
        source_session_id=row.get("source_session_id"),
        supersedes_memory_id=row.get("supersedes_memory_id"),
        last_accessed_at=row.get("last_accessed_at"),
        metadata=_metadata_json(row.get("metadata_json")),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


def _metadata_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    raw = str(value or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    try:
        parsed = ast.literal_eval(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (SyntaxError, ValueError):
        return {"raw_metadata": raw}
