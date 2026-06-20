from __future__ import annotations

import asyncio
import ast
import json
from pathlib import Path
from typing import Any

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
        self.conn.execute(
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
            """
        )

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


def _row_to_fact(row: dict[str, Any]) -> TemporalFact:
    return TemporalFact(
        id=row["id"],
        subject=row["subject"],
        predicate=row["predicate"],
        object=row["object"],
        valid_from=row["valid_from"],
        valid_until=row.get("valid_until"),
        confidence=float(row.get("confidence", 0.7)),
        importance=float(row.get("importance", 0.5)),
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
