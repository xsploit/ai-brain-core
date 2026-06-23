from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from .context import GrilloContextBuilder
from .models import Evidence, EvidenceGap, GrilloEpisode, GrilloMemoryDocument, OpinionEdge, TemporalFact, from_json_dict
from .store import SQLiteGrilloV2Store


ReflectionCompletion = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass(slots=True)
class ReflectionResult:
    episodes: int = 0
    evidence: int = 0
    facts: int = 0
    opinions: int = 0
    memory_docs: int = 0
    invalidated_facts: int = 0
    notes: str = ""


@dataclass(slots=True)
class WorkerTickResult:
    scopes: int = 0
    batches: int = 0
    episodes: int = 0
    evidence: int = 0
    facts: int = 0
    opinions: int = 0
    memory_docs: int = 0
    invalidated_facts: int = 0
    notes: list[str] | None = None

    def add(self, result: ReflectionResult) -> None:
        self.batches += 1
        self.episodes += result.episodes
        self.evidence += result.evidence
        self.facts += result.facts
        self.opinions += result.opinions
        self.memory_docs += result.memory_docs
        self.invalidated_facts += result.invalidated_facts
        if result.notes:
            if self.notes is None:
                self.notes = []
            self.notes.append(result.notes)


class GrilloV2Runtime:
    def __init__(
        self,
        *,
        store: SQLiteGrilloV2Store,
        completion: ReflectionCompletion | None = None,
        persona_id: str = "assistant",
    ):
        self.store = store
        self.completion = completion
        self.persona_id = persona_id
        self.context = GrilloContextBuilder(store)

    def append_episode(self, episode: GrilloEpisode) -> GrilloEpisode:
        return self.store.append_episode(episode)

    def build_context_packet(
        self,
        *,
        scope_key: str,
        actor_id: str | None = None,
        query: str = "",
        channel_id: str | None = None,
    ):
        return self.context.build_packet(
            scope_key=scope_key,
            actor_id=actor_id,
            persona_id=self.persona_id,
            query=query,
            channel_id=channel_id,
        )

    async def reflect_recent(
        self,
        *,
        scope_key: str,
        actor_id: str | None = None,
        channel_id: str | None = None,
        limit: int = 12,
    ) -> ReflectionResult:
        if self.completion is None:
            return ReflectionResult(notes="no_completion")
        episodes = self.store.list_recent_episodes(
            scope_key,
            actor_id=actor_id,
            channel_id=channel_id,
            limit=limit,
        )
        if not episodes:
            return ReflectionResult(notes="no_episodes")
        packet = self.build_context_packet(
            scope_key=scope_key,
            actor_id=actor_id,
            channel_id=channel_id,
            query="\n".join(episode.content for episode in episodes[-3:]),
        )
        payload = await self.completion(
            {
                "schema": GRILLO_V2_REFLECTION_SCHEMA,
                "scope_key": scope_key,
                "actor_id": actor_id,
                "persona_id": self.persona_id,
                "context_packet": packet.as_prompt_text(),
                "episodes": [_episode_payload(episode) for episode in episodes],
            }
        )
        result = self.apply_reflection(scope_key=scope_key, payload=payload)
        result.episodes = len(episodes)
        return result

    async def reflect_unprocessed(
        self,
        *,
        scope_key: str,
        cursor_key: str | None = None,
        limit: int = 12,
    ) -> ReflectionResult:
        if self.completion is None:
            return ReflectionResult(notes="no_completion")
        cursor_key = cursor_key or self.worker_cursor_key(scope_key)
        cursor = _cursor_value(self.store.get_cursor(cursor_key))
        episodes = self.store.list_episodes_after(
            scope_key,
            after_occurred_at=cursor.get("occurred_at"),
            after_episode_id=cursor.get("episode_id"),
            limit=limit,
        )
        if not episodes:
            return ReflectionResult(notes="no_unprocessed_episodes")
        packet = self.build_context_packet(
            scope_key=scope_key,
            actor_id=episodes[-1].actor_id,
            channel_id=episodes[-1].channel_id,
            query="\n".join(episode.content for episode in episodes[-3:]),
        )
        payload = await self.completion(
            {
                "schema": GRILLO_V2_REFLECTION_SCHEMA,
                "mode": "worker_tick",
                "scope_key": scope_key,
                "cursor_key": cursor_key,
                "cursor": cursor,
                "persona_id": self.persona_id,
                "context_packet": packet.as_prompt_text(),
                "episodes": [_episode_payload(episode) for episode in episodes],
            }
        )
        result = self.apply_reflection(scope_key=scope_key, payload=payload)
        result.episodes = len(episodes)
        last = episodes[-1]
        self.store.set_cursor(
            cursor_key,
            scope_key,
            json.dumps(
                {"occurred_at": last.occurred_at, "episode_id": last.episode_id},
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        return result

    async def worker_tick(
        self,
        *,
        scope_key: str | None = None,
        scope_limit: int = 10,
        batch_size: int = 12,
        max_batches: int = 3,
    ) -> WorkerTickResult:
        scopes = [scope_key] if scope_key is not None else self.store.list_episode_scopes(limit=scope_limit)
        result = WorkerTickResult(scopes=len(scopes), notes=[])
        for scope in scopes:
            if result.batches >= max(1, int(max_batches)):
                break
            batch = await self.reflect_unprocessed(
                scope_key=scope,
                limit=max(1, int(batch_size)),
            )
            if batch.notes == "no_unprocessed_episodes":
                continue
            result.add(batch)
        if not result.notes:
            result.notes = ["no_unprocessed_episodes" if scopes else "no_scopes"]
        return result

    def worker_cursor_key(self, scope_key: str) -> str:
        return f"grillo_v2_worker:{self.persona_id}:{scope_key}"

    def apply_reflection(self, *, scope_key: str, payload: dict[str, Any]) -> ReflectionResult:
        result = ReflectionResult(notes=str(payload.get("notes") or ""))
        for item in _list(payload.get("evidence")):
            evidence = Evidence.create(
                scope_key=scope_key,
                episode_id=str(item.get("episode_id") or ""),
                quote=str(item.get("quote") or ""),
                extractor=str(item.get("extractor") or "grillo_v2"),
                confidence=_float(item.get("confidence"), 0.5),
                metadata=_dict(item.get("metadata")),
            )
            if item.get("evidence_id"):
                evidence.evidence_id = str(item["evidence_id"])
            if evidence.episode_id and evidence.quote:
                self.store.append_evidence(evidence)
                result.evidence += 1
        for item in _list(payload.get("facts")):
            fact = TemporalFact.create(
                scope_key=scope_key,
                subject_id=str(item.get("subject_id") or item.get("subject") or ""),
                predicate=str(item.get("predicate") or ""),
                object_value=str(item.get("object") or item.get("object_value") or ""),
                claim=str(item.get("claim") or ""),
                evidence_ids=[str(value) for value in _list(item.get("evidence_ids"))],
                confidence=_float(item.get("confidence"), 0.5),
                valid_from=str(item.get("valid_from") or "") or None,
                valid_to=str(item.get("valid_to") or "") or None,
                contradicts=[str(value) for value in _list(item.get("contradicts"))],
                missing_evidence=[
                    EvidenceGap(
                        question=str(gap.get("question") or ""),
                        why=str(gap.get("why") or ""),
                        needed=str(gap.get("needed") or ""),
                    )
                    for gap in _list(item.get("missing_evidence"))
                    if isinstance(gap, dict)
                ],
                metadata=_dict(item.get("metadata")),
            )
            if item.get("fact_id"):
                fact.fact_id = str(item["fact_id"])
            if fact.subject_id and fact.predicate and fact.claim:
                self.store.upsert_fact(fact)
                result.facts += 1
        for item in _list(payload.get("opinion_edges")):
            edge = OpinionEdge.create(
                scope_key=scope_key,
                source_id=str(item.get("source_id") or self.persona_id),
                target_id=str(item.get("target_id") or ""),
                relation=str(item.get("relation") or ""),
                score=_float(item.get("score"), 0.0),
                rationale=str(item.get("rationale") or ""),
                evidence_ids=[str(value) for value in _list(item.get("evidence_ids"))],
                metadata=_dict(item.get("metadata")),
            )
            if item.get("edge_id"):
                edge.edge_id = str(item["edge_id"])
            if edge.source_id and edge.target_id and edge.relation:
                self.store.upsert_opinion_edge(edge)
                result.opinions += 1
        for item in _list(payload.get("memory_documents")):
            document = GrilloMemoryDocument.create(
                scope_key=scope_key,
                document_type=str(item.get("document_type") or item.get("type") or ""),
                subject_id=str(item.get("subject_id") or item.get("subject") or "") or None,
                title=str(item.get("title") or ""),
                body=str(item.get("body") or item.get("content") or ""),
                evidence_ids=[str(value) for value in _list(item.get("evidence_ids"))],
                importance=_float(item.get("importance"), 0.5),
                metadata=_dict(item.get("metadata")),
            )
            if item.get("memory_id"):
                document.memory_id = str(item["memory_id"])
            if document.document_type and document.title and document.body:
                self.store.upsert_memory_document(document)
                result.memory_docs += 1
        for item in _list(payload.get("invalidate_facts")):
            fact_id = str(item.get("fact_id") or "") if isinstance(item, dict) else str(item)
            if fact_id:
                valid_to = None
                if isinstance(item, dict) and item.get("valid_to"):
                    valid_to = str(item["valid_to"])
                self.store.invalidate_fact(
                    fact_id,
                    valid_to=valid_to,
                    contradicts=[str(value) for value in _list(item.get("contradicts"))] if isinstance(item, dict) else [],
                )
                result.invalidated_facts += 1
        return result


GRILLO_V2_REFLECTION_SCHEMA: dict[str, Any] = {
    "name": "grillo_v2_reflection",
    "schema": {
        "type": "object",
        "properties": {
            "notes": {"type": "string"},
            "evidence": {"type": "array"},
            "facts": {"type": "array"},
            "opinion_edges": {"type": "array"},
            "memory_documents": {"type": "array"},
            "invalidate_facts": {"type": "array"},
        },
        "required": ["notes", "evidence", "facts", "opinion_edges", "memory_documents", "invalidate_facts"],
    },
}


def _episode_payload(episode: GrilloEpisode) -> dict[str, Any]:
    return {
        "episode_id": episode.episode_id,
        "scope_key": episode.scope_key,
        "source": episode.source,
        "actor_id": episode.actor_id,
        "participant_ids": episode.participant_ids,
        "channel_id": episode.channel_id,
        "content": episode.content,
        "occurred_at": episode.occurred_at,
        "metadata": episode.metadata,
    }


def _cursor_value(value: str | None) -> dict[str, str | None]:
    data = from_json_dict(value)
    occurred_at = data.get("occurred_at")
    episode_id = data.get("episode_id")
    return {
        "occurred_at": str(occurred_at) if occurred_at else None,
        "episode_id": str(episode_id) if episode_id else None,
    }


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
