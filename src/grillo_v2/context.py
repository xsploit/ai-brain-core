from __future__ import annotations

from .models import (
    GrilloContextPacket,
    GrilloEntity,
    GrilloEpisode,
    GrilloMemoryDocument,
    OpinionEdge,
    TemporalFact,
    dataclass_dict,
)
from .store import SQLiteGrilloV2Store


DEFAULT_GRILLO_V2_INSTRUCTIONS = [
    "Use current_actor for identity and alias continuity.",
    "Use relationship_state as computed opinion edges, not as objective fact.",
    "Use memory_blocks as durable diary/profile/slot context with provenance.",
    "Use active_facts as evidence-backed semantic memory.",
    "Treat evidence_gaps as uncertainty markers; do not present gaps as known facts.",
    "Use recent_episode_summary only as local continuity, not as durable truth.",
    "If active_facts conflict with the current user message, trust the current message and mark the conflict for reflection.",
]


class GrilloContextBuilder:
    def __init__(self, store: SQLiteGrilloV2Store):
        self.store = store

    def build_packet(
        self,
        *,
        scope_key: str,
        actor_id: str | None = None,
        persona_id: str | None = None,
        query: str = "",
        channel_id: str | None = None,
        fact_limit: int = 8,
        opinion_limit: int = 8,
        episode_limit: int = 6,
    ) -> GrilloContextPacket:
        actor_facts = (
            self.store.list_active_facts(scope_key, subject_id=actor_id, query=query, limit=fact_limit)
            if actor_id
            else []
        )
        query_facts = self.store.search_facts(scope_key, query, limit=fact_limit) if query.strip() else []
        facts = _merge_facts([*actor_facts, *query_facts])[:fact_limit]
        opinions = self.store.list_opinion_edges(
            scope_key,
            source_id=persona_id,
            target_id=actor_id,
            limit=opinion_limit,
        ) if actor_id else []
        episodes = self.store.list_recent_episodes(
            scope_key,
            actor_id=actor_id,
            channel_id=channel_id,
            limit=episode_limit,
        )
        memory_documents = self.store.list_memory_documents(
            scope_key,
            subject_id=actor_id,
            query=query,
            limit=6,
        )
        actor = self.store.get_entity(actor_id)
        return GrilloContextPacket(
            scope_key=scope_key,
            actor_id=actor_id,
            current_actor=_actor_context(actor_id, actor, facts),
            relationship_state=[_opinion_context(edge) for edge in opinions],
            memory_blocks=[_memory_document_context(document) for document in memory_documents],
            active_facts=[_fact_context(fact) for fact in facts],
            evidence_gaps=_evidence_gap_context(facts),
            recent_episode_summary=[_episode_context(episode) for episode in episodes],
            retrieval_notes=_retrieval_notes(
                facts=facts,
                opinions=opinions,
                memory_documents=memory_documents,
                episodes=episodes,
            ),
            instructions=list(DEFAULT_GRILLO_V2_INSTRUCTIONS),
        )


def _merge_facts(facts: list[TemporalFact]) -> list[TemporalFact]:
    seen: set[str] = set()
    merged: list[TemporalFact] = []
    for fact in sorted(facts, key=lambda item: (item.confidence, item.updated_at), reverse=True):
        if fact.fact_id in seen:
            continue
        seen.add(fact.fact_id)
        merged.append(fact)
    return merged


def _actor_context(actor_id: str | None, actor: GrilloEntity | None, facts: list[TemporalFact]) -> dict[str, object]:
    if not actor_id:
        return {}
    aliases = [
        fact.object_value
        for fact in facts
        if fact.subject_id == actor_id and fact.predicate in {"alias", "display_name", "preferred_name"}
    ]
    if actor is not None:
        aliases = [actor.name, *actor.aliases, *aliases]
    return {
        "entity_id": actor_id,
        "entity_type": actor.entity_type if actor is not None else "unknown",
        "name": actor.name if actor is not None else actor_id,
        "known_aliases": _dedupe(aliases)[:8],
        "metadata": actor.metadata if actor is not None else {},
    }


def _fact_context(fact: TemporalFact) -> dict[str, object]:
    return {
        "id": fact.fact_id,
        "subject": fact.subject_id,
        "predicate": fact.predicate,
        "object": fact.object_value,
        "claim": fact.claim,
        "confidence": round(float(fact.confidence), 4),
        "valid_from": fact.valid_from,
        "valid_to": fact.valid_to,
        "evidence_ids": fact.evidence_ids[:8],
        "contradicts": fact.contradicts[:8],
    }


def _opinion_context(edge: OpinionEdge) -> dict[str, object]:
    return {
        "id": edge.edge_id,
        "source": edge.source_id,
        "target": edge.target_id,
        "relation": edge.relation,
        "score": round(float(edge.score), 4),
        "rationale": edge.rationale,
        "evidence_ids": edge.evidence_ids[:8],
        "valid_from": edge.valid_from,
        "valid_to": edge.valid_to,
    }


def _memory_document_context(document: GrilloMemoryDocument) -> dict[str, object]:
    return {
        "id": document.memory_id,
        "type": document.document_type,
        "subject": document.subject_id,
        "title": document.title,
        "body": document.body,
        "importance": round(float(document.importance), 4),
        "evidence_ids": document.evidence_ids[:8],
        "updated_at": document.updated_at,
        "metadata": document.metadata,
    }


def _episode_context(episode: GrilloEpisode) -> dict[str, object]:
    summary = str(episode.metadata.get("summary") or episode.content)
    if len(summary) > 360:
        summary = summary[:357].rstrip() + "..."
    return {
        "id": episode.episode_id,
        "source": episode.source,
        "actor_id": episode.actor_id,
        "participant_ids": episode.participant_ids[:8],
        "channel_id": episode.channel_id,
        "occurred_at": episode.occurred_at,
        "summary": summary,
    }


def _evidence_gap_context(facts: list[TemporalFact]) -> list[dict[str, object]]:
    gaps: list[dict[str, object]] = []
    for fact in facts:
        for gap in fact.missing_evidence:
            payload = dataclass_dict(gap)
            payload["fact_id"] = fact.fact_id
            gaps.append(payload)
    return gaps[:8]


def _retrieval_notes(
    *,
    facts: list[TemporalFact],
    opinions: list[OpinionEdge],
    memory_documents: list[GrilloMemoryDocument],
    episodes: list[GrilloEpisode],
) -> list[str]:
    notes = [
        f"active_facts={len(facts)}",
        f"relationship_state={len(opinions)}",
        f"memory_blocks={len(memory_documents)}",
        f"recent_episodes={len(episodes)}",
    ]
    if not facts:
        notes.append("No evidence-backed facts matched this context packet.")
    return notes


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result
