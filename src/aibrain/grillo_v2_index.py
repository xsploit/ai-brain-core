from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from grillo_v2 import (
    Evidence,
    GrilloMemoryDocument,
    GrilloEntity,
    GrilloEpisode,
    OpinionEdge,
    SQLiteGrilloV2Store,
    TemporalFact as GrilloTemporalFact,
)
from grillo_v2.models import dataclass_dict
from grillo_v2.safety import is_unsafe_memory_payload

from .embeddings import EmbeddingProvider, HashEmbeddingProvider
from .memory_stack.contracts import GraphQuery, RecallHit, RecallItem, TemporalFact
from .memory_stack.stack import _create_graph_store, _create_vector_store


@dataclass(slots=True)
class GrilloV2PackageRecall:
    graph_facts: list[TemporalFact] = field(default_factory=list)
    vector_hits: list[RecallHit] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class GrilloV2LadybugMirror:
    """Structured Ladybug mirror for GRILLO v2 graph diagnostics/traversal."""

    def __init__(self, graph_store: Any, *, persona_id: str):
        self.graph_store = graph_store
        self.conn = graph_store.conn
        self.persona_id = persona_id
        self.last_counts: dict[str, int] = {}
        self._initialized = False

    @classmethod
    def from_graph_store(cls, graph_store: Any, *, persona_id: str) -> "GrilloV2LadybugMirror | None":
        if type(graph_store).__name__ != "LadybugGraphMemoryStore":
            return None
        if getattr(graph_store, "conn", None) is None or not hasattr(graph_store, "_call_locked"):
            return None
        return cls(graph_store, persona_id=persona_id)

    async def sync_scope(
        self,
        *,
        entities: list[GrilloEntity],
        episodes: list[GrilloEpisode],
        evidence: list[Evidence],
        facts: list[GrilloTemporalFact],
        opinions: list[OpinionEdge],
        documents: list[GrilloMemoryDocument],
    ) -> None:
        await self.graph_store._call_locked(
            self._sync_scope_sync,
            entities,
            episodes,
            evidence,
            facts,
            opinions,
            documents,
        )

    def _sync_scope_sync(
        self,
        entities: list[GrilloEntity],
        episodes: list[GrilloEpisode],
        evidence: list[Evidence],
        facts: list[GrilloTemporalFact],
        opinions: list[OpinionEdge],
        documents: list[GrilloMemoryDocument],
    ) -> None:
        self._init_schema()
        for entity in entities:
            self._upsert_entity(entity)
        for episode in episodes:
            self._upsert_episode(episode)
        for item in evidence:
            self._upsert_evidence(item)
        for fact in facts:
            self._upsert_fact(fact)
        for edge in opinions:
            self._upsert_opinion(edge)
        for document in documents:
            self._upsert_document(document)
        self.last_counts = {
            "entities": len(entities),
            "episodes": len(episodes),
            "evidence": len(evidence),
            "facts": len(facts),
            "opinions": len(opinions),
            "memory_docs": len(documents),
        }

    def _init_schema(self) -> None:
        if self._initialized:
            return
        for statement in (
            """
            CREATE NODE TABLE IF NOT EXISTS GrilloEntity(
                id STRING PRIMARY KEY,
                entity_type STRING,
                name STRING,
                aliases_json STRING,
                metadata_json STRING,
                updated_at STRING
            );
            """,
            """
            CREATE NODE TABLE IF NOT EXISTS GrilloEpisode(
                id STRING PRIMARY KEY,
                scope_key STRING,
                source STRING,
                actor_id STRING,
                participant_ids_json STRING,
                channel_id STRING,
                content_preview STRING,
                metadata_json STRING,
                occurred_at STRING
            );
            """,
            """
            CREATE NODE TABLE IF NOT EXISTS GrilloEvidence(
                id STRING PRIMARY KEY,
                scope_key STRING,
                episode_id STRING,
                quote STRING,
                extractor STRING,
                confidence DOUBLE,
                metadata_json STRING,
                created_at STRING
            );
            """,
            """
            CREATE NODE TABLE IF NOT EXISTS GrilloTemporalFact(
                id STRING PRIMARY KEY,
                scope_key STRING,
                subject_id STRING,
                predicate STRING,
                object_value STRING,
                claim STRING,
                confidence DOUBLE,
                valid_from STRING,
                valid_to STRING,
                evidence_ids_json STRING,
                contradicts_json STRING,
                missing_evidence_json STRING,
                metadata_json STRING,
                updated_at STRING
            );
            """,
            """
            CREATE NODE TABLE IF NOT EXISTS GrilloOpinionEdge(
                id STRING PRIMARY KEY,
                scope_key STRING,
                source_id STRING,
                target_id STRING,
                relation STRING,
                score DOUBLE,
                rationale STRING,
                evidence_ids_json STRING,
                valid_from STRING,
                valid_to STRING,
                metadata_json STRING,
                updated_at STRING
            );
            """,
            """
            CREATE NODE TABLE IF NOT EXISTS GrilloMemoryDocument(
                id STRING PRIMARY KEY,
                scope_key STRING,
                document_type STRING,
                subject_id STRING,
                title STRING,
                body_preview STRING,
                importance DOUBLE,
                evidence_ids_json STRING,
                metadata_json STRING,
                updated_at STRING
            );
            """,
            "CREATE REL TABLE IF NOT EXISTS EPISODE_ACTOR(FROM GrilloEpisode TO GrilloEntity);",
            "CREATE REL TABLE IF NOT EXISTS EPISODE_PARTICIPANT(FROM GrilloEpisode TO GrilloEntity);",
            "CREATE REL TABLE IF NOT EXISTS EVIDENCE_FROM_EPISODE(FROM GrilloEvidence TO GrilloEpisode);",
            "CREATE REL TABLE IF NOT EXISTS FACT_SUBJECT(FROM GrilloTemporalFact TO GrilloEntity);",
            "CREATE REL TABLE IF NOT EXISTS FACT_EVIDENCE(FROM GrilloTemporalFact TO GrilloEvidence);",
            "CREATE REL TABLE IF NOT EXISTS OPINION_SOURCE(FROM GrilloOpinionEdge TO GrilloEntity);",
            "CREATE REL TABLE IF NOT EXISTS OPINION_TARGET(FROM GrilloOpinionEdge TO GrilloEntity);",
            "CREATE REL TABLE IF NOT EXISTS OPINION_EVIDENCE(FROM GrilloOpinionEdge TO GrilloEvidence);",
            "CREATE REL TABLE IF NOT EXISTS MEMORY_SUBJECT(FROM GrilloMemoryDocument TO GrilloEntity);",
            "CREATE REL TABLE IF NOT EXISTS MEMORY_EVIDENCE(FROM GrilloMemoryDocument TO GrilloEvidence);",
        ):
            self.conn.execute(statement)
        self._initialized = True

    def _upsert_entity(self, entity: GrilloEntity) -> None:
        self.conn.execute(
            """
            MERGE (e:GrilloEntity {id: $id})
            SET e.entity_type = $entity_type,
                e.name = $name,
                e.aliases_json = $aliases_json,
                e.metadata_json = $metadata_json,
                e.updated_at = $updated_at
            """,
            {
                "id": entity.entity_id,
                "entity_type": entity.entity_type,
                "name": entity.name,
                "aliases_json": _json(entity.aliases),
                "metadata_json": _json(entity.metadata),
                "updated_at": str(entity.metadata.get("updated_at") or ""),
            },
        )

    def _upsert_stub_entity(self, entity_id: str | None) -> None:
        if not entity_id:
            return
        self.conn.execute(
            """
            MERGE (e:GrilloEntity {id: $id})
            """,
            {"id": entity_id},
        )

    def _upsert_episode(self, episode: GrilloEpisode) -> None:
        self._upsert_stub_entity(episode.actor_id)
        for participant_id in episode.participant_ids:
            self._upsert_stub_entity(participant_id)
        self.conn.execute(
            """
            MERGE (e:GrilloEpisode {id: $id})
            SET e.scope_key = $scope_key,
                e.source = $source,
                e.actor_id = $actor_id,
                e.participant_ids_json = $participant_ids_json,
                e.channel_id = $channel_id,
                e.content_preview = $content_preview,
                e.metadata_json = $metadata_json,
                e.occurred_at = $occurred_at
            """,
            {
                "id": episode.episode_id,
                "scope_key": episode.scope_key,
                "source": episode.source,
                "actor_id": episode.actor_id,
                "participant_ids_json": _json(episode.participant_ids),
                "channel_id": episode.channel_id,
                "content_preview": _preview(episode.content),
                "metadata_json": _json(episode.metadata),
                "occurred_at": episode.occurred_at,
            },
        )
        if episode.actor_id:
            self._rel("GrilloEpisode", episode.episode_id, "EPISODE_ACTOR", "GrilloEntity", episode.actor_id)
        for participant_id in episode.participant_ids:
            self._rel("GrilloEpisode", episode.episode_id, "EPISODE_PARTICIPANT", "GrilloEntity", participant_id)

    def _upsert_evidence(self, evidence: Evidence) -> None:
        self.conn.execute(
            """
            MERGE (e:GrilloEvidence {id: $id})
            SET e.scope_key = $scope_key,
                e.episode_id = $episode_id,
                e.quote = $quote,
                e.extractor = $extractor,
                e.confidence = $confidence,
                e.metadata_json = $metadata_json,
                e.created_at = $created_at
            """,
            {
                "id": evidence.evidence_id,
                "scope_key": evidence.scope_key,
                "episode_id": evidence.episode_id,
                "quote": _preview(evidence.quote, limit=1200),
                "extractor": evidence.extractor,
                "confidence": float(evidence.confidence),
                "metadata_json": _json(evidence.metadata),
                "created_at": evidence.created_at,
            },
        )
        self._rel("GrilloEvidence", evidence.evidence_id, "EVIDENCE_FROM_EPISODE", "GrilloEpisode", evidence.episode_id)

    def _upsert_fact(self, fact: GrilloTemporalFact) -> None:
        self._upsert_stub_entity(fact.subject_id)
        self.conn.execute(
            """
            MERGE (f:GrilloTemporalFact {id: $id})
            SET f.scope_key = $scope_key,
                f.subject_id = $subject_id,
                f.predicate = $predicate,
                f.object_value = $object_value,
                f.claim = $claim,
                f.confidence = $confidence,
                f.valid_from = $valid_from,
                f.valid_to = $valid_to,
                f.evidence_ids_json = $evidence_ids_json,
                f.contradicts_json = $contradicts_json,
                f.missing_evidence_json = $missing_evidence_json,
                f.metadata_json = $metadata_json,
                f.updated_at = $updated_at
            """,
            {
                "id": fact.fact_id,
                "scope_key": fact.scope_key,
                "subject_id": fact.subject_id,
                "predicate": fact.predicate,
                "object_value": fact.object_value,
                "claim": fact.claim,
                "confidence": float(fact.confidence),
                "valid_from": fact.valid_from,
                "valid_to": fact.valid_to,
                "evidence_ids_json": _json(fact.evidence_ids),
                "contradicts_json": _json(fact.contradicts),
                "missing_evidence_json": _json([dataclass_dict(gap) for gap in fact.missing_evidence]),
                "metadata_json": _json(fact.metadata),
                "updated_at": fact.updated_at,
            },
        )
        self._rel("GrilloTemporalFact", fact.fact_id, "FACT_SUBJECT", "GrilloEntity", fact.subject_id)
        for evidence_id in fact.evidence_ids:
            self._rel("GrilloTemporalFact", fact.fact_id, "FACT_EVIDENCE", "GrilloEvidence", evidence_id)

    def _upsert_opinion(self, edge: OpinionEdge) -> None:
        self._upsert_stub_entity(edge.source_id)
        self._upsert_stub_entity(edge.target_id)
        self.conn.execute(
            """
            MERGE (o:GrilloOpinionEdge {id: $id})
            SET o.scope_key = $scope_key,
                o.source_id = $source_id,
                o.target_id = $target_id,
                o.relation = $relation,
                o.score = $score,
                o.rationale = $rationale,
                o.evidence_ids_json = $evidence_ids_json,
                o.valid_from = $valid_from,
                o.valid_to = $valid_to,
                o.metadata_json = $metadata_json,
                o.updated_at = $updated_at
            """,
            {
                "id": edge.edge_id,
                "scope_key": edge.scope_key,
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "relation": edge.relation,
                "score": float(edge.score),
                "rationale": edge.rationale,
                "evidence_ids_json": _json(edge.evidence_ids),
                "valid_from": edge.valid_from,
                "valid_to": edge.valid_to,
                "metadata_json": _json(edge.metadata),
                "updated_at": edge.updated_at,
            },
        )
        self._rel("GrilloOpinionEdge", edge.edge_id, "OPINION_SOURCE", "GrilloEntity", edge.source_id)
        self._rel("GrilloOpinionEdge", edge.edge_id, "OPINION_TARGET", "GrilloEntity", edge.target_id)
        for evidence_id in edge.evidence_ids:
            self._rel("GrilloOpinionEdge", edge.edge_id, "OPINION_EVIDENCE", "GrilloEvidence", evidence_id)

    def _upsert_document(self, document: GrilloMemoryDocument) -> None:
        self._upsert_stub_entity(document.subject_id)
        self.conn.execute(
            """
            MERGE (m:GrilloMemoryDocument {id: $id})
            SET m.scope_key = $scope_key,
                m.document_type = $document_type,
                m.subject_id = $subject_id,
                m.title = $title,
                m.body_preview = $body_preview,
                m.importance = $importance,
                m.evidence_ids_json = $evidence_ids_json,
                m.metadata_json = $metadata_json,
                m.updated_at = $updated_at
            """,
            {
                "id": document.memory_id,
                "scope_key": document.scope_key,
                "document_type": document.document_type,
                "subject_id": document.subject_id,
                "title": document.title,
                "body_preview": _preview(document.body),
                "importance": float(document.importance),
                "evidence_ids_json": _json(document.evidence_ids),
                "metadata_json": _json(document.metadata),
                "updated_at": document.updated_at,
            },
        )
        if document.subject_id:
            self._rel("GrilloMemoryDocument", document.memory_id, "MEMORY_SUBJECT", "GrilloEntity", document.subject_id)
        for evidence_id in document.evidence_ids:
            self._rel("GrilloMemoryDocument", document.memory_id, "MEMORY_EVIDENCE", "GrilloEvidence", evidence_id)

    def _rel(self, from_label: str, from_id: str | None, rel: str, to_label: str, to_id: str | None) -> None:
        if not from_id or not to_id:
            return
        self.conn.execute(
            f"""
            MATCH (a:{from_label}), (b:{to_label})
            WHERE a.id = $from_id AND b.id = $to_id
            MERGE (a)-[:{rel}]->(b)
            """,
            {"from_id": from_id, "to_id": to_id},
        )


class GrilloV2PackageIndex:
    """Package-backed graph/vector index for GRILLO v2 state.

    SQLiteGrilloV2Store stays canonical. This index mirrors active facts,
    opinion edges, and memory documents into the existing Ladybug/TurboVec
    adapters so V2 can use the real package-backed retrieval path without a
    risky storage migration.
    """

    def __init__(
        self,
        *,
        graph_store: Any,
        vector_store: Any | None,
        embedding_provider: EmbeddingProvider,
        persona_id: str,
        sync_limit: int = 500,
        recall_top_k: int = 5,
    ):
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.persona_id = persona_id
        self.sync_limit = max(1, int(sync_limit))
        self.recall_top_k = max(1, int(recall_top_k))
        self._synced: dict[str, str] = {}
        self.structured_graph = GrilloV2LadybugMirror.from_graph_store(graph_store, persona_id=persona_id)

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        persona_id: str,
        graph_backend: str = "auto",
        vector_backend: str = "auto",
        embedding_dimensions: int = 256,
        embedding_provider: EmbeddingProvider | None = None,
        sync_limit: int = 500,
        recall_top_k: int = 5,
    ) -> "GrilloV2PackageIndex":
        base = Path(path)
        provider = embedding_provider or HashEmbeddingProvider(dimensions=embedding_dimensions)
        graph_store = _create_graph_store(base, backend=graph_backend)
        vector_store = _create_vector_store(
            base,
            embedding_provider=provider,
            dimensions=embedding_dimensions,
            backend=vector_backend,
        )
        return cls(
            graph_store=graph_store,
            vector_store=vector_store,
            embedding_provider=provider,
            persona_id=persona_id,
            sync_limit=sync_limit,
            recall_top_k=recall_top_k,
        )

    def status(self) -> dict[str, Any]:
        return {
            "graph_backend": type(self.graph_store).__name__,
            "vector_backend": type(self.vector_store).__name__ if self.vector_store is not None else None,
            "embedding_provider": type(self.embedding_provider).__name__,
            "structured_graph": type(self.structured_graph).__name__ if self.structured_graph is not None else None,
            "structured_graph_last_counts": self.structured_graph.last_counts if self.structured_graph is not None else None,
            "synced_items": len(self._synced),
            "sync_limit": self.sync_limit,
            "recall_top_k": self.recall_top_k,
        }

    def close(self) -> None:
        close = getattr(self.graph_store, "close", None)
        if close:
            close()
        if self.vector_store is not None:
            close = getattr(self.vector_store, "close", None)
            if close:
                close()

    async def sync_scope(self, store: SQLiteGrilloV2Store, scope_key: str) -> None:
        active_facts = _filter_safe_facts(store.list_active_facts(scope_key, limit=self.sync_limit))
        all_facts = _filter_safe_facts(
            store.list_temporal_facts(scope_key, include_expired=True, limit=self.sync_limit)
        )
        documents = _filter_safe_documents(store.list_memory_documents(scope_key, limit=self.sync_limit))
        active_opinions = store.list_opinion_edges(scope_key, limit=self.sync_limit)
        all_opinions = store.list_opinion_edges(scope_key, include_expired=True, limit=self.sync_limit)
        if self.structured_graph is not None:
            await self.structured_graph.sync_scope(
                entities=store.list_entities(limit=self.sync_limit),
                episodes=store.list_recent_episodes(scope_key, limit=self.sync_limit),
                evidence=store.list_evidence(scope_key, limit=self.sync_limit),
                facts=all_facts,
                opinions=all_opinions,
                documents=documents,
            )
        for fact in active_facts:
            await self._sync_fact(fact)
        for edge in active_opinions:
            await self._sync_opinion(edge)
        if self.vector_store is not None:
            for document in documents:
                await self._sync_document(document)

    async def recall(
        self,
        *,
        scope_key: str,
        actor_id: str | None,
        query: str,
        top_k: int | None = None,
    ) -> GrilloV2PackageRecall:
        limit = max(1, int(top_k or self.recall_top_k))
        notes = [
            f"package_graph={type(self.graph_store).__name__}",
            f"package_vector={type(self.vector_store).__name__ if self.vector_store is not None else 'none'}",
        ]
        graph_hits: list[TemporalFact] = []
        if query.strip():
            graph_hits.extend(
                await self.graph_store.search_facts(
                    GraphQuery(text=query, thread_id=scope_key, persona_id=self.persona_id, top_k=limit)
                )
            )
        if actor_id:
            graph_hits.extend(
                await self.graph_store.search_facts(
                    GraphQuery(
                        text="",
                        subject=actor_id,
                        thread_id=scope_key,
                        persona_id=self.persona_id,
                        top_k=limit,
                    )
                )
            )
            graph_hits.extend(
                await self.graph_store.search_facts(
                    GraphQuery(
                        text="",
                        object=actor_id,
                        thread_id=scope_key,
                        persona_id=self.persona_id,
                        top_k=limit,
                    )
                )
            )
        graph_hits = _dedupe_graph_hits(graph_hits)[:limit]
        vector_hits: list[RecallHit] = []
        if self.vector_store is not None and query.strip():
            vector_hits = await self.vector_store.search(
                query,
                top_k=limit,
                filters={"scope": scope_key, "persona_id": self.persona_id},
            )
        notes.append(f"package_recall_graph={len(graph_hits)}")
        notes.append(f"package_recall_vector={len(vector_hits)}")
        return GrilloV2PackageRecall(graph_facts=graph_hits, vector_hits=vector_hits, notes=notes)

    async def _sync_fact(self, fact: GrilloTemporalFact) -> None:
        key = f"fact:{fact.fact_id}"
        signature = f"{fact.updated_at}:{fact.valid_to}:{fact.confidence}:{fact.claim}"
        if self._synced.get(key) == signature:
            return
        graph_fact = TemporalFact(
            id=key,
            subject=fact.subject_id,
            predicate=fact.predicate,
            object=fact.object_value,
            valid_from=fact.valid_from,
            valid_until=fact.valid_to,
            confidence=fact.confidence,
            importance=fact.confidence,
            source_event_id=fact.evidence_ids[0] if fact.evidence_ids else None,
            source_session_id=fact.scope_key,
            metadata={
                "grillo_v2_kind": "temporal_fact",
                "persona_id": self.persona_id,
                "scope_key": fact.scope_key,
                "claim": fact.claim,
                "evidence_ids": fact.evidence_ids,
                "contradicts": fact.contradicts,
                "missing_evidence": [dataclass_dict(gap) for gap in fact.missing_evidence],
                "updated_at": fact.updated_at,
                **fact.metadata,
            },
            created_at=fact.valid_from,
            updated_at=fact.updated_at,
        )
        await self.graph_store.upsert_fact(graph_fact)
        if self.vector_store is not None:
            await self.vector_store.add(
                RecallItem(
                    id=key,
                    text=fact.claim or f"{fact.subject_id} {fact.predicate} {fact.object_value}",
                    scope=fact.scope_key,
                    thread_id=fact.scope_key,
                    persona_id=self.persona_id,
                    source_event_id=fact.evidence_ids[0] if fact.evidence_ids else None,
                    source_fact_id=fact.fact_id,
                    importance=fact.confidence,
                    metadata={
                        "grillo_v2_kind": "temporal_fact",
                        "subject_id": fact.subject_id,
                        "predicate": fact.predicate,
                        "object_value": fact.object_value,
                    },
                    created_at=fact.valid_from,
                )
            )
        self._synced[key] = signature

    async def _sync_opinion(self, edge: OpinionEdge) -> None:
        key = f"opinion:{edge.edge_id}"
        signature = f"{edge.updated_at}:{edge.valid_to}:{edge.score}:{edge.rationale}"
        if self._synced.get(key) == signature:
            return
        importance = min(1.0, max(0.1, abs(float(edge.score))))
        graph_fact = TemporalFact(
            id=key,
            subject=edge.source_id,
            predicate=f"opinion:{edge.relation}",
            object=edge.target_id,
            valid_from=edge.valid_from,
            valid_until=edge.valid_to,
            confidence=importance,
            importance=importance,
            source_event_id=edge.evidence_ids[0] if edge.evidence_ids else None,
            source_session_id=edge.scope_key,
            metadata={
                "grillo_v2_kind": "opinion_edge",
                "persona_id": self.persona_id,
                "scope_key": edge.scope_key,
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "relation": edge.relation,
                "score": edge.score,
                "rationale": edge.rationale,
                "evidence_ids": edge.evidence_ids,
                "updated_at": edge.updated_at,
                **edge.metadata,
            },
            created_at=edge.valid_from,
            updated_at=edge.updated_at,
        )
        await self.graph_store.upsert_fact(graph_fact)
        if self.vector_store is not None:
            await self.vector_store.add(
                RecallItem(
                    id=key,
                    text=f"{edge.relation} {edge.target_id}: {edge.rationale}",
                    scope=edge.scope_key,
                    thread_id=edge.scope_key,
                    persona_id=self.persona_id,
                    source_event_id=edge.evidence_ids[0] if edge.evidence_ids else None,
                    source_fact_id=edge.edge_id,
                    importance=importance,
                    metadata={
                        "grillo_v2_kind": "opinion_edge",
                        "source_id": edge.source_id,
                        "target_id": edge.target_id,
                        "relation": edge.relation,
                        "score": edge.score,
                    },
                    created_at=edge.valid_from,
                )
            )
        self._synced[key] = signature

    async def _sync_document(self, document: GrilloMemoryDocument) -> None:
        key = f"memory:{document.memory_id}"
        signature = f"{document.updated_at}:{document.importance}:{document.title}:{document.body}"
        if self._synced.get(key) == signature:
            return
        await self.vector_store.add(
            RecallItem(
                id=key,
                text="\n".join(part for part in [document.title, document.body] if part),
                scope=document.scope_key,
                thread_id=document.scope_key,
                persona_id=self.persona_id,
                source_event_id=document.evidence_ids[0] if document.evidence_ids else None,
                source_fact_id=document.memory_id,
                importance=document.importance,
                metadata={
                    "grillo_v2_kind": "memory_document",
                    "document_type": document.document_type,
                    "subject_id": document.subject_id,
                    "title": document.title,
                },
                created_at=document.updated_at,
            )
        )
        self._synced[key] = signature


def _dedupe_graph_hits(hits: list[TemporalFact]) -> list[TemporalFact]:
    seen: set[str] = set()
    out: list[TemporalFact] = []
    for hit in hits:
        if hit.id in seen:
            continue
        seen.add(hit.id)
        out.append(hit)
    return out


def _filter_safe_facts(facts: list[GrilloTemporalFact]) -> list[GrilloTemporalFact]:
    return [
        fact
        for fact in facts
        if not is_unsafe_memory_payload(
            text=" ".join([fact.predicate, fact.object_value, fact.claim]),
            subject_id=fact.subject_id,
            predicate=fact.predicate,
        )
    ]


def _filter_safe_documents(documents: list[GrilloMemoryDocument]) -> list[GrilloMemoryDocument]:
    return [
        document
        for document in documents
        if not is_unsafe_memory_payload(
            text=" ".join([document.title, document.body]),
            document_type=document.document_type,
            subject_id=document.subject_id or "",
        )
    ]


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _preview(value: str, *, limit: int = 2400) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."
