from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from grillo_v2 import (
    GrilloMemoryDocument,
    OpinionEdge,
    SQLiteGrilloV2Store,
    TemporalFact as GrilloTemporalFact,
)
from grillo_v2.models import dataclass_dict

from .embeddings import HashEmbeddingProvider
from .memory_stack.contracts import GraphQuery, RecallHit, RecallItem, TemporalFact
from .memory_stack.stack import _create_graph_store, _create_vector_store


@dataclass(slots=True)
class GrilloV2PackageRecall:
    graph_facts: list[TemporalFact] = field(default_factory=list)
    vector_hits: list[RecallHit] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


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
        persona_id: str,
        sync_limit: int = 500,
        recall_top_k: int = 5,
    ):
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.persona_id = persona_id
        self.sync_limit = max(1, int(sync_limit))
        self.recall_top_k = max(1, int(recall_top_k))
        self._synced: dict[str, str] = {}

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        persona_id: str,
        graph_backend: str = "auto",
        vector_backend: str = "auto",
        embedding_dimensions: int = 256,
        sync_limit: int = 500,
        recall_top_k: int = 5,
    ) -> "GrilloV2PackageIndex":
        base = Path(path)
        provider = HashEmbeddingProvider(dimensions=embedding_dimensions)
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
            persona_id=persona_id,
            sync_limit=sync_limit,
            recall_top_k=recall_top_k,
        )

    def status(self) -> dict[str, Any]:
        return {
            "graph_backend": type(self.graph_store).__name__,
            "vector_backend": type(self.vector_store).__name__ if self.vector_store is not None else None,
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
        facts = store.list_active_facts(scope_key, limit=self.sync_limit)
        documents = store.list_memory_documents(scope_key, limit=self.sync_limit)
        opinions = store.list_opinion_edges(scope_key, limit=self.sync_limit)
        for fact in facts:
            await self._sync_fact(fact)
        for edge in opinions:
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
