from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..types import MemoryRecord
from .contracts import (
    GraphMemoryStore,
    GraphQuery,
    MemoryReranker,
    RetrievedContext,
    VectorRecallStore,
)


class ScoreReranker:
    async def rerank(
        self,
        query: str,
        contexts: list[RetrievedContext],
    ) -> list[RetrievedContext]:
        return sorted(contexts, key=lambda item: item.score, reverse=True)


class HybridRetriever:
    def __init__(
        self,
        *,
        graph_store: GraphMemoryStore | None = None,
        vector_store: VectorRecallStore | None = None,
        reranker: MemoryReranker | None = None,
    ):
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.reranker = reranker or ScoreReranker()

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        thread_id: str | None = None,
        persona_id: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedContext]:
        contexts: list[RetrievedContext] = []
        if self.graph_store is not None:
            graph_hits = await self.graph_store.search_facts(
                GraphQuery(
                    text=query,
                    thread_id=thread_id,
                    persona_id=persona_id,
                    top_k=max(top_k * 2, top_k),
                )
            )
            for fact in graph_hits:
                contexts.append(
                    RetrievedContext(
                        id=f"graph:{fact.id}",
                        content=fact.content,
                        score=_graph_score(fact.importance, fact.confidence),
                        source="graph",
                        source_id=fact.id,
                        scope="thread" if thread_id else "global",
                        thread_id=thread_id,
                        persona_id=persona_id,
                        importance=fact.importance,
                        metadata={
                            "confidence": fact.confidence,
                            "valid_from": fact.valid_from,
                            "valid_until": fact.valid_until,
                            "source_event_id": fact.source_event_id,
                            **fact.metadata,
                        },
                        created_at=fact.created_at,
                    )
                )
        if self.vector_store is not None:
            vector_filters = dict(filters or {})
            if thread_id and "thread_id" not in vector_filters:
                vector_filters["thread_id"] = thread_id
            if persona_id and "persona_id" not in vector_filters:
                vector_filters["persona_id"] = persona_id
            vector_hits = await self.vector_store.search(
                query,
                top_k=max(top_k * 2, top_k),
                filters=vector_filters or None,
            )
            for hit in vector_hits:
                contexts.append(
                    RetrievedContext(
                        id=f"vector:{hit.id}",
                        content=hit.text,
                        score=hit.score,
                        source="vector",
                        source_id=hit.id,
                        scope=hit.scope,
                        thread_id=hit.thread_id,
                        persona_id=hit.persona_id,
                        importance=hit.importance,
                        metadata={
                            "source_event_id": hit.source_event_id,
                            "source_fact_id": hit.source_fact_id,
                            **hit.metadata,
                        },
                        created_at=hit.created_at,
                    )
                )
        deduped = _dedupe(contexts)
        ranked = await self.reranker.rerank(query, deduped)
        return ranked[: max(top_k, 1)]

    async def retrieve_records(
        self,
        query: str,
        *,
        top_k: int = 5,
        thread_id: str | None = None,
        persona_id: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryRecord]:
        contexts = await self.retrieve(
            query,
            top_k=top_k,
            thread_id=thread_id,
            persona_id=persona_id,
            filters=filters,
        )
        return [
            MemoryRecord(
                id=context.id,
                content=context.content,
                score=context.score,
                scope=context.scope,
                thread_id=context.thread_id,
                persona_id=context.persona_id,
                importance=context.importance,
                metadata={**context.metadata, "memory_source": context.source},
                created_at=context.created_at,
            )
            for context in contexts
        ]


def _graph_score(importance: float, confidence: float) -> float:
    return max(0.0, min(1.0, importance)) * max(0.0, min(1.0, confidence))


def _dedupe(contexts: list[RetrievedContext]) -> list[RetrievedContext]:
    by_key: dict[str, RetrievedContext] = {}
    for context in contexts:
        key = str(
            context.metadata.get("source_fact_id")
            or context.metadata.get("source_event_id")
            or context.content.lower()
        )
        existing = by_key.get(key)
        if existing is None or context.score > existing.score:
            by_key[key] = context
            continue
        if existing.source == "vector" and context.source == "graph":
            by_key[key] = replace(existing, source="graph", score=max(existing.score, context.score))
    return list(by_key.values())
