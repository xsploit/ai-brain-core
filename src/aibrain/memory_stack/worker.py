from __future__ import annotations

import re
from dataclasses import replace
from datetime import datetime, timezone
from uuid import uuid5, NAMESPACE_URL

from .contracts import (
    GraphMemoryStore,
    MemoryExtractor,
    RawEvent,
    RecallItem,
    TemporalFact,
    VectorRecallStore,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RuleBasedMemoryExtractor:
    """Deterministic default extractor used until a GRILLO/LLM extractor is wired."""

    def __init__(self, subject: str = "user"):
        self.subject = subject

    async def extract(self, event: RawEvent) -> list[TemporalFact]:
        text = event.content.strip()
        if not text:
            return []
        explicit = _extract_explicit_memory(text, self.subject)
        if explicit is not None:
            fact = explicit
        else:
            fact = TemporalFact(
                id=_fact_id(self.subject, "said", text, event.id),
                subject=self.subject,
                predicate="said",
                object=_compact_text(text),
                valid_from=event.created_at or utc_now(),
                confidence=0.35,
                importance=float(event.metadata.get("importance", 0.3)),
            )
        fact.source_event_id = event.id
        fact.source_session_id = event.session_id
        fact.metadata = {**fact.metadata, "extractor": "rule_based"}
        return [fact]


class GRILLOMemoryWorker:
    """Memory extraction pipeline.

    GRILLO is represented here as the pluggable extractor brain. Projects can
    pass an LLM-backed extractor later without changing stores or retrieval.
    """

    def __init__(
        self,
        *,
        graph_store: GraphMemoryStore,
        vector_store: VectorRecallStore | None = None,
        extractor: MemoryExtractor | None = None,
    ):
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.extractor = extractor or RuleBasedMemoryExtractor()

    async def process_event(self, event: RawEvent) -> list[TemporalFact]:
        facts = await self.extractor.extract(event)
        stored: list[TemporalFact] = []
        for fact in facts:
            normalized = _normalize_fact(fact, event)
            saved = await self.graph_store.upsert_fact(normalized)
            stored.append(saved)
            if self.vector_store is not None:
                await self.vector_store.add(_fact_to_recall(saved, event))
        return stored


def _normalize_fact(fact: TemporalFact, event: RawEvent) -> TemporalFact:
    subject = fact.subject.strip() or "user"
    predicate = fact.predicate.strip().lower().replace(" ", "_") or "said"
    obj = fact.object.strip()
    fact_id = fact.id or _fact_id(subject, predicate, obj, event.id)
    return replace(
        fact,
        id=fact_id,
        subject=subject,
        predicate=predicate,
        object=obj,
        valid_from=fact.valid_from or event.created_at or utc_now(),
        source_event_id=fact.source_event_id or event.id,
        source_session_id=fact.source_session_id or event.session_id,
    )


def _fact_to_recall(fact: TemporalFact, event: RawEvent) -> RecallItem:
    return RecallItem(
        id=f"fact:{fact.id}",
        text=fact.content,
        scope="thread" if event.thread_id else "global",
        thread_id=event.thread_id,
        persona_id=event.persona_id,
        source_event_id=event.id,
        source_fact_id=fact.id,
        importance=fact.importance,
        metadata={
            "source": "grillo_memory_worker",
            "confidence": fact.confidence,
            **fact.metadata,
        },
        created_at=fact.created_at or event.created_at,
    )


def _extract_explicit_memory(text: str, subject: str) -> TemporalFact | None:
    patterns = [
        (r"\bmy\s+([a-zA-Z0-9 _-]{2,40})\s+is\s+(.+)", "has"),
        (r"\bi\s+like\s+(.+)", "likes"),
        (r"\bi\s+love\s+(.+)", "likes"),
        (r"\bi\s+am\s+working\s+on\s+(.+)", "working_on"),
        (r"\bi(?:'m| am)\s+using\s+(.+)", "uses"),
        (r"\bremember\s+that\s+(.+)", "remembered"),
    ]
    lower = text.lower().strip()
    for pattern, predicate in patterns:
        match = re.search(pattern, lower)
        if not match:
            continue
        if predicate == "has" and len(match.groups()) == 2:
            obj = f"{match.group(1).strip()} is {match.group(2).strip()}"
        else:
            obj = match.group(1).strip()
        obj = _compact_text(obj)
        return TemporalFact(
            id=_fact_id(subject, predicate, obj, text),
            subject=subject,
            predicate=predicate,
            object=obj,
            valid_from=utc_now(),
            confidence=0.65,
            importance=0.7,
        )
    return None


def _compact_text(text: str, limit: int = 320) -> str:
    compacted = re.sub(r"\s+", " ", text).strip()
    return compacted if len(compacted) <= limit else compacted[: limit - 1].rstrip() + "..."


def _fact_id(subject: str, predicate: str, obj: str, salt: str) -> str:
    key = f"{subject}\0{predicate}\0{obj}\0{salt}"
    return str(uuid5(NAMESPACE_URL, key))
