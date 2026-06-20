from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

RawEventType = Literal[
    "message",
    "tool_call",
    "tool_result",
    "response",
    "summary",
    "memory",
]


@dataclass(slots=True)
class RawEvent:
    id: str
    event_type: str
    content: str
    thread_id: str | None = None
    persona_id: str | None = None
    session_id: str | None = None
    actor: str = "user"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None


@dataclass(slots=True)
class TemporalFact:
    id: str
    subject: str
    predicate: str
    object: str
    valid_from: str
    valid_until: str | None = None
    confidence: float = 0.7
    importance: float = 0.5
    source_event_id: str | None = None
    source_session_id: str | None = None
    supersedes_memory_id: str | None = None
    last_accessed_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None

    @property
    def content(self) -> str:
        return f"{self.subject} {self.predicate} {self.object}"


@dataclass(slots=True)
class RecallItem:
    id: str
    text: str
    scope: str = "global"
    thread_id: str | None = None
    persona_id: str | None = None
    source_event_id: str | None = None
    source_fact_id: str | None = None
    importance: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None


@dataclass(slots=True)
class RecallHit:
    id: str
    text: str
    score: float
    scope: str = "global"
    thread_id: str | None = None
    persona_id: str | None = None
    source_event_id: str | None = None
    source_fact_id: str | None = None
    importance: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None


@dataclass(slots=True)
class GraphQuery:
    text: str
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    thread_id: str | None = None
    persona_id: str | None = None
    top_k: int = 5
    include_expired: bool = False


@dataclass(slots=True)
class RetrievedContext:
    id: str
    content: str
    score: float
    source: Literal["graph", "vector", "fts", "raw"]
    source_id: str
    scope: str = "global"
    thread_id: str | None = None
    persona_id: str | None = None
    importance: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None


class RawEventStore(Protocol):
    async def append(self, event: RawEvent) -> RawEvent:
        ...

    async def list_thread_events(self, thread_id: str, limit: int = 100) -> list[RawEvent]:
        ...

    def close(self) -> None:
        ...


class GraphMemoryStore(Protocol):
    async def upsert_fact(self, fact: TemporalFact) -> TemporalFact:
        ...

    async def search_facts(self, query: GraphQuery) -> list[TemporalFact]:
        ...

    async def invalidate_fact(
        self,
        fact_id: str,
        *,
        valid_until: str,
        reason: str,
    ) -> None:
        ...

    def close(self) -> None:
        ...


class VectorRecallStore(Protocol):
    async def add(self, item: RecallItem) -> str:
        ...

    async def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RecallHit]:
        ...

    def close(self) -> None:
        ...


class MemoryExtractor(Protocol):
    async def extract(self, event: RawEvent) -> list[TemporalFact]:
        ...


class MemoryReranker(Protocol):
    async def rerank(
        self,
        query: str,
        contexts: list[RetrievedContext],
    ) -> list[RetrievedContext]:
        ...
