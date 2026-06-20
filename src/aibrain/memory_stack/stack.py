from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from ..embeddings import EmbeddingProvider
from ..types import MemoryRecord, ThreadState
from .contracts import (
    GraphMemoryStore,
    RawEvent,
    RawEventStore,
    VectorRecallStore,
)
from .grillo import GrilloRuntime, SQLiteGrilloStore
from .ladybug_store import LadybugGraphMemoryStore
from .raw_log import SQLiteRawEventStore
from .retriever import HybridRetriever
from .temporal import SQLiteTemporalGraphStore
from .vectors import SQLiteVectorRecallStore, TurboVecRecallStore
from .worker import GRILLOMemoryWorker


class HybridMemoryStack:
    def __init__(
        self,
        *,
        raw_log: RawEventStore,
        graph_store: GraphMemoryStore,
        vector_store: VectorRecallStore | None = None,
        worker: GRILLOMemoryWorker | None = None,
        grillo: GrilloRuntime | None = None,
        retriever: HybridRetriever | None = None,
    ):
        self.raw_log = raw_log
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.worker = worker or GRILLOMemoryWorker(
            graph_store=graph_store,
            vector_store=vector_store,
        )
        self.grillo = grillo
        self.retriever = retriever or HybridRetriever(
            graph_store=graph_store,
            vector_store=vector_store,
        )

    @classmethod
    def sqlite_default(
        cls,
        path: str | Path,
        *,
        embedding_provider: EmbeddingProvider,
    ) -> "HybridMemoryStack":
        return cls.sqlite_paths(
            raw_log_path=path,
            graph_path=path,
            vector_path=path,
            embedding_provider=embedding_provider,
        )

    @classmethod
    def sqlite_paths(
        cls,
        *,
        raw_log_path: str | Path,
        graph_path: str | Path,
        vector_path: str | Path,
        embedding_provider: EmbeddingProvider,
        embedding_dimensions: int = 256,
        graph_backend: str = "auto",
        vector_backend: str = "auto",
    ) -> "HybridMemoryStack":
        raw_log = SQLiteRawEventStore(raw_log_path)
        graph_store = _create_graph_store(graph_path, backend=graph_backend)
        vector_store = _create_vector_store(
            vector_path,
            embedding_provider=embedding_provider,
            dimensions=embedding_dimensions,
            backend=vector_backend,
        )
        grillo_store = SQLiteGrilloStore(raw_log_path)
        grillo = GrilloRuntime(store=grillo_store, vector_store=vector_store)
        return cls(raw_log=raw_log, graph_store=graph_store, vector_store=vector_store, grillo=grillo)

    async def append_event(
        self,
        *,
        event_type: str,
        content: str,
        actor: str,
        thread: ThreadState | None = None,
        persona_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        extract: bool = False,
    ) -> RawEvent:
        event = await self.raw_log.append(
            RawEvent(
                id=str(uuid4()),
                event_type=event_type,
                content=content,
                actor=actor,
                thread_id=thread.thread_id if thread else None,
                persona_id=persona_id or (thread.persona_id if thread else None),
                session_id=session_id,
                metadata=metadata or {},
            )
        )
        if extract:
            await self.worker.process_event(event)
        return event

    async def retrieve_records(
        self,
        query: str,
        *,
        top_k: int = 5,
        thread: ThreadState | None = None,
        persona_id: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryRecord]:
        return await self.retriever.retrieve_records(
            query,
            top_k=top_k,
            thread_id=thread.thread_id if thread else None,
            persona_id=persona_id or (thread.persona_id if thread else None),
            filters=filters,
        )

    def close(self) -> None:
        self.raw_log.close()
        self.graph_store.close()
        if self.vector_store is not None:
            self.vector_store.close()
        if self.grillo is not None:
            self.grillo.store.close()


def _create_graph_store(path: str | Path, *, backend: str) -> GraphMemoryStore:
    normalized = backend.lower().strip()
    if normalized in {"auto", "ladybug"}:
        try:
            return LadybugGraphMemoryStore(_derived_path(path, ".ladybug"))
        except Exception:
            if normalized == "ladybug":
                raise
    return SQLiteTemporalGraphStore(path)


def _create_vector_store(
    path: str | Path,
    *,
    embedding_provider: EmbeddingProvider,
    dimensions: int,
    backend: str,
) -> VectorRecallStore:
    normalized = backend.lower().strip()
    if normalized in {"auto", "turbovec"}:
        try:
            return TurboVecRecallStore(
                _derived_path(path, ".turbovec"),
                _derived_path(path, ".turbovec.sqlite3"),
                embedding_provider=embedding_provider,
                dimensions=dimensions,
            )
        except Exception:
            if normalized == "turbovec":
                raise
    return SQLiteVectorRecallStore(path, embedding_provider=embedding_provider)


def _derived_path(path: str | Path, suffix: str) -> Path:
    source = Path(path)
    if source.suffix:
        return source.with_suffix(suffix)
    return source / f"memory{suffix}"
