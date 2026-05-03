import pytest

from aibrain.embeddings import HashEmbeddingProvider
from aibrain.memory import SQLiteMemoryStore
from aibrain.policy import MemoryPolicy
from aibrain.types import ThreadState


class CountingEmbeddingProvider(HashEmbeddingProvider):
    def __init__(self, dimensions=64):
        super().__init__(dimensions=dimensions)
        self.calls = 0

    async def embed(self, text: str) -> list[float]:
        self.calls += 1
        return await super().embed(text)


@pytest.mark.asyncio
async def test_memory_insert_search_forget(tmp_path):
    store = SQLiteMemoryStore(
        tmp_path / "brain.sqlite3",
        embedding_provider=HashEmbeddingProvider(dimensions=64),
        dimensions=64,
    )
    record = await store.remember(
        "The user likes synthwave music.",
        thread_id="thread-1",
        persona_id="riko",
    )

    results = await store.search(
        "What music does the user like?",
        thread_id="thread-1",
        persona_id="riko",
        top_k=3,
    )

    assert results
    assert results[0].id == record.id
    assert store.forget(record.id) is True
    assert store.forget(record.id) is False


@pytest.mark.asyncio
async def test_memory_policy_reuses_one_query_embedding_across_scopes(tmp_path):
    provider = CountingEmbeddingProvider(dimensions=64)
    store = SQLiteMemoryStore(
        tmp_path / "brain.sqlite3",
        embedding_provider=provider,
        dimensions=64,
    )
    await store.remember("Thread note", scope="thread", thread_id="thread-1", persona_id="riko")
    await store.remember("Persona note", scope="persona", persona_id="riko")
    await store.remember("Global note", scope="global")
    calls_after_writes = provider.calls

    records = await MemoryPolicy(top_k=3).retrieve(
        store,
        "Find notes",
        thread=ThreadState(thread_id="thread-1", persona_id="riko"),
        persona_id="riko",
    )

    assert len(records) == 3
    assert provider.calls == calls_after_writes + 1
