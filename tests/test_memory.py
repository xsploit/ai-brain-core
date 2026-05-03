import pytest

from aibrain.embeddings import HashEmbeddingProvider
from aibrain import memory as memory_module
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


@pytest.mark.asyncio
async def test_memory_fallback_search_runs_off_event_loop(tmp_path, monkeypatch):
    calls = []

    async def fake_to_thread(func, /, *args, **kwargs):
        calls.append(func.__name__)
        return func(*args, **kwargs)

    monkeypatch.setattr(memory_module.asyncio, "to_thread", fake_to_thread)
    store = SQLiteMemoryStore(
        tmp_path / "brain.sqlite3",
        embedding_provider=HashEmbeddingProvider(dimensions=64),
        dimensions=64,
    )
    await store.remember("Fallback memory", scope="global")
    monkeypatch.setattr(store, "_search_rows_with_sqlite_vec", lambda *args, **kwargs: None)

    results = await store.search("Fallback", top_k=1)

    assert results
    assert calls == ["_search_fallback_sync"]
