import pytest

from aibrain.embeddings import HashEmbeddingProvider
from aibrain import memory as memory_module
from aibrain.memory import SQLiteMemoryStore
from aibrain.policy import MemoryPolicy
from aibrain.thread_store import SQLiteThreadStore
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
    assert calls == ["_remember_sync", "_search_sync"]


@pytest.mark.asyncio
async def test_memory_store_reuses_connection_and_closes(tmp_path):
    store = SQLiteMemoryStore(
        tmp_path / "brain.sqlite3",
        embedding_provider=HashEmbeddingProvider(dimensions=64),
        dimensions=64,
    )
    first_conn = store._connect()

    await store.remember("Reusable memory", scope="global")
    await store.search("Reusable", top_k=1)

    assert store._connect() is first_conn
    journal_mode = first_conn.execute("PRAGMA journal_mode").fetchone()[0].lower()
    assert journal_mode == "wal"

    store.close()
    assert store._conn is None


@pytest.mark.asyncio
async def test_memory_insert_without_sqlite_vec_leaves_vector_rowid_null(tmp_path):
    store = SQLiteMemoryStore(
        tmp_path / "brain.sqlite3",
        embedding_provider=HashEmbeddingProvider(dimensions=64),
        dimensions=64,
    )
    store.sqlite_vec_enabled = False

    record = await store.remember("No vector extension yet", scope="global")
    row = store._connect().execute(
        "SELECT vector_rowid FROM brain_memories WHERE id = ?",
        (record.id,),
    ).fetchone()

    assert row["vector_rowid"] is None


@pytest.mark.asyncio
async def test_memory_vec_search_backfills_before_search(tmp_path, monkeypatch):
    store = SQLiteMemoryStore(
        tmp_path / "brain.sqlite3",
        embedding_provider=HashEmbeddingProvider(dimensions=64),
        dimensions=64,
    )
    store.sqlite_vec_enabled = True
    calls = []

    def fake_backfill(conn):
        calls.append("backfill")

    monkeypatch.setattr(store, "_backfill_sqlite_vec", fake_backfill)
    monkeypatch.setattr(store, "_search_rows_with_sqlite_vec", lambda *args, **kwargs: [])

    await store.search("anything", top_k=1)

    assert calls == ["backfill"]


@pytest.mark.asyncio
async def test_memory_vec_score_skips_embedding_json_decode(tmp_path, monkeypatch):
    store = SQLiteMemoryStore(
        tmp_path / "brain.sqlite3",
        embedding_provider=HashEmbeddingProvider(dimensions=64),
        dimensions=64,
    )
    store.sqlite_vec_enabled = True
    monkeypatch.setattr(store, "_backfill_sqlite_vec", lambda conn: None)
    monkeypatch.setattr(
        store,
        "_search_rows_with_sqlite_vec",
        lambda *args, **kwargs: [
            memory_module._DictRow(
                {
                    "id": "mem-1",
                    "scope": "global",
                    "thread_id": None,
                    "persona_id": None,
                    "content": "Vector scored",
                    "metadata_json": "{}",
                    "importance": 0.5,
                    "embedding_json": "not json",
                    "created_at": "now",
                    "vec_score": 0.9,
                }
            )
        ],
    )

    results = await store.search("Vector", top_k=1)

    assert results[0].id == "mem-1"
    assert results[0].score == 0.9


def test_thread_store_reuses_connection_and_closes(tmp_path):
    store = SQLiteThreadStore(tmp_path / "brain.sqlite3")
    first_conn = store._connect()

    state = store.create("persona", thread_id="thread-1")
    assert store.get("thread-1") == state
    assert store._connect() is first_conn
    assert first_conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"

    store.close()
    assert store._conn is None
