import math
from types import SimpleNamespace

import pytest

from aibrain.embeddings import (
    HashEmbeddingProvider,
    OpenAIEmbeddingProvider,
    default_embedding_provider,
)


class FakeEmbeddings:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(data=[SimpleNamespace(embedding=(0.1, 0.2, 0.3))])


class FakeClient:
    def __init__(self):
        self.embeddings = FakeEmbeddings()


@pytest.mark.asyncio
async def test_hash_embedding_is_deterministic_and_normalized():
    provider = HashEmbeddingProvider(dimensions=32)

    first = await provider.embed("Synth wave synth")
    second = await provider.embed("synth wave synth")

    assert first == second
    assert len(first) == 32
    assert math.isclose(
        math.sqrt(sum(value * value for value in first)),
        1.0,
        rel_tol=1e-6,
    )


@pytest.mark.asyncio
async def test_hash_embedding_empty_text_returns_zero_vector():
    provider = HashEmbeddingProvider(dimensions=8)

    embedding = await provider.embed("")

    assert embedding == [0.0] * 8


@pytest.mark.asyncio
async def test_openai_embedding_provider_passes_model_input_and_dimensions():
    client = FakeClient()
    provider = OpenAIEmbeddingProvider(
        client,
        model="text-embedding-test",
        dimensions=3,
    )

    embedding = await provider.embed("remember this")

    assert embedding == [0.1, 0.2, 0.3]
    assert client.embeddings.calls == [
        {
            "model": "text-embedding-test",
            "input": "remember this",
            "dimensions": 3,
        }
    ]


@pytest.mark.asyncio
async def test_openai_embedding_provider_accepts_client_factory():
    client = FakeClient()
    provider = OpenAIEmbeddingProvider(lambda: client, model="text-embedding-test")

    await provider.embed("factory client")

    assert client.embeddings.calls == [
        {
            "model": "text-embedding-test",
            "input": "factory client",
        }
    ]


@pytest.mark.asyncio
async def test_openai_embedding_provider_falls_back_without_embeddings_client():
    provider = OpenAIEmbeddingProvider(SimpleNamespace(), dimensions=4)

    embedding = await provider.embed("fallback")

    assert len(embedding) == 4
    assert any(value != 0 for value in embedding)


def test_default_embedding_provider_uses_hash_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    provider = default_embedding_provider(lambda: FakeClient(), "text-embedding-test", 12)

    assert isinstance(provider, HashEmbeddingProvider)


def test_default_embedding_provider_uses_openai_with_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    provider = default_embedding_provider(lambda: FakeClient(), "text-embedding-test", 12)

    assert isinstance(provider, OpenAIEmbeddingProvider)
