from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Any, Callable, Protocol


class EmbeddingProvider(Protocol):
    async def embed(self, text: str) -> list[float]:
        ...


class HashEmbeddingProvider:
    def __init__(self, dimensions: int = 256):
        self.dimensions = dimensions

    async def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[a-zA-Z0-9_']+", text.lower())
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class OpenAIEmbeddingProvider:
    def __init__(
        self,
        client: Any | Callable[[], Any],
        model: str = "text-embedding-3-small",
        dimensions: int | None = None,
    ):
        self.client = client
        self.model = model
        self.dimensions = dimensions
        self.fallback = HashEmbeddingProvider(dimensions or 256)

    async def embed(self, text: str) -> list[float]:
        client = self.client() if callable(self.client) else self.client
        if not hasattr(client, "embeddings"):
            return await self.fallback.embed(text)
        kwargs: dict[str, Any] = {"model": self.model, "input": text}
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions
        result = await client.embeddings.create(**kwargs)
        return list(result.data[0].embedding)


def default_embedding_provider(
    client_factory: Callable[[], Any],
    model: str,
    dimensions: int,
) -> EmbeddingProvider:
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAIEmbeddingProvider(client_factory, model=model, dimensions=dimensions)
    return HashEmbeddingProvider(dimensions=dimensions)
