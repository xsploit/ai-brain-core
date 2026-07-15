from __future__ import annotations

import time
from dataclasses import dataclass

from .retriever import HybridRetriever


@dataclass(slots=True)
class RetrievalBenchmarkResult:
    rounds: int
    total_seconds: float
    average_ms: float
    hits: int


async def benchmark_retriever(
    retriever: HybridRetriever,
    query: str,
    *,
    rounds: int = 10,
    top_k: int = 5,
) -> RetrievalBenchmarkResult:
    start = time.perf_counter()
    hit_count = 0
    for _ in range(max(rounds, 1)):
        hits = await retriever.retrieve(query, top_k=top_k)
        hit_count += len(hits)
    total = time.perf_counter() - start
    actual_rounds = max(rounds, 1)
    return RetrievalBenchmarkResult(
        rounds=actual_rounds,
        total_seconds=total,
        average_ms=(total / actual_rounds) * 1000,
        hits=hit_count,
    )
