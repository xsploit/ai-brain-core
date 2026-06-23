from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from grillo_v2 import GrilloEpisode, GrilloV2Runtime, GrilloV2Worker, GrilloV2WorkerConfig, SQLiteGrilloV2Store


async def demo_completion(request: dict[str, Any]) -> dict[str, Any]:
    episode = request["episodes"][-1]
    return {
        "notes": "demo memory write",
        "evidence": [],
        "facts": [],
        "opinion_edges": [],
        "memory_documents": [],
        "invalidate_facts": [],
        "tool_calls": [
            {
                "name": "record_evidence",
                "arguments": {
                    "evidence_id": "evidence:demo-preference",
                    "episode_id": episode["episode_id"],
                    "quote": "prefer temporal memory",
                    "confidence": 0.9,
                },
            },
            {
                "name": "upsert_fact",
                "arguments": {
                    "fact_id": "fact:demo-temporal-memory",
                    "subject_id": "user:demo",
                    "predicate": "prefers_memory_architecture",
                    "object": "temporal memory",
                    "claim": "The demo user prefers temporal memory.",
                    "evidence_ids": ["evidence:demo-preference"],
                    "confidence": 0.88,
                },
            },
        ],
    }


async def main() -> None:
    with TemporaryDirectory() as temp_dir:
        store = SQLiteGrilloV2Store(Path(temp_dir) / "grillo_v2.sqlite3")
        try:
            runtime = GrilloV2Runtime(store=store, completion=demo_completion, persona_id="assistant:demo")
            scope_key = "demo:scope:persona:v2"
            runtime.append_episode(
                GrilloEpisode.create(
                    scope_key=scope_key,
                    source="demo",
                    actor_id="user:demo",
                    participant_ids=["user:demo", "assistant:demo"],
                    content="I prefer temporal memory over flat logs.",
                )
            )
            worker = GrilloV2Worker(
                runtime=runtime,
                config=GrilloV2WorkerConfig(scope_key=scope_key, batch_size=4, max_batches=1),
            )
            result = await worker.run_once()
            packet = runtime.build_context_packet(
                scope_key=scope_key,
                actor_id="user:demo",
                query="temporal memory",
            )
            print(f"episodes={result.episodes} facts={result.facts} tool_calls={result.tool_calls}")
            print(packet.as_prompt_text())
        finally:
            store.close()


if __name__ == "__main__":
    asyncio.run(main())
