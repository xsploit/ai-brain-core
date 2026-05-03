from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .types import MemoryRecord, ThreadState


MemoryScope = Literal["thread", "persona", "global"]


class ThreadPolicy:
    @staticmethod
    def discord_channel(guild_id: int | str, channel_id: int | str) -> str:
        return f"discord:guild:{guild_id}:channel:{channel_id}"

    @staticmethod
    def discord_dm(user_id: int | str) -> str:
        return f"discord:dm:{user_id}"

    @staticmethod
    def discord_thread(guild_id: int | str, thread_id: int | str) -> str:
        return f"discord:guild:{guild_id}:thread:{thread_id}"

    @staticmethod
    def discord_voice(guild_id: int | str, voice_channel_id: int | str) -> str:
        return f"discord:guild:{guild_id}:voice:{voice_channel_id}"

    @staticmethod
    def twitch_channel(channel: str) -> str:
        return f"twitch:channel:{channel.strip().lower()}"

    @staticmethod
    def vrm(session_id: str = "local") -> str:
        return f"vrm:{session_id}"


class MemoryPolicy(BaseModel):
    enabled: bool = True
    top_k: int = 5
    min_score: float = 0.05
    scopes: tuple[MemoryScope, ...] = ("thread", "persona", "global")
    inject: bool = True
    inject_role: Literal["developer", "system", "user"] = "developer"
    save_compaction_summary: bool = True
    save_response_summary: bool = False
    metadata_filter: dict[str, Any] = Field(default_factory=dict)

    async def retrieve(
        self,
        memory_store: Any,
        query: str,
        *,
        thread: ThreadState | None,
        persona_id: str,
    ) -> list[MemoryRecord]:
        if not self.enabled:
            return []
        collected: dict[str, MemoryRecord] = {}
        per_scope = max(self.top_k, 1)
        query_embedding: list[float] | None = None
        if hasattr(memory_store, "embed_query"):
            query_embedding = await memory_store.embed_query(query)
        search_kwargs = {
            "top_k": per_scope * max(1, len(self.scopes)),
            "min_score": self.min_score,
            "scope": list(self.scopes),
            "thread_id": thread.thread_id if thread and "thread" in self.scopes else None,
            "persona_id": persona_id
            if any(scope in {"thread", "persona"} for scope in self.scopes)
            else None,
            "metadata_filter": self.metadata_filter,
        }
        if query_embedding is not None:
            search_kwargs["query_embedding"] = query_embedding
        try:
            results = await memory_store.search(query, **search_kwargs)
        except TypeError:
            search_kwargs.pop("query_embedding", None)
            results = await memory_store.search(query, **search_kwargs)
        for record in results:
            collected[record.id] = record
        records = sorted(collected.values(), key=lambda record: record.score, reverse=True)
        return records[: self.top_k]

    def build_injection_message(self, hits: list[MemoryRecord]) -> dict[str, Any] | None:
        if not self.inject or not hits:
            return None
        lines = ["Relevant long-term memory:"]
        for hit in hits:
            lines.append(
                f"- scope={hit.scope} id={hit.id} score={hit.score:.3f}: {hit.content}"
            )
        return {
            "type": "message",
            "role": self.inject_role,
            "content": [{"type": "input_text", "text": "\n".join(lines)}],
        }
