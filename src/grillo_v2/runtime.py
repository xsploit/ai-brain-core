from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from .context import GrilloContextBuilder
from .models import GrilloEpisode, from_json_dict
from .store import SQLiteGrilloV2Store
from .tools import GRILLO_V2_MEMORY_TOOL_NAMES, apply_memory_tool_calls, legacy_payload_tool_calls


ReflectionCompletion = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass(slots=True)
class ReflectionResult:
    episodes: int = 0
    evidence: int = 0
    facts: int = 0
    opinions: int = 0
    memory_docs: int = 0
    invalidated_facts: int = 0
    tool_calls: int = 0
    ignored_tool_calls: int = 0
    notes: str = ""


@dataclass(slots=True)
class WorkerTickResult:
    scopes: int = 0
    batches: int = 0
    episodes: int = 0
    evidence: int = 0
    facts: int = 0
    opinions: int = 0
    memory_docs: int = 0
    invalidated_facts: int = 0
    tool_calls: int = 0
    ignored_tool_calls: int = 0
    notes: list[str] | None = None

    def add(self, result: ReflectionResult) -> None:
        self.batches += 1
        self.episodes += result.episodes
        self.evidence += result.evidence
        self.facts += result.facts
        self.opinions += result.opinions
        self.memory_docs += result.memory_docs
        self.invalidated_facts += result.invalidated_facts
        self.tool_calls += result.tool_calls
        self.ignored_tool_calls += result.ignored_tool_calls
        if result.notes:
            if self.notes is None:
                self.notes = []
            self.notes.append(result.notes)


class GrilloV2Runtime:
    def __init__(
        self,
        *,
        store: SQLiteGrilloV2Store,
        completion: ReflectionCompletion | None = None,
        persona_id: str = "assistant",
    ):
        self.store = store
        self.completion = completion
        self.persona_id = persona_id
        self.context = GrilloContextBuilder(store)

    def append_episode(self, episode: GrilloEpisode) -> GrilloEpisode:
        return self.store.append_episode(episode)

    def build_context_packet(
        self,
        *,
        scope_key: str,
        actor_id: str | None = None,
        query: str = "",
        channel_id: str | None = None,
    ):
        return self.context.build_packet(
            scope_key=scope_key,
            actor_id=actor_id,
            persona_id=self.persona_id,
            query=query,
            channel_id=channel_id,
        )

    async def reflect_recent(
        self,
        *,
        scope_key: str,
        actor_id: str | None = None,
        channel_id: str | None = None,
        limit: int = 12,
    ) -> ReflectionResult:
        if self.completion is None:
            return ReflectionResult(notes="no_completion")
        episodes = self.store.list_recent_episodes(
            scope_key,
            actor_id=actor_id,
            channel_id=channel_id,
            limit=limit,
        )
        if not episodes:
            return ReflectionResult(notes="no_episodes")
        packet = self.build_context_packet(
            scope_key=scope_key,
            actor_id=actor_id,
            channel_id=channel_id,
            query="\n".join(episode.content for episode in episodes[-3:]),
        )
        payload = await self.completion(
            {
                "schema": GRILLO_V2_REFLECTION_SCHEMA,
                "scope_key": scope_key,
                "actor_id": actor_id,
                "persona_id": self.persona_id,
                "available_tools": GRILLO_V2_MEMORY_TOOL_NAMES,
                "context_packet": packet.as_prompt_text(),
                "episodes": [_episode_payload(episode) for episode in episodes],
            }
        )
        result = self.apply_reflection(scope_key=scope_key, payload=payload)
        result.episodes = len(episodes)
        return result

    async def reflect_unprocessed(
        self,
        *,
        scope_key: str,
        cursor_key: str | None = None,
        limit: int = 12,
    ) -> ReflectionResult:
        if self.completion is None:
            return ReflectionResult(notes="no_completion")
        cursor_key = cursor_key or self.worker_cursor_key(scope_key)
        cursor = _cursor_value(self.store.get_cursor(cursor_key))
        episodes = self.store.list_episodes_after(
            scope_key,
            after_occurred_at=cursor.get("occurred_at"),
            after_episode_id=cursor.get("episode_id"),
            limit=limit,
        )
        if not episodes:
            return ReflectionResult(notes="no_unprocessed_episodes")
        packet = self.build_context_packet(
            scope_key=scope_key,
            actor_id=episodes[-1].actor_id,
            channel_id=episodes[-1].channel_id,
            query="\n".join(episode.content for episode in episodes[-3:]),
        )
        payload = await self.completion(
            {
                "schema": GRILLO_V2_REFLECTION_SCHEMA,
                "mode": "worker_tick",
                "scope_key": scope_key,
                "cursor_key": cursor_key,
                "cursor": cursor,
                "persona_id": self.persona_id,
                "available_tools": GRILLO_V2_MEMORY_TOOL_NAMES,
                "context_packet": packet.as_prompt_text(),
                "episodes": [_episode_payload(episode) for episode in episodes],
            }
        )
        result = self.apply_reflection(scope_key=scope_key, payload=payload)
        result.episodes = len(episodes)
        last = episodes[-1]
        self.store.set_cursor(
            cursor_key,
            scope_key,
            json.dumps(
                {"occurred_at": last.occurred_at, "episode_id": last.episode_id},
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        return result

    async def worker_tick(
        self,
        *,
        scope_key: str | None = None,
        scope_limit: int = 10,
        batch_size: int = 12,
        max_batches: int = 3,
    ) -> WorkerTickResult:
        scopes = [scope_key] if scope_key is not None else self.store.list_episode_scopes(limit=scope_limit)
        result = WorkerTickResult(scopes=len(scopes), notes=[])
        for scope in scopes:
            if result.batches >= max(1, int(max_batches)):
                break
            batch = await self.reflect_unprocessed(
                scope_key=scope,
                limit=max(1, int(batch_size)),
            )
            if batch.notes == "no_unprocessed_episodes":
                continue
            result.add(batch)
        if not result.notes:
            result.notes = ["no_unprocessed_episodes" if scopes else "no_scopes"]
        return result

    def worker_cursor_key(self, scope_key: str) -> str:
        return f"grillo_v2_worker:{self.persona_id}:{scope_key}"

    def apply_reflection(self, *, scope_key: str, payload: dict[str, Any]) -> ReflectionResult:
        result = ReflectionResult(notes=str(payload.get("notes") or ""))
        tool_result = apply_memory_tool_calls(
            store=self.store,
            scope_key=scope_key,
            persona_id=self.persona_id,
            tool_calls=[*legacy_payload_tool_calls(payload), *_list(payload.get("tool_calls"))],
        )
        result.tool_calls = tool_result.tool_calls
        result.evidence = tool_result.evidence
        result.facts = tool_result.facts
        result.opinions = tool_result.opinions
        result.memory_docs = tool_result.memory_docs
        result.invalidated_facts = tool_result.invalidated_facts
        result.ignored_tool_calls = tool_result.ignored
        if tool_result.notes:
            notes = [result.notes] if result.notes else []
            notes.extend(tool_result.notes)
            result.notes = "; ".join(notes)
        return result


GRILLO_V2_REFLECTION_SCHEMA: dict[str, Any] = {
    "name": "grillo_v2_reflection",
    "schema": {
        "type": "object",
        "properties": {
            "notes": {"type": "string"},
            "evidence": {"type": "array"},
            "facts": {"type": "array"},
            "opinion_edges": {"type": "array"},
            "memory_documents": {"type": "array"},
            "invalidate_facts": {"type": "array"},
            "tool_calls": {"type": "array"},
        },
        "required": [
            "notes",
            "evidence",
            "facts",
            "opinion_edges",
            "memory_documents",
            "invalidate_facts",
            "tool_calls",
        ],
    },
}


def _episode_payload(episode: GrilloEpisode) -> dict[str, Any]:
    return {
        "episode_id": episode.episode_id,
        "scope_key": episode.scope_key,
        "source": episode.source,
        "actor_id": episode.actor_id,
        "participant_ids": episode.participant_ids,
        "channel_id": episode.channel_id,
        "content": episode.content,
        "occurred_at": episode.occurred_at,
        "metadata": episode.metadata,
    }


def _cursor_value(value: str | None) -> dict[str, str | None]:
    data = from_json_dict(value)
    occurred_at = data.get("occurred_at")
    episode_id = data.get("episode_id")
    return {
        "occurred_at": str(occurred_at) if occurred_at else None,
        "episode_id": str(episode_id) if episode_id else None,
    }


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
