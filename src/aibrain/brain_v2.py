from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from grillo_v2 import GrilloContextPacket, GrilloEntity, GrilloEpisode, GrilloV2Runtime, SQLiteGrilloV2Store
from grillo_v2.backfill import GrilloV2BackfillResult, backfill_discord_identity, backfill_grillo_v1
from grillo_v2.gateway import VERCEL_AI_GATEWAY_BASE_URL, VercelAIGatewayJSONClient
from grillo_v2.runtime import GRILLO_V2_REFLECTION_SCHEMA


@dataclass(slots=True)
class BrainV2Config:
    database_path: Path = Path("brain_v2.sqlite3")
    model: str = "deepseek/deepseek-v4-flash"
    provider: str = "vercel"
    persona_id: str = "neuro-sama"
    persona_name: str = "Neuro-sama"


class BrainV2:
    """Framework 2.0 host adapter for standalone GRILLO v2.

    GRILLO v2 owns memory substrate and context packets. BrainV2 owns the model
    boundary and response assembly.
    """

    def __init__(
        self,
        config: BrainV2Config | None = None,
        *,
        store: SQLiteGrilloV2Store | None = None,
        json_client: VercelAIGatewayJSONClient | None = None,
    ):
        self.config = config or BrainV2Config()
        self.store = store or SQLiteGrilloV2Store(self.config.database_path)
        self.json_client = json_client or VercelAIGatewayJSONClient(
            model=self.config.model,
            api_key=os.getenv("AI_GATEWAY_API_KEY"),
            base_url=self.base_url,
        )
        self.grillo = GrilloV2Runtime(
            store=self.store,
            completion=self._complete_reflection,
            persona_id=self.config.persona_id,
        )

    @property
    def base_url(self) -> str:
        if self.config.provider != "vercel":
            raise ValueError("BrainV2 currently supports provider='vercel' only.")
        return VERCEL_AI_GATEWAY_BASE_URL

    def append_episode(self, episode: GrilloEpisode) -> GrilloEpisode:
        return self.grillo.append_episode(episode)

    def upsert_actor_entity(
        self,
        *,
        actor_id: str,
        name: str | None = None,
        aliases: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> GrilloEntity:
        return self.store.upsert_entity(
            GrilloEntity(
                entity_id=actor_id,
                entity_type="person",
                name=name or actor_id,
                aliases=aliases or [],
                metadata=metadata or {},
            )
        )

    async def respond(
        self,
        *,
        scope_key: str,
        actor_id: str,
        user_text: str,
        source: str = "brain_v2",
        channel_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        self.upsert_actor_entity(
            actor_id=actor_id,
            name=_actor_name_from_metadata(metadata or {}) or actor_id,
            aliases=_actor_aliases_from_metadata(metadata or {}),
            metadata=metadata or {},
        )
        user_episode = GrilloEpisode.create(
            scope_key=scope_key,
            source=source,
            actor_id=actor_id,
            participant_ids=[actor_id, self.config.persona_id],
            channel_id=channel_id,
            content=user_text,
            metadata=metadata or {},
        )
        self.append_episode(user_episode)
        packet = self.build_context_packet(
            scope_key=scope_key,
            actor_id=actor_id,
            query=user_text,
            channel_id=channel_id,
        )
        response_text = await self.json_client.complete_text(
            instructions=_response_instructions(self.config.persona_name),
            prompt=_response_prompt(packet=packet, user_text=user_text),
            store=False,
        )
        if response_text.strip():
            self.append_episode(
                GrilloEpisode.create(
                    scope_key=scope_key,
                    source=f"{source}:assistant",
                    actor_id=self.config.persona_id,
                    participant_ids=[actor_id, self.config.persona_id],
                    channel_id=channel_id,
                    content=response_text.strip(),
                    metadata={"reply_to_episode_id": user_episode.episode_id},
                )
            )
        return response_text.strip()

    def build_context_packet(
        self,
        *,
        scope_key: str,
        actor_id: str | None = None,
        query: str = "",
        channel_id: str | None = None,
    ) -> GrilloContextPacket:
        return self.grillo.build_context_packet(
            scope_key=scope_key,
            actor_id=actor_id,
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
    ):
        return await self.grillo.reflect_recent(
            scope_key=scope_key,
            actor_id=actor_id,
            channel_id=channel_id,
            limit=limit,
        )

    async def worker_tick(
        self,
        *,
        scope_key: str | None = None,
        scope_limit: int = 10,
        batch_size: int = 12,
        max_batches: int = 3,
    ):
        return await self.grillo.worker_tick(
            scope_key=scope_key,
            scope_limit=scope_limit,
            batch_size=batch_size,
            max_batches=max_batches,
        )

    def backfill_from_v1(
        self,
        *,
        grillo_path: str | Path | None = None,
        identity_path: str | Path | None = None,
        limit: int = 10_000,
    ) -> dict[str, GrilloV2BackfillResult]:
        results: dict[str, GrilloV2BackfillResult] = {}
        if grillo_path is not None:
            results["grillo_v1"] = backfill_grillo_v1(
                source_path=grillo_path,
                target=self.store,
                persona_id=self.config.persona_id,
                limit=limit,
            )
        if identity_path is not None:
            results["discord_identity"] = backfill_discord_identity(
                source_path=identity_path,
                target=self.store,
                persona_id=self.config.persona_id,
                limit=limit,
            )
        return results

    def status(self) -> dict[str, Any]:
        return {
            "model": self.config.model,
            "provider": self.config.provider,
            "base_url": self.base_url,
            "persona_id": self.config.persona_id,
            "database_path": str(self.config.database_path),
            "counts": self.store.counts(),
        }

    async def _complete_reflection(self, request: dict[str, Any]) -> dict[str, Any]:
        return await self.json_client.complete_json(
            instructions=GRILLO_V2_REFLECTION_INSTRUCTIONS,
            payload=request,
            schema=GRILLO_V2_REFLECTION_SCHEMA,
        )


GRILLO_V2_REFLECTION_INSTRUCTIONS = "\n".join(
    [
        "You are GRILLO v2, a background memory worker for a persistent AI companion.",
        "You are not writing a chat reply.",
        "Extract only evidence-backed memory from the provided episodes.",
        "Prefer tool_calls for memory writes. Each tool call is {name, arguments}.",
        "Available tools: record_evidence, upsert_fact, upsert_opinion_edge, upsert_memory_document, invalidate_fact.",
        "Write temporal facts with subject, predicate, object, claim, confidence, valid_from, and evidence_ids.",
        "Write opinion_edges only as computed relationship state from the persona to an entity.",
        "Write memory_documents for durable diary/profile/slot/procedural context that should be pinned into future prompts.",
        "Use document_type values like diary, profile, relationship_profile, preference_slot, or procedural_note.",
        "Keep memory_documents compact, first-person when diary-like, and grounded with evidence_ids.",
        "Use missing_evidence when a claim is plausible but not proven.",
        "Invalidate old facts when newer evidence supersedes them.",
        "Return only JSON matching the supplied schema.",
    ]
)


def _response_instructions(persona_name: str) -> str:
    return "\n".join(
        [
            f"You are {persona_name}.",
            "Use the GRILLO v2 context packet as structured memory.",
            "Use memory_blocks for durable diary/profile/slot continuity, while treating active_facts as evidence-backed claims.",
            "Do not treat evidence_gaps as facts.",
            "If memory conflicts with the current message, trust the current message.",
            "Reply naturally and do not expose internal XML tags unless asked for a diagnostic export.",
        ]
    )


def _response_prompt(*, packet: GrilloContextPacket, user_text: str) -> str:
    return "\n\n".join(
        [
            "# GRILLO v2 Context",
            packet.as_prompt_text(),
            "# Current User Message",
            user_text,
        ]
    )


def _actor_name_from_metadata(metadata: dict[str, Any]) -> str | None:
    for key in ("author_display_name", "author_global_name", "author_username", "name"):
        value = metadata.get(key)
        if value:
            return str(value)
    return None


def _actor_aliases_from_metadata(metadata: dict[str, Any]) -> list[str]:
    aliases: list[str] = []
    for key in ("author_display_name", "author_global_name", "author_username", "name"):
        value = metadata.get(key)
        if value:
            aliases.append(str(value))
    author_id = metadata.get("author_id")
    if author_id:
        aliases.extend([str(author_id), f"<@{author_id}>"])
    return _dedupe(aliases)


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result
