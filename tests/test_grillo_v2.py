from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from aibrain.brain_v2 import BrainV2, BrainV2Config
from grillo_v2 import (
    Evidence,
    EvidenceGap,
    GrilloEpisode,
    GrilloV2Runtime,
    OpinionEdge,
    SQLiteGrilloV2Store,
    TemporalFact,
    VercelAIGatewayJSONClient,
)
from grillo_v2.gateway import VERCEL_AI_GATEWAY_BASE_URL


def test_grillo_v2_context_packet_uses_temporal_facts_and_opinion_edges(tmp_path):
    store = SQLiteGrilloV2Store(tmp_path / "grillo-v2.sqlite3")
    episode = store.append_episode(
        GrilloEpisode.create(
            scope_key="discord:guild:1",
            source="discord",
            actor_id="discord_user:subby",
            participant_ids=["discord_user:subby", "persona:neuro"],
            channel_id="bot-chat",
            content="Subby said to call him Subby, not LO.",
            metadata={"summary": "Subby corrected the preferred name."},
        )
    )
    evidence = store.append_evidence(
        Evidence.create(
            scope_key="discord:guild:1",
            episode_id=episode.episode_id,
            quote="call him Subby, not LO",
            extractor="test",
            confidence=0.95,
        )
    )
    store.upsert_fact(
        TemporalFact.create(
            scope_key="discord:guild:1",
            subject_id="discord_user:subby",
            predicate="preferred_name",
            object_value="Subby",
            claim="Subby prefers being called Subby.",
            evidence_ids=[evidence.evidence_id],
            confidence=0.92,
            missing_evidence=[
                EvidenceGap(
                    question="Is LO still acceptable?",
                    why="Recent correction says not to use LO.",
                    needed="A newer explicit confirmation.",
                )
            ],
        )
    )
    store.upsert_opinion_edge(
        OpinionEdge.create(
            scope_key="discord:guild:1",
            source_id="persona:neuro",
            target_id="discord_user:subby",
            relation="familiarity",
            score=0.84,
            rationale="Subby has repeated direct debugging conversations with Neuro.",
            evidence_ids=[evidence.evidence_id],
        )
    )
    runtime = GrilloV2Runtime(store=store, persona_id="persona:neuro")

    packet = runtime.build_context_packet(
        scope_key="discord:guild:1",
        actor_id="discord_user:subby",
        query="who is subby",
        channel_id="bot-chat",
    )
    prompt = packet.as_prompt_text()

    assert '<grillo_context version="2.0" scope="discord:guild:1">' in prompt
    assert "<active_facts>" in prompt
    assert "Subby prefers being called Subby." in prompt
    assert "<relationship_state>" in prompt
    assert "familiarity" in prompt
    assert "<evidence_gaps>" in prompt
    assert "Is LO still acceptable?" in prompt
    assert "<recent_episode_summary>" in prompt
    assert "Subby corrected the preferred name." in prompt


@pytest.mark.asyncio
async def test_grillo_v2_reflection_worker_writes_evidence_facts_and_opinions(tmp_path):
    store = SQLiteGrilloV2Store(tmp_path / "grillo-v2.sqlite3")
    episode = store.append_episode(
        GrilloEpisode.create(
            scope_key="discord:guild:1",
            source="discord",
            actor_id="discord_user:subby",
            content="Subby said memory should be temporal and evidence-backed.",
        )
    )

    async def completion(request):
        assert "context_packet" in request
        assert request["episodes"][0]["episode_id"] == episode.episode_id
        return {
            "notes": "stored temporal design preference",
            "evidence": [
                {
                    "evidence_id": "evidence:manual",
                    "episode_id": episode.episode_id,
                    "quote": "memory should be temporal and evidence-backed",
                    "confidence": 0.9,
                }
            ],
            "facts": [
                {
                    "fact_id": "fact:temporal-memory",
                    "subject_id": "discord_user:subby",
                    "predicate": "prefers_memory_architecture",
                    "object": "temporal evidence-backed memory",
                    "claim": "Subby wants GRILLO memory to be temporal and evidence-backed.",
                    "evidence_ids": ["evidence:manual"],
                    "confidence": 0.88,
                }
            ],
            "opinion_edges": [
                {
                    "edge_id": "opinion:trust-subby",
                    "source_id": "persona:neuro",
                    "target_id": "discord_user:subby",
                    "relation": "trust",
                    "score": 0.7,
                    "rationale": "The user is actively steering the memory architecture.",
                    "evidence_ids": ["evidence:manual"],
                }
            ],
            "invalidate_facts": [],
        }

    runtime = GrilloV2Runtime(store=store, completion=completion, persona_id="persona:neuro")

    result = await runtime.reflect_recent(scope_key="discord:guild:1", actor_id="discord_user:subby")
    packet = runtime.build_context_packet(
        scope_key="discord:guild:1",
        actor_id="discord_user:subby",
        query="temporal memory",
    )

    assert result.episodes == 1
    assert result.evidence == 1
    assert result.facts == 1
    assert result.opinions == 1
    assert packet.active_facts[0]["id"] == "fact:temporal-memory"
    assert packet.relationship_state[0]["id"] == "opinion:trust-subby"


def test_brain_v2_uses_vercel_gateway_and_grillo_v2_store(tmp_path):
    calls = []

    class FakeResponses:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(output_text=json.dumps({"notes": "ok", "evidence": [], "facts": [], "opinion_edges": [], "invalidate_facts": []}))

    fake_client = SimpleNamespace(responses=FakeResponses())
    json_client = VercelAIGatewayJSONClient(client=fake_client, model="deepseek/test")
    brain = BrainV2(
        BrainV2Config(database_path=tmp_path / "brain-v2.sqlite3", model="deepseek/test"),
        json_client=json_client,
    )

    brain.append_episode(
        GrilloEpisode.create(
            scope_key="discord:guild:1",
            source="discord",
            actor_id="discord_user:subby",
            content="hello",
        )
    )
    packet = brain.build_context_packet(
        scope_key="discord:guild:1",
        actor_id="discord_user:subby",
        query="hello",
    )

    assert brain.base_url == VERCEL_AI_GATEWAY_BASE_URL
    assert packet.scope_key == "discord:guild:1"
    assert isinstance(brain.grillo, GrilloV2Runtime)
