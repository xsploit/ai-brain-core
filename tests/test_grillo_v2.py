from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace

import pytest

from aibrain.brain_v2 import BrainV2, BrainV2Config
from aibrain.discord_bot_v2 import (
    _discord_metadata,
    _format_backfill_results,
    _format_status,
    _format_worker_loop_status,
    _format_worker_result,
    _recent_message_item,
    _scope_for_message,
    build_brain_v2,
)
from grillo_v2 import (
    Evidence,
    EvidenceGap,
    GrilloEpisode,
    GrilloEntity,
    GrilloMemoryDocument,
    GrilloV2Runtime,
    GrilloV2Worker,
    GrilloV2WorkerConfig,
    OpinionEdge,
    SQLiteGrilloV2Store,
    TemporalFact,
    VercelAIGatewayJSONClient,
    WorkerTickResult,
    backfill_discord_identity,
    backfill_grillo_v1,
)
from grillo_v2.gateway import VERCEL_AI_GATEWAY_BASE_URL


def test_grillo_v2_context_packet_uses_temporal_facts_and_opinion_edges(tmp_path):
    store = SQLiteGrilloV2Store(tmp_path / "grillo-v2.sqlite3")
    store.upsert_entity(
        GrilloEntity(
            entity_id="discord_user:subby",
            entity_type="person",
            name="Npc",
            aliases=["Subby", "SUBSECT"],
            metadata={"discord_id": "123"},
        )
    )
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
    store.upsert_memory_document(
        GrilloMemoryDocument.create(
            scope_key="discord:guild:1",
            document_type="diary",
            subject_id="discord_user:subby",
            title="Name correction",
            body="I should remember Subby corrected me away from LO and sounded annoyed about it.",
            evidence_ids=[evidence.evidence_id],
            importance=0.9,
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
    assert "known_aliases" in prompt
    assert "SUBSECT" in prompt
    assert "Subby prefers being called Subby." in prompt
    assert "<relationship_state>" in prompt
    assert "familiarity" in prompt
    assert "<memory_blocks>" in prompt
    assert "Name correction" in prompt
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
            "memory_documents": [
                {
                    "memory_id": "memory:diary-subby-temporal",
                    "document_type": "diary",
                    "subject_id": "discord_user:subby",
                    "title": "Subby steering GRILLO v2",
                    "body": "I noticed Subby wants memory to be temporal, evidence-backed, and not a flat log.",
                    "evidence_ids": ["evidence:manual"],
                    "importance": 0.85,
                }
            ],
            "invalidate_facts": [],
            "tool_calls": [],
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
    assert result.memory_docs == 1
    assert packet.active_facts[0]["id"] == "fact:temporal-memory"
    assert packet.relationship_state[0]["id"] == "opinion:trust-subby"
    assert packet.memory_blocks[0]["id"] == "memory:diary-subby-temporal"
    assert "not a flat log" in packet.as_prompt_text()


def test_grillo_v2_reflection_accepts_memory_tool_calls(tmp_path):
    store = SQLiteGrilloV2Store(tmp_path / "grillo-v2.sqlite3")
    episode = store.append_episode(
        GrilloEpisode.create(
            scope_key="discord:guild:1",
            source="discord",
            actor_id="discord_user:subby",
            content="Subby wants GRILLO to use memory tools.",
        )
    )
    old_fact = TemporalFact.create(
        scope_key="discord:guild:1",
        subject_id="discord_user:subby",
        predicate="memory_architecture",
        object_value="flat log",
        claim="Subby wants a flat log memory.",
    )
    old_fact.fact_id = "fact:old-flat-log"
    store.upsert_fact(old_fact)
    runtime = GrilloV2Runtime(store=store, persona_id="persona:neuro")

    result = runtime.apply_reflection(
        scope_key="discord:guild:1",
        payload={
            "notes": "tool path",
            "evidence": [],
            "facts": [],
            "opinion_edges": [],
            "memory_documents": [],
            "invalidate_facts": [],
            "tool_calls": [
                {
                    "function": {
                        "name": "record_evidence",
                        "arguments": json.dumps(
                            {
                                "evidence_id": "evidence:tool",
                                "episode_id": episode.episode_id,
                                "quote": "use memory tools",
                                "confidence": 0.91,
                            }
                        ),
                    }
                },
                {
                    "name": "upsert_fact",
                    "arguments": {
                        "fact_id": "fact:tool-memory",
                        "subject_id": "discord_user:subby",
                        "predicate": "prefers_memory_protocol",
                        "object": "tool calls",
                        "claim": "Subby wants GRILLO to use memory tool calls.",
                        "evidence_ids": ["evidence:tool"],
                        "confidence": 0.9,
                    },
                },
                {
                    "name": "upsert_opinion_edge",
                    "arguments": {
                        "edge_id": "opinion:tool-trust",
                        "target_id": "discord_user:subby",
                        "relation": "collaboration",
                        "score": 0.8,
                        "rationale": "The user is steering the memory protocol.",
                        "evidence_ids": ["evidence:tool"],
                    },
                },
                {
                    "name": "upsert_memory_document",
                    "arguments": {
                        "memory_id": "memory:tool-diary",
                        "document_type": "diary",
                        "subject_id": "discord_user:subby",
                        "title": "Tool protocol",
                        "body": "I should use explicit memory tool calls when reflecting.",
                        "evidence_ids": ["evidence:tool"],
                        "importance": 0.86,
                    },
                },
                {"name": "invalidate_fact", "arguments": {"fact_id": "fact:old-flat-log"}},
            ],
        },
    )
    packet = runtime.build_context_packet(
        scope_key="discord:guild:1",
        actor_id="discord_user:subby",
        query="memory tools",
    )

    assert result.tool_calls == 5
    assert result.evidence == 1
    assert result.facts == 1
    assert result.opinions == 1
    assert result.memory_docs == 1
    assert result.invalidated_facts == 1
    assert result.ignored_tool_calls == 0
    assert packet.active_facts[0]["id"] == "fact:tool-memory"
    assert packet.relationship_state[0]["id"] == "opinion:tool-trust"
    assert packet.memory_blocks[0]["id"] == "memory:tool-diary"
    assert "fact:old-flat-log" not in packet.as_prompt_text()


@pytest.mark.asyncio
async def test_grillo_v2_worker_tick_processes_unreflected_batches(tmp_path):
    store = SQLiteGrilloV2Store(tmp_path / "grillo-v2.sqlite3")
    scope_key = "discord:guild:1"
    episodes = [
        store.append_episode(
            GrilloEpisode.create(
                scope_key=scope_key,
                source="discord",
                actor_id="discord_user:subby",
                content=f"worker message {index}",
                occurred_at=f"2026-06-20T00:00:0{index}+00:00",
            )
        )
        for index in range(3)
    ]
    seen_batches = []

    async def completion(request):
        seen_batches.append([episode["episode_id"] for episode in request["episodes"]])
        return {
            "notes": f"batch {len(seen_batches)}",
            "evidence": [],
            "facts": [],
            "opinion_edges": [],
            "memory_documents": [],
            "invalidate_facts": [],
            "tool_calls": [],
        }

    runtime = GrilloV2Runtime(store=store, completion=completion, persona_id="persona:neuro")

    first = await runtime.worker_tick(scope_key=scope_key, batch_size=2, max_batches=1)
    cursor = json.loads(store.get_cursor(runtime.worker_cursor_key(scope_key)))
    second = await runtime.worker_tick(scope_key=scope_key, batch_size=2, max_batches=1)
    third = await runtime.worker_tick(scope_key=scope_key, batch_size=2, max_batches=1)

    assert seen_batches == [
        [episodes[0].episode_id, episodes[1].episode_id],
        [episodes[2].episode_id],
    ]
    assert first.batches == 1
    assert first.episodes == 2
    assert cursor["episode_id"] == episodes[1].episode_id
    assert second.batches == 1
    assert second.episodes == 1
    assert third.batches == 0
    assert third.notes == ["no_unprocessed_episodes"]


@pytest.mark.asyncio
async def test_grillo_v2_worker_loop_runs_configured_ticks():
    calls = []
    sleeps = []
    tick_notes = []

    class FakeRuntime:
        async def worker_tick(self, **kwargs):
            calls.append(kwargs)
            return WorkerTickResult(scopes=1, batches=1, episodes=len(calls), notes=[f"tick-{len(calls)}"])

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    async def on_tick(result):
        tick_notes.extend(result.notes or [])

    worker = GrilloV2Worker(
        runtime=FakeRuntime(),
        config=GrilloV2WorkerConfig(
            interval_seconds=2.5,
            initial_delay_seconds=0.25,
            scope_key="discord:guild:1:persona:v2",
            scope_limit=7,
            batch_size=3,
            max_batches=2,
        ),
        on_tick=on_tick,
        sleep=fake_sleep,
    )

    await worker.run_forever(max_ticks=2)

    assert calls == [
        {
            "scope_key": "discord:guild:1:persona:v2",
            "scope_limit": 7,
            "batch_size": 3,
            "max_batches": 2,
        },
        {
            "scope_key": "discord:guild:1:persona:v2",
            "scope_limit": 7,
            "batch_size": 3,
            "max_batches": 2,
        },
    ]
    assert sleeps == [0.25, 2.5]
    assert tick_notes == ["tick-1", "tick-2"]
    assert worker.ticks == 2
    assert worker.running is False
    assert worker.last_result is not None
    assert worker.last_result.episodes == 2


def test_brain_v2_uses_vercel_gateway_and_grillo_v2_store(tmp_path):
    calls = []

    class FakeResponses:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                output_text=json.dumps(
                    {
                        "notes": "ok",
                        "evidence": [],
                        "facts": [],
                        "opinion_edges": [],
                        "memory_documents": [],
                        "invalidate_facts": [],
                        "tool_calls": [],
                    }
                )
            )

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


@pytest.mark.asyncio
async def test_brain_v2_respond_compiles_grillo_context_and_stores_assistant_episode(tmp_path):
    calls = []

    class FakeResponses:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(output_text="yo, noted")

    fake_client = SimpleNamespace(responses=FakeResponses())
    json_client = VercelAIGatewayJSONClient(client=fake_client, model="deepseek/test")
    brain = BrainV2(
        BrainV2Config(database_path=tmp_path / "brain-v2.sqlite3", model="deepseek/test"),
        json_client=json_client,
    )
    brain.store.upsert_fact(
        TemporalFact.create(
            scope_key="discord:guild:1",
            subject_id="discord_user:subby",
            predicate="preferred_name",
            object_value="Subby",
            claim="Subby prefers being called Subby.",
            confidence=0.9,
        )
    )

    response = await brain.respond(
        scope_key="discord:guild:1",
        actor_id="discord_user:subby",
        user_text="hello",
        source="discord",
        channel_id="bot-chat",
    )
    episodes = brain.store.list_recent_episodes("discord:guild:1", limit=10)

    assert response == "yo, noted"
    assert len(calls) == 1
    assert calls[0]["model"] == "deepseek/test"
    assert "<grillo_context" in calls[0]["input"]
    assert "Subby prefers being called Subby." in calls[0]["input"]
    assert [episode.source for episode in episodes] == ["discord", "discord:assistant"]


@pytest.mark.asyncio
async def test_brain_v2_respond_includes_persona_prompt(tmp_path):
    calls = []

    class FakeResponses:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(output_text="persona loaded")

    fake_client = SimpleNamespace(responses=FakeResponses())
    json_client = VercelAIGatewayJSONClient(client=fake_client, model="deepseek/test")
    brain = BrainV2(
        BrainV2Config(
            database_path=tmp_path / "brain-v2.sqlite3",
            model="deepseek/test",
            persona_prompt="Keep Neuro's sharp streamer persona intact.",
        ),
        json_client=json_client,
    )

    response = await brain.respond(
        scope_key="discord:guild:1",
        actor_id="discord_user:subby",
        user_text="hello",
    )

    assert response == "persona loaded"
    assert "Keep Neuro's sharp streamer persona intact." in calls[0]["instructions"]


@pytest.mark.asyncio
async def test_brain_v2_respond_includes_metadata_and_rolling_context_without_double_recording(tmp_path):
    calls = []

    class FakeResponses:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(output_text="got context")

    fake_client = SimpleNamespace(responses=FakeResponses())
    json_client = VercelAIGatewayJSONClient(client=fake_client, model="deepseek/test")
    brain = BrainV2(
        BrainV2Config(database_path=tmp_path / "brain-v2.sqlite3", model="deepseek/test"),
        json_client=json_client,
    )
    recorded = brain.record_message(
        scope_key="discord:guild:1",
        actor_id="discord_user:subby",
        user_text="yep",
        source="discord",
        channel_id="bot-chat",
        metadata={
            "message_id": "m2",
            "guild_name": "Test Guild",
            "channel_name": "bot-chat",
            "author_display_name": "Subby",
            "author_username": "subsect",
            "author_id": "123",
            "reply_target": {
                "message_id": "m1",
                "author": "Neuro-sama",
                "author_id": "999",
                "author_is_bot": True,
                "content": "Do you want me to check that?",
            },
        },
    )

    response = await brain.respond(
        scope_key="discord:guild:1",
        actor_id="discord_user:subby",
        user_text="yep",
        source="discord",
        channel_id="bot-chat",
        metadata={
            "message_id": "m2",
            "guild_name": "Test Guild",
            "channel_name": "bot-chat",
            "author_display_name": "Subby",
            "author_username": "subsect",
            "author_id": "123",
            "reply_target": {
                "message_id": "m1",
                "author": "Neuro-sama",
                "author_id": "999",
                "author_is_bot": True,
                "content": "Do you want me to check that?",
            },
        },
        rolling_context=[
            {
                "message_id": "m1",
                "author": "Neuro-sama",
                "author_is_bot": True,
                "content": "Do you want me to check that?",
                "created_at": "2026-06-23T12:00:00+00:00",
            },
            {
                "message_id": "m2",
                "author": "Subby",
                "content": "yep",
                "created_at": "2026-06-23T12:00:02+00:00",
            },
        ],
        record_user_episode=False,
        reply_to_episode_id=recorded.episode_id,
    )
    episodes = brain.store.list_recent_episodes("discord:guild:1", limit=10)

    assert response == "got context"
    assert "# Current Discord Metadata" in calls[0]["input"]
    assert "guild_name: Test Guild" in calls[0]["input"]
    assert "- content: Do you want me to check that?" in calls[0]["input"]
    assert "# Recent Discord Channel Context" in calls[0]["input"]
    assert "Neuro-sama (bot): Do you want me to check that?" in calls[0]["input"]
    assert [episode.source for episode in episodes] == ["discord", "discord:assistant"]


def test_discord_bot_v2_loads_persona_prompt_path(tmp_path, monkeypatch):
    prompt_path = tmp_path / "neuro.persona.txt"
    prompt_path.write_text("Neuro persona from disk.", encoding="utf-8")
    monkeypatch.setenv("DISCORD_BRAIN_V2_DATABASE_PATH", str(tmp_path / "brain.sqlite3"))
    monkeypatch.setenv("DISCORD_BRAIN_V2_PERSONA_PROMPT_PATH", str(prompt_path))
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-key")

    brain = build_brain_v2()

    assert brain.config.persona_prompt == "Neuro persona from disk."


def test_discord_bot_v2_helpers_make_server_scope_and_metadata():
    guild = SimpleNamespace(id=222, name="Test Guild")
    channel = SimpleNamespace(id=333, name="bot-chat")
    author = SimpleNamespace(
        id=123,
        name="subsect",
        display_name="Npc",
        global_name=None,
        bot=False,
    )
    message = SimpleNamespace(
        id=444,
        guild=guild,
        channel=channel,
        author=author,
        jump_url="https://discord.example/message",
    )

    assert _scope_for_message(message) == "discord:guild:222:persona:v2"
    metadata = _discord_metadata(message)
    assert metadata["guild_name"] == "Test Guild"
    assert metadata["channel_name"] == "bot-chat"
    assert metadata["author_username"] == "subsect"
    assert metadata["author_display_name"] == "Npc"


def test_discord_bot_v2_metadata_and_recent_item_include_reply_target():
    bot_author = SimpleNamespace(id=999, name="neuro", display_name="Neuro-sama", global_name=None, bot=True)
    user = SimpleNamespace(id=123, name="subsect", display_name="Subby", global_name=None, bot=False)
    target = SimpleNamespace(
        id=444,
        author=bot_author,
        clean_content="Do you want me to check that?",
        content="Do you want me to check that?",
        jump_url="https://discord.example/target",
    )
    message = SimpleNamespace(
        id=445,
        guild=SimpleNamespace(id=222, name="Test Guild"),
        channel=SimpleNamespace(id=333, name="bot-chat"),
        author=user,
        clean_content="yep",
        content="yep",
        reference=SimpleNamespace(resolved=target, cached_message=None, message_id=444),
        jump_url="https://discord.example/message",
        created_at=None,
    )

    metadata = _discord_metadata(message)
    recent = _recent_message_item(message)

    assert metadata["reply_target"]["author"] == "Neuro-sama"
    assert metadata["reply_target"]["content"] == "Do you want me to check that?"
    assert recent["reply_to_author"] == "Neuro-sama"
    assert recent["reply_to_message_id"] == "444"


def test_grillo_v2_backfills_v1_turns_candidates_and_identity(tmp_path):
    v1_path = tmp_path / "discord_brain.sqlite3"
    conn = sqlite3.connect(v1_path)
    conn.executescript(
        """
        CREATE TABLE grillo_turns (
            turn_id TEXT PRIMARY KEY,
            scope_key TEXT NOT NULL,
            participant_key TEXT NOT NULL,
            role TEXT NOT NULL,
            author_name TEXT NOT NULL,
            channel_id TEXT,
            interface_path TEXT,
            source TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        );
        CREATE TABLE grillo_candidates (
            candidate_id TEXT PRIMARY KEY,
            scope_key TEXT NOT NULL,
            participant_key TEXT NOT NULL,
            type TEXT NOT NULL,
            content TEXT NOT NULL,
            summary TEXT NOT NULL,
            confidence REAL NOT NULL,
            tags_json TEXT NOT NULL DEFAULT '[]',
            source_turn_ids_json TEXT NOT NULL DEFAULT '[]',
            promoted INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );
        CREATE TABLE grillo_slots (
            slot_id TEXT PRIMARY KEY,
            scope_key TEXT NOT NULL,
            participant_key TEXT NOT NULL,
            slot_name TEXT NOT NULL,
            items_json TEXT NOT NULL DEFAULT '[]',
            source_candidate_ids_json TEXT NOT NULL DEFAULT '[]',
            updated_at TEXT NOT NULL
        );
        """
    )
    conn.execute(
        """
        INSERT INTO grillo_turns VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "turn-1",
            "discord:guild:222:user:123:persona:neuro-sama",
            "123",
            "user",
            "SUBSECT",
            "456",
            None,
            "discord",
            "Call me Subby, not LO.",
            json.dumps({"guild_id": 222, "author_id": 123, "author_display_name": "SUBSECT"}),
            "2026-06-19T16:10:00+00:00",
        ),
    )
    conn.execute(
        """
        INSERT INTO grillo_candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "candidate-1",
            "discord:guild:222:user:123:persona:neuro-sama",
            "123",
            "preference",
            "Subby prefers temporal memory.",
            "Subby prefers temporal memory.",
            0.87,
            json.dumps(["memory"]),
            json.dumps(["turn-1"]),
            1,
            "2026-06-19T16:12:00+00:00",
        ),
    )
    conn.execute(
        """
        INSERT INTO grillo_slots VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "slot-1",
            "discord:guild:222:user:123:persona:neuro-sama",
            "123",
            "preferences",
            json.dumps(["Subby wants GRILLO v2 to be evidence-backed."]),
            json.dumps(["candidate-1"]),
            "2026-06-19T16:13:00+00:00",
        ),
    )
    conn.commit()
    conn.close()

    identity_path = tmp_path / "discord_brain.discord-identity.sqlite3"
    conn = sqlite3.connect(identity_path)
    conn.execute(
        """
        CREATE TABLE discord_user_identities (
            guild_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            username TEXT,
            display_name TEXT,
            global_name TEXT,
            mention TEXT,
            is_bot INTEGER NOT NULL DEFAULT 0,
            first_seen_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            message_count INTEGER NOT NULL DEFAULT 0,
            aliases_json TEXT NOT NULL DEFAULT '[]',
            PRIMARY KEY (guild_id, user_id)
        )
        """
    )
    conn.execute(
        """
        INSERT INTO discord_user_identities VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "222",
            "123",
            "subsect",
            "Npc",
            None,
            "<@123>",
            0,
            "2026-06-19T16:10:00+00:00",
            "2026-06-19T16:20:00+00:00",
            12,
            json.dumps(["SUBSECT", "Subby"]),
        ),
    )
    conn.commit()
    conn.close()

    target = SQLiteGrilloV2Store(tmp_path / "grillo-v2.sqlite3")
    v1_result = backfill_grillo_v1(source_path=v1_path, target=target, persona_id="neuro-sama-v2")
    identity_result = backfill_discord_identity(source_path=identity_path, target=target, persona_id="neuro-sama-v2")
    runtime = GrilloV2Runtime(store=target, persona_id="neuro-sama-v2")
    packet = runtime.build_context_packet(
        scope_key="discord:guild:222:persona:v2",
        actor_id="discord_user:123",
        query="temporal memory",
        channel_id="456",
    )
    prompt = packet.as_prompt_text()

    assert v1_result.episodes == 1
    assert v1_result.facts == 2
    assert identity_result.entities == 1
    assert identity_result.facts >= 3
    assert "Subby prefers temporal memory." in prompt
    assert "Npc" in prompt
    assert "SUBSECT" in prompt


def test_brain_v2_backfill_and_status_formatting(tmp_path):
    brain = BrainV2(
        BrainV2Config(database_path=tmp_path / "brain-v2.sqlite3"),
        json_client=SimpleNamespace(),
    )
    brain.store.upsert_entity(
        GrilloEntity(
            entity_id="discord_user:123",
            entity_type="person",
            name="Subby",
            aliases=["SUBSECT"],
            metadata={},
        )
    )
    status = brain.status()

    assert status["base_url"] == VERCEL_AI_GATEWAY_BASE_URL
    assert status["counts"]["entities"] == 1
    assert "entities=`1`" in _format_status(status)
    assert "grillo_v1" in _format_backfill_results(
        {"grillo_v1": SimpleNamespace(episodes=1, entities=0, evidence=2, facts=3, skipped=0)}
    )
    assert "memory_docs=`1`" in _format_worker_result(
        SimpleNamespace(
            scopes=1,
            batches=1,
            episodes=2,
            evidence=0,
            facts=0,
            opinions=0,
            memory_docs=1,
            invalidated_facts=0,
            notes=["ok"],
        )
    )
    assert "ticks=`3`" in _format_worker_loop_status(
        SimpleNamespace(
            worker_enabled=True,
            worker_task=None,
            worker=SimpleNamespace(
                ticks=3,
                consecutive_errors=0,
                last_result=WorkerTickResult(batches=1, episodes=2, notes=["ok"]),
            ),
        )
    )
