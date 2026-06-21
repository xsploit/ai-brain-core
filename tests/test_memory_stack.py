import importlib.util
import json
from types import SimpleNamespace
import pytest

from aibrain import Brain, BrainConfig, MemoryStackConfig
from aibrain.embeddings import HashEmbeddingProvider
from aibrain.memory_stack import (
    GRILLOMemoryWorker,
    GrilloCandidate,
    GrilloRuntime,
    GrilloDiaryEntry,
    GrilloRelationshipProfile,
    GraphQuery,
    HybridMemoryStack,
    HybridRetriever,
    RawEvent,
    RecallItem,
    SQLiteGrilloStore,
    SQLiteRawEventStore,
    SQLiteTemporalGraphStore,
    SQLiteVectorRecallStore,
    LadybugGraphMemoryStore,
    TemporalFact,
    TurboVecRecallStore,
)


class FakeConversations:
    async def create(self, **kwargs):
        return SimpleNamespace(id="conv_memory")


class FakeResponses:
    async def create(self, **kwargs):
        return SimpleNamespace(
            id="resp_memory",
            output=[],
            output_text="noted",
            conversation=kwargs.get("conversation"),
            usage=None,
        )


class FakeOpenAI:
    def __init__(self):
        self.conversations = FakeConversations()
        self.responses = FakeResponses()


class RecordingEmbeddingProvider:
    def __init__(self, *, dimensions: int = 16, max_len: int = 12):
        self.dimensions = dimensions
        self.max_len = max_len
        self.calls = []

    async def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        if len(text) > self.max_len:
            raise AssertionError(f"embedding input too long: {len(text)}")
        return [1.0, *([0.0] * (self.dimensions - 1))]


@pytest.mark.asyncio
async def test_raw_log_appends_and_lists_thread_events(tmp_path):
    store = SQLiteRawEventStore(tmp_path / "brain.sqlite3")
    await store.append(
        RawEvent(
            id="event-1",
            event_type="message",
            actor="user",
            thread_id="thread-1",
            content="hello",
        )
    )

    events = await store.list_thread_events("thread-1")

    assert [event.id for event in events] == ["event-1"]
    assert events[0].content == "hello"


@pytest.mark.asyncio
async def test_grillo_worker_extracts_temporal_fact_and_indexes_recall(tmp_path):
    provider = HashEmbeddingProvider(dimensions=32)
    graph = SQLiteTemporalGraphStore(tmp_path / "brain.sqlite3")
    vector = SQLiteVectorRecallStore(tmp_path / "brain.sqlite3", embedding_provider=provider)
    worker = GRILLOMemoryWorker(graph_store=graph, vector_store=vector)

    facts = await worker.process_event(
        RawEvent(
            id="event-like",
            event_type="message",
            actor="user",
            thread_id="thread-1",
            persona_id="riko",
            content="I like LadybugDB for memory graphs.",
            created_at="2026-06-08T00:00:00+00:00",
        )
    )

    assert facts[0].predicate == "likes"
    graph_hits = await graph.search_facts(GraphQuery(text="LadybugDB", top_k=3))
    vector_hits = await vector.search("memory graph", top_k=3)
    assert graph_hits[0].source_event_id == "event-like"
    assert vector_hits[0].source_fact_id == facts[0].id


@pytest.mark.asyncio
async def test_hybrid_retriever_fuses_graph_and_vector_hits(tmp_path):
    provider = HashEmbeddingProvider(dimensions=32)
    graph = SQLiteTemporalGraphStore(tmp_path / "brain.sqlite3")
    vector = SQLiteVectorRecallStore(tmp_path / "brain.sqlite3", embedding_provider=provider)
    worker = GRILLOMemoryWorker(graph_store=graph, vector_store=vector)
    retriever = HybridRetriever(graph_store=graph, vector_store=vector)

    await worker.process_event(
        RawEvent(
            id="event-graph",
            event_type="message",
            actor="user",
            thread_id="thread-1",
            persona_id="riko",
            content="I am working on ai-brain-core.",
            created_at="2026-06-08T00:00:00+00:00",
        )
    )
    await vector.add(
        RecallItem(
            id="doc-1",
            text="ai-brain-core has a hybrid memory system.",
            thread_id="thread-1",
            persona_id="riko",
            scope="thread",
        )
    )

    hits = await retriever.retrieve_records(
        "what brain memory system",
        top_k=5,
        thread_id="thread-1",
        persona_id="riko",
    )

    assert hits
    assert any("ai-brain-core" in hit.content for hit in hits)
    assert {hit.metadata["memory_source"] for hit in hits} <= {"graph", "vector"}


def test_hybrid_memory_stack_auto_backend_falls_back_when_extras_missing(tmp_path):
    stack = HybridMemoryStack.sqlite_paths(
        raw_log_path=tmp_path / "brain.sqlite3",
        graph_path=tmp_path / "brain.sqlite3",
        vector_path=tmp_path / "brain.sqlite3",
        embedding_provider=HashEmbeddingProvider(dimensions=32),
        embedding_dimensions=32,
        graph_backend="auto",
        vector_backend="auto",
    )

    assert stack.grillo is not None
    assert stack.graph_store is not None
    assert stack.vector_store is not None
    stack.close()


@pytest.mark.asyncio
async def test_grillo_runtime_ingests_diary_slots_and_context_packet(tmp_path):
    provider = HashEmbeddingProvider(dimensions=32)
    vector = SQLiteVectorRecallStore(tmp_path / "brain.sqlite3", embedding_provider=provider)
    runtime = GrilloRuntime(
        store=SQLiteGrilloStore(tmp_path / "brain.sqlite3"),
        vector_store=vector,
    )

    await runtime.ingest_turn_pair(
        scope_key="discord:1:2",
        participant_key="user-1",
        user_text="I like LadybugDB and I am working on TurboVec memory for AI Brain.",
        assistant_text="Noted.",
        author_name="Tyler",
        assistant_name="Neuro-sama",
        source="discord",
    )

    status = await runtime.status()
    packet = await runtime.build_context_packet(
        scope_key="discord:1:2",
        participant_key="user-1",
        query="what memory stack",
        persona_name="Neuro-sama",
    )

    assert status.turns == 2
    assert status.candidates >= 2
    assert status.diary_entries == 1
    assert status.slots >= 2
    assert any("LadybugDB" in item for item in packet.relationship_memory)
    assert packet.thoughts
    assert "grillo_context_packet" in packet.as_prompt_text()


@pytest.mark.asyncio
async def test_grillo_runtime_uses_llm_reflector_for_clean_reflections(tmp_path):
    async def reflector(context):
        user_turn = next(turn for turn in context["turns"] if turn["role"] == "user")
        return {
            "done": True,
            "notes": "stored profile and relationship memory",
            "candidates": [
                {
                    "type": "fact",
                    "content": "LO is a male adult erotica author.",
                    "summary": "LO is a male adult erotica author.",
                    "confidence": 0.91,
                    "tags": ["profile"],
                    "source_turn_ids": [user_turn["turn_id"]],
                },
                {
                    "type": "preference",
                    "content": "LO prefers crude direct language in adult writing contexts.",
                    "summary": "LO prefers crude direct language in adult writing contexts.",
                    "confidence": 0.88,
                    "tags": ["preference"],
                    "source_turn_ids": [user_turn["turn_id"]],
                },
                {
                    "type": "bond_signal",
                    "content": "LO is correcting the assistant toward the real GRILLO reflection design.",
                    "summary": "LO expects GRILLO to be an AI diary/reflection system.",
                    "confidence": 0.84,
                    "tags": ["relationship"],
                    "source_turn_ids": [user_turn["turn_id"]],
                },
            ],
            "diary": {
                "summary": "LO clarified what GRILLO is supposed to be.",
                "personal_thought": "I should treat LO's memory as reflective relationship context, not as pattern-matched labels.",
                "tags": ["grillo", "relationship"],
                "source_turn_ids": [user_turn["turn_id"]],
            },
            "slots": [
                {
                    "slot_name": "user_facts",
                    "items": ["LO is a male adult erotica author."],
                    "operation": "merge",
                },
                {
                    "slot_name": "preferences",
                    "items": ["LO prefers crude direct language in adult writing contexts."],
                    "operation": "merge",
                },
                {
                    "slot_name": "relationship_state",
                    "items": ["LO expects GRILLO to be an AI diary/reflection system."],
                    "operation": "merge",
                },
            ],
            "relationship_profile": {
                "relationship_stage": "familiar",
                "mood": "focused",
                "trust": 6,
                "respect": 6,
                "guard": 12,
                "summary": "LO wants GRILLO implemented as reflective relationship memory.",
                "facts": ["LO expects GRILLO to be an AI diary/reflection system."],
            },
            "profile_patches": [
                {
                    "field": "interaction_style",
                    "operation": "add",
                    "value": "Treat corrections about memory architecture as high-signal.",
                }
            ],
        }

    runtime = GrilloRuntime(
        store=SQLiteGrilloStore(tmp_path / "brain.sqlite3"),
        vector_store=None,
        reflector=reflector,
    )

    await runtime.ingest_turn_pair(
        scope_key="discord:guild:alpha",
        participant_key="user-alpha",
        user_text=(
            "This applies to all chats, always Some things about me: "
            "My name is LO, a male. I am an Adult Erotica Author. "
            "I prefer crude direct language. 500 words minimum length."
        ),
        assistant_text="Noted.",
        source="discord",
    )

    slots = await runtime.store.list_slots("discord:guild:alpha", "user-alpha")
    profile = await runtime.store.get_relationship_profile("discord:guild:alpha")
    packet = await runtime.build_context_packet(
        scope_key="discord:guild:alpha",
        participant_key="user-alpha",
        persona_name="Neuro-sama",
    )
    by_slot = {slot.slot_name: slot.items for slot in slots}
    all_items = [item for slot in slots for item in slot.items]
    relationship_text = "\n".join(packet.relationship_memory)

    assert any("adult erotica author" in item.lower() for item in by_slot["user_facts"])
    assert any("crude direct language" in item for item in by_slot["preferences"])
    assert any("AI diary/reflection system" in item for item in by_slot["relationship_state"])
    assert profile is not None
    assert profile.relationship_stage == "familiar"
    assert profile.mood == "focused"
    assert profile.trust == 6
    assert any("high-signal" in item for item in profile.interaction_style)
    assert not any("This applies to all chats" in item for item in all_items)
    assert "This applies to all chats" not in relationship_text
    assert "Preference signal:" not in relationship_text
    assert "stage=familiar mood=focused" in relationship_text
    assert "pattern-matched labels" in packet.thoughts[0]


@pytest.mark.asyncio
async def test_grillo_runtime_uses_webwaifu_worker_loop_to_reflect_with_context(tmp_path):
    seen_requests = []

    async def worker_completion(request):
        seen_requests.append(request)
        if len(seen_requests) == 1:
            prompt = request["messages"][1]["content"]
            assert "Canonical GRILLO context packet" in prompt
            assert "stage=familiar mood=guarded" in prompt
            assert "I already feel guarded but invested around LO." in prompt
            return {
                "text": json.dumps(
                    {
                        "done": False,
                        "notes": "reflect with tools",
                        "toolCalls": [
                            {"name": "core.worker_memory_read", "args": {}},
                            {
                                "name": "core.worker_candidate_write",
                                "args": {
                                    "type": "bond_signal",
                                    "content": "LO wants the real WebWaifu GRILLO worker loop ported one-to-one.",
                                    "summary": "LO expects one-to-one GRILLO worker parity.",
                                    "confidence": 0.93,
                                    "tags": ["relationship", "grillo"],
                                },
                            },
                            {
                                "name": "core.worker_diary_write",
                                "args": {
                                    "summary": "LO pushed for real GRILLO parity.",
                                    "personal_thought": (
                                        "I should treat LO's correction as a serious relationship signal: "
                                        "he wants the real reflective worker, not a shallow memory parser."
                                    ),
                                    "tags": ["relationship", "grillo"],
                                    "beat_type": "relationship",
                                },
                            },
                            {
                                "name": "core.worker_memory_write",
                                "args": {
                                    "block_name": "relationship_state",
                                    "items": ["LO expects the Discord bot to use the real GRILLO worker loop."],
                                    "operation": "merge",
                                    "reason": "grounded relationship beat",
                                },
                            },
                            {
                                "name": "core.worker_profile_patch",
                                "args": {
                                    "field": "interaction_style",
                                    "operation": "add",
                                    "value": "Treat one-to-one port requests as literal architecture requirements.",
                                },
                            },
                            {
                                "name": "core.worker_emotion_update",
                                "args": {
                                    "intensities": {"focused": 0.82, "guarded": 0.25},
                                    "last_signal_source": "relationship beat",
                                },
                            },
                            {
                                "name": "core.worker_memory_insert_archival",
                                "args": {"text": "One-to-one GRILLO parity requires the worker loop and tool executor."},
                            },
                        ],
                    }
                ),
                "meta": {"provider": "test", "model": "worker-test"},
            }
        assert any("core.worker_memory_read" in message["content"] for message in request["messages"])
        return {"text": json.dumps({"done": True, "notes": "done", "toolCalls": []})}

    store = SQLiteGrilloStore(tmp_path / "brain.sqlite3")
    await store.upsert_relationship_profile(
        GrilloRelationshipProfile(
            profile_id="relationship:discord:guild:alpha",
            scope_key="discord:guild:alpha",
            persona_id="neuro",
            participant_keys=["user-alpha"],
            relationship_stage="familiar",
            mood="guarded",
            trust=5,
            summary="LO wants literal GRILLO parity.",
        )
    )
    await store.append_diary(
        GrilloDiaryEntry(
            diary_id="diary-existing",
            scope_key="discord:guild:alpha",
            participant_key="user-alpha",
            beat_type="relationship",
            summary="Existing relationship reflection.",
            personal_thought="I already feel guarded but invested around LO.",
            tags=["relationship"],
            source_turn_ids=[],
            created_at="2026-06-20T00:00:00+00:00",
        )
    )
    runtime = GrilloRuntime(
        store=store,
        vector_store=None,
        worker_completion=worker_completion,
    )

    await runtime.ingest_turn_pair(
        scope_key="discord:guild:alpha",
        participant_key="user-alpha",
        user_text="Get the WebWaifu GRILLO reflection worker one-to-one.",
        assistant_text="I will port the worker loop.",
        source="discord",
    )

    assert len(seen_requests) == 2
    slots = await store.list_slots("discord:guild:alpha", "user-alpha")
    profile = await store.get_relationship_profile("discord:guild:alpha")
    emotion = await store.get_emotion_state("discord:guild:alpha")
    archival = await store.list_archival_memories("discord:guild:alpha")
    diary = await store.list_diary("discord:guild:alpha", "user-alpha", limit=4)

    assert any(
        "real GRILLO worker loop" in item
        for slot in slots
        if slot.slot_name == "relationship_state"
        for item in slot.items
    )
    assert profile is not None
    assert any("literal architecture requirements" in item for item in profile.interaction_style)
    assert profile.diary_entry.startswith("I should treat LO's correction")
    assert emotion["intensities"]["focused"] == 0.82
    assert archival and "worker loop" in archival[0]["text"]
    assert any("serious relationship signal" in entry.personal_thought for entry in diary)


@pytest.mark.asyncio
async def test_grillo_worker_noop_falls_back_to_diary_write(tmp_path):
    async def worker_completion(request):
        return {"text": json.dumps({"done": True, "notes": "nothing to write", "toolCalls": []})}

    store = SQLiteGrilloStore(tmp_path / "brain.sqlite3")
    runtime = GrilloRuntime(
        store=store,
        vector_store=None,
        worker_completion=worker_completion,
    )

    result = await runtime.ingest_turn_pair(
        scope_key="discord:guild:alpha:user:user-alpha:persona:neuro",
        participant_key="user-alpha",
        user_text="Remember that my name is Subby and the channel move confused context.",
        assistant_text="Got it, I should remember that across channels.",
        source="discord",
        channel_id="222",
    )

    diary = await store.list_diary(
        "discord:guild:alpha:user:user-alpha:persona:neuro",
        "user-alpha",
        limit=4,
    )
    candidates = await store.list_candidates(
        "discord:guild:alpha:user:user-alpha:persona:neuro",
        "user-alpha",
        limit=4,
    )
    turns = await store.list_turns(
        "discord:guild:alpha:user:user-alpha:persona:neuro",
        "user-alpha",
        limit=4,
    )

    assert result[0].channel_id == "222"
    assert diary
    assert "Subby" in diary[0].personal_thought
    assert candidates
    assert any("Subby" in candidate.content for candidate in candidates)
    assert all(turn.channel_id == "222" for turn in turns)


@pytest.mark.asyncio
async def test_grillo_context_packet_filters_semantic_recall_by_scope_and_participant(tmp_path):
    provider = HashEmbeddingProvider(dimensions=32)
    vector = SQLiteVectorRecallStore(tmp_path / "brain.sqlite3", embedding_provider=provider)
    runtime = GrilloRuntime(
        store=SQLiteGrilloStore(tmp_path / "brain.sqlite3"),
        vector_store=vector,
    )

    await runtime.ingest_turn_pair(
        scope_key="discord:guild:alpha",
        participant_key="user-alpha",
        user_text="I like LadybugDB for Alpha project memory.",
        assistant_text="Saved.",
        source="discord",
    )
    await runtime.ingest_turn_pair(
        scope_key="discord:guild:beta",
        participant_key="user-beta",
        user_text="I like TurboVec for Beta project memory.",
        assistant_text="Saved.",
        source="discord",
    )

    packet = await runtime.build_context_packet(
        scope_key="discord:guild:alpha",
        participant_key="user-alpha",
        query="project memory",
        persona_name="Neuro-sama",
    )
    recalled = " ".join(item["text"] for item in packet.recalled_memories)

    assert "Alpha project" in recalled
    assert "Beta project" not in recalled


@pytest.mark.asyncio
async def test_grillo_context_packet_keeps_channel_history_local_but_memory_cross_channel(tmp_path):
    runtime = GrilloRuntime(
        store=SQLiteGrilloStore(tmp_path / "brain.sqlite3"),
        vector_store=None,
    )
    scope = "discord:guild:alpha:user:user-alpha:persona:neuro"
    participant = "user-alpha"

    await runtime.ingest_turn_pair(
        scope_key=scope,
        participant_key=participant,
        user_text="In the first channel we were talking about the Filian wall countdown.",
        assistant_text="I remember the wall context.",
        author_name="Subby",
        assistant_name="Neuro-sama",
        channel_id="111",
        source="discord",
        run_tick=False,
    )
    await runtime.ingest_turn_pair(
        scope_key=scope,
        participant_key=participant,
        user_text="Now we moved channels and I am asking if you remember the wall thing.",
        assistant_text="Yeah, that context should follow you.",
        author_name="Subby",
        assistant_name="Neuro-sama",
        channel_id="222",
        source="discord",
        run_tick=False,
    )
    await runtime.store.append_candidate(
        GrilloCandidate(
            candidate_id="candidate-wall",
            scope_key=scope,
            participant_key=participant,
            type="thread",
            content="Subby discussed the Filian wall countdown in another Discord channel.",
            summary="Subby has an ongoing Filian wall countdown thread.",
            confidence=0.9,
            tags=["discord", "cross_channel"],
        )
    )

    packet = await runtime.build_context_packet(
        scope_key=scope,
        participant_key=participant,
        query="wall countdown",
        current_turn_text="remember the wall thing?",
        channel_id="222",
        persona_name="Neuro-sama",
    )
    history = "\n".join(packet.channel_history)
    recalled = "\n".join(item["text"] for item in packet.recalled_memories)

    assert "Now we moved channels" in history
    assert "first channel" not in history
    assert "Filian wall countdown" in recalled
    assert "history_scope: current channel only (222)" in packet.background_information


@pytest.mark.asyncio
async def test_brain_memory_stack_logs_and_extracts_when_enabled(tmp_path):
    config = BrainConfig(
        database_path=tmp_path / "brain.sqlite3",
        memory_stack=MemoryStackConfig(enabled=True, extract_user_events=True),
    )
    brain = Brain(config, client=FakeOpenAI())

    response = await brain.ask(
        "I like TurboVec for compressed recall.",
        thread_id="thread-memory-stack",
    )

    assert response.text == "noted"
    assert brain.memory_stack is not None
    events = await brain.memory_stack.raw_log.list_thread_events("thread-memory-stack")
    facts = await brain.memory_stack.graph_store.search_facts(
        GraphQuery(text="TurboVec", top_k=3)
    )

    assert [event.event_type for event in events] == ["message", "response"]
    assert facts
    assert facts[0].predicate == "likes"
    assert brain.memory_stack.grillo is not None
    assert brain.memory_stack.grillo.worker_completion is not None
    assert brain.memory_stack.grillo.reflector is not None


@pytest.mark.asyncio
async def test_brain_bounds_memory_retrieval_query_before_embedding(tmp_path, monkeypatch):
    monkeypatch.setenv("AIBRAIN_MEMORY_QUERY_MAX_CHARS", "12")
    provider = RecordingEmbeddingProvider(max_len=12)
    config = BrainConfig(
        database_path=tmp_path / "brain.sqlite3",
        memory_stack=MemoryStackConfig(enabled=True, retrieve=True),
    )
    brain = Brain(config, client=FakeOpenAI(), embedding_provider=provider)

    response = await brain.ask("x" * 200, thread_id="thread-long-query")

    assert response.text == "noted"
    assert provider.calls
    assert all(len(call) <= 12 for call in provider.calls)


@pytest.mark.asyncio
async def test_brain_can_keep_attachment_text_out_of_memory_and_history(tmp_path):
    provider = RecordingEmbeddingProvider(dimensions=256, max_len=80)
    config = BrainConfig(
        database_path=tmp_path / "brain.sqlite3",
        memory_stack=MemoryStackConfig(enabled=True, retrieve=True, extract_user_events=True),
    )
    brain = Brain(config, client=FakeOpenAI(), embedding_provider=provider)

    response = await brain.ask(
        "please summarize\n\n[Readable attachments]\nSECRET FILE TEXT",
        thread_id="thread-file-context",
        memory_query_text="please summarize",
        memory_event_text="please summarize",
        history_text="please summarize",
    )

    assert response.text == "noted"
    assert all("SECRET FILE TEXT" not in call for call in provider.calls)
    events = await brain.memory_stack.raw_log.list_thread_events("thread-file-context")
    assert events[0].content == "please summarize"
    history = brain.chat_store.list("thread-file-context", limit=10)
    assert history[0]["content"] == [{"type": "input_text", "text": "please summarize"}]


@pytest.mark.asyncio
async def test_ladybug_adapter_smoke_when_installed(tmp_path):
    if importlib.util.find_spec("ladybug") is None:
        pytest.skip("ladybug extra not installed")

    graph = LadybugGraphMemoryStore(tmp_path / "graph.ladybug")
    await graph.upsert_fact(
        TemporalFact(
            id="fact-ladybug",
            subject="Tyler",
            predicate="uses",
            object="LadybugDB",
            valid_from="2026-06-08T00:00:00+00:00",
        )
    )

    hits = await graph.search_facts(GraphQuery(text="LadybugDB", top_k=1))

    assert [hit.object for hit in hits] == ["LadybugDB"]

    await graph.upsert_relationship_profile(
        GrilloRelationshipProfile(
            profile_id="relationship:discord:guild:persona:neuro",
            scope_key="discord:guild:persona:neuro",
            persona_id="neuro",
            participant_keys=["discord:guild:lo"],
            relationship_stage="familiar",
            mood="focused",
            trust=6,
            respect=7,
            guard=10,
            facts=["LO expects GRILLO relationship memory."],
            summary="LO wants WebWaifu-style GRILLO memory.",
            updated_at="2026-06-20T00:00:00+00:00",
        )
    )
    profile = await graph.get_relationship_profile_graph("discord:guild:persona:neuro")

    assert profile is not None
    assert profile["relationship_stage"] == "familiar"
    assert profile["mood"] == "focused"


@pytest.mark.asyncio
async def test_turbovec_adapter_smoke_when_installed(tmp_path):
    if importlib.util.find_spec("turbovec") is None or importlib.util.find_spec("numpy") is None:
        pytest.skip("turbovec extra not installed")

    vector = TurboVecRecallStore(
        tmp_path / "vectors.turbovec",
        tmp_path / "metadata.sqlite3",
        embedding_provider=HashEmbeddingProvider(dimensions=16),
        dimensions=16,
    )
    await vector.add(
        RecallItem(
            id="recall-turbovec",
            text="TurboVec stores compressed semantic memory recall.",
        )
    )

    hits = await vector.search("compressed recall", top_k=1)

    assert [hit.id for hit in hits] == ["recall-turbovec"]
