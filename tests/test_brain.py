from types import SimpleNamespace

import asyncio
import pytest

from aibrain import Brain, BrainConfig, Persona
from aibrain.types import ThreadState


class FakeConversations:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(id="conv_123")


class FakeResponses:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            id=f"resp_{len(self.calls)}",
            output=[],
            output_text="ok",
            conversation=kwargs.get("conversation"),
            usage=None,
        )


class FakeStreamResponses:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)

        class Stream:
            async def __aiter__(self_inner):
                yield SimpleNamespace(
                    type="response.completed",
                    response=SimpleNamespace(
                        id=f"resp_stream_{len(self.calls)}",
                        output=[
                            SimpleNamespace(
                                type="function_call",
                                name="slow_a",
                                call_id="call_a",
                                arguments="{}",
                            ),
                            SimpleNamespace(
                                type="function_call",
                                name="slow_b",
                                call_id="call_b",
                                arguments="{}",
                            ),
                        ]
                        if len(self.calls) == 1
                        else [],
                        output_text="" if len(self.calls) == 1 else "done",
                        conversation=kwargs.get("conversation"),
                        usage=None,
                    ),
                )

        return Stream()


class FakeOpenAI:
    def __init__(self):
        self.conversations = FakeConversations()
        self.responses = FakeResponses()


@pytest.mark.asyncio
async def test_ask_passes_responses_state_options(tmp_path):
    client = FakeOpenAI()
    brain = Brain(
        BrainConfig(
            database_path=tmp_path / "brain.sqlite3",
            default_model="gpt-5-nano",
            default_prompt_cache_key="persona:test",
        ),
        client=client,
    )
    persona = Persona(id="riko", instructions="Be Riko.")

    response = await brain.ask(
        "hello",
        thread_id="thread-1",
        persona=persona,
        metadata={"source": "test"},
        service_tier="priority",
    )

    assert response.text == "ok"
    call = client.responses.calls[0]
    assert call["model"] == "gpt-5-nano"
    assert call["conversation"] == "conv_123"
    assert call["instructions"] == "Be Riko."
    assert call["prompt_cache_key"] == "persona:test"
    assert call["prompt_cache_retention"] == "24h"
    assert call["context_management"] == [{"type": "compaction"}]
    assert call["service_tier"] == "priority"


@pytest.mark.asyncio
async def test_thread_lock_cache_evicts_old_unlocked_locks(tmp_path):
    brain = Brain(
        BrainConfig(
            database_path=tmp_path / "brain.sqlite3",
            thread_lock_cache_size=2,
        ),
        client=FakeOpenAI(),
    )

    await brain._get_thread_lock("thread-a")
    await brain._get_thread_lock("thread-b")
    await brain._get_thread_lock("thread-c")

    assert list(brain._thread_locks) == ["thread-b", "thread-c"]


@pytest.mark.asyncio
async def test_tool_loop_continues_with_function_output(tmp_path):
    client = FakeOpenAI()

    async def create(**kwargs):
        client.responses.calls.append(kwargs)
        if len(client.responses.calls) == 1:
            return SimpleNamespace(
                id="resp_tool",
                output=[
                    SimpleNamespace(
                        type="function_call",
                        name="ping",
                        call_id="call_1",
                        arguments="{}",
                    )
                ],
                output_text="",
                conversation=kwargs.get("conversation"),
                usage=None,
            )
        return SimpleNamespace(
            id="resp_final",
            output=[],
            output_text="done",
            conversation=kwargs.get("conversation"),
            usage=None,
        )

    client.responses.create = create
    brain = Brain(BrainConfig(database_path=tmp_path / "brain.sqlite3"), client=client)

    @brain.tools.register
    def ping() -> str:
        """Ping test tool."""
        return "pong"

    response = await brain.ask("call ping", thread_id="thread-tools", tool_names=["ping"])

    assert response.text == "done"
    assert response.tool_results[0]["output"] == "pong"
    assert client.responses.calls[1]["previous_response_id"] == "resp_tool"
    assert client.responses.calls[1]["input"][0]["type"] == "function_call_output"


@pytest.mark.asyncio
async def test_same_thread_turns_serialize_remote_conversation_creation(tmp_path):
    client = FakeOpenAI()

    async def create_conversation(**kwargs):
        client.conversations.calls.append(kwargs)
        await asyncio.sleep(0.01)
        return SimpleNamespace(id="conv_shared")

    client.conversations.create = create_conversation
    brain = Brain(BrainConfig(database_path=tmp_path / "brain.sqlite3"), client=client)

    responses = await asyncio.gather(
        brain.ask("one", thread_id="shared", tool_names=[]),
        brain.ask("two", thread_id="shared", tool_names=[]),
    )

    assert len(client.conversations.calls) == 1
    assert [response.conversation_id for response in responses] == ["conv_shared", "conv_shared"]


def test_update_thread_after_response_does_not_clobber_last_response_with_none(tmp_path):
    brain = Brain(BrainConfig(database_path=tmp_path / "brain.sqlite3"), client=FakeOpenAI())
    state = ThreadState(
        thread_id="thread-state",
        persona_id="persona",
        openai_conversation_id="conv_existing",
        last_response_id="resp_existing",
    )
    brain.thread_store.upsert(state)

    brain._update_thread_after_response(
        state,
        SimpleNamespace(id=None, conversation=None),
    )

    assert state.last_response_id == "resp_existing"
    assert brain.thread_store.get("thread-state").last_response_id == "resp_existing"


@pytest.mark.asyncio
async def test_tool_calls_run_concurrently_and_preserve_output_order(tmp_path):
    client = FakeOpenAI()

    async def create(**kwargs):
        client.responses.calls.append(kwargs)
        if len(client.responses.calls) == 1:
            return SimpleNamespace(
                id="resp_tools",
                output=[
                    SimpleNamespace(
                        type="function_call",
                        name="slow_a",
                        call_id="call_a",
                        arguments="{}",
                    ),
                    SimpleNamespace(
                        type="function_call",
                        name="slow_b",
                        call_id="call_b",
                        arguments="{}",
                    ),
                ],
                output_text="",
                conversation=kwargs.get("conversation"),
                usage=None,
            )
        return SimpleNamespace(
            id="resp_final",
            output=[],
            output_text="done",
            conversation=kwargs.get("conversation"),
            usage=None,
        )

    client.responses.create = create
    brain = Brain(BrainConfig(database_path=tmp_path / "brain.sqlite3"), client=client)
    active = 0
    max_active = 0
    lock = asyncio.Lock()

    async def run_tool(name):
        nonlocal active, max_active
        async with lock:
            active += 1
            max_active = max(max_active, active)
        await asyncio.sleep(0.02)
        async with lock:
            active -= 1
        return name

    @brain.tools.register
    async def slow_a() -> str:
        return await run_tool("a")

    @brain.tools.register
    async def slow_b() -> str:
        return await run_tool("b")

    response = await brain.ask(
        "call both",
        thread_id="thread-parallel-tools",
        tool_names=["slow_a", "slow_b"],
    )

    assert response.text == "done"
    assert max_active == 2
    assert [item["call_id"] for item in response.tool_results] == ["call_a", "call_b"]
    assert [item["call_id"] for item in client.responses.calls[1]["input"]] == [
        "call_a",
        "call_b",
    ]


@pytest.mark.asyncio
async def test_streaming_tool_calls_emit_calls_before_parallel_results(tmp_path):
    client = FakeOpenAI()
    client.responses = FakeStreamResponses()
    brain = Brain(BrainConfig(database_path=tmp_path / "brain.sqlite3"), client=client)

    @brain.tools.register
    async def slow_a() -> str:
        await asyncio.sleep(0.02)
        return "a"

    @brain.tools.register
    async def slow_b() -> str:
        await asyncio.sleep(0.01)
        return "b"

    events = [
        event
        async for event in brain.stream(
            "call both",
            thread_id="thread-stream-tools",
            tool_names=["slow_a", "slow_b"],
        )
    ]
    event_types = [event.type for event in events]

    first_result = event_types.index("tool.result")
    assert event_types[:first_result].count("tool.call") == 2
    assert [item["call_id"] for item in client.responses.calls[1]["input"]] == [
        "call_a",
        "call_b",
    ]


@pytest.mark.asyncio
async def test_parallel_tool_failure_does_not_cancel_other_tool(tmp_path):
    client = FakeOpenAI()

    async def create(**kwargs):
        client.responses.calls.append(kwargs)
        if len(client.responses.calls) == 1:
            return SimpleNamespace(
                id="resp_tools",
                output=[
                    SimpleNamespace(
                        type="function_call",
                        name="fail_tool",
                        call_id="call_fail",
                        arguments="{}",
                    ),
                    SimpleNamespace(
                        type="function_call",
                        name="ok_tool",
                        call_id="call_ok",
                        arguments="{}",
                    ),
                ],
                output_text="",
                conversation=kwargs.get("conversation"),
                usage=None,
            )
        return SimpleNamespace(
            id="resp_final",
            output=[],
            output_text="done",
            conversation=kwargs.get("conversation"),
            usage=None,
        )

    client.responses.create = create
    brain = Brain(BrainConfig(database_path=tmp_path / "brain.sqlite3"), client=client)

    @brain.tools.register
    async def fail_tool() -> str:
        raise RuntimeError("boom")

    @brain.tools.register
    async def ok_tool() -> str:
        await asyncio.sleep(0.01)
        return "ok"

    response = await brain.ask(
        "call both",
        thread_id="thread-tool-failure",
        tool_names=["fail_tool", "ok_tool"],
    )

    by_call_id = {item["call_id"]: item for item in response.tool_results}
    assert by_call_id["call_fail"]["ok"] is False
    assert by_call_id["call_ok"]["ok"] is True
