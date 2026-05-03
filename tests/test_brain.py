from types import SimpleNamespace

import pytest

from aibrain import Brain, BrainConfig, Persona


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
