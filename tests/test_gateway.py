from types import SimpleNamespace

import pytest

from aibrain import OpenAIGateway
from aibrain.gateway import _json_event_to_namespace, _responses_websocket_url, _websocket_headers


class FakeResponses:
    def __init__(self):
        self.input_items = SimpleNamespace(list=self._list_input_items)
        self.input_tokens = SimpleNamespace(count=self._count_input_tokens)

    async def retrieve(self, response_id, **kwargs):
        return ("retrieve", response_id, kwargs)

    async def cancel(self, response_id, **kwargs):
        return ("cancel", response_id, kwargs)

    async def delete(self, response_id, **kwargs):
        return ("delete", response_id, kwargs)

    def _list_input_items(self, response_id, **kwargs):
        async def iterator():
            yield ("input_item", response_id, kwargs)

        return iterator()

    async def _count_input_tokens(self, **kwargs):
        return ("input_tokens", kwargs)


class FakeItems:
    async def create(self, conversation_id, **kwargs):
        return ("items.create", conversation_id, kwargs)


class FakeConversations:
    def __init__(self):
        self.items = FakeItems()

    async def retrieve(self, conversation_id, **kwargs):
        return ("conversation", conversation_id, kwargs)


@pytest.mark.asyncio
async def test_gateway_exposes_response_and_conversation_methods():
    gateway = OpenAIGateway(
        SimpleNamespace(responses=FakeResponses(), conversations=FakeConversations())
    )

    assert await gateway.retrieve_response("resp_1", include=["x"]) == (
        "retrieve",
        "resp_1",
        {"include": ["x"]},
    )
    assert await gateway.cancel_response("resp_1") == ("cancel", "resp_1", {})
    assert await gateway.retrieve_conversation("conv_1") == ("conversation", "conv_1", {})
    assert await gateway.create_conversation_items("conv_1", items=[]) == (
        "items.create",
        "conv_1",
        {"items": []},
    )
    assert await gateway.list_response_input_items("resp_1") == [
        ("input_item", "resp_1", {})
    ]
    assert await gateway.count_response_input_tokens(model="gpt-5-nano", input="hi") == (
        "input_tokens",
        {"model": "gpt-5-nano", "input": "hi"},
    )


def test_gateway_builds_responses_websocket_url_and_headers():
    client = SimpleNamespace(
        base_url="https://api.openai.com/v1/",
        api_key="sk-test",
        default_headers={
            "Authorization": "Bearer sk-test",
            "OpenAI-Organization": object(),
        },
    )

    assert _responses_websocket_url(client) == "wss://api.openai.com/v1/responses"
    assert _websocket_headers(client) == {"Authorization": "Bearer sk-test"}


def test_json_event_to_namespace_recurses():
    event = _json_event_to_namespace(
        {
            "type": "response.completed",
            "response": {"id": "resp_1", "output": [{"type": "message"}]},
        }
    )

    assert event.type == "response.completed"
    assert event.response.id == "resp_1"
    assert event.response.output[0].type == "message"
