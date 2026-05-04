import asyncio
import logging
from types import SimpleNamespace

import pytest

from aibrain import OpenAIGateway
from aibrain.gateway import (
    _json_event_to_namespace,
    _responses_websocket_url,
    _websocket_headers,
    _websocket_is_open,
)


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


class FakeWebSocket:
    def __init__(self, *, recv_items=None, send_error=None, recv_error=None, closed=False):
        self.recv_items = list(recv_items or [])
        self.sent = []
        self.close_calls = 0
        self.send_error = send_error
        self.recv_error = recv_error
        self.closed = closed

    async def send(self, payload):
        if self.send_error is not None:
            raise self.send_error
        self.sent.append(payload)

    async def recv(self):
        if self.recv_error is not None:
            raise self.recv_error
        if not self.recv_items:
            raise RuntimeError("no recv items")
        item = self.recv_items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def close(self):
        self.close_calls += 1
        self.closed = True


class FailingCloseWebSocket:
    async def close(self):
        raise RuntimeError("close failed")


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


def test_websocket_is_open_detects_closed_sockets():
    assert _websocket_is_open(None) is False
    assert _websocket_is_open(SimpleNamespace(closed=True)) is False
    assert _websocket_is_open(SimpleNamespace(close_code=1000)) is False
    assert _websocket_is_open(SimpleNamespace(state=SimpleNamespace(name="CLOSED"))) is False
    assert _websocket_is_open(SimpleNamespace()) is True


@pytest.mark.asyncio
async def test_gateway_websocket_pool_skips_busy_slot():
    gateway = OpenAIGateway(
        SimpleNamespace(responses=FakeResponses(), conversations=FakeConversations()),
        stream_transport="websocket",
        websocket_pool_size=2,
    )

    await gateway._responses_ws_locks[0].acquire()
    try:
        slot = await gateway._next_responses_websocket_slot()
    finally:
        gateway._responses_ws_locks[0].release()

    assert slot == 1


@pytest.mark.asyncio
async def test_gateway_websocket_slot_reservation_locks_selected_slot():
    gateway = OpenAIGateway(
        SimpleNamespace(responses=FakeResponses(), conversations=FakeConversations()),
        stream_transport="websocket",
        websocket_pool_size=2,
    )

    await gateway._responses_ws_locks[0].acquire()
    try:
        slot = await gateway._acquire_responses_websocket_slot()
    finally:
        gateway._responses_ws_locks[0].release()

    try:
        assert slot == 1
        assert gateway._responses_ws_locks[1].locked()
    finally:
        gateway._responses_ws_locks[1].release()
        gateway._responses_ws_slot_sem.release()


@pytest.mark.asyncio
async def test_gateway_slot_acquire_releases_semaphore_when_cancelled():
    gateway = OpenAIGateway(
        SimpleNamespace(responses=FakeResponses(), conversations=FakeConversations()),
        stream_transport="websocket",
        websocket_pool_size=1,
    )

    await gateway._responses_ws_cursor_lock.acquire()
    task = asyncio.create_task(gateway._acquire_responses_websocket_slot())
    for _ in range(10):
        await asyncio.sleep(0)
        if getattr(gateway._responses_ws_slot_sem, "_value", None) == 0:
            break
    assert getattr(gateway._responses_ws_slot_sem, "_value", None) == 0

    task.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        gateway._responses_ws_cursor_lock.release()

    await asyncio.wait_for(gateway._responses_ws_slot_sem.acquire(), timeout=1)
    gateway._responses_ws_slot_sem.release()


@pytest.mark.asyncio
async def test_gateway_reconnects_stale_websocket_before_request(monkeypatch):
    gateway = OpenAIGateway(
        SimpleNamespace(responses=FakeResponses(), conversations=FakeConversations()),
        stream_transport="websocket",
        websocket_pool_size=1,
    )
    stale = FakeWebSocket(closed=True)
    fresh = FakeWebSocket(
        recv_items=[
            '{"type":"response.completed","response":{"id":"resp_1","output":[]}}'
        ]
    )
    gateway._responses_ws_pool[0] = stale

    async def connect():
        return fresh

    monkeypatch.setattr(gateway, "_connect_responses_websocket", connect)

    events = [event async for event in gateway._stream_response_websocket({"model": "gpt"})]

    assert stale.close_calls == 1
    assert fresh.sent
    assert events[0].response.id == "resp_1"


@pytest.mark.asyncio
async def test_gateway_retries_once_if_websocket_fails_before_stream(monkeypatch):
    gateway = OpenAIGateway(
        SimpleNamespace(responses=FakeResponses(), conversations=FakeConversations()),
        stream_transport="websocket",
        websocket_pool_size=1,
    )
    broken = FakeWebSocket(send_error=RuntimeError("stale"))
    fresh = FakeWebSocket(
        recv_items=[
            '{"type":"response.completed","response":{"id":"resp_retry","output":[]}}'
        ]
    )
    sockets = iter([broken, fresh])

    async def connect():
        return next(sockets)

    monkeypatch.setattr(gateway, "_connect_responses_websocket", connect)

    events = [event async for event in gateway._stream_response_websocket({"model": "gpt"})]

    assert broken.close_calls == 1
    assert fresh.sent
    assert events[0].response.id == "resp_retry"


@pytest.mark.asyncio
async def test_gateway_does_not_retry_after_stream_started(monkeypatch):
    gateway = OpenAIGateway(
        SimpleNamespace(responses=FakeResponses(), conversations=FakeConversations()),
        stream_transport="websocket",
        websocket_pool_size=1,
    )
    first = FakeWebSocket(
        recv_items=[
            '{"type":"response.output_text.delta","delta":"hi"}',
            RuntimeError("drop"),
        ]
    )
    second = FakeWebSocket(
        recv_items=[
            '{"type":"response.completed","response":{"id":"resp_duplicate","output":[]}}'
        ]
    )
    sockets = iter([first, second])

    async def connect():
        return next(sockets)

    monkeypatch.setattr(gateway, "_connect_responses_websocket", connect)

    stream = gateway._stream_response_websocket({"model": "gpt"})
    first_event = await anext(stream)
    assert first_event.type == "response.output_text.delta"
    with pytest.raises(RuntimeError, match="drop"):
        await anext(stream)
    assert second.sent == []


@pytest.mark.asyncio
async def test_gateway_closes_websocket_when_stream_is_abandoned(monkeypatch):
    gateway = OpenAIGateway(
        SimpleNamespace(responses=FakeResponses(), conversations=FakeConversations()),
        stream_transport="websocket",
        websocket_pool_size=1,
    )
    socket = FakeWebSocket(
        recv_items=[
            '{"type":"response.output_text.delta","delta":"hi"}',
            '{"type":"response.completed","response":{"id":"resp_1","output":[]}}',
        ]
    )

    async def connect():
        return socket

    monkeypatch.setattr(gateway, "_connect_responses_websocket", connect)

    stream = gateway._stream_response_websocket({"model": "gpt"})
    event = await anext(stream)
    assert event.type == "response.output_text.delta"

    await stream.aclose()

    assert socket.close_calls == 1
    assert gateway._responses_ws_pool[0] is None


@pytest.mark.asyncio
async def test_gateway_logs_websocket_close_failures(caplog):
    gateway = OpenAIGateway(
        SimpleNamespace(responses=FakeResponses(), conversations=FakeConversations()),
        stream_transport="websocket",
        websocket_pool_size=1,
    )
    gateway._responses_ws_pool[0] = FailingCloseWebSocket()

    caplog.set_level(logging.WARNING, logger="aibrain.gateway")
    await gateway._close_responses_websocket_slot(0)

    assert gateway._responses_ws_pool[0] is None
    assert "Failed to close responses websocket slot 0" in caplog.text
