from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from types import SimpleNamespace
from typing import Any, Literal


class OpenAIGateway:
    """Thin async gateway over the official OpenAI SDK.

    This class intentionally does not validate or narrow request parameters.
    The Brain layer can set defaults, while callers can still pass through the
    full Responses/Conversations API surface exposed by the installed SDK.
    """

    def __init__(
        self,
        client: Any,
        *,
        stream_transport: Literal["http", "websocket"] = "http",
        websocket_pool_size: int = 4,
    ):
        self.client = client
        self.stream_transport = stream_transport
        self.websocket_pool_size = max(1, websocket_pool_size)
        self._responses_ws_pool: list[Any | None] = [None] * self.websocket_pool_size
        self._responses_ws_locks = [asyncio.Lock() for _ in range(self.websocket_pool_size)]
        self._responses_ws_cursor = 0
        self._responses_ws_cursor_lock = asyncio.Lock()

    async def create_response(self, **params: Any) -> Any:
        return await self.client.responses.create(**params)

    async def stream_response(self, **params: Any) -> Any:
        if self.stream_transport == "websocket":
            return self._stream_response_websocket(params)
        params["stream"] = True
        return await self.client.responses.create(**params)

    def response_stream_manager(self, **params: Any) -> Any:
        return self.client.responses.stream(**params)

    async def close(self) -> None:
        for slot in range(self.websocket_pool_size):
            await self._close_responses_websocket_slot(slot)

    async def warmup(self) -> None:
        if self.stream_transport == "websocket":
            async with self._responses_ws_locks[0]:
                await self._ensure_responses_websocket(0)

    async def _stream_response_websocket(self, params: dict[str, Any]) -> Any:
        slot = await self._acquire_responses_websocket_slot()
        lock = self._responses_ws_locks[slot]
        payload = dict(params)
        payload.pop("stream", None)
        payload.pop("background", None)
        payload["type"] = "response.create"
        started = False
        last_error: Exception | None = None
        try:
            for attempt in range(2):
                try:
                    websocket = await self._ensure_responses_websocket(slot)
                    await websocket.send(json.dumps(payload))
                    while True:
                        raw = await websocket.recv()
                        event = _json_event_to_namespace(json.loads(raw))
                        started = True
                        yield event
                        event_type = getattr(event, "type", "")
                        if event_type in {
                            "response.completed",
                            "response.failed",
                            "response.cancelled",
                            "error",
                        }:
                            return
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    last_error = exc
                    await self._close_responses_websocket_slot(slot)
                    if started or attempt:
                        raise
                    continue
            if last_error is not None:
                raise last_error
        finally:
            lock.release()

    async def _next_responses_websocket_slot(self) -> int:
        async with self._responses_ws_cursor_lock:
            start = self._responses_ws_cursor
            for offset in range(self.websocket_pool_size):
                slot = (start + offset) % self.websocket_pool_size
                if not self._responses_ws_locks[slot].locked():
                    self._responses_ws_cursor = (slot + 1) % self.websocket_pool_size
                    return slot
            slot = start
            self._responses_ws_cursor = (slot + 1) % self.websocket_pool_size
            return slot

    async def _acquire_responses_websocket_slot(self) -> int:
        wait_slot: int | None = None
        async with self._responses_ws_cursor_lock:
            start = self._responses_ws_cursor
            for offset in range(self.websocket_pool_size):
                slot = (start + offset) % self.websocket_pool_size
                lock = self._responses_ws_locks[slot]
                if not lock.locked():
                    await lock.acquire()
                    self._responses_ws_cursor = (slot + 1) % self.websocket_pool_size
                    return slot
            wait_slot = start
            self._responses_ws_cursor = (wait_slot + 1) % self.websocket_pool_size
        await self._responses_ws_locks[wait_slot].acquire()
        return wait_slot

    async def _ensure_responses_websocket(self, slot: int) -> Any:
        websocket = self._responses_ws_pool[slot]
        if _websocket_is_open(websocket):
            return websocket
        await self._close_responses_websocket_slot(slot)
        websocket = await self._connect_responses_websocket()
        self._responses_ws_pool[slot] = websocket
        return websocket

    async def _connect_responses_websocket(self) -> Any:
        import websockets

        return await websockets.connect(
            _responses_websocket_url(self.client),
            additional_headers=_websocket_headers(self.client),
            max_size=None,
        )

    async def _close_responses_websocket_slot(self, slot: int) -> None:
        websocket = self._responses_ws_pool[slot]
        self._responses_ws_pool[slot] = None
        if websocket is None:
            return
        close = getattr(websocket, "close", None)
        if close is None:
            return
        result = close()
        if hasattr(result, "__await__"):
            with suppress(Exception):
                await result

    async def parse_response(self, *, text_format: type[Any], **params: Any) -> Any:
        return await self.client.responses.parse(text_format=text_format, **params)

    async def compact_response(self, **params: Any) -> Any:
        return await self.client.responses.compact(**params)

    async def retrieve_response(self, response_id: str, **params: Any) -> Any:
        return await self.client.responses.retrieve(response_id, **params)

    async def cancel_response(self, response_id: str, **params: Any) -> Any:
        return await self.client.responses.cancel(response_id, **params)

    async def delete_response(self, response_id: str, **params: Any) -> Any:
        return await self.client.responses.delete(response_id, **params)

    async def list_response_input_items(self, response_id: str, **params: Any) -> list[Any]:
        paginator = self.client.responses.input_items.list(response_id, **params)
        return [item async for item in paginator]

    async def count_response_input_tokens(self, **params: Any) -> Any:
        return await self.client.responses.input_tokens.count(**params)

    async def create_conversation(self, **params: Any) -> Any:
        return await self.client.conversations.create(**params)

    async def retrieve_conversation(self, conversation_id: str, **params: Any) -> Any:
        return await self.client.conversations.retrieve(conversation_id, **params)

    async def update_conversation(self, conversation_id: str, **params: Any) -> Any:
        return await self.client.conversations.update(conversation_id, **params)

    async def delete_conversation(self, conversation_id: str, **params: Any) -> Any:
        return await self.client.conversations.delete(conversation_id, **params)

    async def create_conversation_items(self, conversation_id: str, **params: Any) -> Any:
        return await self.client.conversations.items.create(conversation_id, **params)

    async def list_conversation_items(self, conversation_id: str, **params: Any) -> list[Any]:
        paginator = self.client.conversations.items.list(conversation_id, **params)
        return [item async for item in paginator]

    async def retrieve_conversation_item(
        self,
        conversation_id: str,
        item_id: str,
        **params: Any,
    ) -> Any:
        return await self.client.conversations.items.retrieve(
            item_id,
            conversation_id=conversation_id,
            **params,
        )

    async def delete_conversation_item(
        self,
        conversation_id: str,
        item_id: str,
        **params: Any,
    ) -> Any:
        return await self.client.conversations.items.delete(
            item_id,
            conversation_id=conversation_id,
            **params,
        )


def _responses_websocket_url(client: Any) -> str:
    base_url = str(getattr(client, "base_url", "https://api.openai.com/v1/")).rstrip("/")
    if base_url.endswith("/responses"):
        url = base_url
    else:
        url = f"{base_url}/responses"
    if url.startswith("https://"):
        return "wss://" + url[len("https://") :]
    if url.startswith("http://"):
        return "ws://" + url[len("http://") :]
    return url


def _websocket_headers(client: Any) -> dict[str, str]:
    headers: dict[str, str] = {}
    for key, value in getattr(client, "default_headers", {}).items():
        if isinstance(value, str):
            headers[key] = value
    api_key = getattr(client, "api_key", None)
    if api_key and "Authorization" not in headers:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _websocket_is_open(websocket: Any | None) -> bool:
    if websocket is None:
        return False
    if getattr(websocket, "closed", False):
        return False
    if getattr(websocket, "close_code", None) is not None:
        return False
    state = getattr(websocket, "state", None)
    state_name = str(getattr(state, "name", state)).lower()
    if state_name in {"closing", "closed"}:
        return False
    return True


def _json_event_to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(
            **{key: _json_event_to_namespace(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return [_json_event_to_namespace(item) for item in value]
    return value
