from __future__ import annotations

import asyncio
import json
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
    ):
        self.client = client
        self.stream_transport = stream_transport
        self._responses_ws: Any | None = None
        self._responses_ws_lock = asyncio.Lock()

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
        websocket = self._responses_ws
        self._responses_ws = None
        if websocket is not None:
            await websocket.close()

    async def warmup(self) -> None:
        if self.stream_transport == "websocket":
            async with self._responses_ws_lock:
                await self._ensure_responses_websocket()

    async def _stream_response_websocket(self, params: dict[str, Any]) -> Any:
        async with self._responses_ws_lock:
            websocket = await self._ensure_responses_websocket()
            payload = dict(params)
            payload.pop("stream", None)
            payload.pop("background", None)
            payload["type"] = "response.create"
            await websocket.send(json.dumps(payload))
            while True:
                raw = await websocket.recv()
                event = _json_event_to_namespace(json.loads(raw))
                yield event
                event_type = getattr(event, "type", "")
                if event_type in {"response.completed", "response.failed", "response.cancelled", "error"}:
                    break

    async def _ensure_responses_websocket(self) -> Any:
        if self._responses_ws is not None:
            return self._responses_ws
        import websockets

        self._responses_ws = await websockets.connect(
            _responses_websocket_url(self.client),
            additional_headers=_websocket_headers(self.client),
            max_size=None,
        )
        return self._responses_ws

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


def _json_event_to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(
            **{key: _json_event_to_namespace(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return [_json_event_to_namespace(item) for item in value]
    return value
