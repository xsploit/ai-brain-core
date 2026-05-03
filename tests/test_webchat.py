from fastapi.testclient import TestClient

import asyncio
from types import SimpleNamespace

from aibrain import Brain, BrainConfig, NullSTT, NullTTS, STTConfig, TTSConfig
from aibrain.server import create_app


class FakeModels:
    def __init__(self):
        self.calls = 0

    async def list(self):
        self.calls += 1
        return SimpleNamespace(
            data=[
                SimpleNamespace(id="gpt-test-a"),
                SimpleNamespace(id="text-embedding-3-small"),
                SimpleNamespace(id="gpt-test-b"),
            ]
        )


class FakeClient:
    def __init__(self):
        self.models = FakeModels()


class SlowStreamResponses:
    async def create(self, **kwargs):
        class Stream:
            async def __aiter__(self):
                yield SimpleNamespace(type="response.output_text.delta", delta="started")
                await asyncio.sleep(10)

        return Stream()


class FakeConversations:
    async def create(self, **kwargs):
        return SimpleNamespace(id="conv_cancel")


class SlowStreamClient:
    def __init__(self):
        self.responses = SlowStreamResponses()
        self.conversations = FakeConversations()


def test_webchat_routes_are_served(tmp_path):
    brain = Brain(
        BrainConfig(database_path=tmp_path / "brain.sqlite3"),
        stt_provider=NullSTT(STTConfig(provider="null")),
        tts_provider=NullTTS(TTSConfig(provider="null")),
    )
    app = create_app(brain=brain)

    with TestClient(app) as client:
        root = client.get("/", follow_redirects=False)
        assert root.status_code in {302, 307}
        assert root.headers["location"] == "/webchat"

        page = client.get("/webchat")
        assert page.status_code == 200
        assert "AI Brain Core Console" in page.text

        script = client.get("/webchat/assets/app.js")
        assert script.status_code == 200
        assert "voiceSocket" in script.text
        assert "loadModels" in script.text

        styles = client.get("/webchat/assets/styles.css")
        assert styles.status_code == 200
        assert ".app-shell" in styles.text


def test_models_endpoint_lists_chat_models(tmp_path):
    brain = Brain(
        BrainConfig(database_path=tmp_path / "brain.sqlite3", default_model="gpt-default"),
        client=FakeClient(),
        stt_provider=NullSTT(STTConfig(provider="null")),
        tts_provider=NullTTS(TTSConfig(provider="null")),
    )
    app = create_app(brain=brain)

    with TestClient(app) as client:
        response = client.get("/models")

    assert response.status_code == 200
    model_ids = [model["id"] for model in response.json()]
    assert "gpt-default" in model_ids
    assert "gpt-test-a" in model_ids
    assert "gpt-test-b" in model_ids
    assert "text-embedding-3-small" not in model_ids


def test_models_endpoint_uses_cache(tmp_path):
    openai_client = FakeClient()
    brain = Brain(
        BrainConfig(
            database_path=tmp_path / "brain.sqlite3",
            default_model="gpt-default",
            models_cache_ttl_seconds=300,
        ),
        client=openai_client,
        stt_provider=NullSTT(STTConfig(provider="null")),
        tts_provider=NullTTS(TTSConfig(provider="null")),
    )
    app = create_app(brain=brain)

    with TestClient(app) as client:
        first = client.get("/models")
        second = client.get("/models")

    assert first.status_code == 200
    assert second.status_code == 200
    assert openai_client.models.calls == 1


def test_stream_websocket_cancel_stops_active_turn(tmp_path):
    brain = Brain(
        BrainConfig(database_path=tmp_path / "brain.sqlite3"),
        client=SlowStreamClient(),
        stt_provider=NullSTT(STTConfig(provider="null")),
        tts_provider=NullTTS(TTSConfig(provider="null")),
    )
    app = create_app(brain=brain)

    with TestClient(app) as client:
        with client.websocket_connect("/stream") as ws:
            ws.send_json(
                {
                    "type": "ask",
                    "text": "slow",
                    "thread_id": "cancel:test",
                    "tool_names": [],
                }
            )
            assert ws.receive_json()["type"] == "text.delta"
            ws.send_json({"type": "cancel"})
            assert ws.receive_json()["type"] == "cancelled"
