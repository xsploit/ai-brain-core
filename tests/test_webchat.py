from fastapi.testclient import TestClient

from types import SimpleNamespace

from aibrain import Brain, BrainConfig, NullSTT, NullTTS, STTConfig, TTSConfig
from aibrain.server import create_app


class FakeModels:
    async def list(self):
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
