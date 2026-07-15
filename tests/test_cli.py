import os
from types import SimpleNamespace

from aibrain.cli import FAST_CHAT_MODEL, _chat_brain_options, _chat_config_kwargs, piper_models, serve


def test_fast_chat_preset_uses_low_latency_defaults():
    args = SimpleNamespace(
        database="brain.sqlite3",
        env_file=None,
        stream_transport=None,
        model=None,
        fast=True,
        stateless=False,
        no_tools=False,
        tools=None,
        reasoning_effort=None,
        service_tier=None,
        max_output_tokens=None,
    )

    config_kwargs = _chat_config_kwargs(args)
    brain_options = _chat_brain_options(args)

    assert config_kwargs["openai_stream_transport"] == "http"
    assert config_kwargs["default_model"] == FAST_CHAT_MODEL
    assert config_kwargs["openai_ws_pool_size"] == 1
    assert brain_options["stateless"] is True
    assert brain_options["tool_names"] == []


def test_chat_tool_selection_can_be_explicit():
    args = SimpleNamespace(
        fast=False,
        stateless=False,
        no_tools=False,
        tools="current_time, brain_context",
        reasoning_effort=None,
        service_tier=None,
        max_output_tokens=64,
    )

    options = _chat_brain_options(args)

    assert options["stateless"] is False
    assert options["tool_names"] == ["current_time", "brain_context"]
    assert options["max_output_tokens"] == 64


def test_serve_loads_env_file_before_building_config(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "AI_BRAIN_MODEL=gpt-env",
                "AIBRAIN_OPENAI_STREAM_TRANSPORT=websocket",
            ]
        ),
        encoding="utf-8",
    )
    captured = {}
    monkeypatch.delenv("AI_BRAIN_MODEL", raising=False)
    monkeypatch.delenv("AIBRAIN_OPENAI_STREAM_TRANSPORT", raising=False)

    def fake_create_app(*, config):
        captured["config"] = config
        return object()

    def fake_run(app, *, host, port):
        captured["host"] = host
        captured["port"] = port

    monkeypatch.setattr("aibrain.cli.create_app", fake_create_app)
    monkeypatch.setattr("aibrain.cli.uvicorn.run", fake_run)

    serve(
        SimpleNamespace(
            database=tmp_path / "brain.sqlite3",
            env_file=env_file,
            model=None,
            stream_transport=None,
            host="127.0.0.1",
            port=8765,
        )
    )

    assert captured["config"].default_model == "gpt-env"
    assert captured["config"].openai_stream_transport == "websocket"
    assert captured["port"] == 8765
    os.environ.pop("AI_BRAIN_MODEL", None)
    os.environ.pop("AIBRAIN_OPENAI_STREAM_TRANSPORT", None)


def test_serve_defaults_to_client_chat_transport(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.delenv("AIBRAIN_OPENAI_STREAM_TRANSPORT", raising=False)

    def fake_create_app(*, config):
        captured["config"] = config
        return object()

    monkeypatch.setattr("aibrain.cli.create_app", fake_create_app)
    monkeypatch.setattr("aibrain.cli.uvicorn.run", lambda app, *, host, port: None)

    serve(
        SimpleNamespace(
            database=tmp_path / "brain.sqlite3",
            env_file=None,
            model=None,
            stream_transport=None,
            host="127.0.0.1",
            port=8765,
        )
    )

    assert captured["config"].openai_stream_transport == "http"


def test_piper_models_command_lists_discovered_voice(tmp_path, capsys):
    model = tmp_path / "voice.onnx"
    config = tmp_path / "voice.onnx.json"
    model.write_bytes(b"model")
    config.write_text('{"audio":{"sample_rate":22050}}', encoding="utf-8")

    piper_models(
        SimpleNamespace(
            root=[str(tmp_path)],
            manifest=None,
            refresh=True,
            json=False,
        )
    )

    output = capsys.readouterr().out
    assert "voice" in output
    assert str(model) in output
    assert str(config) in output


def test_piper_models_command_can_emit_json(tmp_path, capsys):
    model = tmp_path / "json_voice.onnx"
    model.write_bytes(b"model")

    piper_models(
        SimpleNamespace(
            root=[str(tmp_path)],
            manifest=None,
            refresh=True,
            json=True,
        )
    )

    output = capsys.readouterr().out
    assert '"slug": "json_voice"' in output
    assert str(model).replace("\\", "\\\\") in output
