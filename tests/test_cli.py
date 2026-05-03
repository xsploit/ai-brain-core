from types import SimpleNamespace

from aibrain.cli import FAST_CHAT_MODEL, _chat_brain_options, _chat_config_kwargs


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

    assert config_kwargs["openai_stream_transport"] == "websocket"
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
