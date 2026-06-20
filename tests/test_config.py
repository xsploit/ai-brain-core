from aibrain import BrainConfig


def test_brain_config_reads_memory_and_voice_socket_env(monkeypatch):
    monkeypatch.setenv("AIBRAIN_MEMORY_VEC_OVERFETCH", "7")
    monkeypatch.setenv("AIBRAIN_VOICE_SOCKET_MAX_MESSAGE_BYTES", "1234")

    config = BrainConfig()

    assert config.memory_vec_overfetch == 7
    assert config.voice_socket_max_message_bytes == 1234


def test_brain_config_defaults_to_vercel_local_deepseek(monkeypatch):
    monkeypatch.delenv("AIBRAIN_PROVIDER", raising=False)
    monkeypatch.delenv("AIBRAIN_STATE_MODE", raising=False)
    monkeypatch.delenv("AI_BRAIN_MODEL", raising=False)
    monkeypatch.delenv("AIBRAIN_MODEL", raising=False)
    monkeypatch.delenv("AIBRAIN_EMBEDDING_MODEL", raising=False)

    config = BrainConfig()

    assert config.provider == "vercel"
    assert config.state_mode == "local"
    assert config.default_model == "deepseek/deepseek-v4-flash"
    assert config.embedding_model == "openai/text-embedding-3-small"
    assert config.store is False
