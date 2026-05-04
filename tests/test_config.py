from aibrain import BrainConfig


def test_brain_config_reads_memory_and_voice_socket_env(monkeypatch):
    monkeypatch.setenv("AIBRAIN_MEMORY_VEC_OVERFETCH", "7")
    monkeypatch.setenv("AIBRAIN_VOICE_SOCKET_MAX_MESSAGE_BYTES", "1234")

    config = BrainConfig()

    assert config.memory_vec_overfetch == 7
    assert config.voice_socket_max_message_bytes == 1234
