import asyncio
import json
from types import SimpleNamespace

import pytest

from aibrain import BaseTTSProvider, Brain, BrainConfig, TTSChunk
from aibrain import tts as tts_module
from aibrain.tts import (
    PiperExecutableTTS,
    PiperProcessTTS,
    PiperVoice,
    SentenceChunker,
    TTSConfig,
    split_tts_text,
    tts_config_for_voice,
    with_env_overrides,
)


class FakeTTS(BaseTTSProvider):
    async def stream(self, text: str, **options):
        if text.startswith("Hello"):
            await asyncio.sleep(0.02)
        yield TTSChunk(audio=b"audio:" + text.encode(), sample_rate=22050, index=0, final=True)


class FakeStreamResponses:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)

        class Stream:
            async def __aiter__(self):
                yield SimpleNamespace(type="response.output_text.delta", delta="Hello. ")
                yield SimpleNamespace(type="response.output_text.delta", delta="Bye.")
                yield SimpleNamespace(
                    type="response.completed",
                    response=SimpleNamespace(
                        id="resp_1",
                        output=[],
                        output_text="Hello. Bye.",
                        conversation=kwargs.get("conversation"),
                        usage=None,
                    ),
                )

        return Stream()


class FakeConversations:
    async def create(self, **kwargs):
        return SimpleNamespace(id="conv_1")


class FakeClient:
    def __init__(self):
        self.responses = FakeStreamResponses()
        self.conversations = FakeConversations()


class SlowStoryResponses:
    async def create(self, **kwargs):
        class Stream:
            async def __aiter__(self):
                yield SimpleNamespace(type="response.output_text.delta", delta="First sentence. ")
                await asyncio.sleep(0.05)
                yield SimpleNamespace(type="response.output_text.delta", delta="Second sentence.")
                yield SimpleNamespace(
                    type="response.completed",
                    response=SimpleNamespace(
                        id="resp_story",
                        output=[],
                        output_text="First sentence. Second sentence.",
                        conversation=kwargs.get("conversation"),
                        usage=None,
                    ),
                )

        return Stream()


class SlowStoryClient:
    def __init__(self):
        self.responses = SlowStoryResponses()
        self.conversations = FakeConversations()


class BurstAfterFirstSentenceResponses:
    async def create(self, **kwargs):
        class Stream:
            async def __aiter__(self):
                yield SimpleNamespace(type="response.output_text.delta", delta="First sentence. ")
                for index in range(40):
                    yield SimpleNamespace(type="response.output_text.delta", delta=f"word{index} ")
                yield SimpleNamespace(
                    type="response.completed",
                    response=SimpleNamespace(
                        id="resp_burst",
                        output=[],
                        output_text="First sentence. " + " ".join(f"word{index}" for index in range(40)),
                        conversation=kwargs.get("conversation"),
                        usage=None,
                    ),
                )

        return Stream()


class BurstAfterFirstSentenceClient:
    def __init__(self):
        self.responses = BurstAfterFirstSentenceResponses()
        self.conversations = FakeConversations()


def test_sentence_chunker_splits_early():
    chunker = SentenceChunker(max_chars=12)
    assert chunker.feed("Hello. next") == ["Hello."]
    assert chunker.flush() == "next"


def test_split_tts_text_keeps_sentence_chunks():
    config = TTSConfig(provider="null", chunk_chars=180, min_chunk_chars=6)
    assert split_tts_text("First sentence. Second sentence.", config) == [
        "First sentence.",
        "Second sentence.",
    ]


def test_split_tts_text_returns_no_chunks_for_empty_text():
    assert split_tts_text("   ", TTSConfig(provider="null")) == []


def test_piper_env_overrides(monkeypatch):
    config = TTSConfig(
        piper_executable_path="C:/piper/piper.exe",
        piper_model_path="C:/piper/voice.onnx",
        piper_config_path="C:/piper/voice.onnx.json",
        piper_espeak_data_path=None,
    )
    monkeypatch.setenv("PIPER_MODEL", "C:/override/voice.onnx")
    merged = with_env_overrides(config)
    assert str(merged.piper_model_path).replace("\\", "/") == "C:/override/voice.onnx"


def test_default_piper_config_matches_explicit_model_env(tmp_path, monkeypatch):
    model = tmp_path / "voice.onnx"
    config_path = tmp_path / "voice.onnx.json"
    model.write_bytes(b"model")
    config_path.write_text('{"audio":{"sample_rate":22050}}', encoding="utf-8")
    monkeypatch.setenv("PIPER_MODEL", str(model))
    monkeypatch.delenv("PIPER_CONFIG", raising=False)
    monkeypatch.delenv("AIBRAIN_TTS_VOICE", raising=False)
    monkeypatch.delenv("PIPER_VOICE", raising=False)

    config = TTSConfig(provider="null")

    assert config.piper_model_path == model
    assert config.piper_config_path == config_path
    assert config.resolved_sample_rate() == 22050


def test_tts_defaults_do_not_use_user_specific_paths(monkeypatch):
    for name in [
        "PIPER_EXE",
        "PIPER_MODEL",
        "PIPER_CONFIG",
        "PIPER_ESPEAK_DATA",
        "AIBRAIN_TTS_VOICE_ROOTS",
        "AIBRAIN_TTS_MANIFESTS",
    ]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(tts_module.shutil, "which", lambda name: None)

    config = TTSConfig(provider="null")

    assert config.piper_executable_path is None
    assert config.piper_model_path is None
    assert config.piper_config_path is None
    assert config.piper_espeak_data_path is None


def test_tts_provider_env_controls_enabled(monkeypatch):
    monkeypatch.setenv("AIBRAIN_TTS_PROVIDER", "null")
    config = TTSConfig()
    assert config.provider == "null"
    assert config.enabled is False

    monkeypatch.delenv("AIBRAIN_TTS_PROVIDER", raising=False)
    monkeypatch.setenv("TTS_PROVIDER", "none")
    legacy = TTSConfig()
    assert legacy.provider == "null"
    assert legacy.enabled is False


def test_piper_idle_timeout_env_override(monkeypatch):
    monkeypatch.setenv("PIPER_PROCESS_IDLE_TIMEOUT", "0.7")
    assert TTSConfig(provider="null").process_idle_timeout == 0.7


def test_discover_piper_voices_uses_env_roots_and_manifests(tmp_path, monkeypatch):
    manifest_model = tmp_path / "manifest_voice.onnx"
    manifest_config = tmp_path / "manifest_voice.onnx.json"
    manifest_model.write_bytes(b"model")
    manifest_config.write_text('{"audio":{"sample_rate":22050}}', encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "slug": "manifest",
                    "label": "Manifest",
                    "onnx": str(manifest_model),
                    "config": str(manifest_config),
                }
            ]
        ),
        encoding="utf-8",
    )
    root = tmp_path / "voices"
    root.mkdir()
    rooted_model = root / "rooted.onnx"
    rooted_config = root / "rooted.onnx.json"
    rooted_model.write_bytes(b"model")
    rooted_config.write_text('{"audio":{"sample_rate":22050}}', encoding="utf-8")
    monkeypatch.setenv("AIBRAIN_TTS_MANIFESTS", str(manifest))
    monkeypatch.setenv("AIBRAIN_TTS_VOICE_ROOTS", str(root))

    voices = tts_module.discover_piper_voices()
    slugs = {voice.slug for voice in voices}

    assert {"manifest", "rooted"} <= slugs


class FakePiperProcess(PiperProcessTTS):
    def __init__(self):
        super().__init__(TTSConfig(provider="null"))
        self.segments = []
        self.used_models = []

    async def _stream_process(self, text: str, *, config=None, start_index: int = 0):
        runtime_config = config or self.config
        self.segments.append(text)
        self.used_models.append(runtime_config.piper_model_path)
        yield TTSChunk(
            audio=text.encode(),
            sample_rate=runtime_config.resolved_sample_rate(),
            index=start_index,
            final=True,
            text=text,
            voice=str(runtime_config.piper_model_path) if runtime_config.piper_model_path else None,
        )


class PartialFailurePiperProcess(PiperProcessTTS):
    def __init__(self):
        super().__init__(TTSConfig(provider="null"))
        self.fallback_calls = 0

    async def _stream_process(self, text: str, *, config=None, start_index: int = 0):
        yield TTSChunk(
            audio=b"partial",
            sample_rate=22050,
            index=start_index,
            final=False,
            text=text,
        )
        raise RuntimeError("process failed after audio")

    async def _run_piper(self, text: str, *, output_raw: bool, config=None) -> bytes:
        self.fallback_calls += 1
        return b"fallback"


class ConcurrentFakePiperProcess(PiperProcessTTS):
    def __init__(self):
        super().__init__(TTSConfig(provider="null"))
        self.active = 0
        self.max_active = 0
        self.entered = asyncio.Event()
        self.concurrent = asyncio.Event()
        self.release = asyncio.Event()

    async def _stream_process(self, text: str, *, config=None, start_index: int = 0):
        runtime_config = config or self.config
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.entered.set()
        if self.active >= 2:
            self.concurrent.set()
        await self.release.wait()
        try:
            yield TTSChunk(
                audio=text.encode(),
                sample_rate=runtime_config.resolved_sample_rate(),
                index=start_index,
                final=True,
                text=text,
                voice=str(runtime_config.piper_model_path) if runtime_config.piper_model_path else None,
            )
        finally:
            self.active -= 1


@pytest.mark.asyncio
async def test_piper_process_stream_splits_multi_sentence_text():
    provider = FakePiperProcess()

    chunks = [chunk async for chunk in provider.stream("One sentence. Two sentence.")]

    assert provider.segments == ["One sentence.", "Two sentence."]
    assert [chunk.index for chunk in chunks] == [0, 1]


@pytest.mark.asyncio
async def test_piper_process_does_not_fallback_after_partial_audio():
    provider = PartialFailurePiperProcess()
    stream = provider.stream("One sentence.")

    first = await anext(stream)
    assert first.audio == b"partial"
    with pytest.raises(RuntimeError, match="process failed"):
        await anext(stream)
    assert provider.fallback_calls == 0


@pytest.mark.asyncio
async def test_piper_process_locks_per_voice(tmp_path):
    first_model = tmp_path / "first.onnx"
    first_config = tmp_path / "first.onnx.json"
    second_model = tmp_path / "second.onnx"
    second_config = tmp_path / "second.onnx.json"
    for model, config in [(first_model, first_config), (second_model, second_config)]:
        model.write_bytes(b"model")
        config.write_text('{"audio":{"sample_rate":22050}}', encoding="utf-8")
    provider = ConcurrentFakePiperProcess()

    async def collect(text, model, config):
        return [
            chunk
            async for chunk in provider.stream(
                text,
                voice={"path": model, "config": config},
            )
        ]

    first = asyncio.create_task(collect("one", first_model, first_config))
    await asyncio.wait_for(provider.entered.wait(), timeout=1)
    second = asyncio.create_task(collect("two", second_model, second_config))
    await asyncio.wait_for(provider.concurrent.wait(), timeout=1)
    provider.release.set()
    await asyncio.gather(first, second)

    assert provider.max_active == 2


def test_tts_config_for_voice_resolves_requested_voice(tmp_path, monkeypatch):
    model = tmp_path / "voice.onnx"
    config_path = tmp_path / "voice.onnx.json"
    model.write_bytes(b"model")
    config_path.write_text('{"audio":{"sample_rate":44100}}', encoding="utf-8")

    monkeypatch.setattr(
        tts_module,
        "resolve_piper_voice",
        lambda voice_id=None: PiperVoice(
            slug="voice-a",
            label="Voice A",
            onnx=model,
            config=config_path,
        ),
    )

    resolved = tts_config_for_voice(TTSConfig(provider="null"), "voice-a")

    assert resolved.piper_model_path == model
    assert resolved.piper_config_path == config_path
    assert resolved.resolved_sample_rate() == 44100


def test_tts_config_for_voice_accepts_direct_model_path(tmp_path):
    model = tmp_path / "direct.onnx"
    config_path = tmp_path / "direct.onnx.json"
    model.write_bytes(b"model")
    config_path.write_text('{"audio":{"sample_rate":24000}}', encoding="utf-8")

    resolved = tts_config_for_voice(
        TTSConfig(provider="null"),
        {"path": model, "config": config_path},
    )

    assert resolved.piper_model_path == model
    assert resolved.piper_config_path == config_path
    assert resolved.resolved_sample_rate() == 24000


class FakeExecutable(PiperExecutableTTS):
    def __init__(self, config):
        super().__init__(config)
        self.used_models = []

    async def _run_piper(self, text: str, *, output_raw: bool, config=None) -> bytes:
        self.used_models.append(config.piper_model_path)
        return b"audio"


@pytest.mark.asyncio
async def test_piper_executable_uses_requested_voice(tmp_path, monkeypatch):
    default_model = tmp_path / "default.onnx"
    selected_model = tmp_path / "selected.onnx"
    selected_config = tmp_path / "selected.onnx.json"
    default_model.write_bytes(b"default")
    selected_model.write_bytes(b"selected")
    selected_config.write_text('{"audio":{"sample_rate":32000}}', encoding="utf-8")
    monkeypatch.setattr(
        tts_module,
        "resolve_piper_voice",
        lambda voice_id=None: PiperVoice(
            slug="selected",
            label="Selected",
            onnx=selected_model,
            config=selected_config,
        ),
    )
    provider = FakeExecutable(
        TTSConfig(
            provider="piper_executable",
            piper_executable_path=tmp_path / "piper.exe",
            piper_model_path=default_model,
            piper_config_path=None,
        )
    )

    chunks = [chunk async for chunk in provider.stream("hello", voice="selected")]

    assert provider.used_models == [selected_model]
    assert chunks[0].sample_rate == 32000
    assert chunks[0].voice == str(selected_model)


@pytest.mark.asyncio
async def test_piper_process_uses_requested_voice(tmp_path, monkeypatch):
    selected_model = tmp_path / "selected.onnx"
    selected_config = tmp_path / "selected.onnx.json"
    selected_model.write_bytes(b"selected")
    selected_config.write_text('{"audio":{"sample_rate":32000}}', encoding="utf-8")
    monkeypatch.setattr(
        tts_module,
        "resolve_piper_voice",
        lambda voice_id=None: PiperVoice(
            slug="selected",
            label="Selected",
            onnx=selected_model,
            config=selected_config,
        ),
    )
    provider = FakePiperProcess()

    audio = await provider.synthesize("hello", voice="selected")

    assert provider.used_models == [selected_model]
    assert audio.sample_rate == 32000
    assert audio.voice == str(selected_model)


@pytest.mark.asyncio
async def test_stream_with_tts_emits_audio(tmp_path):
    brain = Brain(
        BrainConfig(database_path=tmp_path / "brain.sqlite3"),
        client=FakeClient(),
        tts_provider=FakeTTS(TTSConfig(provider="null")),
    )

    events = [
        event
        async for event in brain.stream_with_tts(
            "hello",
            thread_id="thread-tts",
            tool_names=[],
        )
    ]

    assert "text.delta" in [event.type for event in events]
    assert "tts.audio" in [event.type for event in events]
    assert events[0].type == "tts.playlist.start"
    audio_segments = [
        event.data["segment_index"]
        for event in events
        if event.type == "tts.audio"
    ]
    assert audio_segments == sorted(audio_segments)
    assert audio_segments == [0, 1]
    assert [event.type for event in events].count("tts.playlist.done") == 1
    assert "response.done" in [event.type for event in events]
    assert [event.type for event in events].index("response.done") < [event.type for event in events].index(
        "tts.playlist.done"
    )
    assert events[-1].type == "tts.playlist.done"


@pytest.mark.asyncio
async def test_stream_with_tts_starts_audio_before_response_finishes(tmp_path):
    brain = Brain(
        BrainConfig(database_path=tmp_path / "brain.sqlite3"),
        client=SlowStoryClient(),
        tts_provider=FakeTTS(TTSConfig(provider="null")),
    )

    events = [
        event
        async for event in brain.stream_with_tts(
            "tell me a story",
            thread_id="thread-story",
            tool_names=[],
        )
    ]

    first_audio_index = next(
        index for index, event in enumerate(events) if event.type == "tts.audio"
    )
    second_delta_index = next(
        index
        for index, event in enumerate(events)
        if event.type == "text.delta" and event.data.get("text") == "Second sentence."
    )
    response_done_index = next(
        index for index, event in enumerate(events) if event.type == "response.done"
    )

    assert first_audio_index < second_delta_index
    assert first_audio_index < response_done_index


@pytest.mark.asyncio
async def test_stream_with_tts_does_not_starve_audio_behind_text_burst(tmp_path):
    brain = Brain(
        BrainConfig(database_path=tmp_path / "brain.sqlite3"),
        client=BurstAfterFirstSentenceClient(),
        tts_provider=FakeTTS(TTSConfig(provider="null")),
    )

    events = [
        event
        async for event in brain.stream_with_tts(
            "burst",
            thread_id="thread-burst",
            tool_names=[],
        )
    ]

    first_audio_index = next(
        index for index, event in enumerate(events) if event.type == "tts.audio"
    )
    text_before_audio = sum(
        1 for event in events[:first_audio_index] if event.type == "text.delta"
    )

    assert text_before_audio <= 9


@pytest.mark.asyncio
async def test_stream_with_tts_reports_queue_overflow(tmp_path):
    brain = Brain(
        BrainConfig(database_path=tmp_path / "brain.sqlite3", stream_event_queue_max=2),
        client=BurstAfterFirstSentenceClient(),
        tts_provider=FakeTTS(TTSConfig(provider="null")),
    )

    events = []
    async for event in brain.stream_with_tts(
        "burst",
        thread_id="thread-overflow",
        tool_names=[],
    ):
        events.append(event)
        await asyncio.sleep(0.01)
        if event.type == "error":
            break

    assert any(
        event.type == "error" and "overflowed" in event.data.get("message", "")
        for event in events
    )
