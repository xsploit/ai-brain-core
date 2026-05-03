import base64
import wave
from array import array
from io import BytesIO
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from aibrain import (
    BaseSTTProvider,
    BaseTTSProvider,
    Brain,
    BrainConfig,
    EnergyVAD,
    NullTTS,
    STTResult,
    TTSChunk,
    TTSConfig,
    UtteranceBuffer,
    VADConfig,
)
from aibrain.server import create_app
from aibrain import stt as stt_module
from aibrain.stt import decode_wav_bytes, pcm_s16le_to_float32, resample_linear


def pcm(samples: list[int]) -> bytes:
    return array("h", samples).tobytes()


def wav_bytes(samples: list[int], *, sample_rate: int = 16000) -> bytes:
    output = BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm(samples))
    return output.getvalue()


class FakeSTT(BaseSTTProvider):
    async def transcribe(self, audio: bytes, **kwargs):
        return STTResult(
            text=kwargs.get("text", "hello from speech"),
            language=kwargs.get("language"),
            duration=0.1,
            provider="fake",
        )


class FakeTTS(BaseTTSProvider):
    async def stream(self, text: str, **options):
        yield TTSChunk(audio=b"audio:" + text.encode(), sample_rate=16000, final=True)


class FakeVAD:
    def __init__(self):
        self.warmed = False

    async def warmup(self):
        self.warmed = True

    def is_speech(self, audio: bytes, **kwargs):
        return False


class FakeStreamResponses:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)

        class Stream:
            async def __aiter__(self):
                yield SimpleNamespace(type="response.output_text.delta", delta="voice answer.")
                yield SimpleNamespace(
                    type="response.completed",
                    response=SimpleNamespace(
                        id="resp_voice",
                        output=[],
                        output_text="voice answer.",
                        conversation=kwargs.get("conversation"),
                        usage=None,
                    ),
                )

        return Stream()


class FakeConversations:
    async def create(self, **kwargs):
        return SimpleNamespace(id="conv_voice")


class FakeClient:
    def __init__(self):
        self.responses = FakeStreamResponses()
        self.conversations = FakeConversations()


def test_pcm_and_wav_decode():
    decoded = pcm_s16le_to_float32(pcm([0, 32767, -32768]))
    assert decoded[0] == 0
    assert decoded[1] > 0.99
    assert decoded[2] == -1.0

    wav = decode_wav_bytes(wav_bytes([1000, -1000], sample_rate=8000))
    assert wav.sample_rate == 8000
    assert len(wav.samples) == 2


def test_pcm_decode_and_resample_numpy_fast_path_matches_shape():
    decoded = pcm_s16le_to_float32(pcm([0, 16384, -16384, 32767]), channels=1)
    assert len(decoded) == 4
    assert abs(float(decoded[1]) - 0.5) < 0.01

    resampled = resample_linear(decoded, 16000, 8000)
    assert len(resampled) == 2
    assert abs(float(resampled[0])) < 0.01


def test_utterance_buffer_vad_endpointing():
    config = VADConfig(
        provider="energy",
        threshold=0.01,
        min_speech_ms=20,
        end_silence_ms=50,
        padding_ms=0,
    )
    buffer = UtteranceBuffer(vad=EnergyVAD(config), config=config, sample_rate=16000)

    first = buffer.push(pcm([9000] * 1600))
    assert first.speech_started
    assert not first.speech_ended

    second = buffer.push(pcm([0] * 1600))
    assert second.speech_ended
    assert second.utterance_audio is not None


@pytest.mark.asyncio
async def test_utterance_buffer_push_async_offloads_vad(monkeypatch):
    calls = []

    async def fake_to_thread(func, /, *args, **kwargs):
        calls.append(func.__name__)
        return func(*args, **kwargs)

    monkeypatch.setattr(stt_module.asyncio, "to_thread", fake_to_thread)
    buffer = UtteranceBuffer(
        vad=EnergyVAD(VADConfig(provider="energy", threshold=0.01)),
        sample_rate=16000,
    )

    await buffer.push_async(pcm([0] * 1600))

    assert calls == ["push"]


@pytest.mark.asyncio
async def test_brain_warmup_warms_existing_vad(tmp_path):
    vad = FakeVAD()
    brain = Brain(
        BrainConfig(database_path=tmp_path / "brain.sqlite3"),
        stt_provider=FakeSTT(),
        vad_detector=vad,
        tts_provider=NullTTS(TTSConfig(provider="null")),
    )

    await brain.warmup(stt=True, tts=False)

    assert vad.warmed is True


@pytest.mark.asyncio
async def test_voice_stream_transcribes_then_runs_brain_and_tts(tmp_path):
    brain = Brain(
        BrainConfig(database_path=tmp_path / "brain.sqlite3"),
        client=FakeClient(),
        stt_provider=FakeSTT(),
        tts_provider=FakeTTS(TTSConfig(provider="null")),
    )

    events = [
        event
        async for event in brain.voice_stream(
            pcm([1000] * 1600),
            thread_id="voice:test",
            stt_options={"text": "what did I say"},
        )
    ]

    assert events[0].type == "stt.final"
    assert events[0].data["text"] == "what did I say"
    assert "text.delta" in [event.type for event in events]
    assert "tts.audio" in [event.type for event in events]
    assert "response.done" in [event.type for event in events]
    assert events[-1].type == "tts.playlist.done"


def test_stt_http_endpoint_uses_brain_provider(tmp_path):
    brain = Brain(
        BrainConfig(database_path=tmp_path / "brain.sqlite3"),
        stt_provider=FakeSTT(),
        tts_provider=NullTTS(TTSConfig(provider="null")),
    )
    app = create_app(brain=brain)
    client = TestClient(app)
    payload = {
        "audio": base64.b64encode(pcm([0] * 1600)).decode("ascii"),
        "format": "pcm_s16le",
        "sample_rate": 16000,
        "channels": 1,
        "options": {"text": "server transcript"},
    }

    response = client.post("/stt", json=payload)

    assert response.status_code == 200
    assert response.json()["text"] == "server transcript"


def test_voice_websocket_endpoint_runs_full_loop(tmp_path):
    brain = Brain(
        BrainConfig(database_path=tmp_path / "brain.sqlite3"),
        client=FakeClient(),
        stt_provider=FakeSTT(),
        tts_provider=FakeTTS(TTSConfig(provider="null")),
    )
    app = create_app(brain=brain)
    client = TestClient(app)

    with client.websocket_connect("/voice") as ws:
        ws.send_json(
            {
                "type": "audio.start",
                "thread_id": "voice:socket",
                "sample_rate": 16000,
                "channels": 1,
                "encoding": "pcm_s16le",
                "vad": {
                    "provider": "energy",
                    "threshold": 0.01,
                    "min_speech_ms": 20,
                    "end_silence_ms": 50,
                    "padding_ms": 0,
                },
            }
        )
        assert ws.receive_json()["type"] == "audio.started"

        ws.send_bytes(pcm([9000] * 1600))
        assert ws.receive_json()["type"] == "vad.speech.start"

        ws.send_bytes(pcm([0] * 1600))
        events = []
        for _ in range(20):
            event = ws.receive_json()
            events.append(event["type"])
            if event["type"] == "tts.playlist.done":
                break

    assert "vad.speech.end" in events
    assert "stt.final" in events
    assert "text.delta" in events
    assert "tts.audio" in events
    assert "response.done" in events
    assert events[-1] == "tts.playlist.done"
