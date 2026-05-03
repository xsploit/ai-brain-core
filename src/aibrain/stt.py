from __future__ import annotations

import asyncio
import base64
import io
import math
import os
import sys
import wave
from array import array
from collections import deque
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field


AudioEncoding = Literal["pcm_s16le", "wav", "flac"]
STTProviderName = Literal["faster_whisper", "null"]
VADProviderName = Literal["silero", "energy", "none"]


class VADConfig(BaseModel):
    provider: VADProviderName = Field(
        default_factory=lambda: os.environ.get("AIBRAIN_VAD_PROVIDER", "silero")  # type: ignore[arg-type]
    )
    threshold: float = Field(default_factory=lambda: _env_float("AIBRAIN_VAD_THRESHOLD", 0.5))
    min_speech_ms: int = Field(default_factory=lambda: _env_int("AIBRAIN_VAD_MIN_SPEECH_MS", 250))
    end_silence_ms: int = Field(default_factory=lambda: _env_int("AIBRAIN_VAD_END_SILENCE_MS", 700))
    padding_ms: int = Field(default_factory=lambda: _env_int("AIBRAIN_VAD_PADDING_MS", 300))
    max_utterance_seconds: float = Field(
        default_factory=lambda: _env_float("AIBRAIN_VAD_MAX_UTTERANCE_SECONDS", 30.0)
    )


class STTConfig(BaseModel):
    enabled: bool = Field(default_factory=lambda: os.environ.get("AIBRAIN_STT_ENABLED", "1") != "0")
    provider: STTProviderName = Field(
        default_factory=lambda: os.environ.get("AIBRAIN_STT_PROVIDER", "faster_whisper")  # type: ignore[arg-type]
    )
    model: str = Field(default_factory=lambda: os.environ.get("AIBRAIN_STT_MODEL", "base.en"))
    device: str = Field(default_factory=lambda: os.environ.get("AIBRAIN_STT_DEVICE", "auto"))
    compute_type: str = Field(default_factory=lambda: os.environ.get("AIBRAIN_STT_COMPUTE_TYPE", "auto"))
    language: str | None = Field(default_factory=lambda: os.environ.get("AIBRAIN_STT_LANGUAGE"))
    sample_rate: int = Field(default_factory=lambda: _env_int("AIBRAIN_STT_SAMPLE_RATE", 16000))
    channels: int = Field(default_factory=lambda: _env_int("AIBRAIN_STT_CHANNELS", 1))
    encoding: AudioEncoding = "pcm_s16le"
    auto_resample: bool = Field(
        default_factory=lambda: os.environ.get("AIBRAIN_STT_AUTO_RESAMPLE", "1") != "0"
    )
    vad_config: VADConfig = Field(default_factory=VADConfig)


class STTSegment(BaseModel):
    text: str
    start: float | None = None
    end: float | None = None
    confidence: float | None = None


class STTResult(BaseModel):
    text: str
    language: str | None = None
    duration: float | None = None
    segments: list[STTSegment] = Field(default_factory=list)
    provider: str | None = None


@dataclass(slots=True)
class AudioData:
    samples: Any
    sample_rate: int
    channels: int = 1
    encoding: str = "float32"

    @property
    def duration(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return len(self.samples) / self.sample_rate


@dataclass(slots=True)
class VADFrameResult:
    speech_detected: bool = False
    speech_started: bool = False
    speech_ended: bool = False
    utterance_audio: bytes | None = None
    discarded: bool = False


class BaseSTTProvider:
    def __init__(self, config: STTConfig | None = None):
        self.config = config or STTConfig()

    async def transcribe(
        self,
        audio: bytes,
        *,
        format: AudioEncoding = "pcm_s16le",
        sample_rate: int = 16000,
        channels: int = 1,
        language: str | None = None,
        **options: Any,
    ) -> STTResult:
        raise NotImplementedError

    async def warmup(self) -> None:
        return None

    async def close(self) -> None:
        return None


class NullSTT(BaseSTTProvider):
    async def transcribe(
        self,
        audio: bytes,
        *,
        format: AudioEncoding = "pcm_s16le",
        sample_rate: int = 16000,
        channels: int = 1,
        language: str | None = None,
        **options: Any,
    ) -> STTResult:
        text = str(options.get("text", ""))
        duration = audio_duration(audio, format=format, sample_rate=sample_rate, channels=channels)
        return STTResult(text=text, language=language, duration=duration, provider="null")


class FasterWhisperSTT(BaseSTTProvider):
    def __init__(self, config: STTConfig | None = None):
        super().__init__(config)
        self._model: Any | None = None

    async def warmup(self) -> None:
        await asyncio.to_thread(self._ensure_model)

    async def transcribe(
        self,
        audio: bytes,
        *,
        format: AudioEncoding = "pcm_s16le",
        sample_rate: int = 16000,
        channels: int = 1,
        language: str | None = None,
        **options: Any,
    ) -> STTResult:
        return await asyncio.to_thread(
            self._transcribe_sync,
            audio,
            format,
            sample_rate,
            channels,
            language,
            options,
        )

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed. Install with `python -m pip install -e .[stt]`."
            ) from exc
        kwargs: dict[str, Any] = {"device": self.config.device}
        if self.config.compute_type != "auto":
            kwargs["compute_type"] = self.config.compute_type
        self._model = WhisperModel(self.config.model, **kwargs)
        return self._model

    def _transcribe_sync(
        self,
        audio: bytes,
        format: AudioEncoding,
        sample_rate: int,
        channels: int,
        language: str | None,
        options: dict[str, Any],
    ) -> STTResult:
        audio_data = decode_audio_bytes(
            audio,
            format=format,
            sample_rate=sample_rate,
            channels=channels,
            target_sample_rate=self.config.sample_rate if self.config.auto_resample else None,
        )
        model = self._ensure_model()
        np_audio = audio_to_numpy(audio_data)
        kwargs = dict(options)
        kwargs.setdefault("vad_filter", False)
        resolved_language = language or self.config.language
        if resolved_language:
            kwargs["language"] = resolved_language
        segments_iter, info = model.transcribe(np_audio, **kwargs)
        segments = [
            STTSegment(
                text=str(getattr(segment, "text", "")).strip(),
                start=getattr(segment, "start", None),
                end=getattr(segment, "end", None),
                confidence=_segment_confidence(segment),
            )
            for segment in segments_iter
        ]
        text = " ".join(segment.text for segment in segments if segment.text).strip()
        return STTResult(
            text=text,
            language=getattr(info, "language", resolved_language),
            duration=audio_data.duration,
            segments=segments,
            provider="faster_whisper",
        )


class BaseVAD:
    def __init__(self, config: VADConfig | None = None):
        self.config = config or VADConfig()

    def is_speech(
        self,
        audio: bytes,
        *,
        sample_rate: int = 16000,
        channels: int = 1,
        encoding: AudioEncoding = "pcm_s16le",
    ) -> bool:
        raise NotImplementedError

    async def warmup(self) -> None:
        return None


class NoVAD(BaseVAD):
    def is_speech(
        self,
        audio: bytes,
        *,
        sample_rate: int = 16000,
        channels: int = 1,
        encoding: AudioEncoding = "pcm_s16le",
    ) -> bool:
        return bool(audio)


class EnergyVAD(BaseVAD):
    def is_speech(
        self,
        audio: bytes,
        *,
        sample_rate: int = 16000,
        channels: int = 1,
        encoding: AudioEncoding = "pcm_s16le",
    ) -> bool:
        if not audio:
            return False
        samples = decode_audio_bytes(
            audio,
            format=encoding,
            sample_rate=sample_rate,
            channels=channels,
        ).samples
        if len(samples) == 0:
            return False
        try:
            import numpy as np

            rms = float(np.sqrt(np.mean(np.asarray(samples, dtype=np.float32) ** 2)))
        except ImportError:
            rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
        return rms >= self.config.threshold


class SileroVAD(BaseVAD):
    def __init__(self, config: VADConfig | None = None):
        super().__init__(config)
        self._model: Any | None = None
        self._get_speech_timestamps: Any | None = None

    def is_speech(
        self,
        audio: bytes,
        *,
        sample_rate: int = 16000,
        channels: int = 1,
        encoding: AudioEncoding = "pcm_s16le",
    ) -> bool:
        if not audio:
            return False
        model, get_speech_timestamps = self._ensure_model()
        audio_data = decode_audio_bytes(
            audio,
            format=encoding,
            sample_rate=sample_rate,
            channels=channels,
            target_sample_rate=16000,
        )
        np_audio = audio_to_numpy(audio_data)
        timestamps = get_speech_timestamps(
            np_audio,
            model,
            sampling_rate=16000,
            threshold=self.config.threshold,
        )
        return bool(timestamps)

    async def warmup(self) -> None:
        await asyncio.to_thread(self._ensure_model)

    def _ensure_model(self) -> tuple[Any, Any]:
        if self._model is not None and self._get_speech_timestamps is not None:
            return self._model, self._get_speech_timestamps
        try:
            from silero_vad import get_speech_timestamps, load_silero_vad
        except ImportError as exc:
            raise RuntimeError(
                "silero-vad is not installed. Install with `python -m pip install -e .[stt]`."
            ) from exc
        self._model = load_silero_vad()
        self._get_speech_timestamps = get_speech_timestamps
        return self._model, self._get_speech_timestamps


class UtteranceBuffer:
    def __init__(
        self,
        *,
        vad: BaseVAD,
        config: VADConfig | None = None,
        sample_rate: int = 16000,
        channels: int = 1,
        encoding: AudioEncoding = "pcm_s16le",
    ):
        self.vad = vad
        self.config = config or vad.config
        self.sample_rate = sample_rate
        self.channels = channels
        self.encoding = encoding
        self._in_speech = False
        self._speech_samples = 0
        self._silence_samples = 0
        self._chunks: list[bytes] = []
        self._prebuffer: deque[bytes] = deque()
        self._prebuffer_samples = 0

    def push(self, audio: bytes) -> VADFrameResult:
        if not audio:
            return VADFrameResult()
        sample_count = pcm_sample_count(audio, channels=self.channels)
        detected = self.vad.is_speech(
            audio,
            sample_rate=self.sample_rate,
            channels=self.channels,
            encoding=self.encoding,
        )
        started = False
        if detected:
            if not self._in_speech:
                started = True
                self._in_speech = True
                self._chunks = list(self._prebuffer)
                self._prebuffer.clear()
                self._prebuffer_samples = 0
            self._speech_samples += sample_count
            self._silence_samples = 0
            self._chunks.append(audio)
        elif self._in_speech:
            self._silence_samples += sample_count
            self._chunks.append(audio)
        else:
            self._add_prebuffer(audio, sample_count)

        if self._in_speech and self._should_end():
            return self._finalize(speech_detected=detected, started=started)
        return VADFrameResult(speech_detected=detected, speech_started=started)

    async def push_async(self, audio: bytes) -> VADFrameResult:
        return await asyncio.to_thread(self.push, audio)

    def flush(self) -> VADFrameResult:
        if not self._in_speech:
            self.reset()
            return VADFrameResult()
        return self._finalize(speech_detected=False, started=False)

    def reset(self) -> None:
        self._in_speech = False
        self._speech_samples = 0
        self._silence_samples = 0
        self._chunks = []
        self._prebuffer.clear()
        self._prebuffer_samples = 0

    def _should_end(self) -> bool:
        if self._speech_samples >= seconds_to_samples(
            self.config.max_utterance_seconds,
            self.sample_rate,
        ):
            return True
        return self._silence_samples >= ms_to_samples(self.config.end_silence_ms, self.sample_rate)

    def _finalize(self, *, speech_detected: bool, started: bool) -> VADFrameResult:
        min_samples = ms_to_samples(self.config.min_speech_ms, self.sample_rate)
        utterance = b"".join(self._chunks)
        discarded = self._speech_samples < min_samples
        self.reset()
        if discarded:
            return VADFrameResult(
                speech_detected=speech_detected,
                speech_started=started,
                speech_ended=True,
                discarded=True,
            )
        return VADFrameResult(
            speech_detected=speech_detected,
            speech_started=started,
            speech_ended=True,
            utterance_audio=utterance,
        )

    def _add_prebuffer(self, audio: bytes, sample_count: int) -> None:
        max_samples = ms_to_samples(self.config.padding_ms, self.sample_rate)
        if max_samples <= 0:
            return
        self._prebuffer.append(audio)
        self._prebuffer_samples += sample_count
        while self._prebuffer and self._prebuffer_samples > max_samples:
            removed = self._prebuffer.popleft()
            self._prebuffer_samples -= pcm_sample_count(removed, channels=self.channels)


def create_stt_provider(config: STTConfig | None = None) -> BaseSTTProvider:
    resolved = config or STTConfig()
    if not resolved.enabled or resolved.provider == "null":
        return NullSTT(resolved)
    return FasterWhisperSTT(resolved)


def create_vad_detector(config: VADConfig | None = None) -> BaseVAD:
    resolved = config or VADConfig()
    if resolved.provider == "none":
        return NoVAD(resolved)
    if resolved.provider == "energy":
        return EnergyVAD(resolved)
    return SileroVAD(resolved)


def decode_audio_base64(value: str) -> bytes:
    return base64.b64decode(value)


def decode_audio_bytes(
    audio: bytes,
    *,
    format: AudioEncoding = "pcm_s16le",
    sample_rate: int = 16000,
    channels: int = 1,
    target_sample_rate: int | None = None,
) -> AudioData:
    if format == "pcm_s16le":
        samples = pcm_s16le_to_float32(audio, channels=channels)
        resolved = AudioData(samples=samples, sample_rate=sample_rate, channels=1)
    elif format == "wav":
        resolved = decode_wav_bytes(audio)
    elif format == "flac":
        resolved = decode_with_soundfile(audio, format="FLAC")
    else:
        raise ValueError(f"Unsupported audio format: {format}")
    if target_sample_rate and resolved.sample_rate != target_sample_rate:
        resolved = AudioData(
            samples=resample_linear(resolved.samples, resolved.sample_rate, target_sample_rate),
            sample_rate=target_sample_rate,
            channels=1,
        )
    return resolved


def decode_wav_bytes(audio: bytes) -> AudioData:
    with wave.open(io.BytesIO(audio), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())
    if sample_width != 2:
        return decode_with_soundfile(audio, format="WAV")
    return AudioData(
        samples=pcm_s16le_to_float32(frames, channels=channels),
        sample_rate=sample_rate,
        channels=1,
    )


def decode_with_soundfile(audio: bytes, *, format: str | None = None) -> AudioData:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "soundfile is required for this audio format. Install with `python -m pip install -e .[stt]`."
        ) from exc
    data, sample_rate = sf.read(io.BytesIO(audio), dtype="float32", always_2d=True, format=format)
    channels = data.shape[1]
    if channels == 1:
        mono = data[:, 0]
    else:
        mono = data.mean(axis=1)
    return AudioData(samples=mono, sample_rate=int(sample_rate), channels=1)


def pcm_s16le_to_float32(audio: bytes, *, channels: int = 1) -> Any:
    if not audio:
        return []
    if len(audio) % 2:
        audio = audio[:-1]
    try:
        import numpy as np

        values = np.frombuffer(audio, dtype="<i2").astype(np.float32)
        if channels > 1:
            frame_count = len(values) // channels
            values = values[: frame_count * channels].reshape(frame_count, channels).mean(axis=1)
        return np.clip(values / 32768.0, -1.0, 1.0)
    except ImportError:
        pass
    values = array("h")
    values.frombytes(audio)
    if sys.byteorder != "little":
        values.byteswap()
    if channels <= 1:
        return [max(-1.0, min(1.0, sample / 32768.0)) for sample in values]
    mono = []
    frame_count = len(values) // channels
    for index in range(frame_count):
        start = index * channels
        mono.append(sum(values[start : start + channels]) / channels / 32768.0)
    return [max(-1.0, min(1.0, sample)) for sample in mono]


def audio_to_numpy(audio: AudioData) -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "numpy is required for this STT backend. Install with `python -m pip install -e .[stt]`."
        ) from exc
    return np.asarray(audio.samples, dtype=np.float32)


def resample_linear(samples: Any, source_rate: int, target_rate: int) -> Any:
    if source_rate <= 0 or target_rate <= 0 or source_rate == target_rate or len(samples) == 0:
        return samples
    target_len = max(1, int(round(len(samples) * target_rate / source_rate)))
    if target_len == 1:
        return [samples[0]]
    try:
        import numpy as np

        source = np.asarray(samples, dtype=np.float32)
        positions = np.linspace(0, len(source) - 1, target_len)
        return np.interp(positions, np.arange(len(source)), source).astype(np.float32)
    except ImportError:
        pass
    scale = (len(samples) - 1) / (target_len - 1)
    output = []
    for index in range(target_len):
        source_pos = index * scale
        left = int(source_pos)
        right = min(left + 1, len(samples) - 1)
        fraction = source_pos - left
        output.append(samples[left] * (1.0 - fraction) + samples[right] * fraction)
    return output


def audio_duration(
    audio: bytes,
    *,
    format: AudioEncoding = "pcm_s16le",
    sample_rate: int = 16000,
    channels: int = 1,
) -> float:
    if format == "pcm_s16le":
        return pcm_sample_count(audio, channels=channels) / sample_rate if sample_rate > 0 else 0.0
    return decode_audio_bytes(audio, format=format, sample_rate=sample_rate, channels=channels).duration


def pcm_sample_count(audio: bytes, *, channels: int = 1) -> int:
    if channels <= 0:
        channels = 1
    return len(audio) // 2 // channels


def ms_to_samples(ms: int | float, sample_rate: int) -> int:
    return int(sample_rate * (float(ms) / 1000.0))


def seconds_to_samples(seconds: int | float, sample_rate: int) -> int:
    return int(sample_rate * float(seconds))


def _segment_confidence(segment: Any) -> float | None:
    avg_logprob = getattr(segment, "avg_logprob", None)
    if avg_logprob is None:
        return None
    try:
        return max(0.0, min(1.0, math.exp(float(avg_logprob))))
    except (TypeError, ValueError):
        return None


def _env_float(name: str, fallback: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return fallback
    try:
        return float(value)
    except ValueError:
        return fallback


def _env_int(name: str, fallback: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return fallback
    try:
        return int(value)
    except ValueError:
        return fallback
