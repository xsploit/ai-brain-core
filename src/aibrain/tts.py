from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import tempfile
import threading
import wave
from collections import OrderedDict
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field


def _env_path(name: str, fallback: Path | None = None) -> Path | None:
    value = os.environ.get(name)
    if value:
        return Path(value)
    return fallback if fallback and fallback.exists() else None


def _env_float(name: str, fallback: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return fallback
    try:
        return float(value)
    except ValueError:
        return fallback


def _env_int(name: str, fallback: int | None = None) -> int | None:
    value = os.environ.get(name)
    if value is None:
        return fallback
    try:
        return int(value)
    except ValueError:
        return fallback


def _default_tts_provider() -> Literal["piper_process", "piper_executable", "piper_http", "null"]:
    value = (
        os.environ.get("AIBRAIN_TTS_PROVIDER")
        or os.environ.get("TTS_PROVIDER")
        or "piper_process"
    ).strip().lower()
    aliases = {
        "none": "null",
        "off": "null",
        "disabled": "null",
        "piper": "piper_process",
        "process": "piper_process",
        "executable": "piper_executable",
        "http": "piper_http",
    }
    normalized = aliases.get(value, value)
    if normalized in {"piper_process", "piper_executable", "piper_http", "null"}:
        return normalized  # type: ignore[return-value]
    return "piper_process"


class TTSConfig(BaseModel):
    enabled: bool = Field(default_factory=lambda: _default_tts_provider() != "null")
    provider: Literal["piper_process", "piper_executable", "piper_http", "null"] = Field(
        default_factory=lambda: _default_tts_provider()
    )
    piper_executable_path: Path | None = Field(
        default_factory=lambda: _default_piper_executable_path()
    )
    piper_model_path: Path | None = Field(default_factory=lambda: _default_piper_model_path())
    piper_config_path: Path | None = Field(default_factory=lambda: _default_piper_config_path())
    piper_espeak_data_path: Path | None = Field(
        default_factory=lambda: _env_path("PIPER_ESPEAK_DATA")
    )
    piper_http_url: str = Field(
        default_factory=lambda: os.environ.get("PIPER_HTTP_URL", "http://127.0.0.1:5000")
    )
    speaker: int | None = Field(default_factory=lambda: _env_int("PIPER_SPEAKER"))
    noise_scale: float = Field(default_factory=lambda: _env_float("PIPER_NOISE_SCALE", 0.667))
    length_scale: float = Field(default_factory=lambda: _env_float("PIPER_LENGTH_SCALE", 1.0))
    noise_w: float = Field(default_factory=lambda: _env_float("PIPER_NOISE_W", 0.8))
    sentence_silence: float = Field(default_factory=lambda: _env_float("PIPER_SENTENCE_SILENCE", 0.05))
    json_input: bool = Field(default_factory=lambda: os.environ.get("PIPER_JSON_INPUT", "1") != "0")
    output_format: Literal["pcm_s16le", "wav"] = "pcm_s16le"
    sample_rate: int | None = None
    chunk_chars: int = 180
    min_chunk_chars: int = 6
    comma_chunk_chars: int = 48
    process_idle_timeout: float = Field(
        default_factory=lambda: _env_float("PIPER_PROCESS_IDLE_TIMEOUT", 0.45)
    )
    process_first_chunk_timeout: float = 10.0
    process_pool_max: int = Field(default_factory=lambda: _env_int("PIPER_PROCESS_POOL_MAX", 4))
    warmup_on_start: bool = True

    def resolved_config_path(self) -> Path | None:
        if self.piper_config_path:
            return self.piper_config_path
        if self.piper_model_path:
            candidate = Path(str(self.piper_model_path) + ".json")
            return candidate if candidate.exists() else None
        return None

    def resolved_sample_rate(self) -> int:
        if self.sample_rate:
            return self.sample_rate
        config_path = self.resolved_config_path()
        if config_path and config_path.exists():
            try:
                data = json.loads(config_path.read_text(encoding="utf-8"))
                return int(data.get("audio", {}).get("sample_rate") or 22050)
            except Exception:
                return 22050
        return 22050


@dataclass(slots=True)
class TTSAudio:
    audio: bytes
    sample_rate: int
    encoding: str = "pcm_s16le"
    voice: str | None = None

    def to_event_data(self) -> dict[str, Any]:
        return {
            "audio": base64.b64encode(self.audio).decode("ascii"),
            "sample_rate": self.sample_rate,
            "encoding": self.encoding,
            "voice": self.voice,
        }


@dataclass(slots=True)
class PiperVoice:
    slug: str
    label: str
    onnx: Path
    config: Path | None = None
    created_at: str | None = None

    def model_dump(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "label": self.label,
            "onnx": str(self.onnx),
            "config": str(self.config) if self.config else None,
            "created_at": self.created_at,
        }


@dataclass(slots=True)
class TTSChunk:
    audio: bytes
    sample_rate: int
    encoding: str = "pcm_s16le"
    index: int = 0
    final: bool = False
    text: str | None = None
    voice: str | None = None

    def to_event_data(self) -> dict[str, Any]:
        return {
            "audio": base64.b64encode(self.audio).decode("ascii"),
            "sample_rate": self.sample_rate,
            "encoding": self.encoding,
            "index": self.index,
            "final": self.final,
            "text": self.text,
            "voice": self.voice,
        }


_VOICE_CACHE_LOCK = threading.Lock()
_VOICE_CACHE: dict[tuple[tuple[str, ...], tuple[str, ...]], list[PiperVoice]] = {}


class BaseTTSProvider:
    def __init__(self, config: TTSConfig | None = None):
        self.config = config or TTSConfig()

    async def synthesize(self, text: str, **options: Any) -> TTSAudio:
        chunks = [chunk async for chunk in self.stream(text, **options)]
        return TTSAudio(
            audio=b"".join(chunk.audio for chunk in chunks),
            sample_rate=chunks[0].sample_rate if chunks else self.config.resolved_sample_rate(),
            encoding=chunks[0].encoding if chunks else self.config.output_format,
            voice=(chunks[0].voice if chunks and chunks[0].voice else options.get("voice")),
        )

    async def stream(self, text: str, **options: Any) -> AsyncIterator[TTSChunk]:
        raise NotImplementedError

    async def warmup(self, **options: Any) -> None:
        return None

    async def close(self) -> None:
        return None


class NullTTS(BaseTTSProvider):
    async def stream(self, text: str, **options: Any) -> AsyncIterator[TTSChunk]:
        if False:
            yield TTSChunk(audio=b"", sample_rate=self.config.resolved_sample_rate(), text=text)


class PiperExecutableTTS(BaseTTSProvider):
    async def stream(self, text: str, **options: Any) -> AsyncIterator[TTSChunk]:
        runtime_config = tts_config_for_voice(self.config, options.get("voice"))
        audio = await self._run_piper(
            text,
            output_raw=runtime_config.output_format == "pcm_s16le",
            config=runtime_config,
        )
        yield TTSChunk(
            audio=audio,
            sample_rate=runtime_config.resolved_sample_rate(),
            encoding=runtime_config.output_format,
            index=0,
            final=True,
            text=text,
            voice=tts_voice_name(runtime_config),
        )

    async def _run_piper(
        self,
        text: str,
        *,
        output_raw: bool,
        config: TTSConfig | None = None,
    ) -> bytes:
        runtime_config = config or self.config
        command = _piper_command(runtime_config, output_raw=output_raw, output_stdout=True)
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate(_piper_stdin_line(runtime_config, text))
        if process.returncode != 0:
            raise RuntimeError(stderr.decode("utf-8", errors="ignore") or "Piper failed")
        return stdout


class PiperProcessTTS(PiperExecutableTTS):
    def __init__(self, config: TTSConfig | None = None):
        super().__init__(config)
        self._processes: OrderedDict[tuple[Any, ...], asyncio.subprocess.Process] = OrderedDict()
        self._locks: OrderedDict[tuple[Any, ...], asyncio.Lock] = OrderedDict()
        self._locks_guard = asyncio.Lock()

    async def stream(self, text: str, **options: Any) -> AsyncIterator[TTSChunk]:
        runtime_config = tts_config_for_voice(self.config, options.get("voice"))
        key = tts_config_process_key(runtime_config)
        lock = await self._process_lock(key)
        async with lock:
            yielded_anything = False
            try:
                index = 0
                for segment in split_tts_text(text, runtime_config):
                    async for chunk in self._stream_process(
                        segment,
                        config=runtime_config,
                        start_index=index,
                    ):
                        yielded_anything = True
                        yield chunk
                        index = chunk.index + 1
            except Exception:
                await self._close_process(key)
                if yielded_anything:
                    raise
                async for chunk in super().stream(text, **options):
                    yield chunk

    async def _process_lock(self, key: tuple[Any, ...]) -> asyncio.Lock:
        async with self._locks_guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[key] = lock
                self._trim_lock_cache()
            else:
                self._locks.move_to_end(key)
            return lock

    def _trim_lock_cache(self) -> None:
        max_locks = max(1, self.config.process_pool_max * 4)
        while len(self._locks) > max_locks:
            removed = False
            for key, lock in list(self._locks.items()):
                if lock.locked() or key in self._processes:
                    continue
                self._locks.pop(key, None)
                removed = True
                break
            if not removed:
                break

    async def _stream_process(
        self,
        text: str,
        *,
        config: TTSConfig | None = None,
        start_index: int = 0,
    ) -> AsyncIterator[TTSChunk]:
        runtime_config = config or self.config
        process = await self._ensure_process(runtime_config)
        if process.stdin is None or process.stdout is None:
            raise RuntimeError("Piper process has no stdin/stdout pipe")
        process.stdin.write(self._stdin_line(text, runtime_config))
        await process.stdin.drain()
        index = start_index
        got_audio = False
        sample_rate = runtime_config.resolved_sample_rate()
        voice_name = tts_voice_name(runtime_config)
        while True:
            timeout = (
                runtime_config.process_idle_timeout
                if got_audio
                else runtime_config.process_first_chunk_timeout
            )
            try:
                data = await asyncio.wait_for(process.stdout.read(4096), timeout=timeout)
            except asyncio.TimeoutError:
                if got_audio:
                    break
                raise RuntimeError("Timed out waiting for first Piper audio chunk")
            if not data:
                break
            got_audio = True
            yield TTSChunk(
                audio=data,
                sample_rate=sample_rate,
                encoding="pcm_s16le",
                index=index,
                final=False,
                text=text,
                voice=voice_name,
            )
            index += 1
        if got_audio:
            yield TTSChunk(
                audio=b"",
                sample_rate=sample_rate,
                encoding="pcm_s16le",
                index=index,
                final=True,
                text=text,
                voice=voice_name,
            )

    async def _ensure_process(self, config: TTSConfig | None = None) -> asyncio.subprocess.Process:
        runtime_config = config or self.config
        key = tts_config_process_key(runtime_config)
        process = self._processes.get(key)
        if process and process.returncode is None:
            self._processes.move_to_end(key)
            return process
        command = _piper_command(runtime_config, output_raw=True, output_stdout=False)
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        self._processes[key] = process
        self._processes.move_to_end(key)
        await self._evict_process_pool(keep_key=key)
        return process

    def _stdin_line(self, text: str, config: TTSConfig | None = None) -> bytes:
        return _piper_stdin_line(config or self.config, text)

    async def warmup(self, **options: Any) -> None:
        runtime_config = tts_config_for_voice(self.config, options.get("voice"))
        key = tts_config_process_key(runtime_config)
        lock = await self._process_lock(key)
        async with lock:
            await self._ensure_process(runtime_config)

    async def _close_process(self, key: tuple[Any, ...]) -> None:
        process = self._processes.pop(key, None)
        if process is None or process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=2)
        except asyncio.TimeoutError:
            process.kill()

    async def _evict_process_pool(self, *, keep_key: tuple[Any, ...]) -> None:
        max_processes = self.config.process_pool_max
        if max_processes <= 0:
            return
        while len(self._processes) > max_processes:
            evict_key: tuple[Any, ...] | None = None
            for key, process in list(self._processes.items()):
                if key == keep_key:
                    continue
                if process.returncode is not None:
                    evict_key = key
                    break
                lock = self._locks.get(key)
                if lock is not None and lock.locked():
                    continue
                evict_key = key
                break
            if evict_key is None:
                break
            await self._close_process(evict_key)

    async def close(self) -> None:
        processes = list(self._processes.values())
        self._processes.clear()
        self._locks.clear()
        for process in processes:
            if process.returncode is not None:
                continue
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=2)
            except asyncio.TimeoutError:
                process.kill()


class PiperHttpTTS(BaseTTSProvider):
    async def stream(self, text: str, **options: Any) -> AsyncIterator[TTSChunk]:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                self.config.piper_http_url,
                json={"text": text, **options},
            )
            response.raise_for_status()
            yield TTSChunk(
                audio=response.content,
                sample_rate=self.config.resolved_sample_rate(),
                encoding=response.headers.get("x-audio-encoding", "wav"),
                index=0,
                final=True,
                text=text,
                voice=options.get("voice"),
            )


class SentenceChunker:
    def __init__(self, max_chars: int = 180, min_chars: int = 6, comma_chars: int = 48):
        self.max_chars = max_chars
        self.min_chars = min_chars
        self.comma_chars = comma_chars
        self.buffer = ""
        self._terminal_scan_pos = 0
        self._soft_scan_pos = 0

    def feed(self, text: str) -> list[str]:
        self.buffer += text
        chunks: list[str] = []
        while True:
            terminal = self._find_split(
                {".", "!", "?", "。", "！", "？"},
                self.min_chars,
                "_terminal_scan_pos",
            )
            if terminal is not None:
                chunks.append(self.buffer[:terminal].strip())
                self.buffer = self.buffer[terminal:].lstrip()
                self._reset_scan_positions()
                continue
            soft = self._find_split(
                {",", ";", ":", "，", "；", "："},
                self.comma_chars,
                "_soft_scan_pos",
            )
            if soft is not None:
                chunks.append(self.buffer[:soft].strip())
                self.buffer = self.buffer[soft:].lstrip()
                self._reset_scan_positions()
                continue
            if len(self.buffer) >= self.max_chars:
                split_at = self.buffer.rfind(" ", 0, self.max_chars)
                if split_at <= 0:
                    split_at = self.max_chars
                chunks.append(self.buffer[:split_at].strip())
                self.buffer = self.buffer[split_at:].lstrip()
                self._reset_scan_positions()
                continue
            break
        return [chunk for chunk in chunks if chunk]

    def _find_split(self, split_chars: set[str], min_chars: int, cursor_attr: str) -> int | None:
        start = max(0, int(getattr(self, cursor_attr)))
        for index in range(start, len(self.buffer)):
            if self.buffer[index] not in split_chars:
                continue
            end = index + 1
            if end >= min_chars:
                return end
        setattr(self, cursor_attr, len(self.buffer))
        return None

    def _reset_scan_positions(self) -> None:
        self._terminal_scan_pos = 0
        self._soft_scan_pos = 0

    def flush(self) -> str | None:
        text = self.buffer.strip()
        self.buffer = ""
        self._reset_scan_positions()
        return text or None


def create_tts_provider(config: TTSConfig | None = None) -> BaseTTSProvider:
    resolved = with_env_overrides(config or TTSConfig())
    if not resolved.enabled or resolved.provider == "null":
        return NullTTS(resolved)
    if resolved.provider == "piper_http":
        return PiperHttpTTS(resolved)
    if resolved.provider == "piper_executable":
        return PiperExecutableTTS(resolved)
    return PiperProcessTTS(resolved)


def split_tts_text(text: str, config: TTSConfig) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    chunker = SentenceChunker(
        max_chars=config.chunk_chars,
        min_chars=config.min_chunk_chars,
        comma_chars=config.comma_chunk_chars,
    )
    chunks = chunker.feed(text)
    tail = chunker.flush()
    if tail:
        chunks.append(tail)
    return chunks or [stripped]


def tts_config_for_voice(config: TTSConfig, voice: Any | None) -> TTSConfig:
    if not voice:
        return config
    if isinstance(voice, dict):
        model_value = voice.get("model") or voice.get("onnx") or voice.get("path")
        config_value = voice.get("config")
        if model_value:
            model_path = Path(str(model_value))
            if model_path.exists():
                return config.model_copy(
                    update={
                        "piper_model_path": model_path,
                        "piper_config_path": Path(str(config_value)) if config_value else _matching_config(model_path),
                    }
                )
        voice_id = voice.get("slug") or voice.get("id") or model_value
    else:
        voice_id = str(voice)
    if not voice_id:
        return config
    direct_model = Path(str(voice_id))
    if direct_model.exists():
        return config.model_copy(
            update={
                "piper_model_path": direct_model,
                "piper_config_path": _matching_config(direct_model),
            }
        )
    resolved = resolve_piper_voice(str(voice_id))
    if resolved is None:
        raise ValueError(f"Unknown Piper voice: {voice_id}")
    return config.model_copy(
        update={
            "piper_model_path": resolved.onnx,
            "piper_config_path": resolved.config,
        }
    )


def tts_voice_name(config: TTSConfig) -> str | None:
    return str(config.piper_model_path) if config.piper_model_path else None


def tts_config_process_key(config: TTSConfig) -> tuple[Any, ...]:
    return (
        str(config.piper_executable_path) if config.piper_executable_path else None,
        str(config.piper_model_path) if config.piper_model_path else None,
        str(config.resolved_config_path()) if config.resolved_config_path() else None,
        str(config.piper_espeak_data_path) if config.piper_espeak_data_path else None,
        config.speaker,
        config.noise_scale,
        config.length_scale,
        config.noise_w,
        config.sentence_silence,
        config.json_input,
        config.output_format,
    )


def with_env_overrides(config: TTSConfig) -> TTSConfig:
    updates: dict[str, Any] = {}
    env_map = {
        "AIBRAIN_TTS_PROVIDER": "provider",
        "PIPER_EXE": "piper_executable_path",
        "PIPER_MODEL": "piper_model_path",
        "PIPER_CONFIG": "piper_config_path",
        "PIPER_ESPEAK_DATA": "piper_espeak_data_path",
        "PIPER_HTTP_URL": "piper_http_url",
    }
    for env_name, field_name in env_map.items():
        value = os.environ.get(env_name)
        if value:
            updates[field_name] = Path(value) if "path" in field_name else value
    if os.environ.get("PIPER_MODEL") and not os.environ.get("PIPER_CONFIG"):
        model_path = Path(os.environ["PIPER_MODEL"])
        updates["piper_config_path"] = _matching_config(model_path)
    voice_id = os.environ.get("AIBRAIN_TTS_VOICE") or os.environ.get("PIPER_VOICE")
    if voice_id and not os.environ.get("PIPER_MODEL"):
        voice = resolve_piper_voice(voice_id)
        if voice:
            updates["piper_model_path"] = voice.onnx
            updates["piper_config_path"] = voice.config
    if os.environ.get("PIPER_SPEAKER"):
        updates["speaker"] = _env_int("PIPER_SPEAKER")
    for env_name, field_name in {
        "PIPER_NOISE_SCALE": "noise_scale",
        "PIPER_LENGTH_SCALE": "length_scale",
        "PIPER_NOISE_W": "noise_w",
        "PIPER_SENTENCE_SILENCE": "sentence_silence",
    }.items():
        if os.environ.get(env_name):
            updates[field_name] = _env_float(env_name, getattr(config, field_name))
    return config.model_copy(update=updates) if updates else config


def discover_piper_voices(
    *,
    manifest_paths: list[Path] | None = None,
    search_roots: list[Path] | None = None,
    refresh: bool = False,
) -> list[PiperVoice]:
    voices: dict[str, PiperVoice] = {}
    resolved_manifests = manifest_paths if manifest_paths is not None else _env_path_list(
        "AIBRAIN_TTS_MANIFESTS"
    )
    resolved_roots = search_roots if search_roots is not None else _env_path_list(
        "AIBRAIN_TTS_VOICE_ROOTS"
    )
    cache_key = (
        tuple(str(path) for path in resolved_manifests),
        tuple(str(path) for path in resolved_roots),
    )
    use_cache = manifest_paths is None and search_roots is None
    if use_cache and not refresh:
        with _VOICE_CACHE_LOCK:
            cached = _VOICE_CACHE.get(cache_key)
            if cached is not None:
                return list(cached)
    for manifest_path in resolved_manifests:
        if not manifest_path.exists():
            continue
        try:
            items = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for item in items:
            onnx = _existing_path(item.get("onnx"))
            if not onnx:
                continue
            config = _existing_path(item.get("config")) or _matching_config(onnx)
            slug = str(item.get("slug") or onnx.stem)
            voices[slug] = PiperVoice(
                slug=slug,
                label=str(item.get("label") or slug),
                onnx=onnx,
                config=config,
                created_at=item.get("created_at"),
            )
    for root in resolved_roots:
        if not root.exists():
            continue
        for onnx in root.rglob("*.onnx"):
            if "\\cache\\" in str(onnx).lower():
                continue
            slug = onnx.stem.replace("en_US-", "").replace("-medium", "").replace("-high", "")
            voices.setdefault(
                slug,
                PiperVoice(
                    slug=slug,
                    label=slug.replace("_", " ").title(),
                    onnx=onnx,
                    config=_matching_config(onnx),
                ),
            )
    discovered = sorted(voices.values(), key=lambda voice: voice.slug)
    if use_cache:
        with _VOICE_CACHE_LOCK:
            _VOICE_CACHE[cache_key] = list(discovered)
    return discovered


def resolve_piper_voice(voice_id: str | None = None) -> PiperVoice | None:
    voices = discover_piper_voices()
    if not voices:
        return None
    requested = voice_id or os.environ.get("AIBRAIN_TTS_VOICE") or os.environ.get("PIPER_VOICE")
    if requested:
        requested_lower = requested.lower()
        for voice in voices:
            if requested_lower in {voice.slug.lower(), voice.label.lower(), str(voice.onnx).lower()}:
                return voice
        for voice in voices:
            if requested_lower in voice.slug.lower() or requested_lower in str(voice.onnx).lower():
                return voice
    return voices[0]


def _default_piper_executable_path() -> Path | None:
    explicit = _env_path("PIPER_EXE")
    if explicit:
        return explicit
    discovered = shutil.which("piper")
    return Path(discovered) if discovered else None


def _default_piper_model_path() -> Path | None:
    explicit = _env_path("PIPER_MODEL")
    if explicit:
        return explicit
    voice = resolve_piper_voice()
    if voice:
        return voice.onnx
    return None


def _default_piper_config_path() -> Path | None:
    explicit = _env_path("PIPER_CONFIG")
    if explicit:
        return explicit
    model = _env_path("PIPER_MODEL")
    if model:
        return _matching_config(model)
    voice = resolve_piper_voice()
    if voice:
        return voice.config
    return None


def _existing_path(value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    return path if path.exists() else None


def _matching_config(onnx: Path) -> Path | None:
    candidate = Path(str(onnx) + ".json")
    return candidate if candidate.exists() else None


def _env_path_list(name: str) -> list[Path]:
    value = os.environ.get(name, "")
    return [Path(item) for item in value.split(os.pathsep) if item.strip()]


def _piper_command(config: TTSConfig, *, output_raw: bool, output_stdout: bool) -> list[str]:
    exe = config.piper_executable_path
    model = config.piper_model_path
    if exe is None or not exe.exists():
        raise FileNotFoundError("Piper executable not found. Set PIPER_EXE or tts_config.")
    if model is None or not model.exists():
        raise FileNotFoundError("Piper model not found. Set PIPER_MODEL or tts_config.")
    command = [str(exe), "--model", str(model), "--quiet"]
    config_path = config.resolved_config_path()
    if config_path:
        command.extend(["--config", str(config_path)])
    if config.piper_espeak_data_path and config.piper_espeak_data_path.exists():
        command.extend(["--espeak_data", str(config.piper_espeak_data_path)])
    if config.speaker is not None:
        command.extend(["--speaker", str(config.speaker)])
    command.extend(["--noise_scale", str(config.noise_scale)])
    command.extend(["--length_scale", str(config.length_scale)])
    command.extend(["--noise_w", str(config.noise_w)])
    command.extend(["--sentence_silence", str(config.sentence_silence)])
    if config.json_input:
        command.append("--json-input")
    if output_raw:
        command.append("--output_raw")
    if output_stdout:
        command.extend(["--output_file", "-"])
    return command


def _piper_stdin_line(config: TTSConfig, text: str) -> bytes:
    cleaned = text.strip()
    if config.json_input:
        payload: dict[str, Any] = {"text": cleaned}
        if config.speaker is not None:
            payload["speaker_id"] = config.speaker
        return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
    return (cleaned + "\n").encode("utf-8")


def write_wav_bytes(audio: bytes, *, sample_rate: int, channels: int = 1, sample_width: int = 2) -> bytes:
    with tempfile.SpooledTemporaryFile() as file:
        with wave.open(file, "wb") as wav:
            wav.setnchannels(channels)
            wav.setsampwidth(sample_width)
            wav.setframerate(sample_rate)
            wav.writeframes(audio)
        file.seek(0)
        return file.read()
