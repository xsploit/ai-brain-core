from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import os
import time
from typing import Any
from uuid import uuid4

import httpx

from grillo_v2.gateway import VERCEL_AI_GATEWAY_BASE_URL, VercelAIGatewayJSONClient


TREBLO_BASE_URL = "https://api.treblo.com/v1"
TREBLO_STREAM_BASE_URL = "https://api-stream.treblo.com"
logger = logging.getLogger("aibrain.treblo_song")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class TrebloSongConfig:
    api_key: str | None = None
    base_url: str = TREBLO_BASE_URL
    stream_base_url: str = TREBLO_STREAM_BASE_URL
    user_cooldown_seconds: float = 1800.0
    max_queue_size: int = 10
    poll_interval_seconds: float = 4.0
    max_poll_attempts: int = 180
    request_timeout_seconds: float = 60.0
    prompt_max_chars: int = 800
    style_scale: float = 3.5
    prompt_strength: float = 1.0
    output_format: str = "ogg"
    enable_streaming: bool = True
    align_lyrics: bool = True
    length_min_seconds: int = 30
    length_max_seconds: int = 120
    max_attachment_bytes: int = 8 * 1024 * 1024
    ai_gateway_api_key: str | None = None
    ai_gateway_model: str = "deepseek/deepseek-v4-pro"

    @classmethod
    def from_env(cls) -> "TrebloSongConfig":
        return cls(
            api_key=os.getenv("TREBLO_API_KEY"),
            base_url=os.getenv("TREBLO_BASE_URL", TREBLO_BASE_URL).rstrip("/"),
            stream_base_url=os.getenv("TREBLO_STREAM_BASE_URL", TREBLO_STREAM_BASE_URL).rstrip("/"),
            user_cooldown_seconds=_env_float("DISCORD_BRAIN_V2_TREBLO_USER_COOLDOWN_SECONDS", 1800.0),
            max_queue_size=_env_int("DISCORD_BRAIN_V2_TREBLO_MAX_QUEUE_SIZE", 10),
            poll_interval_seconds=_env_float("DISCORD_BRAIN_V2_TREBLO_POLL_INTERVAL_SECONDS", 4.0),
            max_poll_attempts=_env_int("DISCORD_BRAIN_V2_TREBLO_MAX_POLL_ATTEMPTS", 180),
            request_timeout_seconds=_env_float("DISCORD_BRAIN_V2_TREBLO_TIMEOUT_SECONDS", 60.0),
            prompt_max_chars=_env_int("DISCORD_BRAIN_V2_TREBLO_PROMPT_MAX_CHARS", 800),
            style_scale=_env_float("DISCORD_BRAIN_V2_TREBLO_STYLE_SCALE", 3.5),
            prompt_strength=_env_float("DISCORD_BRAIN_V2_TREBLO_PROMPT_STRENGTH", 1.0),
            output_format=os.getenv("DISCORD_BRAIN_V2_TREBLO_OUTPUT_FORMAT", "ogg").strip() or "ogg",
            enable_streaming=_env_bool("DISCORD_BRAIN_V2_TREBLO_ENABLE_STREAMING", True),
            align_lyrics=_env_bool("DISCORD_BRAIN_V2_TREBLO_ALIGN_LYRICS", True),
            length_min_seconds=_env_int("DISCORD_BRAIN_V2_TREBLO_LENGTH_MIN_SECONDS", 30),
            length_max_seconds=_env_int("DISCORD_BRAIN_V2_TREBLO_LENGTH_MAX_SECONDS", 120),
            max_attachment_bytes=_env_int("DISCORD_BRAIN_V2_TREBLO_MAX_ATTACHMENT_BYTES", 8 * 1024 * 1024),
            ai_gateway_api_key=os.getenv("AI_GATEWAY_API_KEY") or os.getenv("VERCEL_OIDC_TOKEN"),
            ai_gateway_model=os.getenv("AI_GATEWAY_MODEL", "deepseek/deepseek-v4-pro"),
        )


@dataclass(slots=True)
class TrebloSongJob:
    job_id: str
    user_id: int
    channel_id: int
    prompt: str
    mode: str = "prompt_only"
    author_name: str = ""
    lyrics: str | None = None
    task_id: str | None = None
    status: str = "queued"
    generation_status: str | None = None
    stream_url: str | None = None
    audio_url: str | None = None
    error: str | None = None
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    attempts: int = 0


class TrebloSongClient:
    def __init__(self, config: TrebloSongConfig):
        self.config = config
        self.lyric_client = VercelAIGatewayJSONClient(
            model=config.ai_gateway_model,
            api_key=config.ai_gateway_api_key,
            base_url=VERCEL_AI_GATEWAY_BASE_URL,
        ) if config.ai_gateway_api_key else None

    async def generate(self, prompt: str, *, mode: str = "prompt_only") -> tuple[dict[str, Any], str | None]:
        payload = await self.payload(prompt, mode=mode)
        response = await self._request(
            "POST",
            "/generations/v3",
            json=payload,
        )
        return response, payload.get("lyrics") if isinstance(payload.get("lyrics"), str) else None

    async def status(self, task_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/generations/status/{task_id}?include_alignment=true")

    async def result(self, task_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/generations/{task_id}")

    async def download_audio(self, url: str) -> tuple[bytes | None, str | None]:
        if self.config.max_attachment_bytes <= 0:
            return None, None
        async with httpx.AsyncClient(timeout=self.config.request_timeout_seconds, follow_redirects=True) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                content_type = response.headers.get("content-type")
                length = response.headers.get("content-length")
                if length and int(length) > self.config.max_attachment_bytes:
                    return None, content_type
                chunks: list[bytes] = []
                total = 0
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > self.config.max_attachment_bytes:
                        return None, content_type
                    chunks.append(chunk)
        return b"".join(chunks), content_type

    async def payload(self, prompt: str, *, mode: str = "prompt_only") -> dict[str, Any]:
        mode = normalize_mode(mode)
        prompt = prompt[: self.config.prompt_max_chars]
        lyrics = await self.generate_lyrics(prompt) if mode == "auto_lyrics" else None
        if mode == "instrumental":
            prompt = f"Instrumental track with no vocals and no lyrics. {prompt}".strip()
        payload: dict[str, Any] = {
            "prompt": prompt,
            "style_scale": self.config.style_scale,
            "prompt_strength": self.config.prompt_strength,
            "output_format": self.config.output_format,
            "enable_streaming": self.config.enable_streaming,
            "align_lyrics": False if mode == "instrumental" else self.config.align_lyrics,
            "length_range": [self.config.length_min_seconds, self.config.length_max_seconds],
        }
        if lyrics:
            payload["lyrics"] = lyrics
        return payload

    async def generate_lyrics(self, prompt: str) -> str:
        if self.lyric_client is None:
            raise TrebloSongError("AI_GATEWAY_API_KEY is required for auto_lyrics mode.")
        response = await self.lyric_client.complete_json(
            instructions=(
                "Write compact, singable original song lyrics for Treblo. "
                "Return only JSON. Keep lines short and chorus-forward. Do not include tags."
            ),
            payload={"brief": prompt, "requirements": ["original lyrics", "compact lines", "strong chorus"]},
            schema={
                "name": "treblo_auto_lyrics",
                "strict": False,
                "schema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "lyrics": {"type": "string"},
                    },
                    "required": ["lyrics"],
                    "additionalProperties": True,
                },
            },
        )
        lyrics = str(response.get("lyrics") or "").strip()
        if not lyrics:
            raise TrebloSongError("AI Gateway did not return usable lyrics.")
        return lyrics

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        if not self.config.api_key:
            raise TrebloSongError("TREBLO_API_KEY is not configured.")
        headers = {"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=self.config.request_timeout_seconds) as client:
            response = await client.request(method, f"{self.config.base_url}{path}", headers=headers, **kwargs)
        text = response.text
        if response.status_code >= 400:
            raise TrebloSongError(f"Treblo request failed ({response.status_code}): {text[:400]}")
        if not text.strip():
            return {}
        try:
            return response.json()
        except ValueError as exc:
            raise TrebloSongError(f"Treblo returned non-JSON response: {text[:400]}") from exc


class TrebloSongQueue:
    def __init__(self, config: TrebloSongConfig | None = None, *, owner_user_ids: set[int] | None = None):
        self.config = config or TrebloSongConfig.from_env()
        self.client = TrebloSongClient(self.config)
        self.owner_user_ids = owner_user_ids or set()
        self.pending: deque[TrebloSongJob] = deque()
        self.jobs: dict[str, TrebloSongJob] = {}
        self.last_submit_at: dict[int, float] = {}
        self.active_job_id: str | None = None
        self._event = asyncio.Event()
        self._closed = False
        self._lock = asyncio.Lock()

    async def submit(
        self,
        *,
        user_id: int,
        channel_id: int,
        prompt: str,
        author_name: str = "",
        mode: str = "prompt_only",
    ) -> TrebloSongJob:
        prompt = " ".join((prompt or "").split())
        mode = normalize_mode(mode)
        if not prompt:
            raise TrebloSongError("usage: `/song prompt:<prompt>`")
        if not self.config.api_key:
            raise TrebloSongError("TREBLO_API_KEY is not configured.")
        if mode == "auto_lyrics" and not self.config.ai_gateway_api_key:
            raise TrebloSongError("AI_GATEWAY_API_KEY is required for auto_lyrics mode.")
        if len(prompt) > self.config.prompt_max_chars:
            prompt = prompt[: self.config.prompt_max_chars].rstrip()
        async with self._lock:
            if len(self.pending) >= self.config.max_queue_size:
                raise TrebloSongError(f"song queue is full ({self.config.max_queue_size}). try again later.")
            now = time.monotonic()
            last = self.last_submit_at.get(user_id)
            if user_id not in self.owner_user_ids and last is not None:
                remaining = self.config.user_cooldown_seconds - (now - last)
                if remaining > 0:
                    raise TrebloSongRateLimited(remaining)
            job = TrebloSongJob(
                job_id=uuid4().hex[:10],
                user_id=user_id,
                channel_id=channel_id,
                prompt=prompt,
                mode=mode,
                author_name=author_name,
            )
            self.jobs[job.job_id] = job
            self.pending.append(job)
            if user_id not in self.owner_user_ids:
                self.last_submit_at[user_id] = now
            self._event.set()
            return job

    def queue_position(self, job_id: str) -> int | None:
        for index, job in enumerate(self.pending, start=1):
            if job.job_id == job_id:
                return index
        return None

    def snapshot(self) -> dict[str, Any]:
        active = self.jobs.get(self.active_job_id or "")
        recent = sorted(self.jobs.values(), key=lambda job: job.updated_at, reverse=True)[:8]
        return {
            "active": active,
            "pending": list(self.pending),
            "recent": recent,
            "max_queue_size": self.config.max_queue_size,
            "cooldown_seconds": self.config.user_cooldown_seconds,
            "owner_user_ids": sorted(self.owner_user_ids),
        }

    async def run(self, bot: Any) -> None:
        while not self._closed:
            job = await self._next_job()
            if job is None:
                continue
            self.active_job_id = job.job_id
            try:
                await self._run_job(bot, job)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Treblo song job %s failed", job.job_id)
                job.status = "failed"
                job.error = str(exc)
                job.updated_at = _utc_now()
                await _send_channel_message(bot, job.channel_id, f"song `{job.job_id}` failed: {job.error}")
            finally:
                self.active_job_id = None

    async def close(self) -> None:
        self._closed = True
        self._event.set()

    async def _next_job(self) -> TrebloSongJob | None:
        while not self._closed:
            async with self._lock:
                if self.pending:
                    return self.pending.popleft()
                self._event.clear()
            await self._event.wait()
        return None

    async def _run_job(self, bot: Any, job: TrebloSongJob) -> None:
        job.status = "submitting"
        job.updated_at = _utc_now()
        await _send_channel_message(bot, job.channel_id, f"song `{job.job_id}` submitting to Treblo (`{job.mode}`).")
        generation, lyrics = await self.client.generate(job.prompt, mode=job.mode)
        job.lyrics = lyrics
        task_id = extract_task_id(generation)
        if not task_id:
            raise TrebloSongError(f"Treblo returned no task id: {_preview(generation)}")
        job.task_id = task_id
        job.status = "generating"
        job.generation_status = "RECEIVED"
        job.stream_url = stream_url_for_task(task_id, self.config.stream_base_url)
        job.updated_at = _utc_now()
        await _send_channel_message(
            bot,
            job.channel_id,
            f"song `{job.job_id}` queued on Treblo task `{task_id}`. polling every {self.config.poll_interval_seconds:g}s.",
        )
        stream_announced = False
        for attempt in range(1, self.config.max_poll_attempts + 1):
            job.attempts = attempt
            status_payload = await self.client.status(task_id)
            status = normalize_status(status_payload)
            job.generation_status = status
            job.updated_at = _utc_now()
            if status == "GENERATING_STREAMING_READY" and not stream_announced:
                stream_announced = True
                await _send_channel_message(bot, job.channel_id, f"song `{job.job_id}` stream ready: {job.stream_url}")
            if status == "SUCCESS":
                await self._finish_success(bot, job)
                return
            if status == "FAILURE":
                result = await self.client.result(task_id)
                raise TrebloSongError(extract_error(result) or "Treblo generation failed.")
            await asyncio.sleep(self.config.poll_interval_seconds)
        raise TrebloSongError(f"polling timed out after {self.config.max_poll_attempts} attempts.")

    async def _finish_success(self, bot: Any, job: TrebloSongJob) -> None:
        result = await self.client.result(job.task_id or "")
        audio_url = extract_audio_url(result)
        job.status = "ready"
        job.generation_status = "SUCCESS"
        job.audio_url = audio_url
        job.updated_at = _utc_now()
        if not audio_url:
            await _send_channel_message(bot, job.channel_id, f"song `{job.job_id}` finished, but Treblo returned no audio URL.")
            return
        audio, content_type = await self.client.download_audio(audio_url)
        if audio:
            extension = "ogg" if "ogg" in (content_type or "").lower() else self.config.output_format
            await _send_channel_file(
                bot,
                job.channel_id,
                filename=f"treblo-{job.job_id}.{extension}",
                data=audio,
                message=f"song `{job.job_id}` complete.",
            )
            return
        await _send_channel_message(bot, job.channel_id, f"song `{job.job_id}` complete: {audio_url}")


class TrebloSongError(RuntimeError):
    pass


class TrebloSongRateLimited(TrebloSongError):
    def __init__(self, remaining_seconds: float):
        self.remaining_seconds = max(0.0, remaining_seconds)
        super().__init__(f"rate limited. try again in {_format_seconds(self.remaining_seconds)}.")


def normalize_status(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        status = value.get("status")
        return status if isinstance(status, str) else "UNKNOWN"
    return "UNKNOWN"


def normalize_mode(value: str) -> str:
    normalized = (value or "prompt_only").strip().lower().replace("-", "_")
    aliases = {
        "prompt": "prompt_only",
        "plain": "prompt_only",
        "auto": "auto_lyrics",
        "lyrics": "auto_lyrics",
        "auto_lyric": "auto_lyrics",
        "autolyrics": "auto_lyrics",
        "instrumental": "instrumental",
        "no_vocals": "instrumental",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"prompt_only", "auto_lyrics", "instrumental"}:
        raise TrebloSongError("mode must be prompt_only, auto_lyrics, or instrumental.")
    return normalized


def extract_task_id(value: Any) -> str | None:
    return _first_string_for_keys(value, {"task_id", "taskId", "task", "id", "generation_id", "generationId"})


def extract_audio_url(value: Any) -> str | None:
    song_paths = _value_for_key(value, "song_paths")
    if isinstance(song_paths, list):
        for item in song_paths:
            if isinstance(item, str) and item:
                return item
    return _first_string_for_keys(value, {"audio_url", "audioUrl", "song_url", "songUrl", "url", "download_url", "downloadUrl"})


def extract_error(value: Any) -> str | None:
    return _first_string_for_keys(value, {"error_message", "errorMessage", "error", "message", "detail"})


def stream_url_for_task(task_id: str, base_url: str = TREBLO_STREAM_BASE_URL) -> str:
    return f"{base_url.rstrip('/')}/stream/{task_id}"


def _first_string_for_keys(value: Any, keys: set[str]) -> str | None:
    if isinstance(value, dict):
        for key in keys:
            found = value.get(key)
            if isinstance(found, str) and found.strip():
                return found.strip()
        for found in value.values():
            nested = _first_string_for_keys(found, keys)
            if nested:
                return nested
    elif isinstance(value, list):
        for item in value:
            nested = _first_string_for_keys(item, keys)
            if nested:
                return nested
    return None


def _value_for_key(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for item in value.values():
            found = _value_for_key(item, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _value_for_key(item, key)
            if found is not None:
                return found
    return None


async def _send_channel_message(bot: Any, channel_id: int, content: str) -> None:
    channel = bot.get_channel(channel_id) if hasattr(bot, "get_channel") else None
    if channel is None and hasattr(bot, "fetch_channel"):
        channel = await bot.fetch_channel(channel_id)
    if channel is None:
        return
    await channel.send(content[:1900])


async def _send_channel_file(bot: Any, channel_id: int, *, filename: str, data: bytes, message: str) -> None:
    import discord
    import io

    channel = bot.get_channel(channel_id) if hasattr(bot, "get_channel") else None
    if channel is None and hasattr(bot, "fetch_channel"):
        channel = await bot.fetch_channel(channel_id)
    if channel is None:
        return
    await channel.send(message[:1900], file=discord.File(io.BytesIO(data), filename=filename))


def _preview(value: Any, *, limit: int = 400) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _format_seconds(seconds: float) -> str:
    seconds = int(max(0, round(seconds)))
    minutes, sec = divmod(seconds, 60)
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{sec}s"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
