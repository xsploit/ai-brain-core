from __future__ import annotations

import asyncio
import base64
import importlib.util
import sys
from dataclasses import dataclass
from typing import Any, Literal

from .tts import write_wav_bytes
from .types import BrainEvent


@dataclass(slots=True)
class AudioFrame:
    audio: bytes
    sample_rate: int
    encoding: str = "pcm_s16le"
    channels: int = 1
    segment_index: int | None = None

    @classmethod
    def from_event(cls, event: BrainEvent | dict[str, Any]) -> AudioFrame | None:
        data = event.data if isinstance(event, BrainEvent) else event
        audio_value = data.get("audio")
        if not audio_value:
            return None
        return cls(
            audio=base64.b64decode(audio_value),
            sample_rate=int(data.get("sample_rate") or 22050),
            encoding=str(data.get("encoding") or "pcm_s16le"),
            channels=int(data.get("channels") or 1),
            segment_index=data.get("segment_index"),
        )


class BaseAudioPlayer:
    name = "base"

    async def play(self, frame: AudioFrame) -> None:
        raise NotImplementedError

    async def close(self) -> None:
        return None


class NullAudioPlayer(BaseAudioPlayer):
    name = "none"

    async def play(self, frame: AudioFrame) -> None:
        return None


class WinsoundAudioPlayer(BaseAudioPlayer):
    name = "winsound"

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise RuntimeError("winsound playback is only available on Windows")
        import winsound

        self._winsound = winsound
        self._lock = asyncio.Lock()

    async def play(self, frame: AudioFrame) -> None:
        if frame.encoding == "pcm_s16le":
            payload = write_wav_bytes(
                frame.audio,
                sample_rate=frame.sample_rate,
                channels=frame.channels,
            )
        elif frame.encoding == "wav":
            payload = frame.audio
        else:
            raise ValueError(f"Unsupported audio encoding: {frame.encoding}")
        async with self._lock:
            await asyncio.to_thread(
                self._winsound.PlaySound,
                payload,
                self._winsound.SND_MEMORY,
            )


class SoundDeviceAudioPlayer(BaseAudioPlayer):
    name = "sounddevice"

    def __init__(self) -> None:
        import sounddevice as sounddevice

        self._sounddevice = sounddevice
        self._stream = None
        self._stream_key: tuple[int, int] | None = None
        self._lock = asyncio.Lock()

    async def play(self, frame: AudioFrame) -> None:
        if frame.encoding != "pcm_s16le":
            raise ValueError(f"Unsupported audio encoding: {frame.encoding}")
        async with self._lock:
            await asyncio.to_thread(self._play_sync, frame)

    def _play_sync(self, frame: AudioFrame) -> None:
        key = (frame.sample_rate, frame.channels)
        if self._stream_key != key:
            self._close_sync()
            self._stream = self._sounddevice.RawOutputStream(
                samplerate=frame.sample_rate,
                channels=frame.channels,
                dtype="int16",
                blocksize=0,
            )
            self._stream.start()
            self._stream_key = key
        self._stream.write(frame.audio)

    async def close(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._close_sync)

    def _close_sync(self) -> None:
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None
        self._stream_key = None


def create_audio_player(
    kind: Literal["auto", "sounddevice", "winsound", "none"] = "auto",
) -> BaseAudioPlayer:
    if kind == "none":
        return NullAudioPlayer()
    if kind in {"auto", "sounddevice"} and importlib.util.find_spec("sounddevice") is not None:
        return SoundDeviceAudioPlayer()
    if kind == "sounddevice":
        raise RuntimeError("sounddevice is not installed. Install ai-brain-core[audio].")
    if kind in {"auto", "winsound"} and sys.platform == "win32":
        return WinsoundAudioPlayer()
    if kind == "winsound":
        raise RuntimeError("winsound playback is only available on Windows.")
    return NullAudioPlayer()


class AudioPlaybackWorker:
    def __init__(self, player: BaseAudioPlayer | None = None):
        self.player = player or create_audio_player()
        self.queue: asyncio.Queue[AudioFrame] = asyncio.Queue()
        self.task: asyncio.Task[None] | None = None
        self.frames = 0
        self.bytes = 0

    async def start(self) -> None:
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self._run())

    async def enqueue_event(self, event: BrainEvent | dict[str, Any]) -> None:
        frame = AudioFrame.from_event(event)
        if frame is None:
            return
        await self.enqueue(frame)

    async def enqueue(self, frame: AudioFrame) -> None:
        await self.queue.put(frame)

    async def drain(self) -> None:
        await self.queue.join()

    async def close(self) -> None:
        await self.drain()
        if self.task is not None:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        await self.player.close()

    async def _run(self) -> None:
        while True:
            frame = await self.queue.get()
            batch = [frame]
            try:
                while True:
                    try:
                        next_frame = self.queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if _can_merge(batch[-1], next_frame):
                        batch.append(next_frame)
                    else:
                        await self._play_batch(batch)
                        _mark_done(self.queue, batch)
                        batch = [next_frame]
                await self._play_batch(batch)
            finally:
                _mark_done(self.queue, batch)

    async def _play_batch(self, frames: list[AudioFrame]) -> None:
        if not frames:
            return
        frame = _merge_frames(frames)
        await self.player.play(frame)
        self.frames += len(frames)
        self.bytes += len(frame.audio)


def _mark_done(queue: asyncio.Queue[AudioFrame], frames: list[AudioFrame]) -> None:
    for _ in frames:
        queue.task_done()


def _can_merge(left: AudioFrame, right: AudioFrame) -> bool:
    return (
        left.encoding == right.encoding == "pcm_s16le"
        and left.sample_rate == right.sample_rate
        and left.channels == right.channels
    )


def _merge_frames(frames: list[AudioFrame]) -> AudioFrame:
    first = frames[0]
    if len(frames) == 1 or first.encoding != "pcm_s16le":
        return first
    return AudioFrame(
        audio=b"".join(frame.audio for frame in frames),
        sample_rate=first.sample_rate,
        encoding=first.encoding,
        channels=first.channels,
        segment_index=first.segment_index,
    )
