from __future__ import annotations

import asyncio
import base64
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .autonomy import AutonomyAction, HeartbeatConfig
from .config import BrainConfig, Persona
from .core import Brain
from .inputs import ImageInput
from .policy import MemoryPolicy
from .stt import AudioEncoding, VADConfig, decode_audio_base64


AudioTransport = Literal["json_base64", "binary"]


class ThreadRequest(BaseModel):
    thread_id: str | None = None
    persona: Persona | dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    create_remote: bool = False


class AskRequest(BaseModel):
    text: str
    thread_id: str | None = None
    persona: Persona | dict[str, Any] | None = None
    images: list[ImageInput | dict[str, Any] | str] | None = None
    use_memory: bool | MemoryPolicy | dict[str, Any] = True
    tool_names: list[str] | None = None
    tts: bool = False
    tts_options: dict[str, Any] = Field(default_factory=dict)
    audio_transport: AudioTransport = "json_base64"
    options: dict[str, Any] = Field(default_factory=dict)


class TTSRequest(BaseModel):
    text: str
    options: dict[str, Any] = Field(default_factory=dict)
    audio_transport: AudioTransport = "json_base64"


class STTRequest(BaseModel):
    audio: str
    format: AudioEncoding = "pcm_s16le"
    sample_rate: int = 16000
    channels: int = 1
    language: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class VoiceStartRequest(BaseModel):
    type: str = "audio.start"
    thread_id: str
    persona: Persona | dict[str, Any] | None = None
    encoding: AudioEncoding = "pcm_s16le"
    sample_rate: int = 16000
    channels: int = 1
    language: str | None = None
    tts: bool = True
    use_memory: bool | MemoryPolicy | dict[str, Any] = True
    tool_names: list[str] | None = None
    vad: VADConfig | dict[str, Any] | None = None
    stt_options: dict[str, Any] = Field(default_factory=dict)
    tts_options: dict[str, Any] = Field(default_factory=dict)
    audio_transport: AudioTransport = "json_base64"
    options: dict[str, Any] = Field(default_factory=dict)


class HeartbeatRequest(BaseModel):
    thread_id: str
    persona: Persona | dict[str, Any] | None = None
    context: dict[str, Any] = Field(default_factory=dict)
    actions: list[AutonomyAction | dict[str, Any]] = Field(default_factory=list)
    config: HeartbeatConfig | dict[str, Any] | None = None
    options: dict[str, Any] = Field(default_factory=dict)


FALLBACK_MODELS = [
    "gpt-5-nano",
    "gpt-5-mini",
    "gpt-5",
    "gpt-4.1-mini",
    "gpt-4.1",
]


def create_app(brain: Brain | None = None, config: BrainConfig | None = None) -> FastAPI:
    brain_instance = brain or Brain(config=config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.brain = brain_instance
        if app.state.brain.config.tts_config.warmup_on_start:
            await app.state.brain.warmup()
        try:
            yield
        finally:
            await app.state.brain.close()

    app = FastAPI(title="AI Brain Core", version="0.1.0", lifespan=lifespan)
    app.state.brain = brain_instance
    app.state.models_cache = {"expires_at": 0.0, "ids": None}
    app.state.tts_voice_cache = None
    webchat_dir = Path(__file__).with_name("webchat")

    if webchat_dir.exists():
        app.mount(
            "/webchat/assets",
            StaticFiles(directory=webchat_dir),
            name="webchat-assets",
        )

        @app.get("/", include_in_schema=False)
        async def root() -> RedirectResponse:
            return RedirectResponse(url="/webchat")

        @app.get("/webchat", include_in_schema=False)
        async def webchat() -> FileResponse:
            return FileResponse(webchat_dir / "index.html")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/models")
    async def models() -> list[dict[str, Any]]:
        model_ids = await _list_openai_models(
            app.state.brain,
            cache=app.state.models_cache,
            ttl_seconds=app.state.brain.config.models_cache_ttl_seconds,
        )
        default_model = app.state.brain.config.default_model
        if default_model not in model_ids:
            model_ids.insert(0, default_model)
        return [
            {
                "id": model_id,
                "label": model_id,
                "default": model_id == default_model,
            }
            for model_id in model_ids
        ]

    @app.post("/threads")
    async def create_thread(request: ThreadRequest) -> dict[str, Any]:
        state = await app.state.brain.open_thread(
            thread_id=request.thread_id,
            persona=request.persona,
            metadata=request.metadata,
            create_remote=request.create_remote,
        )
        return {
            "thread_id": state.thread_id,
            "persona_id": state.persona_id,
            "openai_conversation_id": state.openai_conversation_id,
            "last_response_id": state.last_response_id,
            "metadata": state.metadata,
        }

    @app.post("/ask")
    async def ask(request: AskRequest) -> dict[str, Any]:
        if request.tts:
            response = await app.state.brain.ask(
                request.text,
                thread_id=request.thread_id,
                persona=request.persona,
                images=request.images,
                use_memory=request.use_memory,
                tool_names=request.tool_names,
                **request.options,
            )
            audio = await app.state.brain.speak(response.text, **request.tts_options)
        else:
            response = await app.state.brain.ask(
                request.text,
                thread_id=request.thread_id,
                persona=request.persona,
                images=request.images,
                use_memory=request.use_memory,
                tool_names=request.tool_names,
                **request.options,
            )
            audio = None
        return {
            "text": response.text,
            "response_id": response.response_id,
            "conversation_id": response.conversation_id,
            "thread_id": response.thread_id,
            "tool_results": response.tool_results,
            "audio": audio.to_event_data() if audio else None,
            "memory_hits": [
                {"id": hit.id, "content": hit.content, "score": hit.score}
                for hit in response.memory_hits
            ],
        }

    @app.post("/tts")
    async def tts(request: TTSRequest) -> dict[str, Any]:
        audio = await app.state.brain.speak(request.text, **request.options)
        return audio.to_event_data()

    @app.post("/stt")
    async def stt(request: STTRequest) -> dict[str, Any]:
        audio = await asyncio.to_thread(decode_audio_base64, request.audio)
        result = await app.state.brain.transcribe(
            audio,
            format=request.format,
            sample_rate=request.sample_rate,
            channels=request.channels,
            language=request.language,
            **request.options,
        )
        return result.model_dump()

    @app.post("/heartbeat")
    async def heartbeat(request: HeartbeatRequest) -> dict[str, Any]:
        result = await app.state.brain.heartbeat_tick(
            thread_id=request.thread_id,
            persona=request.persona,
            context=request.context,
            actions=request.actions,
            config=request.config,
            **request.options,
        )
        return result.model_dump()

    @app.get("/tts/voices")
    async def tts_voices(refresh: bool = False) -> list[dict[str, Any]]:
        from .tts import discover_piper_voices

        if refresh or app.state.tts_voice_cache is None:
            app.state.tts_voice_cache = [
                voice.model_dump() for voice in discover_piper_voices()
            ]
        return list(app.state.tts_voice_cache)

    @app.websocket("/stream")
    async def stream(websocket: WebSocket) -> None:
        await _brain_socket(app.state.brain, websocket, default_tts=False)

    @app.websocket("/tts")
    async def tts_socket(websocket: WebSocket) -> None:
        await websocket.accept()
        send_lock = asyncio.Lock()
        try:
            while True:
                payload = await websocket.receive_json()
                request = TTSRequest.model_validate(payload)
                async for event in app.state.brain.tts_stream(request.text, **request.options):
                    await _safe_send_event(
                        websocket,
                        send_lock,
                        event.model_dump(),
                        audio_transport=request.audio_transport,
                    )
        except WebSocketDisconnect:
            return

    @app.websocket("/brain")
    async def brain_socket(websocket: WebSocket) -> None:
        await _brain_socket(app.state.brain, websocket, default_tts=True)

    @app.websocket("/voice")
    async def voice_socket(websocket: WebSocket) -> None:
        await _voice_socket(app.state.brain, websocket)

    return app


async def _list_openai_models(
    brain: Brain,
    *,
    cache: dict[str, Any] | None = None,
    ttl_seconds: int = 300,
) -> list[str]:
    now = time.monotonic()
    if cache is not None and cache.get("ids") is not None and cache.get("expires_at", 0) > now:
        return list(cache["ids"])
    try:
        result = await brain.client.models.list()
        ids = sorted(
            {
                str(getattr(model, "id", ""))
                for model in getattr(result, "data", [])
                if _is_chat_model(str(getattr(model, "id", "")))
            }
        )
        model_ids = ids or list(FALLBACK_MODELS)
    except Exception:
        model_ids = list(FALLBACK_MODELS)
    if cache is not None:
        cache["ids"] = list(model_ids)
        cache["expires_at"] = now + max(0, ttl_seconds)
    return model_ids


def _is_chat_model(model_id: str) -> bool:
    return model_id.startswith(("gpt-", "o", "chatgpt-"))


async def _brain_socket(brain: Brain, websocket: WebSocket, *, default_tts: bool) -> None:
    await websocket.accept()
    send_lock = asyncio.Lock()
    active_task: asyncio.Task[None] | None = None
    try:
        while True:
            payload = await websocket.receive_json()
            message_type = payload.get("type", "ask")
            if message_type == "cancel":
                active_task = await _cancel_task(
                    active_task,
                    websocket,
                    send_lock,
                    notify=True,
                )
                continue
            if message_type == "heartbeat":
                if active_task is not None and not active_task.done():
                    await _safe_send_json(
                        websocket,
                        send_lock,
                        {"type": "error", "message": "Cannot run heartbeat while a turn is active"},
                    )
                    continue
                request = HeartbeatRequest.model_validate(payload)
                result = await brain.heartbeat_tick(
                    thread_id=request.thread_id,
                    persona=request.persona,
                    context=request.context,
                    actions=request.actions,
                    config=request.config,
                    **request.options,
                )
                event_type = "heartbeat.skipped" if result.skipped else "heartbeat.decision"
                await _safe_send_json(websocket, send_lock, {"type": event_type, **result.model_dump()})
                continue
            if message_type != "ask":
                await _safe_send_json(
                    websocket,
                    send_lock,
                    {"type": "error", "message": f"Unsupported message type: {message_type}"}
                )
                continue
            active_task = await _cancel_task(active_task, websocket, send_lock, notify=False)
            request = AskRequest.model_validate(payload)
            active_task = asyncio.create_task(
                _send_brain_turn(brain, websocket, send_lock, request, default_tts=default_tts)
            )
    except WebSocketDisconnect:
        await _cancel_task(active_task, websocket, send_lock, notify=False)
        return


async def _voice_socket(brain: Brain, websocket: WebSocket) -> None:
    await websocket.accept()
    send_lock = asyncio.Lock()
    session: VoiceStartRequest | None = None
    buffer = None
    active_task: asyncio.Task[None] | None = None
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                await _cancel_task(active_task, websocket, send_lock, notify=False)
                return
            if message.get("bytes") is not None:
                if session is None or buffer is None:
                    await _safe_send_json(
                        websocket,
                        send_lock,
                        {"type": "error", "message": "audio.start is required before binary audio"}
                    )
                    continue
                result = await _handle_voice_audio(websocket, send_lock, buffer, message["bytes"])
                if result.speech_started and active_task is not None and not active_task.done():
                    active_task = await _cancel_task(active_task, websocket, send_lock, notify=True)
                if result.utterance_audio:
                    active_task = await _start_voice_turn(
                        brain,
                        websocket,
                        send_lock,
                        session,
                        result.utterance_audio,
                        active_task,
                    )
                continue
            text = message.get("text")
            if text is None:
                continue
            payload = json.loads(text)
            message_type = payload.get("type", "audio.chunk")
            if message_type == "audio.start":
                active_task = await _cancel_task(active_task, websocket, send_lock, notify=False)
                session = VoiceStartRequest.model_validate(payload)
                if session.encoding != "pcm_s16le":
                    await _safe_send_json(
                        websocket,
                        send_lock,
                        {
                            "type": "error",
                            "message": "Streaming voice v1 expects pcm_s16le audio",
                        }
                    )
                    session = None
                    buffer = None
                    continue
                buffer = brain.utterance_buffer(
                    vad_config=session.vad,
                    sample_rate=session.sample_rate,
                    channels=session.channels,
                    encoding=session.encoding,
                )
                await _safe_send_json(
                    websocket,
                    send_lock,
                    {
                        "type": "audio.started",
                        "sample_rate": session.sample_rate,
                        "channels": session.channels,
                        "encoding": session.encoding,
                    }
                )
                continue
            if message_type == "audio.cancel":
                if buffer is not None:
                    buffer.reset()
                active_task = await _cancel_task(active_task, websocket, send_lock, notify=True)
                continue
            if session is None or buffer is None:
                await _safe_send_json(
                    websocket,
                    send_lock,
                    {"type": "error", "message": "audio.start is required before audio chunks"}
                )
                continue
            if message_type == "audio.stop":
                result = buffer.flush()
                if result.speech_ended:
                    await _safe_send_json(
                        websocket,
                        send_lock,
                        {"type": "vad.speech.end", "discarded": result.discarded},
                    )
                if result.utterance_audio:
                    active_task = await _start_voice_turn(
                        brain,
                        websocket,
                        send_lock,
                        session,
                        result.utterance_audio,
                        active_task,
                    )
                continue
            if message_type == "audio.chunk":
                audio = base64.b64decode(payload.get("audio", ""))
                result = await _handle_voice_audio(websocket, send_lock, buffer, audio)
                if result.speech_started and active_task is not None and not active_task.done():
                    active_task = await _cancel_task(active_task, websocket, send_lock, notify=True)
                if result.utterance_audio:
                    active_task = await _start_voice_turn(
                        brain,
                        websocket,
                        send_lock,
                        session,
                        result.utterance_audio,
                        active_task,
                    )
                continue
            await _safe_send_json(
                websocket,
                send_lock,
                {"type": "error", "message": f"Unsupported voice message type: {message_type}"}
            )
    except WebSocketDisconnect:
        await _cancel_task(active_task, websocket, send_lock, notify=False)
        return


async def _safe_send_json(
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    payload: dict[str, Any],
) -> None:
    async with send_lock:
        await websocket.send_json(payload)


async def _safe_send_event(
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    payload: dict[str, Any],
    *,
    audio_transport: AudioTransport = "json_base64",
) -> None:
    if payload.get("type") != "tts.audio" or audio_transport != "binary":
        if payload.get("type") == "tts.audio":
            payload = {**payload, "audio_transport": "json_base64"}
        await _safe_send_json(websocket, send_lock, payload)
        return

    audio_b64 = payload.pop("audio", "")
    audio = base64.b64decode(audio_b64) if audio_b64 else b""
    payload.update({"audio_transport": "binary", "binary_bytes": len(audio)})
    async with send_lock:
        await websocket.send_json(payload)
        if audio:
            await websocket.send_bytes(audio)


async def _cancel_task(
    task: asyncio.Task[None] | None,
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    *,
    notify: bool,
) -> asyncio.Task[None] | None:
    if task is not None:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        else:
            try:
                task.result()
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                if notify:
                    await _safe_send_json(
                        websocket,
                        send_lock,
                        {"type": "error", "message": str(exc)},
                    )
    if notify:
        await _safe_send_json(websocket, send_lock, {"type": "cancelled"})
    return None


async def _send_brain_turn(
    brain: Brain,
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    request: AskRequest,
    *,
    default_tts: bool,
) -> None:
    event_stream = (
        brain.stream_with_tts(
            request.text,
            thread_id=request.thread_id,
            persona=request.persona,
            images=request.images,
            use_memory=request.use_memory,
            tool_names=request.tool_names,
            tts_options=request.tts_options,
            **request.options,
        )
        if request.tts or default_tts
        else brain.stream(
            request.text,
            thread_id=request.thread_id,
            persona=request.persona,
            images=request.images,
            use_memory=request.use_memory,
            tool_names=request.tool_names,
            **request.options,
        )
    )
    try:
        async for event in event_stream:
            await _safe_send_event(
                websocket,
                send_lock,
                event.model_dump(),
                audio_transport=request.audio_transport,
            )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        await _safe_send_json(websocket, send_lock, {"type": "error", "message": str(exc)})


async def _handle_voice_audio(
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    buffer: Any,
    audio: bytes,
) -> Any:
    if hasattr(buffer, "push_async"):
        result = await buffer.push_async(audio)
    else:
        result = await asyncio.to_thread(buffer.push, audio)
    if result.speech_started:
        await _safe_send_json(websocket, send_lock, {"type": "vad.speech.start"})
    if result.speech_ended:
        await _safe_send_json(
            websocket,
            send_lock,
            {"type": "vad.speech.end", "discarded": result.discarded},
        )
    return result


async def _start_voice_turn(
    brain: Brain,
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    session: VoiceStartRequest,
    audio: bytes,
    active_task: asyncio.Task[None] | None,
) -> asyncio.Task[None]:
    await _cancel_task(active_task, websocket, send_lock, notify=False)
    return asyncio.create_task(_run_voice_turn(brain, websocket, send_lock, session, audio))


async def _run_voice_turn(
    brain: Brain,
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    session: VoiceStartRequest,
    audio: bytes,
) -> None:
    stt_options = {
        "format": session.encoding,
        "sample_rate": session.sample_rate,
        "channels": session.channels,
        "language": session.language,
        **session.stt_options,
    }
    try:
        async for event in brain.voice_stream(
            audio,
            thread_id=session.thread_id,
            persona=session.persona,
            tts=session.tts,
            stt_options=stt_options,
            tts_options=session.tts_options,
            use_memory=session.use_memory,
            tool_names=session.tool_names,
            **session.options,
        ):
            await _safe_send_event(
                websocket,
                send_lock,
                event.model_dump(),
                audio_transport=session.audio_transport,
            )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        await _safe_send_json(websocket, send_lock, {"type": "error", "message": str(exc)})
