from __future__ import annotations

import base64
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

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
    options: dict[str, Any] = Field(default_factory=dict)


class TTSRequest(BaseModel):
    text: str
    options: dict[str, Any] = Field(default_factory=dict)


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
        model_ids = await _list_openai_models(app.state.brain)
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
        audio = decode_audio_base64(request.audio)
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
    async def tts_voices() -> list[dict[str, Any]]:
        from .tts import discover_piper_voices

        return [voice.model_dump() for voice in discover_piper_voices()]

    @app.websocket("/stream")
    async def stream(websocket: WebSocket) -> None:
        await _brain_socket(app.state.brain, websocket, default_tts=False)

    @app.websocket("/tts")
    async def tts_socket(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                payload = await websocket.receive_json()
                request = TTSRequest.model_validate(payload)
                async for event in app.state.brain.tts_stream(request.text, **request.options):
                    await websocket.send_json(event.model_dump())
        except WebSocketDisconnect:
            return

    @app.websocket("/brain")
    async def brain_socket(websocket: WebSocket) -> None:
        await _brain_socket(app.state.brain, websocket, default_tts=True)

    @app.websocket("/voice")
    async def voice_socket(websocket: WebSocket) -> None:
        await _voice_socket(app.state.brain, websocket)

    return app


async def _list_openai_models(brain: Brain) -> list[str]:
    try:
        result = await brain.client.models.list()
        ids = sorted(
            {
                str(getattr(model, "id", ""))
                for model in getattr(result, "data", [])
                if _is_chat_model(str(getattr(model, "id", "")))
            }
        )
        return ids or list(FALLBACK_MODELS)
    except Exception:
        return list(FALLBACK_MODELS)


def _is_chat_model(model_id: str) -> bool:
    return model_id.startswith(("gpt-", "o", "chatgpt-"))


async def _brain_socket(brain: Brain, websocket: WebSocket, *, default_tts: bool) -> None:
    await websocket.accept()
    try:
        while True:
            payload = await websocket.receive_json()
            message_type = payload.get("type", "ask")
            if message_type == "cancel":
                await websocket.send_json({"type": "cancelled"})
                continue
            if message_type == "heartbeat":
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
                await websocket.send_json({"type": event_type, **result.model_dump()})
                continue
            if message_type != "ask":
                await websocket.send_json(
                    {"type": "error", "message": f"Unsupported message type: {message_type}"}
                )
                continue
            request = AskRequest.model_validate(payload)
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
            async for event in event_stream:
                await websocket.send_json(event.model_dump())
    except WebSocketDisconnect:
        return


async def _voice_socket(brain: Brain, websocket: WebSocket) -> None:
    await websocket.accept()
    session: VoiceStartRequest | None = None
    buffer = None
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                return
            if message.get("bytes") is not None:
                if session is None or buffer is None:
                    await websocket.send_json(
                        {"type": "error", "message": "audio.start is required before binary audio"}
                    )
                    continue
                await _handle_voice_audio(brain, websocket, session, buffer, message["bytes"])
                continue
            text = message.get("text")
            if text is None:
                continue
            payload = json.loads(text)
            message_type = payload.get("type", "audio.chunk")
            if message_type == "audio.start":
                session = VoiceStartRequest.model_validate(payload)
                if session.encoding != "pcm_s16le":
                    await websocket.send_json(
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
                await websocket.send_json(
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
                await websocket.send_json({"type": "cancelled"})
                continue
            if session is None or buffer is None:
                await websocket.send_json(
                    {"type": "error", "message": "audio.start is required before audio chunks"}
                )
                continue
            if message_type == "audio.stop":
                result = buffer.flush()
                if result.speech_ended:
                    await websocket.send_json({"type": "vad.speech.end", "discarded": result.discarded})
                if result.utterance_audio:
                    await _run_voice_turn(brain, websocket, session, result.utterance_audio)
                continue
            if message_type == "audio.chunk":
                audio = base64.b64decode(payload.get("audio", ""))
                await _handle_voice_audio(brain, websocket, session, buffer, audio)
                continue
            await websocket.send_json(
                {"type": "error", "message": f"Unsupported voice message type: {message_type}"}
            )
    except WebSocketDisconnect:
        return


async def _handle_voice_audio(
    brain: Brain,
    websocket: WebSocket,
    session: VoiceStartRequest,
    buffer: Any,
    audio: bytes,
) -> None:
    result = buffer.push(audio)
    if result.speech_started:
        await websocket.send_json({"type": "vad.speech.start"})
    if result.speech_ended:
        await websocket.send_json({"type": "vad.speech.end", "discarded": result.discarded})
    if result.utterance_audio:
        await _run_voice_turn(brain, websocket, session, result.utterance_audio)


async def _run_voice_turn(
    brain: Brain,
    websocket: WebSocket,
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
        await websocket.send_json(event.model_dump())
