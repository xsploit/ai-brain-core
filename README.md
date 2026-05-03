# AI Brain Core

Reusable Python brain for chatbots, waifus, VRM apps, Twitch/Discord bots, and local tools.

The core uses OpenAI Responses + Conversations through the official `openai` Python SDK. It exposes a high-level API for normal use while keeping raw Responses API options available through keyword pass-through.

## Features

- Responses API wrapper with Conversations, `previous_response_id`, prompt caching, compaction, tools, structured outputs, vision inputs, and streaming.
- Local SQLite thread metadata.
- SQLite vector memory with optional `sqlite-vec` detection and a Python cosine fallback.
- Agentic function-tool loop with Python tool registration.
- FastAPI HTTP + WebSocket adapter for bots, apps, VRM clients, and future TTS.
- Piper TTS with persistent process, executable, HTTP, and null providers.
- OpenClaw-style heartbeat autonomy with `HEARTBEAT.md`, interval tasks, active hours, random wake probability, and silent `HEARTBEAT_OK` acknowledgments.
- Windows-native speech input path with VAD endpointing, optional `faster-whisper`, and full voice loop streaming into ordered Piper TTS.

## Quickstart

```powershell
git clone https://github.com/your-org/ai-brain-core.git
cd ai-brain-core
.\scripts\setup.ps1
.\scripts\run-server.ps1
```

On Linux/macOS:

```bash
git clone https://github.com/your-org/ai-brain-core.git
cd ai-brain-core
./scripts/setup.sh
./scripts/run-server.sh
```

Setup uses `uv` and installs the full framework by default: core brain,
FastAPI/WebSockets, vector memory, STT/VAD, TTS support, and test tooling. It
also creates `.env` from `.env.example` if you do not have one yet.
Python 3.11 is the default because the full local STT/VAD stack depends on
audio wheels that are not consistently published for Python 3.10.

Lightweight core-only install:

```powershell
.\scripts\setup.ps1 -Core
```

Manual uv commands:

```powershell
uv sync --extra all --extra dev
uv run pytest -q
uv run aibrain serve --env-file .env
```

Responses streaming defaults to the official SDK HTTP/SSE transport. To test
OpenAI Responses WebSocket mode for persistent `/v1/responses` streaming:

```powershell
$env:AIBRAIN_OPENAI_STREAM_TRANSPORT="websocket"
uv run aibrain chat --tts --stats --env-file .env
```

Server mode uses a small Responses WebSocket pool so one long stream does not
block every other client:

```powershell
$env:AIBRAIN_OPENAI_WS_POOL_SIZE="4"
$env:AIBRAIN_STREAM_EVENT_QUEUE_MAX="256"
$env:AIBRAIN_MODELS_CACHE_TTL_SECONDS="300"
$env:AIBRAIN_THREAD_LOCK_CACHE_SIZE="4096"
```

Stateful turns for the same `thread_id` are serialized so concurrent Discord or
Twitch events do not split a channel across duplicate OpenAI conversations.
`cancel` messages on `/stream`, `/brain`, and `/voice` cancel active work instead
of only acknowledging the button click.

If `sqlite-vec` is unavailable on your machine, memory still works with the built-in fallback.

Set your OpenAI key:

```powershell
$env:OPENAI_API_KEY="sk-..."
```

Or point the brain at an existing `.env` file:

```powershell
$env:AIBRAIN_ENV_FILE="C:\path\to\.env"
uv run aibrain serve --env-file "C:\path\to\.env"
```

## Minimal Usage

```python
import asyncio
from aibrain import Brain, Persona

async def main():
    brain = Brain()
    persona = Persona(
        id="riko",
        name="Riko",
        instructions="You are Riko, a direct and playful AI companion.",
        model="gpt-5-nano",
    )

    response = await brain.ask(
        "Remember that I like synthwave.",
        persona=persona,
        thread_id="discord:user:123",
    )
    print(response.text)

asyncio.run(main())
```

## Streaming

```python
async for event in brain.stream("Say hi quickly.", thread_id="twitch:channel:main"):
    if event.type == "text.delta":
        print(event.data["text"], end="", flush=True)
```

## Thread Policies

```python
from aibrain import ThreadPolicy

thread_id = ThreadPolicy.discord_channel(guild_id, channel_id)
thread_id = ThreadPolicy.discord_dm(user_id)
thread_id = ThreadPolicy.twitch_channel("mychannel")
```

## Heartbeat Autonomy

The brain owns the reusable decision loop. Platform adapters own delivery. A
Discord bot can call `heartbeat_tick()`, then post only when
`result.decision.should_act` is true.

```python
from aibrain import AutonomyAction, Brain, HeartbeatConfig

brain = Brain()

result = await brain.heartbeat_tick(
    thread_id="discord:channel:123",
    config=HeartbeatConfig(
        heartbeat_path="HEARTBEAT.md",
        interval_seconds=1800,
        jitter_seconds=300,
        run_probability=0.35,
        active_hours={
            "start": "09:00",
            "end": "22:00",
            "timezone": "America/Los_Angeles",
        },
    ),
    actions=[
        AutonomyAction(
            name="message",
            description="Send a short proactive message to the current channel.",
        )
    ],
    context={"recent_chat": "adapter supplied summary goes here"},
)

if result.decision and result.decision.should_act:
    await discord_channel.send(result.decision.message)
```

`HEARTBEAT.md` can be a tiny checklist or include interval tasks:

```text
tasks:
- name: inbox-triage
  interval: 30m
  prompt: "Check for urgent unread messages."

- If nothing needs attention, reply HEARTBEAT_OK.
```

Use `run_probability` when you want a heartbeat cadence without every tick
calling the model. `jitter_seconds` randomizes timing; `run_probability`
randomizes whether a scheduled tick wakes the model at all.

## Piper TTS

Configure Piper with env vars:

```powershell
$env:AIBRAIN_TTS_VOICE="riko_fish_s2_200_rvc_32k_2259"
$env:PIPER_EXE="C:\path\to\piper.exe"
$env:PIPER_MODEL="C:\path\to\voice.onnx"
$env:PIPER_CONFIG="C:\path\to\voice.onnx.json"
$env:AIBRAIN_TTS_VOICE_ROOTS="D:\voices;D:\more-voices"
$env:AIBRAIN_TTS_MANIFESTS="D:\voices\manifest.json"
```

The runtime keeps Piper processes hot per selected voice. `GET /tts/voices`
lists voices from `AIBRAIN_TTS_VOICE_ROOTS` and `AIBRAIN_TTS_MANIFESTS`.
Voice discovery is cached between requests; pass `refresh=true` to the endpoint
after adding or changing voice files. For bots that switch between many voices,
cap the hot Piper process pool with `PIPER_PROCESS_POOL_MAX`.
Persistent Piper raw streaming uses `PIPER_PROCESS_IDLE_TIMEOUT` to detect when
one text segment is done. The default is `0.15`, which was faster in local tests
without dropping audio; if a specific voice cuts off words, raise it toward
`0.25` or `0.45`.

Terminal smoke test with streaming text and immediate TTS playback:

```powershell
uv run aibrain chat --tts --stats --env-file .env
```

The chat CLI uses OpenAI Responses WebSocket streaming by default. For live
voice latency testing, use the stripped path:

```powershell
uv run aibrain chat --fast --tts --stats --env-file .env
```

`--fast` keeps the WebSocket and Piper process hot, uses `gpt-4.1-mini` unless
you pass `--model`, and disables conversation state, memory retrieval, and tool
schemas for that turn path. That is the right mode for measuring first-token and
first-audio latency without framework features adding work before the stream.

Use `--audio-player none` to measure backend first-text/first-audio timing
without local playback, or `--voice <slug>` to select a discovered Piper voice.

Compare Responses HTTP/SSE and WebSocket latency:

```powershell
uv run aibrain bench --transport both --rounds 5 --fast --tts --env-file .env
```

Use text + audio streaming:

```python
async for event in brain.stream_with_tts("Talk fast.", thread_id="vrm:local"):
    if event.type == "text.delta":
        print(event.data["text"], end="")
    if event.type == "tts.audio":
        audio_b64 = event.data["audio"]
```

### TTS Playback Contract

`stream_with_tts` emits a playlist:

```text
tts.playlist.start
tts.start        segment_index=0
tts.audio        segment_index=0
tts.done         segment_index=0
tts.start        segment_index=1
tts.audio        segment_index=1
tts.done         segment_index=1
response.done
tts.playlist.done
```

The server emits TTS audio in order, while `response.done` can arrive before all
audio finishes. Clients should keep the socket open until `tts.playlist.done`,
play each audio chunk in order, and avoid blocking text rendering on audio
decode/playback.

## Speech Input

The first STT backend is Windows-native `faster-whisper`; Silero VAD is used for
speech endpointing when the `stt` extra is installed. Raw streaming audio is
`pcm_s16le`, 16 kHz, mono.

```python
result = await brain.transcribe(
    audio_bytes,
    format="pcm_s16le",
    sample_rate=16000,
    channels=1,
)
print(result.text)
```

For the full voice loop:

```python
async for event in brain.voice_stream(
    audio_bytes,
    thread_id="discord:channel:123",
    stt_options={"format": "pcm_s16le", "sample_rate": 16000, "channels": 1},
):
    if event.type == "stt.final":
        print("heard:", event.data["text"])
    if event.type == "tts.audio":
        play_or_queue(event.data)
```

Runtime overrides:

```powershell
$env:AIBRAIN_STT_MODEL="small.en"
$env:AIBRAIN_STT_DEVICE="auto"
$env:AIBRAIN_STT_COMPUTE_TYPE="auto"
$env:AIBRAIN_VAD_PROVIDER="silero"
```

## FastAPI Server

```powershell
uv run aibrain serve --host 127.0.0.1 --port 8765 --env-file .env
```

Open the built-in test console:

```text
http://127.0.0.1:8765/webchat
```

WebSocket clients connect to:

```text
ws://127.0.0.1:8765/stream
ws://127.0.0.1:8765/brain
ws://127.0.0.1:8765/tts
ws://127.0.0.1:8765/voice
```

HTTP adapters can also call `POST /heartbeat`. WebSocket adapters can send a
message with `"type": "heartbeat"` to `/brain` or `/stream`.
Use `POST /stt` for base64 PCM/WAV/FLAC transcription. Use `/voice` for
`audio.start` + binary PCM frames + `audio.stop`; finalized utterances emit
`stt.final`, LLM streaming events, and ordered `tts.*` playlist events.
For TTS output, clients can request `"audio_transport": "binary"` on `/brain`,
`/stream`, `/tts`, or `/voice`; JSON/base64 remains the default.

Send:

```json
{
  "type": "ask",
  "thread_id": "vrm:local:user",
  "text": "Explain what you see.",
  "images": [{"url": "https://example.com/screenshot.png"}]
}
```
