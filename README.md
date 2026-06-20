# AI Brain Core

Reusable Python brain for bots, avatars, desktop agents, voice apps, and local tools.

[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB)](https://www.python.org/)
[![Vercel AI Gateway](https://img.shields.io/badge/Vercel%20AI%20Gateway-Responses%20API-111111)](https://vercel.com/docs/ai-gateway)
[![FastAPI](https://img.shields.io/badge/FastAPI-HTTP%20%2B%20WebSocket-009688)](https://fastapi.tiangolo.com/)
[![CI](https://github.com/xsploit/ai-brain-core/actions/workflows/tests.yml/badge.svg)](https://github.com/xsploit/ai-brain-core/actions/workflows/tests.yml)
[![Tests](https://img.shields.io/badge/tests-116%20passing-2E7D32)](#development)

AI Brain Core is a provider-routed brain framework. By default it uses Vercel
AI Gateway with the OpenAI-compatible Responses API, but the framework owns
conversation state locally instead of depending on a provider Conversations API.
It gives you one importable "brain" that can be dropped into a Discord bot,
Twitch bot, VRM app, web chat, desktop assistant, or voice loop.

Repository: https://github.com/xsploit/ai-brain-core

The core goal is simple: keep the AI state, memory, tools, streaming, and voice
pipeline in one reusable Python package so every new bot does not start from
zero.

## What You Get

| Area | Built in |
| --- | --- |
| Model gateway | Vercel AI Gateway by default, OpenAI-compatible Responses API, structured output, vision input, streaming, optional legacy OpenAI mode |
| Agent loop | Python tool registry, function calling, parallel tool execution, max-step guard, streaming tool events |
| State | Local SQLite conversation history, SQLite thread store, per-thread turn locks, Discord/Twitch-friendly thread policies |
| Memory | GRILLO, Ladybug/TurboVec auto backends, AI Gateway embeddings, SQLite/hash fallback, configurable memory policy |
| Voice out | Piper executable, hot Piper process mode, Piper HTTP, ordered TTS playlist events |
| Voice in | PCM/WAV/FLAC decode, Silero or energy VAD, faster-whisper STT provider |
| Server | FastAPI HTTP routes, WebSocket `/stream`, `/brain`, `/tts`, `/voice`, built-in web test console |
| Autonomy | OpenClaw-style heartbeat loop with `HEARTBEAT.md`, active hours, jitter, random wake probability |

## Latency Snapshot

Local benchmark from the current fast path, using Responses streaming,
Piper process TTS, memory off, tools off, and no speaker playback:

```text
WebSocket + Piper:
  "hey"       avg first_text ~462ms   avg first_audio ~584ms
  "whats new" avg first_text ~539ms   avg first_audio ~654ms

Transport comparison:
  HTTP/SSE   avg first_text ~1194ms   avg first_audio ~1449ms
  WebSocket  avg first_text ~493ms    avg first_audio ~612ms
```

Your numbers will vary by model, region, output length, hardware, and voice
model. The important part is that the framework exposes the timing hooks and
benchmark command so you can measure your own setup instead of guessing.

```powershell
uv run aibrain bench --transport both --rounds 5 --fast --tts --env-file .env
```

## Quickstart

```powershell
git clone https://github.com/xsploit/ai-brain-core.git
cd ai-brain-core
.\scripts\setup.ps1
.\scripts\run-server.ps1
```

Linux/macOS:

```bash
git clone https://github.com/xsploit/ai-brain-core.git
cd ai-brain-core
./scripts/setup.sh
./scripts/run-server.sh
```

The setup scripts use `uv`, install the full framework by default, and create
`.env` from `.env.example` when needed.

Core-only install:

```powershell
.\scripts\setup.ps1 -Core
```

Manual setup:

```powershell
uv sync --extra all --extra dev
uv run pytest -q
uv run aibrain serve --env-file .env
```

Set your API key:

```powershell
$env:OPENAI_API_KEY="sk-..."
```

Or use an env file:

```powershell
uv run aibrain serve --env-file "C:\path\to\.env"
```

## Minimal Brain

```python
import asyncio
from aibrain import Brain, Persona


async def main():
    brain = Brain()
    persona = Persona(
        id="riko",
        name="Riko",
        instructions="You are Riko. Be direct, useful, and quick.",
        model="gpt-5-nano",
    )

    response = await brain.ask(
        "Remember that I like synthwave.",
        persona=persona,
        thread_id="discord:user:123",
    )

    print(response.text)
    await brain.close()


asyncio.run(main())
```

## Streaming Text

```python
async for event in brain.stream("Say hi quickly.", thread_id="twitch:channel:main"):
    if event.type == "text.delta":
        print(event.data["text"], end="", flush=True)
```

## Streaming Text + TTS

`stream_with_tts()` emits text deltas as they arrive, splits text into TTS-safe
chunks, renders each chunk through Piper, and emits ordered audio events.

```python
async for event in brain.stream_with_tts("Give me the short version.", thread_id="vrm:local"):
    if event.type == "text.delta":
        print(event.data["text"], end="", flush=True)

    if event.type == "tts.audio":
        audio_b64 = event.data["audio"]
        segment = event.data["segment_index"]
        # Queue/play this segment in order.
```

Playlist contract:

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

Clients should keep the socket open until `tts.playlist.done`, play each audio
chunk in order, and never block text rendering on audio playback.

## Fast Voice Test

Use the CLI to test the actual voice path:

```powershell
uv run aibrain chat --fast --tts --stats --env-file .env
```

`--fast` sets the low-latency path:

- OpenAI Responses WebSocket transport
- stateless turn
- memory off
- tools off
- hot Piper process
- `gpt-4.1-mini` unless you pass `--model`

Measure backend timing without speaker hardware:

```powershell
uv run aibrain chat --fast --tts --stats --audio-player none --env-file .env
```

## FastAPI Server

```powershell
uv run aibrain serve --host 127.0.0.1 --port 8765 --env-file .env
```

Built-in test console:

```text
http://127.0.0.1:8765/webchat
```

The browser console records mic audio through `AudioWorklet` when available and
falls back to `ScriptProcessor` only on older browsers. Both paths send the same
binary `pcm_s16le`, 16 kHz mono frames to `/voice`.

Endpoints:

| Endpoint | Use |
| --- | --- |
| `GET /health` | health check |
| `GET /models` | OpenAI model dropdown source |
| `GET /tts/voices` | discovered Piper voices |
| `POST /ask` | non-streaming brain turn |
| `POST /tts` | TTS render |
| `POST /stt` | speech-to-text |
| `POST /heartbeat` | one autonomy tick |
| `WS /stream` | streaming text |
| `WS /brain` | streaming text + default TTS |
| `WS /tts` | streaming TTS only |
| `WS /voice` | mic/audio stream -> VAD -> STT -> brain -> TTS |

Vision input example:

```json
{
  "type": "ask",
  "thread_id": "desktop:user",
  "text": "Explain what you see.",
  "images": [{"url": "https://example.com/screenshot.png"}]
}
```

## Thread IDs

Thread IDs are how you map platform state to brain state.

```python
from aibrain import ThreadPolicy

discord_channel = ThreadPolicy.discord_channel(guild_id, channel_id)
discord_dm = ThreadPolicy.discord_dm(user_id)
twitch_channel = ThreadPolicy.twitch_channel("mychannel")
```

Stateful turns for the same `thread_id` are serialized so two Discord messages
cannot accidentally create two remote conversations for the same channel.

## Memory

Memory is optional and configurable. The default store is SQLite. If
`sqlite-vec` is installed, vector search uses it; otherwise the framework falls
back to Python cosine similarity.

```python
await brain.remember(
    "The user likes short answers and synthwave.",
    thread_id="discord:user:123",
    importance=0.8,
)

hits = await brain.search_memory(
    "How should I answer this user?",
    thread_id="discord:user:123",
    top_k=5,
)
```

### Hybrid Memory Stack

For richer agents, enable the v2 memory stack:

```python
from aibrain import Brain, BrainConfig, MemoryStackConfig

brain = Brain(
    BrainConfig(
        memory_stack=MemoryStackConfig(
            enabled=True,
            extract_user_events=True,
        )
    )
)
```

The stack is layered so each part has one job:

| Layer | Role |
| --- | --- |
| Raw log | Append-only event history for messages, responses, tools, and summaries |
| GRILLO worker | Extracts durable facts, scores importance, and prepares recall text |
| Temporal graph | Stores entities/facts with `valid_from`, `valid_until`, confidence, source IDs, and supersession metadata |
| Vector recall | Finds semantically similar memories and links them back to raw events/facts |
| Hybrid retriever | Fuses graph facts, vector hits, importance, confidence, and recency into `MemoryRecord` results |

The built-in implementation uses SQLite so the framework works without extra
services. Optional adapters are exposed for LadybugDB graph memory and TurboVec
compressed vector recall when those packages are installed.

### Discord Brain Bot

The Brain-native Discord runtime is a thin demo bot that uses:

- one Brain thread per DM, channel, or Discord thread
- local SQLite conversation history
- hybrid memory stack event extraction and retrieval
- GRILLO scoped memory packets with candidates, diary thoughts, slots, and recall
- backend auto-selection: Ladybug graph + TurboVec recall when installed, SQLite fallback
- streaming Discord replies by editing a live message
- vision input from image attachments
- readable text and PDF attachments
- agentic Discord tools for guild/channel/member/message/thread/role/moderation actions
- Brain tools: `discord_context`, `remember`, `search_memory`, `current_time`, `brain_context`, Tavily tools, and Discord tools

```powershell
uv run --extra discord aibrain-discord
```

Useful env values:

```env
DISCORD_BRAIN_BOT_TOKEN=
DISCORD_BRAIN_DATABASE_PATH=discord_brain.sqlite3
DISCORD_BRAIN_MODEL=deepseek/deepseek-v4-flash
DISCORD_BRAIN_RESPOND_TO_DMS=true
DISCORD_BRAIN_RESPOND_TO_MENTIONS=true
DISCORD_BRAIN_RESPOND_TO_ALL=false
DISCORD_BRAIN_COMMAND_PREFIX=!brain
DISCORD_BRAIN_ALLOWED_USER_IDS=120418341775998976
DISCORD_BRAIN_OWNER_USER_IDS=120418341775998976
DISCORD_BRAIN_MEMBERS_INTENT=true
DISCORD_BRAIN_TOOL_CHANNEL_IDS=
DISCORD_BRAIN_TOOL_AUDIT_CHANNEL_ID=
DISCORD_BRAIN_TEXT_ATTACHMENT_MAX_BYTES=300000
DISCORD_BRAIN_PDF_ATTACHMENT_MAX_PAGES=16
DISCORD_BRAIN_MEMORY_TOP_K=8
DISCORD_BRAIN_GRAPH_BACKEND=auto
DISCORD_BRAIN_VECTOR_BACKEND=auto
```

Enable the Discord Developer Portal privileged Members Intent when using member
list/search tools. `DISCORD_BRAIN_OWNER_USER_IDS` controls owner-only Discord
tools such as guild list, channel/role structure changes, invites, and webhooks.
Discord admins can use lower-risk model/admin commands, but owner-only tools
stay restricted to the configured owner IDs.

Discord commands:

- `!brain status`
- `!brain remember <durable fact>`
- `!brain recall <query>`
- `!brain grillo`
- `!brain grillo tick [beat_type]`
- `!brain grillo context [query]`
- `!model`, `!model set <model-id>`, `!model refresh`, `!model info [model-id]`, `!model export`

The model-facing Discord tool suite includes guild discovery, channel and role
inventory, member list/search, permissions inspection, bounded channel history
search, sending/editing/deleting/pinning messages, reactions, thread management,
timeouts/kicks/bans/unbans, member role assignment, and owner-only server
structure tools. Tools still check the bot's actual Discord permissions before
mutating anything.

GRILLO uses the same Discord scope as the Brain thread. Before each turn, the bot
builds a compact context packet for that DM/channel/thread and prepends it to the
model input; after the reply, the completed user/assistant pair is queued into
the background memory tick.

## Tools

Register Python functions directly. The framework exposes them to the model as
function tools and executes parallel tool calls concurrently.

```python
from aibrain import Brain

brain = Brain()


@brain.tools.register
async def current_song() -> str:
    """Return the currently playing song."""
    return "FM-84 - Running in the Night"


response = await brain.ask(
    "What am I listening to?",
    thread_id="desktop:music",
    tool_names=["current_song"],
)
```

## Heartbeat Autonomy

The brain owns the reusable decision loop. Platform adapters own delivery.

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
    context={"recent_chat": "adapter-supplied summary"},
)

if result.decision and result.decision.should_act:
    await discord_channel.send(result.decision.message)
```

Example `HEARTBEAT.md`:

```text
tasks:
- name: inbox-triage
  interval: 30m
  prompt: "Check for urgent unread messages."

- If nothing needs attention, reply HEARTBEAT_OK.
```

## Piper TTS

Configure Piper through env vars or `TTSConfig`.

```powershell
$env:AIBRAIN_TTS_PROVIDER="piper_process"
$env:PIPER_EXE="C:\path\to\piper.exe"
$env:PIPER_MODEL="C:\path\to\voice.onnx"
$env:PIPER_CONFIG="C:\path\to\voice.onnx.json"
$env:AIBRAIN_TTS_VOICE_ROOTS="D:\voices;D:\more-voices"
$env:AIBRAIN_TTS_MANIFESTS="D:\voices\manifest.json"
```

Important Piper knobs:

| Env var | Default | Notes |
| --- | ---: | --- |
| `PIPER_PROCESS_IDLE_TIMEOUT` | `0.15` | Detects when one raw Piper segment is done. Raise to `0.25` or `0.45` if a voice clips words. |
| `PIPER_PROCESS_POOL_MAX` | `4` | Max hot Piper processes kept for voice switching. |
| `PIPER_LENGTH_SCALE` | `1.0` | Lower is faster speech, higher is slower. |
| `PIPER_SENTENCE_SILENCE` | `0.05` | Silence inserted by Piper between sentences. |

`GET /tts/voices` lists voices discovered from `AIBRAIN_TTS_VOICE_ROOTS` and
`AIBRAIN_TTS_MANIFESTS`, plus bundled Piper voices shipped under
`src/aibrain/voices/piper`. Discovery is cached; call `/tts/voices?refresh=true`
after changing files.

List discovered Piper voices from the CLI:

```powershell
uv run aibrain piper-models
uv run aibrain piper-models --json
```

## Speech Input

Raw streaming audio is `pcm_s16le`, 16 kHz, mono. The default local stack is:

```text
mic/audio -> VAD -> speech chunk -> faster-whisper -> brain stream -> Piper TTS
```

```python
result = await brain.transcribe(
    audio_bytes,
    format="pcm_s16le",
    sample_rate=16000,
    channels=1,
)

print(result.text)
```

Runtime overrides:

```powershell
$env:AIBRAIN_STT_PROVIDER="faster_whisper"
$env:AIBRAIN_STT_MODEL="base.en"
$env:AIBRAIN_STT_DEVICE="auto"
$env:AIBRAIN_STT_COMPUTE_TYPE="auto"
$env:AIBRAIN_VAD_PROVIDER="silero"
$env:AIBRAIN_VAD_THRESHOLD="0.5"
```

## Configuration Reference

| Setting | Default | Purpose |
| --- | --- | --- |
| `AIBRAIN_PROVIDER` | `vercel` | `vercel` or legacy `openai` |
| `AI_GATEWAY_API_KEY` | unset | Vercel AI Gateway key |
| `AIBRAIN_BASE_URL` | Vercel AI Gateway URL | Override OpenAI-compatible base URL |
| `AI_BRAIN_MODEL` / `AIBRAIN_MODEL` | `deepseek/deepseek-v4-flash` | Default model id |
| `AIBRAIN_STATE_MODE` | `local` | `local`, `stateless`, legacy `conversation`, or legacy `previous_response_id` |
| `AIBRAIN_LOCAL_HISTORY_LIMIT` | `24` | Number of local chat messages replayed per turn |
| `AIBRAIN_OPENAI_STREAM_TRANSPORT` | `http` | `http` or `websocket` |
| `AIBRAIN_OPENAI_WS_POOL_SIZE` | `4` | Responses WebSocket pool size |
| `AIBRAIN_STREAM_EVENT_QUEUE_MAX` | `256` | Backpressure bound for streaming events |
| `AIBRAIN_THREAD_LOCK_CACHE_SIZE` | `4096` | Max remembered per-thread locks |
| `AIBRAIN_MODELS_CACHE_TTL_SECONDS` | `300` | `/models` cache TTL |
| `AIBRAIN_MEMORY_VEC_OVERFETCH` | `5` | sqlite-vec candidate multiplier before metadata filters |
| `AIBRAIN_VOICE_SOCKET_MAX_MESSAGE_BYTES` | `1048576` | Max binary `/voice` audio chunk size; `0` disables the guard |
| `AIBRAIN_TTS_PROVIDER` | `piper_process` | `piper_process`, `piper_executable`, `piper_http`, `null` |
| `AIBRAIN_STT_PROVIDER` | `faster_whisper` | `faster_whisper` or `null` |
| `AIBRAIN_VAD_PROVIDER` | `silero` | `silero`, `energy`, or `none` |

## Development

```powershell
uv sync --extra all --extra dev
uv run pytest -q -p no:cacheprovider
uv run python -m compileall -q src tests examples
```

GitHub Actions runs Windows/Python 3.11 coverage for the core suite, the
`vector` extra, the `stt` extra, and the full `all` extra. Embeddings are core
package behavior and are covered by the base test suite.

Current verification:

```text
106 passed
```

## Project Shape

```text
src/aibrain/core.py        Brain API, streaming, tool loop, voice loop
src/aibrain/gateway.py     OpenAI Responses/Conversations gateway
src/aibrain/memory.py      SQLite + optional sqlite-vec memory
src/aibrain/thread_store.py SQLite thread state
src/aibrain/tts.py         Piper providers and TTS chunking
src/aibrain/stt.py         STT, VAD, audio decoding
src/aibrain/server.py      FastAPI HTTP/WebSocket adapter
src/aibrain/webchat/       Browser test console
```

## Status

This is a framework starter, not a single bot. The intended workflow is:

1. Build or import the brain once.
2. Give each platform a stable `thread_id`.
3. Register project-specific tools.
4. Pick memory policy and voice settings.
5. Connect Discord, Twitch, VRM, web, or desktop UI on top.

The framework handles the shared AI pipeline. Your adapter handles the platform.
