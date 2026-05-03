from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

import uvicorn

from .config import BrainConfig
from .core import Brain
from .playback import AudioPlaybackWorker, create_audio_player
from .server import create_app


FAST_CHAT_MODEL = "gpt-4.1-mini"


async def chat(args: argparse.Namespace) -> None:
    brain = Brain(BrainConfig(**_chat_config_kwargs(args)))
    thread_id = args.thread
    audio_worker: AudioPlaybackWorker | None = None
    tts_options = {"voice": args.voice} if args.voice else {}
    if args.tts:
        player = create_audio_player(args.audio_player)
        audio_worker = AudioPlaybackWorker(player)
        await audio_worker.start()
        print(f"[audio:{player.name}]")
        if player.name == "winsound":
            print("[audio:winsound is a fallback; install/use sounddevice for raw PCM streaming]")
        if not args.no_tts_warmup:
            await brain.warmup(openai=True, tts=True, tts_options=tts_options)
    elif brain.config.openai_stream_transport == "websocket":
        await brain.warmup(openai=True, tts=False)
    if args.fast or args.stateless or args.no_memory or args.no_tools:
        modes = []
        if args.fast:
            modes.append("fast")
        if args.stateless or args.fast:
            modes.append("stateless")
        if args.no_memory or args.fast:
            modes.append("memory-off")
        if args.no_tools or args.fast:
            modes.append("tools-off")
        modes.append(f"transport={brain.config.openai_stream_transport}")
        print(f"[mode:{', '.join(modes)}]")
    try:
        while True:
            try:
                text = (await asyncio.to_thread(input, "> ")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if not text:
                continue
            started_at = time.perf_counter()
            first_text_at = None
            first_audio_at = None
            last_text_at = None
            response_done_at = None
            playlist_done_at = None
            audio_drained_at = None
            audio_bytes_start = audio_worker.bytes if audio_worker is not None else 0
            audio_frames_start = audio_worker.frames if audio_worker is not None else 0
            brain_options = _chat_brain_options(args)
            stream = (
                brain.stream_with_tts(
                    text,
                    thread_id=thread_id,
                    tts_options=tts_options,
                    use_memory=not (args.no_memory or args.fast),
                    **brain_options,
                )
                if args.tts
                else brain.stream(
                    text,
                    thread_id=thread_id,
                    use_memory=not (args.no_memory or args.fast),
                    **brain_options,
                )
            )
            async for event in stream:
                if args.trace_events:
                    _trace_event(event, started_at)
                if event.type == "text.delta":
                    if first_text_at is None:
                        first_text_at = time.perf_counter()
                    last_text_at = time.perf_counter()
                    sys.stdout.write(event.data["text"])
                    sys.stdout.flush()
                elif event.type == "tts.audio" and audio_worker is not None:
                    if first_audio_at is None:
                        first_audio_at = time.perf_counter()
                    await audio_worker.enqueue_event(event)
                elif event.type == "response.done":
                    response_done_at = time.perf_counter()
                    print()
                elif event.type == "tts.playlist.done" and audio_worker is not None:
                    playlist_done_at = time.perf_counter()
                    await audio_worker.drain()
                    audio_drained_at = time.perf_counter()
            if args.stats:
                timings = []
                if first_text_at is not None:
                    timings.append(f"first_text={int((first_text_at - started_at) * 1000)}ms")
                if last_text_at is not None:
                    timings.append(f"last_text={int((last_text_at - started_at) * 1000)}ms")
                if response_done_at is not None:
                    timings.append(f"response_done={int((response_done_at - started_at) * 1000)}ms")
                if first_audio_at is not None:
                    timings.append(f"first_audio={int((first_audio_at - started_at) * 1000)}ms")
                if playlist_done_at is not None:
                    timings.append(f"tts_done={int((playlist_done_at - started_at) * 1000)}ms")
                if audio_drained_at is not None:
                    timings.append(f"audio_done={int((audio_drained_at - started_at) * 1000)}ms")
                if audio_worker is not None:
                    timings.append(f"audio_frames={audio_worker.frames - audio_frames_start}")
                    timings.append(f"audio_bytes={audio_worker.bytes - audio_bytes_start}")
                if timings:
                    print(f"[{', '.join(timings)}]")
    finally:
        if audio_worker is not None:
            await audio_worker.close()
        await brain.close()


async def tts(args: argparse.Namespace) -> None:
    brain = Brain(
        BrainConfig(
            database_path=args.database,
            env_file=args.env_file,
            **({"default_model": args.model} if args.model else {}),
        )
    )
    try:
        audio = await brain.speak(args.text)
        output = args.output
        if audio.encoding == "pcm_s16le" and output.lower().endswith(".wav"):
            from .tts import write_wav_bytes

            data = write_wav_bytes(audio.audio, sample_rate=audio.sample_rate)
        else:
            data = audio.audio
        with open(output, "wb") as file:
            file.write(data)
        print(output)
    finally:
        await brain.close()


@dataclass(slots=True)
class BenchmarkTurn:
    transport: str
    round: int
    prompt: str
    first_text_ms: int | None = None
    last_text_ms: int | None = None
    response_done_ms: int | None = None
    first_audio_ms: int | None = None
    tts_done_ms: int | None = None
    playback_done_ms: int | None = None
    text_chunks: int = 0
    audio_frames: int = 0
    audio_bytes: int = 0


async def bench(args: argparse.Namespace) -> None:
    if not args.prompt:
        args.prompt = ["hey"]
    transports = ["http", "websocket"] if args.transport == "both" else [args.transport]
    all_stats: list[BenchmarkTurn] = []
    for transport in transports:
        config_kwargs = _benchmark_config_kwargs(args, transport)
        brain = Brain(BrainConfig(**config_kwargs))
        audio_worker: AudioPlaybackWorker | None = None
        tts_options = {"voice": args.voice} if args.voice else {}
        try:
            if args.tts and args.audio_player != "none":
                player = create_audio_player(args.audio_player)
                audio_worker = AudioPlaybackWorker(player)
                await audio_worker.start()
            await brain.warmup(openai=True, tts=args.tts, tts_options=tts_options)
            for round_index in range(args.rounds):
                prompt = args.prompt[round_index % len(args.prompt)]
                stats = await _run_benchmark_turn(
                    brain,
                    prompt,
                    transport=transport,
                    round_index=round_index + 1,
                    args=args,
                    audio_worker=audio_worker,
                    tts_options=tts_options,
                )
                all_stats.append(stats)
                if not args.json:
                    print(_format_benchmark_turn(stats), flush=True)
        finally:
            if audio_worker is not None:
                await audio_worker.close()
            await brain.close()
    if args.json:
        print(json.dumps([asdict(item) for item in all_stats], indent=2))


async def _run_benchmark_turn(
    brain: Brain,
    prompt: str,
    *,
    transport: str,
    round_index: int,
    args: argparse.Namespace,
    audio_worker: AudioPlaybackWorker | None,
    tts_options: dict[str, Any],
) -> BenchmarkTurn:
    started_at = time.perf_counter()
    stats = BenchmarkTurn(transport=transport, round=round_index, prompt=prompt)
    audio_bytes_start = audio_worker.bytes if audio_worker is not None else 0
    audio_frames_start = audio_worker.frames if audio_worker is not None else 0
    stream_options = _benchmark_brain_options(args)
    stream = (
        brain.stream_with_tts(
            prompt,
            thread_id=f"bench:{transport}",
            tts_options=tts_options,
            use_memory=not args.fast,
            **stream_options,
        )
        if args.tts
        else brain.stream(
            prompt,
            thread_id=f"bench:{transport}",
            use_memory=not args.fast,
            **stream_options,
        )
    )
    async for event in stream:
        now_ms = int((time.perf_counter() - started_at) * 1000)
        if event.type == "text.delta":
            stats.text_chunks += 1
            if stats.first_text_ms is None:
                stats.first_text_ms = now_ms
            stats.last_text_ms = now_ms
        elif event.type == "response.done":
            stats.response_done_ms = now_ms
        elif event.type == "tts.audio":
            if stats.first_audio_ms is None:
                stats.first_audio_ms = now_ms
            audio = event.data.get("audio", "")
            if isinstance(audio, str):
                stats.audio_bytes += (len(audio) * 3) // 4
            if audio_worker is not None:
                await audio_worker.enqueue_event(event)
        elif event.type == "tts.playlist.done":
            stats.tts_done_ms = now_ms
            if audio_worker is not None:
                await audio_worker.drain()
                stats.playback_done_ms = int((time.perf_counter() - started_at) * 1000)
    if audio_worker is not None:
        stats.audio_frames = audio_worker.frames - audio_frames_start
        stats.audio_bytes = audio_worker.bytes - audio_bytes_start
    return stats


def _benchmark_config_kwargs(args: argparse.Namespace, transport: str) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "database_path": args.database,
        "env_file": args.env_file,
        "openai_stream_transport": transport,
    }
    model = args.model or (FAST_CHAT_MODEL if args.fast else None)
    if model:
        kwargs["default_model"] = model
    if args.fast:
        kwargs["openai_ws_pool_size"] = 1
    return kwargs


def _benchmark_brain_options(args: argparse.Namespace) -> dict[str, object]:
    options: dict[str, object] = {"stateless": args.fast}
    if args.fast:
        options["tool_names"] = []
    if args.max_output_tokens is not None:
        options["max_output_tokens"] = args.max_output_tokens
    if args.service_tier:
        options["service_tier"] = args.service_tier
    return options


def _format_benchmark_turn(stats: BenchmarkTurn) -> str:
    fields = [
        f"{stats.transport} round={stats.round}",
        f"first_text={stats.first_text_ms}ms",
        f"last_text={stats.last_text_ms}ms",
        f"response_done={stats.response_done_ms}ms",
    ]
    if stats.first_audio_ms is not None:
        fields.extend(
            [
                f"first_audio={stats.first_audio_ms}ms",
                f"tts_done={stats.tts_done_ms}ms",
                f"playback_done={stats.playback_done_ms}ms",
                f"audio_frames={stats.audio_frames}",
                f"audio_bytes={stats.audio_bytes}",
            ]
        )
    fields.append(f"text_chunks={stats.text_chunks}")
    return "[" + ", ".join(fields) + "]"


def serve(args: argparse.Namespace) -> None:
    kwargs: dict[str, Any] = {
        "database_path": args.database,
        "env_file": args.env_file,
    }
    if args.model:
        kwargs["default_model"] = args.model
    if args.stream_transport:
        kwargs["openai_stream_transport"] = args.stream_transport
    config = BrainConfig(**kwargs)
    app = create_app(config=config)
    uvicorn.run(app, host=args.host, port=args.port)


def main() -> None:
    parser = argparse.ArgumentParser(prog="aibrain")
    subparsers = parser.add_subparsers(dest="command", required=True)

    chat_parser = subparsers.add_parser("chat")
    chat_parser.add_argument("--database", default="brain.sqlite3")
    chat_parser.add_argument("--model", default=None)
    chat_parser.add_argument("--thread", default="cli:default")
    chat_parser.add_argument("--env-file", default=None)
    chat_parser.add_argument(
        "--fast",
        action="store_true",
        help="Low-latency live voice preset: WebSocket, stateless, memory off, tools off.",
    )
    chat_parser.add_argument("--tts", action="store_true")
    chat_parser.add_argument("--voice", default=None)
    chat_parser.add_argument("--no-tts-warmup", action="store_true")
    chat_parser.add_argument(
        "--audio-player",
        default="sounddevice",
        choices=["auto", "sounddevice", "winsound", "none"],
    )
    chat_parser.add_argument("--no-memory", action="store_true")
    chat_parser.add_argument("--no-tools", action="store_true")
    chat_parser.add_argument("--tools", default=None, help="Comma-separated tool names to expose.")
    chat_parser.add_argument("--stateless", action="store_true")
    chat_parser.add_argument(
        "--stream-transport",
        default=None,
        choices=["http", "websocket"],
    )
    chat_parser.add_argument("--stats", action="store_true")
    chat_parser.add_argument("--trace-events", action="store_true")
    chat_parser.add_argument("--max-output-tokens", type=int, default=None)
    chat_parser.add_argument(
        "--reasoning-effort",
        choices=["minimal", "low", "medium", "high"],
        default=None,
    )
    chat_parser.add_argument(
        "--service-tier",
        choices=["auto", "default", "flex", "scale", "priority"],
        default=None,
    )
    chat_parser.set_defaults(func=lambda args: asyncio.run(chat(args)))

    serve_parser = subparsers.add_parser("serve")
    serve_parser.add_argument("--database", default="brain.sqlite3")
    serve_parser.add_argument("--model", default=None)
    serve_parser.add_argument("--env-file", default=None)
    serve_parser.add_argument(
        "--stream-transport",
        default=None,
        choices=["http", "websocket"],
    )
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", default=8765, type=int)
    serve_parser.set_defaults(func=serve)

    tts_parser = subparsers.add_parser("tts")
    tts_parser.add_argument("text")
    tts_parser.add_argument("--output", default="speech.wav")
    tts_parser.add_argument("--database", default="brain.sqlite3")
    tts_parser.add_argument("--model", default=None)
    tts_parser.add_argument("--env-file", default=None)
    tts_parser.set_defaults(func=lambda args: asyncio.run(tts(args)))

    bench_parser = subparsers.add_parser("bench")
    bench_parser.add_argument("--database", default="brain.sqlite3")
    bench_parser.add_argument("--model", default=None)
    bench_parser.add_argument("--env-file", default=None)
    bench_parser.add_argument(
        "--transport",
        default="both",
        choices=["http", "websocket", "both"],
    )
    bench_parser.add_argument("--rounds", type=int, default=5)
    bench_parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="Prompt to benchmark. Pass more than once to rotate prompts.",
    )
    bench_parser.add_argument("--tts", action="store_true")
    bench_parser.add_argument("--voice", default=None)
    bench_parser.add_argument(
        "--audio-player",
        default="none",
        choices=["auto", "sounddevice", "winsound", "none"],
    )
    bench_parser.add_argument("--fast", action="store_true")
    bench_parser.add_argument("--json", action="store_true")
    bench_parser.add_argument("--max-output-tokens", type=int, default=96)
    bench_parser.add_argument(
        "--service-tier",
        choices=["auto", "default", "flex", "scale", "priority"],
        default=None,
    )
    bench_parser.set_defaults(
        func=lambda args: asyncio.run(bench(args)),
    )

    args = parser.parse_args()
    args.func(args)


def _trace_event(event: object, started_at: float) -> None:
    event_type = getattr(event, "type", "")
    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    data = getattr(event, "data", {}) or {}
    if event_type == "text.delta":
        text = str(data.get("text", ""))
        detail = f" chars={len(text)} text={text[:32]!r}"
    elif event_type == "tts.audio":
        audio = data.get("audio", "")
        detail = f" segment={data.get('segment_index')} b64_chars={len(audio)}"
    elif event_type in {"tts.start", "tts.done"}:
        detail = f" segment={data.get('segment_index')}"
    else:
        detail = ""
    print(f"\n[{elapsed_ms}ms {event_type}{detail}]", file=sys.stderr, flush=True)


def _chat_config_kwargs(args: argparse.Namespace) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "database_path": args.database,
        "env_file": args.env_file,
        "openai_stream_transport": args.stream_transport or "websocket",
    }
    model = args.model or (FAST_CHAT_MODEL if args.fast else None)
    if model:
        kwargs["default_model"] = model
    if args.fast:
        kwargs["openai_ws_pool_size"] = 1
    return kwargs


def _chat_brain_options(args: argparse.Namespace) -> dict[str, object]:
    options: dict[str, object] = {"stateless": args.stateless or args.fast}
    if args.fast or args.no_tools:
        options["tool_names"] = []
    elif args.tools:
        options["tool_names"] = [
            tool.strip() for tool in args.tools.split(",") if tool.strip()
        ]
    if args.reasoning_effort:
        options["reasoning"] = {"effort": args.reasoning_effort}
    if args.service_tier:
        options["service_tier"] = args.service_tier
    if args.max_output_tokens is not None:
        options["max_output_tokens"] = args.max_output_tokens
    return options


if __name__ == "__main__":
    main()
