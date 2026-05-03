import asyncio
import json

import websockets


async def main() -> None:
    async with websockets.connect("ws://127.0.0.1:8765/brain") as socket:
        await socket.send(
            json.dumps(
                {
                    "type": "ask",
                    "thread_id": "vrm:local",
                    "text": "Say hello in one short sentence.",
                    "tts": True,
                }
            )
        )
        async for raw in socket:
            event = json.loads(raw)
            if event["type"] == "text.delta":
                print(event["text"], end="", flush=True)
            elif event["type"] == "tts.audio":
                print(f"\n[audio chunk {event['index']}]")
            elif event["type"] == "response.done":
                print()
                return


if __name__ == "__main__":
    asyncio.run(main())
