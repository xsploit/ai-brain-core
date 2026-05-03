import asyncio
import json

import websockets


async def main() -> None:
    async with websockets.connect("ws://127.0.0.1:8765/stream") as socket:
        await socket.send(
            json.dumps(
                {
                    "type": "ask",
                    "thread_id": "example:websocket",
                    "text": "Say hello in one sentence.",
                }
            )
        )
        async for raw in socket:
            event = json.loads(raw)
            if event["type"] == "text.delta":
                print(event["text"], end="", flush=True)
            elif event["type"] == "response.done":
                print()
                return


if __name__ == "__main__":
    asyncio.run(main())
