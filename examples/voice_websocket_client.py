import asyncio
import json
import wave

import websockets


async def main() -> None:
    uri = "ws://127.0.0.1:8765/voice"
    wav_path = "speech.wav"
    async with websockets.connect(uri) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "audio.start",
                    "thread_id": "voice:local",
                    "sample_rate": 16000,
                    "channels": 1,
                    "encoding": "pcm_s16le",
                    "tts": True,
                }
            )
        )
        with wave.open(wav_path, "rb") as wav:
            if wav.getframerate() != 16000 or wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                raise ValueError("speech.wav must be 16 kHz mono pcm16")
            while chunk := wav.readframes(960):
                await ws.send(chunk)
                await asyncio.sleep(0.03)
        await ws.send(json.dumps({"type": "audio.stop"}))

        while True:
            event = json.loads(await ws.recv())
            print(event["type"])
            if event["type"] == "response.done":
                break


if __name__ == "__main__":
    asyncio.run(main())
