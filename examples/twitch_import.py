import asyncio

from aibrain import Brain, Persona, ThreadPolicy


brain = Brain()
persona = Persona(
    id="twitch-waifu",
    name="Twitch Waifu",
    instructions="You are a fast Twitch chat companion. Keep replies stream-safe and short.",
)


async def handle_twitch_chat(channel: str, chatter: str, message: str) -> str:
    thread_id = ThreadPolicy.twitch_channel(channel)
    response = await brain.ask(
        f"{chatter}: {message}",
        thread_id=thread_id,
        persona=persona,
        metadata={"channel": channel, "chatter": chatter},
    )
    return response.text


if __name__ == "__main__":
    print(asyncio.run(handle_twitch_chat("mychannel", "viewer", "hello")))
