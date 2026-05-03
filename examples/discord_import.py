import asyncio

from aibrain import Brain, Persona, ThreadPolicy


brain = Brain()
persona = Persona(
    id="discord-waifu",
    name="Discord Waifu",
    instructions="You are a concise Discord companion. Keep live replies natural.",
    model="gpt-5-nano",
)


async def handle_discord_message(guild_id: int, channel_id: int, message: str) -> str:
    thread_id = ThreadPolicy.discord_channel(guild_id, channel_id)
    response = await brain.ask(message, thread_id=thread_id, persona=persona)
    return response.text


async def handle_discord_stream(guild_id: int, channel_id: int, message: str):
    thread_id = ThreadPolicy.discord_channel(guild_id, channel_id)
    async for event in brain.stream_with_tts(message, thread_id=thread_id, persona=persona):
        yield event


async def main() -> None:
    print(await handle_discord_message(123, 456, "Say hi."))


if __name__ == "__main__":
    asyncio.run(main())
