import asyncio

from aibrain import Brain, Persona


async def main() -> None:
    brain = Brain()
    persona = Persona(
        id="waifu",
        name="Waifu",
        instructions="You are a fast, expressive AI companion. Keep replies natural.",
        model="gpt-5-nano",
    )
    while True:
        text = input("> ")
        async for event in brain.stream(text, thread_id="example:cli", persona=persona):
            if event.type == "text.delta":
                print(event.data["text"], end="", flush=True)
            elif event.type == "response.done":
                print()


if __name__ == "__main__":
    asyncio.run(main())
