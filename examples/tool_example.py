import asyncio

from aibrain import Brain


brain = Brain()


@brain.tools.register
def get_room_state() -> dict:
    """Return the current local room state."""
    return {"lights": "dim", "music": "synthwave", "viewer_present": True}


async def main() -> None:
    response = await brain.ask(
        "What is happening in the room?",
        thread_id="example:tools",
        tool_names=["get_room_state"],
    )
    print(response.text)


if __name__ == "__main__":
    asyncio.run(main())
