import asyncio

from aibrain import AutonomyAction, Brain, HeartbeatConfig


async def main() -> None:
    brain = Brain()
    loop = brain.autonomy_loop(
        thread_id="discord:channel:123",
        config=HeartbeatConfig(
            interval_seconds=1800,
            jitter_seconds=300,
            run_probability=0.35,
            heartbeat_path="HEARTBEAT.md",
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
        context_provider=lambda: {
            "recent_chat_summary": "Adapter should provide this from Discord/Twitch/etc.",
            "platform": "discord",
        },
    )
    try:
        async for event in loop.run():
            if event.type == "heartbeat.decision":
                decision = event.data["decision"]
                if decision["should_act"] and decision["action"] == "message":
                    print(decision["message"])
            elif event.type == "heartbeat.skipped":
                print(f"skipped: {event.data['reason']}")
    finally:
        await brain.close()


if __name__ == "__main__":
    asyncio.run(main())
