import pytest
import asyncio

from aibrain import tools as tools_module
from aibrain.tools import ToolRegistry
from aibrain.types import ToolContext


@pytest.mark.asyncio
async def test_tool_registry_generates_schema_and_executes():
    registry = ToolRegistry()

    @registry.register
    def add(left: int, right: int) -> int:
        """Add two numbers."""
        return left + right

    schema = registry.schemas(["add"])[0]
    assert schema["name"] == "add"
    assert schema["parameters"]["properties"]["left"]["type"] == "integer"

    result = await registry.execute(
        "add",
        '{"left": 2, "right": 3}',
        context=ToolContext(brain=None, thread=None, persona_id="default"),
    )
    assert result == 5


@pytest.mark.asyncio
async def test_tool_registry_timeout():
    registry = ToolRegistry()

    @registry.register(timeout_seconds=0.01)
    async def slow() -> str:
        await asyncio.sleep(1)
        return "done"

    with pytest.raises(asyncio.TimeoutError):
        await registry.execute(
            "slow",
            "{}",
            context=ToolContext(brain=None, thread=None, persona_id="default"),
        )


@pytest.mark.asyncio
async def test_sync_tools_run_through_thread_offload(monkeypatch):
    calls = []

    async def fake_to_thread(func, /, *args, **kwargs):
        calls.append(func.__name__)
        return func(*args, **kwargs)

    monkeypatch.setattr(tools_module.asyncio, "to_thread", fake_to_thread)
    registry = ToolRegistry()

    @registry.register
    def blocking_tool() -> str:
        return "ok"

    result = await registry.execute(
        "blocking_tool",
        "{}",
        context=ToolContext(brain=None, thread=None, persona_id="default"),
    )

    assert result == "ok"
    assert calls == ["blocking_tool"]
