import asyncio

import pytest

from aibrain.discord_bot import DEFAULT_DISCORD_TOOL_NAMES, build_brain, build_persona
from aibrain.tavily_tools import (
    TavilyConfigError,
    _tavily_api_key,
    register_tavily_tools,
    tavily_crawl,
    tavily_search,
)
from aibrain.tools import ToolRegistry


def test_register_tavily_tools_exposes_all_agentic_tools():
    registry = ToolRegistry()

    register_tavily_tools(registry)

    assert {
        "tavily_search",
        "tavily_extract",
        "tavily_crawl",
        "tavily_map",
        "tavily_research",
        "tavily_research_status",
    }.issubset(registry._tools)


def test_discord_brain_exposes_tavily_schemas_to_model():
    brain = build_brain()
    persona = build_persona()

    schemas = brain._tool_schemas(persona, DEFAULT_DISCORD_TOOL_NAMES)

    assert {
        "tavily_search",
        "tavily_extract",
        "tavily_crawl",
        "tavily_map",
        "tavily_research",
        "tavily_research_status",
    }.issubset({schema["name"] for schema in schemas})


def test_tavily_api_key_required(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_TOKEN", raising=False)
    monkeypatch.delenv("TVLY_API_KEY", raising=False)

    with pytest.raises(TavilyConfigError):
        _tavily_api_key()


def test_tavily_search_payload_is_bounded(monkeypatch):
    calls = []

    async def fake_post(path, payload, *, timeout_seconds=None):
        calls.append((path, payload, timeout_seconds))
        return {"ok": True}

    monkeypatch.setattr("aibrain.tavily_tools._tavily_post", fake_post)

    result = asyncio.run(tavily_search("latest ai", max_results=99, include_raw_content=True))

    assert result == {"ok": True}
    assert calls[0][0] == "/search"
    assert calls[0][1]["max_results"] == 10
    assert calls[0][1]["include_raw_content"] is True


def test_tavily_crawl_payload_is_bounded(monkeypatch):
    calls = []

    async def fake_post(path, payload, *, timeout_seconds=None):
        calls.append((path, payload, timeout_seconds))
        return {"ok": True}

    monkeypatch.setenv("TAVILY_TOOL_CRAWL_LIMIT", "7")
    monkeypatch.setattr("aibrain.tavily_tools._tavily_post", fake_post)

    asyncio.run(tavily_crawl("https://docs.example.com", max_depth=20, limit=99))

    assert calls[0][0] == "/crawl"
    assert calls[0][1]["max_depth"] == 3
    assert calls[0][1]["limit"] == 7
