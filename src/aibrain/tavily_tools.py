from __future__ import annotations

import os
from typing import Any

import httpx


TAVILY_BASE_URL = "https://api.tavily.com"


class TavilyConfigError(RuntimeError):
    pass


async def tavily_search(
    query: str,
    search_depth: str = "basic",
    max_results: int = 5,
    include_answer: bool = True,
    include_raw_content: bool = False,
    include_images: bool = False,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    topic: str | None = None,
    time_range: str | None = None,
) -> dict[str, Any]:
    """Search the live web with Tavily and return answer, sources, and snippets."""
    payload = _compact_payload(
        {
            "query": query,
            "search_depth": search_depth,
            "max_results": _clamp(max_results, 1, 10),
            "include_answer": include_answer,
            "include_raw_content": include_raw_content,
            "include_images": include_images,
            "include_domains": include_domains,
            "exclude_domains": exclude_domains,
            "topic": topic,
            "time_range": time_range,
        }
    )
    return await _tavily_post("/search", payload)


async def tavily_extract(
    urls: list[str],
    extract_depth: str = "basic",
    format: str = "markdown",
    include_images: bool = False,
) -> dict[str, Any]:
    """Extract clean page content from one or more URLs with Tavily."""
    payload = _compact_payload(
        {
            "urls": urls[: _env_int("TAVILY_TOOL_MAX_URLS", 5)],
            "extract_depth": extract_depth,
            "format": format,
            "include_images": include_images,
        }
    )
    return await _tavily_post("/extract", payload)


async def tavily_crawl(
    url: str,
    instructions: str | None = None,
    max_depth: int = 1,
    max_breadth: int = 20,
    limit: int = 10,
    extract_depth: str = "basic",
    format: str = "markdown",
    select_paths: list[str] | None = None,
    exclude_paths: list[str] | None = None,
    select_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
) -> dict[str, Any]:
    """Crawl a website from a root URL and extract clean content."""
    payload = _compact_payload(
        {
            "url": url,
            "instructions": instructions,
            "max_depth": _clamp(max_depth, 1, 3),
            "max_breadth": _clamp(max_breadth, 1, 50),
            "limit": _clamp(limit, 1, _env_int("TAVILY_TOOL_CRAWL_LIMIT", 25)),
            "extract_depth": extract_depth,
            "format": format,
            "select_paths": select_paths,
            "exclude_paths": exclude_paths,
            "select_domains": select_domains,
            "exclude_domains": exclude_domains,
        }
    )
    return await _tavily_post("/crawl", payload)


async def tavily_map(
    url: str,
    instructions: str | None = None,
    max_depth: int = 1,
    max_breadth: int = 20,
    limit: int = 50,
    select_paths: list[str] | None = None,
    exclude_paths: list[str] | None = None,
    select_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
) -> dict[str, Any]:
    """Map/discover URLs from a website without extracting full page content."""
    payload = _compact_payload(
        {
            "url": url,
            "instructions": instructions,
            "max_depth": _clamp(max_depth, 1, 3),
            "max_breadth": _clamp(max_breadth, 1, 50),
            "limit": _clamp(limit, 1, _env_int("TAVILY_TOOL_MAP_LIMIT", 75)),
            "select_paths": select_paths,
            "exclude_paths": exclude_paths,
            "select_domains": select_domains,
            "exclude_domains": exclude_domains,
        }
    )
    return await _tavily_post("/map", payload)


async def tavily_research(
    input: str,
    model: str = "auto",
    citation_format: str = "numbered",
    output_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a Tavily deep research task. Poll with tavily_research_status."""
    payload = _compact_payload(
        {
            "input": input,
            "model": model,
            "citation_format": citation_format,
            "output_schema": output_schema,
        }
    )
    return await _tavily_post("/research", payload, timeout_seconds=_env_int("TAVILY_TOOL_RESEARCH_TIMEOUT", 60))


async def tavily_research_status(request_id: str) -> dict[str, Any]:
    """Get the status and result of a Tavily research task."""
    return await _tavily_get(f"/research/{request_id}")


def register_tavily_tools(registry: Any) -> None:
    registry.register(tavily_search, name="tavily_search", timeout_seconds=_env_int("TAVILY_TOOL_TIMEOUT", 30))
    registry.register(tavily_extract, name="tavily_extract", timeout_seconds=_env_int("TAVILY_TOOL_TIMEOUT", 30))
    registry.register(tavily_crawl, name="tavily_crawl", timeout_seconds=_env_int("TAVILY_TOOL_TIMEOUT", 30))
    registry.register(tavily_map, name="tavily_map", timeout_seconds=_env_int("TAVILY_TOOL_TIMEOUT", 30))
    registry.register(tavily_research, name="tavily_research", timeout_seconds=_env_int("TAVILY_TOOL_RESEARCH_TIMEOUT", 60))
    registry.register(tavily_research_status, name="tavily_research_status", timeout_seconds=_env_int("TAVILY_TOOL_TIMEOUT", 30))


async def _tavily_post(path: str, payload: dict[str, Any], *, timeout_seconds: int | None = None) -> dict[str, Any]:
    api_key = _tavily_api_key()
    async with httpx.AsyncClient(timeout=timeout_seconds or _env_int("TAVILY_TOOL_TIMEOUT", 30)) as client:
        response = await client.post(
            f"{_base_url()}{path}",
            headers=_headers(api_key),
            json=payload,
        )
        response.raise_for_status()
        return response.json()


async def _tavily_get(path: str) -> dict[str, Any]:
    api_key = _tavily_api_key()
    async with httpx.AsyncClient(timeout=_env_int("TAVILY_TOOL_TIMEOUT", 30)) as client:
        response = await client.get(f"{_base_url()}{path}", headers=_headers(api_key))
        response.raise_for_status()
        return response.json()


def _headers(api_key: str) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    project_id = os.getenv("TAVILY_PROJECT") or os.getenv("TAVILY_PROJECT_ID")
    if project_id:
        headers["X-Project-ID"] = project_id
    return headers


def _tavily_api_key() -> str:
    api_key = os.getenv("TAVILY_API_KEY") or os.getenv("TAVILY_API_TOKEN") or os.getenv("TVLY_API_KEY")
    if not api_key:
        raise TavilyConfigError("Set TAVILY_API_KEY to use Tavily web tools.")
    return api_key


def _base_url() -> str:
    return os.getenv("TAVILY_BASE_URL", TAVILY_BASE_URL).rstrip("/")


def _compact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value not in (None, "", [], {})}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(value)))
