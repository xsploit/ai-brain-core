from __future__ import annotations

import json
import os
from typing import Any

from openai import AsyncOpenAI


VERCEL_AI_GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh/v1"


class VercelAIGatewayJSONClient:
    def __init__(
        self,
        *,
        model: str = "deepseek/deepseek-v4-flash",
        api_key: str | None = None,
        base_url: str = VERCEL_AI_GATEWAY_BASE_URL,
        client: Any | None = None,
    ):
        self.model = model
        self.base_url = base_url
        self.client = client or AsyncOpenAI(
            api_key=api_key or os.getenv("AI_GATEWAY_API_KEY"),
            base_url=base_url,
        )

    async def complete_json(
        self,
        *,
        instructions: str,
        payload: dict[str, Any],
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        text_config: dict[str, Any] | None = None
        if schema is not None:
            text_config = {
                "format": {
                    "type": "json_schema",
                    "name": schema.get("name", "grillo_v2_json"),
                    "schema": schema.get("schema", schema),
                    "strict": bool(schema.get("strict", False)),
                }
            }
        response = await self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=json.dumps(payload, ensure_ascii=True, sort_keys=True),
            text=text_config,
            store=False,
        )
        output_text = getattr(response, "output_text", None)
        if output_text is None:
            output_text = _response_text(response)
        return _parse_json_object(output_text)


def _response_text(response: Any) -> str:
    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                parts.append(str(text))
    return "\n".join(parts)


def _parse_json_object(text: str) -> dict[str, Any]:
    value = (text or "").strip()
    if value.startswith("```"):
        value = value.strip("`").strip()
        if value.lower().startswith("json"):
            value = value[4:].strip()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        start = value.find("{")
        end = value.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            parsed = json.loads(value[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return parsed if isinstance(parsed, dict) else {}
