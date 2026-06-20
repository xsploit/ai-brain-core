from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any


FALLBACK_MODELS = [
    "deepseek/deepseek-v4-flash",
    "gpt-5-nano",
    "gpt-5-mini",
    "gpt-5",
    "gpt-4.1-mini",
    "gpt-4.1",
]

_NON_CHAT_TOKENS = (
    "embedding",
    "moderation",
    "dall-e",
    "gpt-image",
    "image-generation",
    "tts",
    "whisper",
    "transcribe",
)

_KNOWN_CHAT_PREFIXES = (
    "anthropic/",
    "chatgpt-",
    "claude-",
    "deepseek/",
    "gemini-",
    "google/",
    "gpt-",
    "grok-",
    "meta/",
    "mistral/",
    "o",
    "openai/",
    "qwen/",
    "xai/",
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ModelChoice:
    id: str
    label: str
    owned_by: str | None = None
    created: int | None = None
    metadata: dict[str, Any] | None = None


async def list_model_choices(
    brain: Any,
    *,
    cache: dict[str, Any] | None = None,
    cache_lock: asyncio.Lock | None = None,
    ttl_seconds: int = 300,
    refresh: bool = False,
    default_models: list[str | None] | tuple[str | None, ...] = (),
    log: logging.Logger | None = None,
) -> list[ModelChoice]:
    now = time.monotonic()
    if not refresh and cache is not None and cache.get("models") is not None and cache.get("expires_at", 0) > now:
        return list(cache["models"])
    if cache_lock is not None:
        async with cache_lock:
            return await list_model_choices(
                brain,
                cache=cache,
                cache_lock=None,
                ttl_seconds=ttl_seconds,
                refresh=refresh,
                default_models=default_models,
                log=log,
            )
    try:
        result = await brain.client.models.list()
        choices = _model_choices_from_result(result)
        if not choices:
            choices = _fallback_model_choices()
    except Exception:
        (log or logger).warning("Failed to list OpenAI models, using fallback list", exc_info=True)
        choices = _fallback_model_choices()
    choices = _ensure_model_choices(choices, default_models)
    if cache is not None:
        cache["models"] = list(choices)
        cache["expires_at"] = now + max(0, ttl_seconds)
    return choices


def model_choice_ids(choices: list[ModelChoice]) -> list[str]:
    return [choice.id for choice in choices]


def is_chat_model_id(model_id: str) -> bool:
    normalized = model_id.strip().lower()
    if not normalized:
        return False
    if any(token in normalized for token in _NON_CHAT_TOKENS):
        return False
    if normalized.startswith(_KNOWN_CHAT_PREFIXES):
        return True
    return "/" in normalized


def _model_choices_from_result(result: Any) -> list[ModelChoice]:
    choices: dict[str, ModelChoice] = {}
    for model in getattr(result, "data", []) or []:
        model_id = str(getattr(model, "id", "")).strip()
        if not is_chat_model_id(model_id):
            continue
        choices[model_id] = ModelChoice(
            id=model_id,
            label=model_id,
            owned_by=_optional_string(getattr(model, "owned_by", None)),
            created=_optional_int(getattr(model, "created", None)),
            metadata=_model_metadata(model),
        )
    return sorted(choices.values(), key=lambda choice: choice.id.lower())


def _fallback_model_choices() -> list[ModelChoice]:
    return [ModelChoice(id=model_id, label=model_id, owned_by="fallback") for model_id in FALLBACK_MODELS]


def _ensure_model_choices(
    choices: list[ModelChoice],
    default_models: list[str | None] | tuple[str | None, ...],
) -> list[ModelChoice]:
    by_id = {choice.id: choice for choice in choices}
    for model_id in default_models:
        normalized = (model_id or "").strip()
        if normalized and normalized not in by_id:
            by_id[normalized] = ModelChoice(id=normalized, label=normalized, owned_by="configured")
    return sorted(by_id.values(), key=lambda choice: choice.id.lower())


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _model_metadata(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        try:
            dumped = model.model_dump(mode="json")
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass
    if isinstance(model, dict):
        return dict(model)
    raw = getattr(model, "__dict__", None)
    if isinstance(raw, dict):
        return {key: value for key, value in raw.items() if _jsonish(value)}
    metadata: dict[str, Any] = {}
    for key in ("id", "object", "created", "owned_by", "provider", "context_window", "max_output_tokens"):
        if hasattr(model, key):
            value = getattr(model, key)
            if _jsonish(value):
                metadata[key] = value
    return metadata


def _jsonish(value: Any) -> bool:
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, list):
        return all(_jsonish(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _jsonish(item) for key, item in value.items())
    return False
