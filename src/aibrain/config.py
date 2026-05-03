import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .policy import MemoryPolicy
from .stt import STTConfig
from .tts import TTSConfig


class Persona(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = "default"
    name: str = "Default"
    instructions: str = "You are a helpful AI companion."
    model: str | None = None
    prompt_cache_key: str | None = None
    prompt_cache_retention: Literal["in-memory", "24h"] | None = None
    tools: list[str] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BrainConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    env_file: Path | None = Field(default_factory=lambda: _default_env_file())
    database_path: Path = Field(default_factory=lambda: Path("brain.sqlite3"))
    default_model: str = Field(default_factory=lambda: os.environ.get("AI_BRAIN_MODEL", "gpt-5-nano"))
    openai_stream_transport: Literal["http", "websocket"] = Field(
        default_factory=lambda: os.environ.get("AIBRAIN_OPENAI_STREAM_TRANSPORT", "http")  # type: ignore[arg-type]
    )
    openai_ws_pool_size: int = Field(
        default_factory=lambda: _env_int("AIBRAIN_OPENAI_WS_POOL_SIZE", 4)
    )
    stream_event_queue_max: int = Field(
        default_factory=lambda: _env_int("AIBRAIN_STREAM_EVENT_QUEUE_MAX", 256)
    )
    thread_lock_cache_size: int = Field(
        default_factory=lambda: _env_int("AIBRAIN_THREAD_LOCK_CACHE_SIZE", 4096)
    )
    models_cache_ttl_seconds: int = Field(
        default_factory=lambda: _env_int("AIBRAIN_MODELS_CACHE_TTL_SECONDS", 300)
    )
    state_mode: Literal["conversation", "previous_response_id", "stateless"] = "conversation"
    store: bool | None = True
    truncation: Literal["auto", "disabled"] | None = "auto"
    service_tier: Literal["auto", "default", "flex", "scale", "priority"] | None = None
    reasoning: dict[str, Any] | None = None
    text: dict[str, Any] | None = None
    default_prompt_cache_key: str | None = None
    prompt_cache_retention: Literal["in-memory", "24h"] | None = "24h"
    context_management: list[dict[str, Any]] | None = Field(
        default_factory=lambda: [{"type": "compaction"}]
    )
    memory_top_k: int = 5
    memory_min_score: float = 0.05
    memory_policy: MemoryPolicy = Field(default_factory=MemoryPolicy)
    auto_memory_tools: bool = True
    default_persona: Persona = Field(default_factory=Persona)
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 256
    max_agent_steps: int = 8
    tool_timeout_seconds: float = 30.0
    stt_config: STTConfig = Field(default_factory=STTConfig)
    tts_config: TTSConfig = Field(default_factory=TTSConfig)


def _default_env_file() -> Path | None:
    value = os.environ.get("AIBRAIN_ENV_FILE")
    return Path(value) if value else None


def _env_int(name: str, fallback: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return fallback
    try:
        return int(value)
    except ValueError:
        return fallback
