from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ThreadState:
    thread_id: str
    persona_id: str
    openai_conversation_id: str | None = None
    last_response_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None


@dataclass(slots=True)
class MemoryRecord:
    id: str
    content: str
    score: float = 0.0
    scope: str = "global"
    thread_id: str | None = None
    persona_id: str | None = None
    importance: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None


@dataclass(slots=True)
class ToolContext:
    brain: Any
    thread: ThreadState | None
    persona_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BrainEvent:
    type: str
    data: dict[str, Any] = field(default_factory=dict)

    def model_dump(self) -> dict[str, Any]:
        return {"type": self.type, **self.data}


@dataclass(slots=True)
class BrainResponse:
    text: str
    response_id: str | None = None
    conversation_id: str | None = None
    thread_id: str | None = None
    parsed: Any = None
    raw_response: Any = None
    usage: Any = None
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    memory_hits: list[MemoryRecord] = field(default_factory=list)
    audio: Any = None
