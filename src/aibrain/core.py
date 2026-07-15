from __future__ import annotations

import asyncio
import json
import os
import random
import threading
from collections import OrderedDict, deque
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from openai import AsyncOpenAI

from .autonomy import (
    AutonomyAction,
    AutonomyDecision,
    AutonomyLoop,
    HeartbeatFile,
    HeartbeatConfig,
    HeartbeatResult,
    heartbeat_prompt,
    load_heartbeat_file,
    strip_heartbeat_ack,
    within_active_hours,
)
from .chat_store import SQLiteChatHistoryStore
from .config import BrainConfig, Persona
from .embeddings import EmbeddingProvider, default_embedding_provider
from .env import load_env_file
from .gateway import OpenAIGateway
from .inputs import FileInput, ImageInput, build_user_message
from .memory import SQLiteMemoryStore
from .memory_stack import HybridMemoryStack
from .policy import MemoryPolicy
from .stt import (
    AudioEncoding,
    BaseSTTProvider,
    BaseVAD,
    STTResult,
    UtteranceBuffer,
    VADConfig,
    create_stt_provider,
    create_vad_detector,
)
from .thread_store import SQLiteThreadStore
from .tools import ToolRegistry
from .tts import BaseTTSProvider, SentenceChunker, TTSAudio, create_tts_provider
from .types import BrainEvent, BrainResponse, MemoryRecord, ThreadState, ToolContext


_HELD_THREAD_LOCKS: ContextVar[frozenset[str]] = ContextVar(
    "aibrain_held_thread_locks",
    default=frozenset(),
)
DEFAULT_MEMORY_QUERY_MAX_CHARS = 6000

_GRILLO_REFLECTION_INSTRUCTIONS = "\n".join(
    [
        "You are the background sleep-time memory agent for this AI companion.",
        "You are not writing a user-facing chat reply.",
        "Read the JSON payload and return only JSON matching the schema.",
        "Write durable memory candidates for explicit preferences, facts, goals, boundaries, bond signals, and ongoing threads.",
        "Write relationship memory when the turn changes trust, tone, attachment, recurring context, or how the assistant should understand the participant.",
        "Use relationship_profile for grounded stage, mood, numeric relationship scores, summary, diary, and known facts.",
        "Use profile_patches for grounded tone_preferences, interaction_style, boundaries, and active_threads.",
        "Write diary entries only when the turn meaningfully changes relationship, mood, goals, or stream/server context.",
        "Diary entries are private first-person reflections from the avatar perspective, not receipts or summaries of every reply.",
        "A good diary personal_thought says how the speaker or chat made the avatar feel, what changed, and what to remember next time.",
        "Do not write mechanical diary text like 'Processed N turns' or 'I noticed X and answered as Y'.",
        "Use slots for consolidated grounded memory, not every transient message.",
        "Prefer concise, clean memory text. Remove wrapper labels like Preference signal or Durable fact signal.",
    ]
)

_GRILLO_REFLECTION_SCHEMA: dict[str, Any] = {
    "name": "grillo_reflection",
    "strict": False,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "done": {"type": "boolean"},
            "notes": {"type": "string"},
            "candidates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["preference", "fact", "goal", "boundary", "bond_signal", "thread"],
                        },
                        "content": {"type": "string"},
                        "summary": {"type": "string"},
                        "confidence": {"type": "number"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "source_turn_ids": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["type", "content", "summary", "confidence", "tags"],
                },
            },
            "diary": {
                "anyOf": [
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "beat_type": {"type": "string"},
                            "summary": {"type": "string"},
                            "personal_thought": {"type": "string"},
                            "tags": {"type": "array", "items": {"type": "string"}},
                            "source_turn_ids": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["summary", "personal_thought", "tags"],
                    },
                    {"type": "null"},
                ]
            },
            "slots": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "slot_name": {
                            "type": "string",
                            "enum": [
                                "core_identity",
                                "relationship_state",
                                "user_facts",
                                "preferences",
                                "boundaries",
                                "ongoing_threads",
                                "open_threads",
                                "verified_facts",
                                "tone_preferences",
                                "working_scratchpad",
                            ],
                        },
                        "items": {"type": "array", "items": {"type": "string"}},
                        "operation": {"type": "string", "enum": ["merge", "replace"]},
                        "source_candidate_ids": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["slot_name", "items", "operation"],
                },
            },
            "relationship_profile": {
                "anyOf": [
                    {
                        "type": "object",
                        "additionalProperties": True,
                        "properties": {
                            "relationship_stage": {"type": "string"},
                            "mood": {"type": "string"},
                            "trust": {"type": "number"},
                            "attraction": {"type": "number"},
                            "respect": {"type": "number"},
                            "irritation": {"type": "number"},
                            "jealousy": {"type": "number"},
                            "guard": {"type": "number"},
                            "summary": {"type": "string"},
                            "diary_entry": {"type": "string"},
                            "facts": {"type": "array", "items": {"type": "string"}},
                            "tone_preferences": {"type": "array", "items": {"type": "string"}},
                            "interaction_style": {"type": "array", "items": {"type": "string"}},
                            "boundaries": {"type": "array", "items": {"type": "string"}},
                            "active_threads": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                    {"type": "null"},
                ]
            },
            "profile_patches": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "field": {
                            "type": "string",
                            "enum": [
                                "tone_preferences",
                                "interaction_style",
                                "boundaries",
                                "active_threads",
                            ],
                        },
                        "operation": {"type": "string", "enum": ["add", "remove"]},
                        "value": {"type": "string"},
                    },
                    "required": ["field", "operation", "value"],
                },
            },
        },
        "required": [
            "done",
            "notes",
            "candidates",
            "diary",
            "slots",
            "relationship_profile",
            "profile_patches",
        ],
    },
}

_GRILLO_WORKER_RESPONSE_SCHEMA: dict[str, Any] = {
    "name": "grillo_worker_response",
    "strict": False,
    "schema": {
        "type": "object",
        "additionalProperties": True,
        "properties": {
            "done": {"type": "boolean"},
            "notes": {"type": "string"},
            "toolCalls": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": True,
                    "properties": {
                        "name": {
                            "type": "string",
                            "enum": [
                                "core.worker_memory_read",
                                "core.worker_memory_search",
                                "core.worker_candidate_list",
                                "core.worker_candidate_write",
                                "core.worker_diary_write",
                                "core.worker_memory_write",
                                "core.worker_profile_patch",
                                "core.worker_emotion_read",
                                "core.worker_emotion_update",
                                "core.worker_memory_insert_archival",
                            ],
                        },
                        "args": {"type": "object", "additionalProperties": True},
                    },
                    "required": ["name", "args"],
                },
            },
            "candidate": {"type": "object", "additionalProperties": True},
            "diary": {"type": "object", "additionalProperties": True},
            "memory": {"type": "object", "additionalProperties": True},
        },
        "required": ["done", "notes", "toolCalls"],
    },
}

_CONTINUATION_PARAM_KEYS = frozenset(
    {
        "model",
        "conversation",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "max_tool_calls",
        "metadata",
        "store",
        "truncation",
        "context_management",
        "prompt_cache_key",
        "prompt_cache_retention",
        "reasoning",
        "service_tier",
        "temperature",
        "top_p",
        "text",
        "include",
        "instructions",
        "prompt",
        "safety_identifier",
        "user",
    }
)


class _StreamQueueOverflow(RuntimeError):
    pass


def _parse_json_object(text: str) -> dict[str, Any] | None:
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
            return None
        try:
            parsed = json.loads(value[start : end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


class Brain:
    def __init__(
        self,
        config: BrainConfig | None = None,
        *,
        client: Any | None = None,
        thread_store: SQLiteThreadStore | None = None,
        chat_store: SQLiteChatHistoryStore | None = None,
        memory_store: SQLiteMemoryStore | None = None,
        memory_stack: HybridMemoryStack | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        tools: ToolRegistry | None = None,
        openai_gateway: OpenAIGateway | None = None,
        stt_provider: BaseSTTProvider | None = None,
        vad_detector: BaseVAD | None = None,
        tts_provider: BaseTTSProvider | None = None,
    ):
        self.config = config or BrainConfig()
        load_env_file(self.config.env_file)
        self._client = client
        self._openai = openai_gateway
        self._client_lock = threading.Lock()
        self._openai_lock = threading.Lock()
        self.thread_store = thread_store or SQLiteThreadStore(self.config.database_path)
        self.chat_store = chat_store or SQLiteChatHistoryStore(self.config.database_path)
        provider = embedding_provider or default_embedding_provider(
            self._get_client,
            self.config.embedding_model,
            self.config.embedding_dimensions,
        )
        self.memory = memory_store or SQLiteMemoryStore(
            self.config.database_path,
            embedding_provider=provider,
            dimensions=self.config.embedding_dimensions,
            vec_overfetch=self.config.memory_vec_overfetch,
        )
        self.memory_stack = memory_stack
        if self.memory_stack is None and self.config.memory_stack.enabled:
            self.memory_stack = HybridMemoryStack.sqlite_paths(
                raw_log_path=self.config.memory_stack.raw_log_path or self.config.database_path,
                graph_path=self.config.memory_stack.graph_path or self.config.database_path,
                vector_path=self.config.memory_stack.vector_path or self.config.database_path,
                embedding_provider=provider,
                embedding_dimensions=self.config.embedding_dimensions,
                graph_backend=self.config.memory_stack.graph_backend,
                vector_backend=self.config.memory_stack.vector_backend,
            )
        self.tools = tools or ToolRegistry()
        self.stt = stt_provider or create_stt_provider(self.config.stt_config)
        self.vad = vad_detector
        self.tts = tts_provider or create_tts_provider(self.config.tts_config)
        self._thread_locks: OrderedDict[str, asyncio.Lock] = OrderedDict()
        self._thread_locks_guard = asyncio.Lock()
        if self.config.auto_memory_tools:
            self._register_memory_tools()
            self._register_context_tools()
        self._wire_grillo_reflector()

    def _get_client(self) -> Any:
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    kwargs: dict[str, Any] = {}
                    if self.config.api_key:
                        kwargs["api_key"] = self.config.api_key
                    base_url = self._client_base_url()
                    if base_url:
                        kwargs["base_url"] = base_url
                    self._client = AsyncOpenAI(**kwargs)
        return self._client

    def _client_base_url(self) -> str | None:
        if self.config.base_url:
            return self.config.base_url
        if self.config.provider == "vercel":
            return "https://ai-gateway.vercel.sh/v1"
        return None

    @property
    def client(self) -> Any:
        return self._get_client()

    @property
    def openai(self) -> OpenAIGateway:
        if self._openai is None:
            with self._openai_lock:
                if self._openai is None:
                    self._openai = OpenAIGateway(
                        self.client,
                        stream_transport=self.config.openai_stream_transport,
                        websocket_pool_size=self.config.openai_ws_pool_size,
                    )
        return self._openai

    async def close(self) -> None:
        await self.stt.close()
        await self.tts.close()
        if hasattr(self.memory, "close"):
            self.memory.close()
        if self.memory_stack is not None:
            self.memory_stack.close()
        if hasattr(self.thread_store, "close"):
            self.thread_store.close()
        if hasattr(self.chat_store, "close"):
            self.chat_store.close()
        if self._openai is not None:
            await self._openai.close()
        client = self._client
        if client is not None and hasattr(client, "close"):
            result = client.close()
            if hasattr(result, "__await__"):
                await result

    def _wire_grillo_reflector(self) -> None:
        if self.memory_stack is None or self.memory_stack.grillo is None:
            return
        enabled = os.environ.get("AIBRAIN_GRILLO_LLM_REFLECTOR", "true").strip().lower()
        if enabled in {"0", "false", "no", "off"}:
            return
        self.memory_stack.grillo.worker_completion = self._complete_grillo_worker
        self.memory_stack.grillo.reflector = self._reflect_grillo_memory

    async def _complete_grillo_worker(self, request: dict[str, Any]) -> dict[str, Any]:
        messages = request.get("messages") if isinstance(request.get("messages"), list) else []
        system_prompt = ""
        conversation: list[dict[str, Any]] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "").strip()
            content = str(message.get("content") or "")
            if role == "system" and not system_prompt:
                system_prompt = content
            else:
                conversation.append({"role": role or "user", "content": content})
        persona = Persona(
            id="grillo-backend-worker",
            name="GRILLO Backend Worker",
            instructions=system_prompt or _GRILLO_REFLECTION_INSTRUCTIONS,
            model=self.config.default_model,
            prompt_cache_key="aibrain:grillo-backend-worker",
            tools=[],
        )
        payload = {
            "messages": conversation,
            "response_format": request.get("responseFormat") or {"type": "json_object"},
            "state_key": request.get("stateKey"),
            "state_scope": request.get("stateScope"),
            "tool_choice_mode": request.get("toolChoiceMode"),
        }
        response = await self.structured(
            json.dumps(payload, ensure_ascii=False),
            json_schema=_GRILLO_WORKER_RESPONSE_SCHEMA,
            persona=persona,
            use_memory=False,
            tool_names=[],
            stateless=True,
            memory_event_text="",
            memory_stack_record=False,
            temperature=float(request.get("temperature") or 0.25),
            max_output_tokens=int(request.get("maxTokens") or 900),
        )
        return {
            "text": response.text,
            "meta": {
                "model": self.config.default_model,
                "provider": self.config.provider,
            },
        }

    async def _reflect_grillo_memory(self, context: dict[str, Any]) -> dict[str, Any]:
        persona = Persona(
            id="grillo-memory-worker",
            name="GRILLO Memory Worker",
            instructions=_GRILLO_REFLECTION_INSTRUCTIONS,
            model=self.config.default_model,
            prompt_cache_key="aibrain:grillo-memory-worker",
            tools=[],
        )
        payload = {
            "scope_key": context.get("scope_key"),
            "participant_key": context.get("participant_key"),
            "beat_type": context.get("beat_type"),
            "current_time_iso": context.get("current_time_iso"),
            "turns": context.get("turns") or [],
            "memory_slots": context.get("memory_slots") or [],
            "recent_diary": context.get("recent_diary") or [],
            "recent_candidates": context.get("recent_candidates") or [],
            "relationship_profile": context.get("relationship_profile"),
        }
        response = await self.structured(
            json.dumps(payload, ensure_ascii=False),
            json_schema=_GRILLO_REFLECTION_SCHEMA,
            persona=persona,
            use_memory=False,
            tool_names=[],
            stateless=True,
            memory_event_text="",
            memory_stack_record=False,
            temperature=0.25,
            max_output_tokens=1800,
        )
        parsed = _parse_json_object(response.text)
        if parsed is None:
            raise ValueError("GRILLO reflector returned invalid JSON")
        return parsed

    async def warmup(
        self,
        *,
        openai: bool = False,
        stt: bool = False,
        tts: bool = True,
        tts_options: dict[str, Any] | None = None,
    ) -> None:
        if openai:
            await self.openai.warmup()
        if stt:
            await self.stt.warmup()
            if self.vad is None:
                self.vad = create_vad_detector(self.config.stt_config.vad_config)
            await self.vad.warmup()
        if tts:
            await self.tts.warmup(**(tts_options or {}))

    @asynccontextmanager
    async def _thread_turn(
        self,
        thread_id: str | None,
        response_options: dict[str, Any] | None = None,
        *,
        force: bool = False,
    ):
        lock_id = self._thread_turn_lock_id(thread_id, response_options, force=force)
        if lock_id is None:
            yield
            return
        held = _HELD_THREAD_LOCKS.get()
        if lock_id in held:
            yield
            return
        lock = await self._get_thread_lock(lock_id)
        async with lock:
            token = _HELD_THREAD_LOCKS.set(held | {lock_id})
            try:
                yield
            finally:
                _HELD_THREAD_LOCKS.reset(token)

    def _thread_turn_lock_id(
        self,
        thread_id: str | None,
        response_options: dict[str, Any] | None = None,
        *,
        force: bool = False,
    ) -> str | None:
        if not thread_id:
            return None
        if self.config.state_mode == "stateless" and not force:
            return None
        if response_options and response_options.get("stateless") and not force:
            return None
        return thread_id

    async def _get_thread_lock(self, thread_id: str) -> asyncio.Lock:
        async with self._thread_locks_guard:
            lock = self._thread_locks.get(thread_id)
            if lock is None:
                lock = asyncio.Lock()
                self._thread_locks[thread_id] = lock
                self._trim_thread_lock_cache()
            else:
                self._thread_locks.move_to_end(thread_id)
            return lock

    def _trim_thread_lock_cache(self) -> None:
        max_locks = self.config.thread_lock_cache_size
        if max_locks <= 0:
            return
        while len(self._thread_locks) > max_locks:
            removed = False
            for key, lock in list(self._thread_locks.items()):
                if lock.locked():
                    continue
                self._thread_locks.pop(key, None)
                removed = True
                break
            if not removed:
                break

    def _resolve_persona(self, persona: Persona | dict[str, Any] | None) -> Persona:
        if persona is None:
            return self.config.default_persona
        if isinstance(persona, Persona):
            return persona
        return Persona.model_validate(persona)

    def _register_memory_tools(self) -> None:
        if "remember" not in self.tools:

            @self.tools.register(name="remember")
            async def remember(
                content: str,
                importance: float = 0.5,
                scope: str = "global",
                context: ToolContext | None = None,
            ) -> dict[str, Any]:
                """Store a long-term memory for later recall."""
                thread = context.thread if context else None
                resolved_thread_id = thread.thread_id if thread and scope == "thread" else None
                resolved_persona_id = (
                    context.persona_id if context and scope in {"thread", "persona"} else None
                )
                record = await self.memory.remember(
                    content,
                    scope=scope,
                    thread_id=resolved_thread_id,
                    persona_id=resolved_persona_id,
                    importance=importance,
                )
                return {"id": record.id, "content": record.content}

        if "search_memory" not in self.tools:

            @self.tools.register(name="search_memory")
            async def search_memory(
                query: str,
                top_k: int = 5,
                context: ToolContext | None = None,
            ) -> list[dict[str, Any]]:
                """Search long-term memory for relevant facts."""
                thread = context.thread if context else None
                records = await self.memory.search(
                    query,
                    top_k=top_k,
                    scope=["thread", "persona", "global"],
                    thread_id=thread.thread_id if thread else None,
                    persona_id=context.persona_id if context else None,
                )
                return [
                    {"id": record.id, "content": record.content, "score": record.score}
                    for record in records
                ]

        if "forget_memory" not in self.tools:

            @self.tools.register(name="forget_memory")
            def forget_memory(memory_id: str) -> dict[str, Any]:
                """Delete one long-term memory by id."""
                deleted = self.memory.forget(memory_id)
                return {"id": memory_id, "deleted": deleted}

    def _register_context_tools(self) -> None:
        if "current_time" not in self.tools:

            @self.tools.register(name="current_time")
            def current_time() -> dict[str, Any]:
                """Return the current UTC timestamp."""
                return {"utc": datetime.now(timezone.utc).isoformat()}

        if "brain_context" not in self.tools:

            @self.tools.register(name="brain_context")
            def brain_context(context: ToolContext | None = None) -> dict[str, Any]:
                """Return the current brain thread and persona context."""
                thread = context.thread if context else None
                return {
                    "persona_id": context.persona_id if context else None,
                    "thread_id": thread.thread_id if thread else None,
                    "conversation_id": thread.openai_conversation_id if thread else None,
                    "metadata": thread.metadata if thread else {},
                }

    async def open_thread(
        self,
        *,
        thread_id: str | None = None,
        persona: Persona | dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        create_remote: bool = False,
    ) -> ThreadState:
        async with self._thread_turn(thread_id, force=True):
            resolved = self._resolve_persona(persona)
            if thread_id:
                existing = self.thread_store.get(thread_id)
                if existing:
                    return existing
            conversation_id = None
            if create_remote and self.config.state_mode == "conversation":
                conversation = await self.openai.create_conversation(metadata=metadata or {})
                conversation_id = getattr(conversation, "id", None)
            return self.thread_store.create(
                persona_id=resolved.id,
                thread_id=thread_id,
                openai_conversation_id=conversation_id,
                metadata=metadata,
            )

    async def ask(
        self,
        text: str,
        *,
        thread_id: str | None = None,
        persona: Persona | dict[str, Any] | None = None,
        images: list[ImageInput | dict[str, Any] | str] | None = None,
        files: list[FileInput | dict[str, Any] | str] | None = None,
        use_memory: bool | MemoryPolicy | dict[str, Any] = True,
        tool_names: Sequence[str] | None = None,
        openai_tools: list[dict[str, Any]] | None = None,
        max_agent_steps: int | None = None,
        **response_options: Any,
        ) -> BrainResponse:
        memory_stack_record = response_options.pop("memory_stack_record", True)
        async with self._thread_turn(thread_id, response_options):
            params, state, resolved, memory_hits = await self._build_response_params(
                text,
                thread_id=thread_id,
                persona=persona,
                images=images,
                files=files,
                use_memory=use_memory,
                tool_names=tool_names,
                openai_tools=openai_tools,
                response_options=response_options,
            )
            context = ToolContext(brain=self, thread=state, persona_id=resolved.id)
            response = await self._run_agent_loop(
                params,
                state=state,
                context=context,
                memory_hits=memory_hits,
                max_agent_steps=max_agent_steps or self.config.max_agent_steps,
            )
            self._append_local_chat_response(state, response.text, response.response_id)
            if memory_stack_record:
                await self._append_memory_stack_response_event(response, state, resolved.id)
            return response

    async def vision(
        self,
        text: str,
        *,
        images: list[ImageInput | dict[str, Any] | str],
        **kwargs: Any,
    ) -> BrainResponse:
        return await self.ask(text, images=images, **kwargs)

    async def structured(
        self,
        text: str,
        *,
        output_model: type[Any] | None = None,
        json_schema: dict[str, Any] | None = None,
        thread_id: str | None = None,
        persona: Persona | dict[str, Any] | None = None,
        images: list[ImageInput | dict[str, Any] | str] | None = None,
        files: list[FileInput | dict[str, Any] | str] | None = None,
        **response_options: Any,
    ) -> BrainResponse:
        response_options.pop("memory_stack_record", True)
        if output_model is None and json_schema is None:
            raise ValueError("structured() requires output_model or json_schema")
        if json_schema is not None:
            response_options.setdefault(
                "text",
                {
                    "format": {
                        "type": "json_schema",
                        "name": json_schema.get("name", "structured_output"),
                        "schema": json_schema.get("schema", json_schema),
                        "strict": json_schema.get("strict", True),
                    }
                },
            )
        async with self._thread_turn(thread_id, response_options):
            params, state, _, memory_hits = await self._build_response_params(
                text,
                thread_id=thread_id,
                persona=persona,
                images=images,
                files=files,
                use_memory=response_options.pop("use_memory", True),
                tool_names=response_options.pop("tool_names", None),
                openai_tools=response_options.pop("openai_tools", None),
                response_options=response_options,
            )
            if output_model is not None:
                response = await self.openai.parse_response(text_format=output_model, **params)
            else:
                response = await self.openai.create_response(**params)
            self._update_thread_after_response(state, response)
            parsed = getattr(response, "output_parsed", None)
            return BrainResponse(
                text=self._extract_text(response),
                response_id=getattr(response, "id", None),
                conversation_id=self._extract_conversation_id(response, state),
                thread_id=state.thread_id if state else None,
                parsed=parsed,
                raw_response=response,
                usage=getattr(response, "usage", None),
                memory_hits=memory_hits,
            )

    async def stream(
        self,
        text: str,
        *,
        thread_id: str | None = None,
        persona: Persona | dict[str, Any] | None = None,
        images: list[ImageInput | dict[str, Any] | str] | None = None,
        files: list[FileInput | dict[str, Any] | str] | None = None,
        use_memory: bool | MemoryPolicy | dict[str, Any] = True,
        tool_names: Sequence[str] | None = None,
        openai_tools: list[dict[str, Any]] | None = None,
        max_agent_steps: int | None = None,
        **response_options: Any,
    ) -> AsyncIterator[BrainEvent]:
        async with self._thread_turn(thread_id, response_options):
            async for event in self._stream_unlocked(
                text,
                thread_id=thread_id,
                persona=persona,
                images=images,
                files=files,
                use_memory=use_memory,
                tool_names=tool_names,
                openai_tools=openai_tools,
                max_agent_steps=max_agent_steps,
                **response_options,
            ):
                yield event

    async def _stream_unlocked(
        self,
        text: str,
        *,
        thread_id: str | None = None,
        persona: Persona | dict[str, Any] | None = None,
        images: list[ImageInput | dict[str, Any] | str] | None = None,
        files: list[FileInput | dict[str, Any] | str] | None = None,
        use_memory: bool | MemoryPolicy | dict[str, Any] = True,
        tool_names: Sequence[str] | None = None,
        openai_tools: list[dict[str, Any]] | None = None,
        max_agent_steps: int | None = None,
        **response_options: Any,
    ) -> AsyncIterator[BrainEvent]:
        memory_stack_record = response_options.pop("memory_stack_record", True)
        params, state, resolved, memory_hits = await self._build_response_params(
            text,
            thread_id=thread_id,
            persona=persona,
            images=images,
            files=files,
            use_memory=use_memory,
            tool_names=tool_names,
            openai_tools=openai_tools,
            response_options=response_options,
        )
        context = ToolContext(brain=self, thread=state, persona_id=resolved.id)
        for hit in memory_hits:
            yield BrainEvent(
                "memory.hit",
                {"id": hit.id, "content": hit.content, "score": hit.score},
            )

        tool_results: list[dict[str, Any]] = []
        steps = 0
        while True:
            response = None
            stream = await self.openai.stream_response(**params)
            async for event in stream:
                normalized = self._normalize_stream_event(event)
                if normalized is not None:
                    yield normalized
                if getattr(event, "type", None) == "response.completed":
                    response = getattr(event, "response", None)
            if response is None:
                yield BrainEvent("error", {"message": "OpenAI stream ended without response.completed"})
                return
            self._update_thread_after_response(state, response)
            calls = self._extract_function_calls(response)
            if not calls:
                self._append_local_chat_response(
                    state,
                    self._extract_text(response),
                    getattr(response, "id", None),
                )
                if memory_stack_record:
                    await self._append_memory_stack_raw_response_event(response, state, resolved.id)
                yield BrainEvent(
                    "response.done",
                    {
                        "response_id": getattr(response, "id", None),
                        "conversation_id": self._extract_conversation_id(response, state),
                        "thread_id": state.thread_id if state else None,
                        "tool_results": tool_results,
                    },
                )
                return
            steps += 1
            if steps > (max_agent_steps or self.config.max_agent_steps):
                yield BrainEvent("error", {"message": "Max agent tool steps exceeded"})
                return
            for call in calls:
                yield BrainEvent("tool.call", {"name": call["name"], "call_id": call["call_id"]})
            results: list[dict[str, Any] | None] = [None] * len(calls)
            tasks = [
                asyncio.create_task(self._execute_tool_call_index(index, call, context))
                for index, call in enumerate(calls)
            ]
            try:
                for completed in asyncio.as_completed(tasks):
                    index, result = await completed
                    results[index] = result
                    tool_results.append(result)
                    yield BrainEvent("tool.result", result)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            ordered_results = [result for result in results if result is not None]
            params = self._continuation_params(
                params,
                response,
                self._tool_outputs(ordered_results),
            )

    async def speak(self, text: str, **tts_options: Any) -> TTSAudio:
        return await self.tts.synthesize(text, **tts_options)

    async def transcribe(
        self,
        audio: bytes,
        *,
        format: AudioEncoding = "pcm_s16le",
        sample_rate: int | None = None,
        channels: int | None = None,
        language: str | None = None,
        **options: Any,
    ) -> STTResult:
        return await self.stt.transcribe(
            audio,
            format=format,
            sample_rate=sample_rate or self.config.stt_config.sample_rate,
            channels=channels or self.config.stt_config.channels,
            language=language or self.config.stt_config.language,
            **options,
        )

    def utterance_buffer(
        self,
        *,
        vad_config: VADConfig | dict[str, Any] | None = None,
        sample_rate: int | None = None,
        channels: int | None = None,
        encoding: AudioEncoding | None = None,
    ) -> UtteranceBuffer:
        resolved_vad_config = (
            vad_config
            if isinstance(vad_config, VADConfig)
            else VADConfig.model_validate(vad_config or self.config.stt_config.vad_config.model_dump())
        )
        return UtteranceBuffer(
            vad=self.vad or create_vad_detector(resolved_vad_config),
            config=resolved_vad_config,
            sample_rate=sample_rate or self.config.stt_config.sample_rate,
            channels=channels or self.config.stt_config.channels,
            encoding=encoding or self.config.stt_config.encoding,
        )

    async def voice_turn(
        self,
        audio_or_text: bytes | str,
        *,
        thread_id: str | None = None,
        persona: Persona | dict[str, Any] | None = None,
        tts: bool = True,
        stt_options: dict[str, Any] | None = None,
        tts_options: dict[str, Any] | None = None,
        **brain_options: Any,
    ) -> BrainResponse:
        transcript = (
            STTResult(text=audio_or_text, provider="text")
            if isinstance(audio_or_text, str)
            else await self.transcribe(audio_or_text, **(stt_options or {}))
        )
        if not transcript.text.strip():
            return BrainResponse(text="", parsed=transcript)
        response = await self.ask(
            transcript.text,
            thread_id=thread_id,
            persona=persona,
            **brain_options,
        )
        response.parsed = transcript
        if tts:
            response.audio = await self.speak(response.text, **(tts_options or {}))
        return response

    async def voice_stream(
        self,
        audio_or_text: bytes | str,
        *,
        thread_id: str | None = None,
        persona: Persona | dict[str, Any] | None = None,
        tts: bool = True,
        stt_options: dict[str, Any] | None = None,
        tts_options: dict[str, Any] | None = None,
        **brain_options: Any,
    ) -> AsyncIterator[BrainEvent]:
        transcript = (
            STTResult(text=audio_or_text, provider="text")
            if isinstance(audio_or_text, str)
            else await self.transcribe(audio_or_text, **(stt_options or {}))
        )
        yield BrainEvent("stt.final", transcript.model_dump())
        if not transcript.text.strip():
            yield BrainEvent("stt.empty", {"reason": "empty-transcript"})
            return
        event_stream = (
            self.stream_with_tts(
                transcript.text,
                thread_id=thread_id,
                persona=persona,
                tts_options=tts_options or {},
                **brain_options,
            )
            if tts
            else self.stream(
                transcript.text,
                thread_id=thread_id,
                persona=persona,
                **brain_options,
            )
        )
        async for event in event_stream:
            yield event

    async def tts_stream(self, text: str, **tts_options: Any) -> AsyncIterator[BrainEvent]:
        playback_id = str(uuid4())
        yield BrainEvent("tts.playlist.start", {"playback_id": playback_id, "ordered": True})
        async for event in self._tts_events(
            text,
            playback_id=playback_id,
            segment_index=0,
            **tts_options,
        ):
            yield event
        yield BrainEvent("tts.playlist.done", {"playback_id": playback_id, "segments": 1})

    async def stream_with_tts(
        self,
        text: str,
        *,
        tts_options: dict[str, Any] | None = None,
        emit_text: bool = True,
        **kwargs: Any,
    ) -> AsyncIterator[BrainEvent]:
        import asyncio

        playback_id = str(uuid4())
        tts_done = object()
        control_events: deque[Any] = deque()
        tts_events: deque[Any] = deque()
        event_condition = asyncio.Condition()
        max_pending_events = max(1, self.config.stream_event_queue_max)
        text_queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=max_pending_events)
        producer_done = False
        queue_overflowed = False

        def is_tts_event(item: Any) -> bool:
            return (
                isinstance(item, BrainEvent)
                and item.type.startswith("tts.")
                and item.type != "tts.playlist.start"
            )

        async def put_event(item: Any) -> None:
            nonlocal producer_done, queue_overflowed
            async with event_condition:
                if len(control_events) + len(tts_events) >= max_pending_events:
                    if not queue_overflowed:
                        queue_overflowed = True
                        control_events.append(
                            BrainEvent(
                                "error",
                                {
                                    "message": "Stream event queue overflowed",
                                    "stage": "stream",
                                },
                            )
                        )
                    producer_done = True
                    event_condition.notify_all()
                    raise _StreamQueueOverflow("Stream event queue overflowed")
                target = tts_events if is_tts_event(item) else control_events
                target.append(item)
                event_condition.notify()

        async def produce_tts() -> int:
            segment_index = 0
            while True:
                chunk_text = await text_queue.get()
                if chunk_text is tts_done:
                    return segment_index
                async for tts_event in self._tts_events(
                    str(chunk_text),
                    playback_id=playback_id,
                    segment_index=segment_index,
                    **(tts_options or {}),
                ):
                    await put_event(tts_event)
                segment_index += 1

        async def produce_stream() -> None:
            chunker = SentenceChunker(
                max_chars=self.config.tts_config.chunk_chars,
                min_chars=self.config.tts_config.min_chunk_chars,
                comma_chars=self.config.tts_config.comma_chunk_chars,
            )
            tts_worker = asyncio.create_task(produce_tts())
            done_event: BrainEvent | None = None
            try:
                await put_event(
                    BrainEvent("tts.playlist.start", {"playback_id": playback_id, "ordered": True})
                )
                async for event in self.stream(text, **kwargs):
                    if event.type == "response.done":
                        done_event = event
                        break
                    if emit_text or event.type not in {"text.delta", "text.done"}:
                        await put_event(event)
                    if event.type == "text.delta":
                        for chunk in chunker.feed(str(event.data.get("text", ""))):
                            await text_queue.put(chunk)
                            await asyncio.sleep(0)
                tail = chunker.flush()
                if tail:
                    await text_queue.put(tail)
                await text_queue.put(tts_done)
                if done_event:
                    await put_event(done_event)
                segment_count = await tts_worker
                await put_event(
                    BrainEvent(
                        "tts.playlist.done",
                        {"playback_id": playback_id, "segments": segment_count},
                    )
                )
            except _StreamQueueOverflow:
                return
            finally:
                nonlocal producer_done
                if not tts_worker.done():
                    tts_worker.cancel()
                    try:
                        await tts_worker
                    except asyncio.CancelledError:
                        pass
                async with event_condition:
                    producer_done = True
                    event_condition.notify_all()

        producer = asyncio.create_task(produce_stream())
        control_burst = 0
        max_control_burst = 4
        try:
            while True:
                async with event_condition:
                    await event_condition.wait_for(
                        lambda: control_events or tts_events or producer_done
                    )
                    if not control_events and not tts_events and producer_done:
                        break
                    if control_events and (control_burst < max_control_burst or not tts_events):
                        item = control_events.popleft()
                        control_burst += 1
                    elif tts_events:
                        item = tts_events.popleft()
                        control_burst = 0
                    else:
                        item = control_events.popleft()
                        control_burst += 1
                if item is None:
                    break
                yield item
        finally:
            if not producer.done():
                producer.cancel()
                try:
                    await producer
                except asyncio.CancelledError:
                    pass

    async def _tts_events(
        self,
        text: str,
        *,
        playback_id: str | None = None,
        segment_index: int | None = None,
        **tts_options: Any,
    ) -> AsyncIterator[BrainEvent]:
        segment_id = (
            f"{playback_id}:{segment_index}"
            if playback_id is not None and segment_index is not None
            else None
        )
        base = {
            "text": text,
            "playback_id": playback_id,
            "segment_index": segment_index,
            "segment_id": segment_id,
        }
        yield BrainEvent("tts.start", base)
        try:
            async for chunk in self.tts.stream(text, **tts_options):
                if chunk.audio:
                    data = chunk.to_event_data()
                    data.update(base)
                    yield BrainEvent("tts.audio", data)
            yield BrainEvent("tts.done", base)
        except Exception as exc:
            yield BrainEvent(
                "error",
                {"message": f"TTS failed: {exc}", "stage": "tts", **base},
            )

    async def compact(
        self,
        *,
        thread_id: str | None = None,
        input: str | list[dict[str, Any]] | None = None,
        instructions: str | None = None,
        model: str | None = None,
        **options: Any,
    ) -> Any:
        state = self.thread_store.get(thread_id) if thread_id else None
        save_to_memory = bool(options.pop("save_to_memory", False))
        kwargs: dict[str, Any] = {
            "model": model or self.config.default_model,
            **options,
        }
        if input is not None:
            kwargs["input"] = input
        if instructions is not None:
            kwargs["instructions"] = instructions
        if (
            state
            and state.last_response_id
            and self.config.state_mode == "previous_response_id"
        ):
            kwargs.setdefault("previous_response_id", state.last_response_id)
        result = await self.openai.compact_response(**kwargs)
        if state and save_to_memory:
            summary = self._extract_text(result) or str(result)
            if summary:
                await self.memory.remember(
                    summary,
                    scope="thread",
                    thread_id=state.thread_id,
                    persona_id=state.persona_id,
                    metadata={"source": "compaction"},
                    importance=0.8,
                )
        return result

    async def retrieve_response(self, response_id: str, **options: Any) -> Any:
        return await self.openai.retrieve_response(response_id, **options)

    async def cancel_response(self, response_id: str, **options: Any) -> Any:
        return await self.openai.cancel_response(response_id, **options)

    async def delete_response(self, response_id: str, **options: Any) -> Any:
        return await self.openai.delete_response(response_id, **options)

    async def list_response_input_items(self, response_id: str, **options: Any) -> list[Any]:
        return await self.openai.list_response_input_items(response_id, **options)

    async def count_response_input_tokens(self, **options: Any) -> Any:
        return await self.openai.count_response_input_tokens(**options)

    def response_stream_manager(self, **options: Any) -> Any:
        return self.openai.response_stream_manager(**options)

    async def autonomy_tick(
        self,
        *,
        thread_id: str,
        persona: Persona | dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        actions: list[AutonomyAction | dict[str, Any]] | None = None,
        config: HeartbeatConfig | dict[str, Any] | None = None,
        heartbeat: HeartbeatFile | dict[str, Any] | None = None,
        **response_options: Any,
    ) -> AutonomyDecision:
        heartbeat_config = (
            config
            if isinstance(config, HeartbeatConfig)
            else HeartbeatConfig.model_validate(config or {})
        )
        normalized_actions = [
            action if isinstance(action, AutonomyAction) else AutonomyAction.model_validate(action)
            for action in actions or []
        ]
        heartbeat_file = (
            heartbeat
            if isinstance(heartbeat, HeartbeatFile) or heartbeat is None
            else HeartbeatFile.model_validate(heartbeat)
        )
        prompt = heartbeat_prompt(
            context=context or {},
            actions=normalized_actions,
            instruction=f"{heartbeat_config.prompt}\n\n{heartbeat_config.idle_instruction}",
            heartbeat=heartbeat_file,
            ack_token=heartbeat_config.ack_token,
        )
        response = await self.structured(
            prompt,
            output_model=AutonomyDecision,
            thread_id=thread_id,
            persona=persona,
            use_memory=response_options.pop("use_memory", True),
            tool_names=response_options.pop("tool_names", []),
            **response_options,
        )
        decision = response.parsed
        if not isinstance(decision, AutonomyDecision):
            try:
                decision = AutonomyDecision.model_validate_json(response.text)
            except Exception:
                decision = AutonomyDecision(
                    should_act=False,
                    action="none",
                    confidence=0.0,
                    reason="Could not parse autonomy decision.",
                )
        if decision.confidence < heartbeat_config.act_threshold:
            decision.should_act = False
        if not decision.should_act:
            decision.action = "none"
        ack, _ = strip_heartbeat_ack(
            decision.message,
            ack_token=heartbeat_config.ack_token,
            ack_max_chars=heartbeat_config.ack_max_chars,
        )
        if ack:
            decision.should_act = False
            decision.action = "none"
            decision.message = ""
        return decision

    async def heartbeat_tick(
        self,
        *,
        thread_id: str,
        persona: Persona | dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        actions: list[AutonomyAction | dict[str, Any]] | None = None,
        config: HeartbeatConfig | dict[str, Any] | None = None,
        **response_options: Any,
    ) -> HeartbeatResult:
        heartbeat_config = (
            config
            if isinstance(config, HeartbeatConfig)
            else HeartbeatConfig.model_validate(config or {})
        )
        if not heartbeat_config.enabled:
            return HeartbeatResult(skipped=True, reason="disabled")
        if not within_active_hours(heartbeat_config.active_hours):
            return HeartbeatResult(skipped=True, reason="outside-active-hours")
        if heartbeat_config.run_probability <= 0:
            return HeartbeatResult(skipped=True, reason="random-chance")
        if heartbeat_config.run_probability < 1 and random.random() > heartbeat_config.run_probability:
            return HeartbeatResult(skipped=True, reason="random-chance")

        async with self._thread_turn(thread_id, response_options, force=True):
            resolved = self._resolve_persona(persona)
            state = await self._ensure_thread(thread_id, resolved)
            task_state = {}
            if state and isinstance(state.metadata.get(heartbeat_config.task_state_key), dict):
                task_state = state.metadata[heartbeat_config.task_state_key]
            heartbeat = await asyncio.to_thread(
                load_heartbeat_file,
                heartbeat_config.heartbeat_path,
                task_state=task_state,
                skip_empty_file=heartbeat_config.skip_empty_file,
            )
            if heartbeat.skipped_reason:
                return HeartbeatResult(
                    skipped=True,
                    reason=heartbeat.skipped_reason,
                    heartbeat=heartbeat,
                )

            response_options.setdefault("use_memory", not heartbeat_config.light_context)
            if heartbeat_config.isolated_session:
                response_options.setdefault("stateless", True)
            decision = await self.autonomy_tick(
                thread_id=thread_id,
                persona=resolved,
                context=context or {},
                actions=actions,
                config=heartbeat_config,
                heartbeat=heartbeat,
                **response_options,
            )
            if state and heartbeat.due_tasks:
                now = datetime.now(timezone.utc).isoformat()
                updated_task_state = dict(task_state)
                for task in heartbeat.due_tasks:
                    updated_task_state[task.name] = now
                state.metadata[heartbeat_config.task_state_key] = updated_task_state
                self.thread_store.upsert(state)
            return HeartbeatResult(decision=decision, heartbeat=heartbeat)

    def autonomy_loop(
        self,
        *,
        thread_id: str,
        persona: Persona | dict[str, Any] | None = None,
        config: HeartbeatConfig | dict[str, Any] | None = None,
        actions: list[AutonomyAction | dict[str, Any]] | None = None,
        context_provider: Any = None,
    ) -> AutonomyLoop:
        heartbeat_config = (
            config
            if isinstance(config, HeartbeatConfig)
            else HeartbeatConfig.model_validate(config or {})
        )
        return AutonomyLoop(
            self,
            thread_id=thread_id,
            persona=persona,
            config=heartbeat_config,
            actions=actions,
            context_provider=context_provider,
        )

    async def retrieve_conversation(self, conversation_id: str, **options: Any) -> Any:
        return await self.openai.retrieve_conversation(conversation_id, **options)

    async def update_conversation(self, conversation_id: str, **options: Any) -> Any:
        return await self.openai.update_conversation(conversation_id, **options)

    async def delete_conversation(self, conversation_id: str, **options: Any) -> Any:
        return await self.openai.delete_conversation(conversation_id, **options)

    async def create_conversation_items(self, conversation_id: str, **options: Any) -> Any:
        return await self.openai.create_conversation_items(conversation_id, **options)

    async def list_conversation_items(self, conversation_id: str, **options: Any) -> list[Any]:
        return await self.openai.list_conversation_items(conversation_id, **options)

    async def _build_response_params(
        self,
        text: str,
        *,
        thread_id: str | None,
        persona: Persona | dict[str, Any] | None,
        images: list[ImageInput | dict[str, Any] | str] | None,
        files: list[FileInput | dict[str, Any] | str] | None,
        use_memory: bool | MemoryPolicy | dict[str, Any],
        tool_names: Sequence[str] | None,
        openai_tools: list[dict[str, Any]] | None,
        response_options: dict[str, Any],
    ) -> tuple[dict[str, Any], ThreadState | None, Persona, list[MemoryRecord]]:
        resolved = self._resolve_persona(persona)
        force_stateless = bool(response_options.pop("stateless", False))
        state = None if force_stateless else await self._ensure_thread(thread_id, resolved)
        memory_hits: list[MemoryRecord] = []
        input_items: list[dict[str, Any]] = []
        memory_query_source = response_options.pop("memory_query_text", text)
        memory_event_text = response_options.pop("memory_event_text", text)
        history_text = response_options.pop("history_text", text)
        memory_policy = self._resolve_memory_policy(
            use_memory,
            response_options.pop("memory_policy", None),
        )
        if memory_policy.enabled and state is not None:
            memory_query = _memory_query_text(memory_query_source)
            memory_hits = await memory_policy.retrieve(
                self.memory,
                memory_query,
                thread=state,
                persona_id=resolved.id,
            )
            if self.memory_stack is not None and self.config.memory_stack.retrieve:
                stack_hits = await self.memory_stack.retrieve_records(
                    memory_query,
                    top_k=memory_policy.top_k,
                    thread=state,
                    persona_id=resolved.id,
                    filters=memory_policy.metadata_filter,
                )
                memory_hits = _dedupe_memory_records([*memory_hits, *stack_hits])[
                    : memory_policy.top_k
                ]
            memory_message = memory_policy.build_injection_message(memory_hits)
            if memory_message:
                input_items.append(memory_message)
        if self.memory_stack is not None and state is not None and str(memory_event_text or "").strip():
            await self.memory_stack.append_event(
                event_type="message",
                actor="user",
                content=str(memory_event_text),
                thread=state,
                persona_id=resolved.id,
                metadata={"source": "brain.ask"},
                extract=self.config.memory_stack.extract_user_events,
            )
        user_message = await asyncio.to_thread(build_user_message, text, images=images, files=files)
        if state is not None and self.config.state_mode == "local":
            input_items.extend(self._local_history_input(state))
            history_message = (
                user_message
                if history_text == text
                else await asyncio.to_thread(build_user_message, str(history_text), images=images, files=files)
            )
            self.chat_store.append(
                state.thread_id,
                "user",
                self._history_content_from_user_message(history_message),
                metadata={"source": "brain.user"},
            )
        input_items.append(user_message)

        params: dict[str, Any] = {
            "model": response_options.pop("model", resolved.model or self.config.default_model),
            "input": response_options.pop("input", input_items),
            "instructions": response_options.pop("instructions", resolved.instructions),
        }
        if self.config.store is not None:
            params["store"] = response_options.pop("store", self.config.store)
        if self.config.truncation is not None:
            params["truncation"] = response_options.pop("truncation", self.config.truncation)
        if self.config.context_management is not None and self.config.provider == "openai":
            params["context_management"] = response_options.pop(
                "context_management",
                self.config.context_management,
            )
        if self.config.service_tier is not None and self.config.provider == "openai":
            params["service_tier"] = response_options.pop("service_tier", self.config.service_tier)
        if self.config.reasoning is not None:
            params["reasoning"] = response_options.pop("reasoning", self.config.reasoning)
        if self.config.text is not None:
            params["text"] = response_options.pop("text", self.config.text)
        prompt_cache_key = (
            resolved.prompt_cache_key
            or self.config.default_prompt_cache_key
            or f"aibrain:{resolved.id}"
        )
        if prompt_cache_key and self.config.provider == "openai":
            params["prompt_cache_key"] = response_options.pop("prompt_cache_key", prompt_cache_key)
        retention = resolved.prompt_cache_retention or self.config.prompt_cache_retention
        if retention and self.config.provider == "openai":
            params["prompt_cache_retention"] = response_options.pop(
                "prompt_cache_retention",
                retention,
            )

        if state is not None:
            if self.config.state_mode == "local":
                params["store"] = response_options.pop("store", False)
            elif self.config.state_mode == "conversation":
                await self._ensure_remote_conversation(state)
                params["conversation"] = response_options.pop(
                    "conversation",
                    state.openai_conversation_id,
                )
            elif self.config.state_mode == "previous_response_id" and state.last_response_id:
                params["previous_response_id"] = response_options.pop(
                    "previous_response_id",
                    state.last_response_id,
                )

        tool_schemas = self._tool_schemas(resolved, tool_names)
        if openai_tools:
            tool_schemas.extend(openai_tools)
        raw_tools = response_options.pop("tools", None)
        if raw_tools:
            tool_schemas.extend(raw_tools)
        if tool_schemas:
            params["tools"] = tool_schemas
            params["parallel_tool_calls"] = response_options.pop("parallel_tool_calls", True)

        params.update({key: value for key, value in response_options.items() if value is not None})
        return params, state, resolved, memory_hits

    async def _ensure_thread(self, thread_id: str | None, persona: Persona) -> ThreadState | None:
        if self.config.state_mode == "stateless":
            return None
        if thread_id:
            existing = self.thread_store.get(thread_id)
            if existing:
                if existing.persona_id != persona.id:
                    existing.persona_id = persona.id
                    self.thread_store.upsert(existing)
                return existing
        return self.thread_store.create(persona_id=persona.id, thread_id=thread_id or str(uuid4()))

    async def _ensure_remote_conversation(self, state: ThreadState) -> None:
        if state.openai_conversation_id:
            return
        conversation = await self.openai.create_conversation(metadata=state.metadata)
        conversation_id = getattr(conversation, "id", None)
        if conversation_id:
            state.openai_conversation_id = conversation_id
            self.thread_store.upsert(state)

    def _memory_message(self, hits: list[MemoryRecord]) -> dict[str, Any]:
        lines = ["Relevant long-term memory:"]
        for hit in hits:
            lines.append(f"- ({hit.id}, score={hit.score:.3f}) {hit.content}")
        return {
            "type": "message",
            "role": "developer",
            "content": [{"type": "input_text", "text": "\n".join(lines)}],
        }

    def _resolve_memory_policy(
        self,
        use_memory: bool | MemoryPolicy | dict[str, Any],
        override: MemoryPolicy | dict[str, Any] | None = None,
    ) -> MemoryPolicy:
        if override is not None:
            return override if isinstance(override, MemoryPolicy) else MemoryPolicy.model_validate(override)
        if isinstance(use_memory, MemoryPolicy):
            return use_memory
        if isinstance(use_memory, dict):
            return MemoryPolicy.model_validate(use_memory)
        if not use_memory:
            return MemoryPolicy(enabled=False)
        return self.config.memory_policy.model_copy(
            update={
                "top_k": self.config.memory_top_k,
                "min_score": self.config.memory_min_score,
            }
        )

    def _tool_schemas(self, persona: Persona, names: Sequence[str] | None) -> list[dict[str, Any]]:
        selected = names if names is not None else persona.tools
        if selected is None:
            return self.tools.schemas()
        return self.tools.schemas(selected)

    async def _run_agent_loop(
        self,
        params: dict[str, Any],
        *,
        state: ThreadState | None,
        context: ToolContext,
        memory_hits: list[MemoryRecord],
        max_agent_steps: int,
    ) -> BrainResponse:
        tool_results: list[dict[str, Any]] = []
        steps = 0
        while True:
            response = await self.openai.create_response(**params)
            self._update_thread_after_response(state, response)
            calls = self._extract_function_calls(response)
            if not calls:
                return BrainResponse(
                    text=self._extract_text(response),
                    response_id=getattr(response, "id", None),
                    conversation_id=self._extract_conversation_id(response, state),
                    thread_id=state.thread_id if state else None,
                    raw_response=response,
                    usage=getattr(response, "usage", None),
                    tool_results=tool_results,
                    memory_hits=memory_hits,
                )
            steps += 1
            if steps > max_agent_steps:
                raise RuntimeError("Max agent tool steps exceeded")
            results = await self._execute_tool_calls(calls, context)
            tool_results.extend(results)
            params = self._continuation_params(params, response, self._tool_outputs(results))

    async def _execute_tool_calls(
        self,
        calls: list[dict[str, Any]],
        context: ToolContext,
    ) -> list[dict[str, Any]]:
        indexed_results = await asyncio.gather(
            *[
                self._execute_tool_call_index(index, call, context)
                for index, call in enumerate(calls)
            ]
        )
        return [result for _, result in sorted(indexed_results, key=lambda item: item[0])]

    async def _execute_tool_call_index(
        self,
        index: int,
        call: dict[str, Any],
        context: ToolContext,
    ) -> tuple[int, dict[str, Any]]:
        return index, await self._execute_tool_call(call, context)

    def _tool_outputs(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function_call_output",
                "call_id": result["call_id"],
                "output": json.dumps(result["output"]),
            }
            for result in results
        ]

    async def _execute_tool_call(
        self,
        call: dict[str, Any],
        context: ToolContext,
    ) -> dict[str, Any]:
        try:
            output = await self.tools.execute(
                call["name"],
                call["arguments"],
                context=context,
                timeout_seconds=self.config.tool_timeout_seconds,
            )
            return {
                "name": call["name"],
                "call_id": call["call_id"],
                "ok": True,
                "output": output,
            }
        except Exception as exc:
            return {
                "name": call["name"],
                "call_id": call["call_id"],
                "ok": False,
                "output": {"error": str(exc)},
            }

    def _continuation_params(
        self,
        previous_params: dict[str, Any],
        response: Any,
        outputs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        params = {
            key: value
            for key, value in previous_params.items()
            if key in _CONTINUATION_PARAM_KEYS
        }
        params["input"] = outputs
        response_id = getattr(response, "id", None)
        if self.config.state_mode == "local":
            params["input"] = [
                *list(previous_params.get("input") or []),
                *self._response_output_items(response),
                *outputs,
            ]
            params.pop("previous_response_id", None)
            params.pop("conversation", None)
            params.pop("prompt_cache_key", None)
            params.pop("prompt_cache_retention", None)
            params.pop("context_management", None)
        elif response_id:
            params["previous_response_id"] = response_id
        return params

    def _local_history_input(self, state: ThreadState) -> list[dict[str, Any]]:
        return [
            {
                "type": "message",
                "role": message["role"],
                "content": self._normalize_local_history_content(message["content"]),
            }
            for message in self.chat_store.list(
                state.thread_id,
                limit=self.config.local_history_limit,
            )
        ]

    def _append_local_chat_response(
        self,
        state: ThreadState | None,
        text: str,
        response_id: str | None,
    ) -> None:
        if state is None or self.config.state_mode != "local" or not text:
            return
        self.chat_store.append(
            state.thread_id,
            "assistant",
            [{"type": "input_text", "text": text}],
            metadata={"source": "brain.assistant", "response_id": response_id},
        )

    def _response_output_items(self, response: Any) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for item in getattr(response, "output", None) or []:
            if isinstance(item, dict):
                items.append(item)
                continue
            item_type = self._field(item, "type")
            if item_type == "function_call":
                items.append(
                    {
                        "type": "function_call",
                        "call_id": self._field(item, "call_id"),
                        "name": self._field(item, "name"),
                        "arguments": self._field(item, "arguments") or "{}",
                    }
                )
        return items

    def _normalize_local_history_content(self, content: Any) -> list[dict[str, Any]]:
        if isinstance(content, str):
            return [{"type": "input_text", "text": content}]
        if isinstance(content, list):
            return content
        return [{"type": "input_text", "text": str(content)}]

    def _history_content_from_user_message(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            item
            for item in message.get("content", [])
            if isinstance(item, dict) and item.get("type") == "input_text"
        ]

    def _extract_function_calls(self, response: Any) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        for item in getattr(response, "output", None) or []:
            item_type = self._field(item, "type")
            if item_type == "function_call":
                calls.append(
                    {
                        "name": self._field(item, "name"),
                        "call_id": self._field(item, "call_id"),
                        "arguments": self._field(item, "arguments") or "{}",
                    }
                )
        return calls

    def _extract_text(self, response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if output_text:
            return output_text
        parts: list[str] = []
        for item in getattr(response, "output", None) or []:
            for content in self._field(item, "content") or []:
                if self._field(content, "type") == "output_text":
                    text = self._field(content, "text")
                    if text:
                        parts.append(text)
        return "".join(parts)

    def _extract_conversation_id(self, response: Any, state: ThreadState | None) -> str | None:
        conversation = getattr(response, "conversation", None)
        if isinstance(conversation, str):
            return conversation
        conversation_id = getattr(conversation, "id", None)
        if conversation_id:
            return conversation_id
        if isinstance(conversation, dict):
            return conversation.get("id")
        return state.openai_conversation_id if state else None

    def _update_thread_after_response(self, state: ThreadState | None, response: Any) -> None:
        if state is None:
            return
        conversation_id = self._extract_conversation_id(response, state)
        response_id = getattr(response, "id", None)
        updated_state = self.thread_store.update_remote_ids(
            state.thread_id,
            openai_conversation_id=conversation_id,
            last_response_id=response_id,
        )
        state.openai_conversation_id = conversation_id
        if response_id is not None:
            state.last_response_id = response_id
        state.updated_at = updated_state.updated_at

    async def _append_memory_stack_response_event(
        self,
        response: BrainResponse,
        state: ThreadState | None,
        persona_id: str,
    ) -> None:
        if self.memory_stack is None or state is None or not response.text:
            return
        await self.memory_stack.append_event(
            event_type="response",
            actor="assistant",
            content=response.text,
            thread=state,
            persona_id=persona_id,
            metadata={
                "source": "brain.response",
                "response_id": response.response_id,
                "conversation_id": response.conversation_id,
            },
            extract=self.config.memory_stack.extract_response_events,
        )

    async def _append_memory_stack_raw_response_event(
        self,
        response: Any,
        state: ThreadState | None,
        persona_id: str,
    ) -> None:
        if self.memory_stack is None or state is None:
            return
        text = self._extract_text(response)
        if not text:
            return
        await self.memory_stack.append_event(
            event_type="response",
            actor="assistant",
            content=text,
            thread=state,
            persona_id=persona_id,
            metadata={
                "source": "brain.stream.response",
                "response_id": getattr(response, "id", None),
                "conversation_id": self._extract_conversation_id(response, state),
            },
            extract=self.config.memory_stack.extract_response_events,
        )

    def _normalize_stream_event(self, event: Any) -> BrainEvent | None:
        event_type = getattr(event, "type", "")
        if event_type in {"response.output_text.delta", "response.text.delta"}:
            return BrainEvent("text.delta", {"text": getattr(event, "delta", "")})
        if event_type in {"response.output_text.done", "response.text.done"}:
            return BrainEvent("text.done", {"text": getattr(event, "text", "")})
        if event_type == "response.created":
            response = getattr(event, "response", None)
            return BrainEvent("response.started", {"response_id": getattr(response, "id", None)})
        if event_type == "response.failed":
            response = getattr(event, "response", None)
            return BrainEvent("error", {"message": str(getattr(response, "error", response))})
        return None

    @staticmethod
    def _field(item: Any, name: str) -> Any:
        if isinstance(item, dict):
            return item.get(name)
        return getattr(item, name, None)


def _memory_query_text(text: str) -> str:
    limit = _env_int("AIBRAIN_MEMORY_QUERY_MAX_CHARS", DEFAULT_MEMORY_QUERY_MAX_CHARS)
    return str(text or "").strip()[: max(1, limit)]


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _dedupe_memory_records(records: list[MemoryRecord]) -> list[MemoryRecord]:
    deduped: dict[str, MemoryRecord] = {}
    for record in sorted(records, key=lambda item: item.score, reverse=True):
        key = str(
            record.metadata.get("source_fact_id")
            or record.metadata.get("source_event_id")
            or record.id
        )
        if key not in deduped:
            deduped[key] = record
    return list(deduped.values())
