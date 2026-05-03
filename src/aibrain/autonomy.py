from __future__ import annotations

import asyncio
import os
import random
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field

from .config import Persona
from .types import BrainEvent


class AutonomyAction(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str
    description: str = ""
    schema_: dict[str, Any] = Field(default_factory=dict, alias="schema")


class AutonomyDecision(BaseModel):
    should_act: bool = False
    action: str = "none"
    message: str = ""
    confidence: float = 0.0
    reason: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)


class HeartbeatActiveHours(BaseModel):
    start: str = "09:00"
    end: str = "22:00"
    timezone: str | None = None


class HeartbeatTask(BaseModel):
    name: str
    prompt: str
    interval_seconds: float | None = None
    last_run_at: str | None = None


class HeartbeatFile(BaseModel):
    path: str | None = None
    exists: bool = False
    content: str = ""
    instructions: str = ""
    tasks: list[HeartbeatTask] = Field(default_factory=list)
    due_tasks: list[HeartbeatTask] = Field(default_factory=list)
    skipped_reason: str | None = None

    @property
    def has_tasks(self) -> bool:
        return bool(self.tasks)


class HeartbeatConfig(BaseModel):
    enabled: bool = True
    interval_seconds: float = 60.0
    jitter_seconds: float = 15.0
    run_probability: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Chance that a scheduled heartbeat tick actually runs the model.",
    )
    max_ticks: int | None = None
    act_threshold: float = 0.5
    heartbeat_path: Path | None = Field(default_factory=lambda: _default_heartbeat_path())
    ack_token: str = "HEARTBEAT_OK"
    ack_max_chars: int = 300
    skip_empty_file: bool = True
    task_state_key: str = "heartbeat_tasks"
    light_context: bool = False
    isolated_session: bool = False
    active_hours: HeartbeatActiveHours | dict[str, Any] | None = None
    prompt: str = (
        "Read HEARTBEAT.md if it exists. Follow it strictly. "
        "Do not infer or repeat old tasks from prior chats. "
        "If nothing needs attention, acknowledge with HEARTBEAT_OK."
    )
    idle_instruction: str = (
        "Decide whether the AI should proactively do something now. "
        "Stay quiet unless there is a useful, timely, or socially natural reason to act."
    )


class HeartbeatResult(BaseModel):
    skipped: bool = False
    reason: str | None = None
    decision: AutonomyDecision | None = None
    heartbeat: HeartbeatFile | None = None


ContextProvider = Callable[[], dict[str, Any] | Awaitable[dict[str, Any]]]


async def maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


def heartbeat_prompt(
    *,
    context: dict[str, Any],
    actions: list[AutonomyAction],
    instruction: str,
    heartbeat: HeartbeatFile | None = None,
    ack_token: str = "HEARTBEAT_OK",
) -> str:
    normalized_actions = actions or [
        AutonomyAction(
            name="message",
            description="Return a proactive message for the host adapter to deliver.",
        )
    ]
    action_lines = [
        f"- {action.name}: {action.description or 'No description'}"
        for action in normalized_actions
    ]
    heartbeat_lines = []
    if heartbeat is not None:
        if heartbeat.exists:
            heartbeat_lines.append(f"HEARTBEAT.md path: {heartbeat.path}")
            if heartbeat.due_tasks:
                heartbeat_lines.append("Due HEARTBEAT.md tasks:")
                for task in heartbeat.due_tasks:
                    heartbeat_lines.append(f"- {task.name}: {task.prompt}")
            if heartbeat.instructions:
                heartbeat_lines.append("Additional HEARTBEAT.md instructions:")
                heartbeat_lines.append(heartbeat.instructions)
        else:
            heartbeat_lines.append("HEARTBEAT.md was not found; use current context only.")
    return (
        f"{instruction}\n\n"
        f"Heartbeat response contract:\n"
        f"- If nothing needs attention, set should_act=false, action=\"none\", "
        f"and message=\"{ack_token}\".\n"
        f"- If the host adapter should say something, set should_act=true, "
        f"action=\"message\", and put the outward text in message.\n"
        f"- If a listed adapter action is needed, set action to that action name and put "
        f"arguments in payload.\n"
        f"- Do not invent platform abilities that are not listed.\n\n"
        f"{chr(10).join(heartbeat_lines)}\n\n"
        f"Available actions:\n{chr(10).join(action_lines) if action_lines else '- none'}\n\n"
        f"Current context:\n{context}"
    )


def load_heartbeat_file(
    path: Path | str | None,
    *,
    task_state: dict[str, str] | None = None,
    now: datetime | None = None,
    skip_empty_file: bool = True,
) -> HeartbeatFile:
    if path is None:
        return HeartbeatFile(exists=False)
    resolved = Path(path)
    if not resolved.exists():
        return HeartbeatFile(path=str(resolved), exists=False)
    content = resolved.read_text(encoding="utf-8")
    if skip_empty_file and is_effectively_empty_heartbeat(content):
        return HeartbeatFile(
            path=str(resolved),
            exists=True,
            content=content,
            skipped_reason="empty-heartbeat-file",
        )
    heartbeat = parse_heartbeat_content(content)
    heartbeat.path = str(resolved)
    heartbeat.exists = True
    heartbeat.content = content
    current_time = now or datetime.now(timezone.utc)
    state = task_state or {}
    due_tasks = []
    for task in heartbeat.tasks:
        last_run = state.get(task.name)
        task.last_run_at = last_run
        if _task_is_due(task, last_run, current_time):
            due_tasks.append(task)
    heartbeat.due_tasks = due_tasks
    if heartbeat.tasks and not heartbeat.due_tasks:
        heartbeat.skipped_reason = "no-tasks-due"
    return heartbeat


def parse_heartbeat_content(content: str) -> HeartbeatFile:
    lines = content.splitlines()
    tasks_index = next(
        (
            index
            for index, line in enumerate(lines)
            if line.strip().lower() == "tasks:"
        ),
        None,
    )
    if tasks_index is None:
        return HeartbeatFile(instructions=content.strip())

    before = lines[:tasks_index]
    after = lines[tasks_index + 1 :]
    tasks: list[HeartbeatTask] = []
    additional: list[str] = []
    current: dict[str, Any] | None = None
    in_task_block = True

    for line in after:
        stripped = line.strip()
        if in_task_block and not stripped:
            continue
        if in_task_block and stripped.startswith("- name:"):
            if current:
                tasks.append(_task_from_mapping(current))
            current = {"name": _parse_scalar(stripped.split(":", 1)[1])}
            continue
        if in_task_block and current is not None:
            match = re.match(r"^([A-Za-z_][\w-]*)\s*:\s*(.*)$", stripped)
            if match:
                key, value = match.groups()
                if key in {"prompt", "interval"}:
                    current[key] = _parse_scalar(value)
                    continue
        if current:
            tasks.append(_task_from_mapping(current))
            current = None
        in_task_block = False
        additional.append(line)

    if current:
        tasks.append(_task_from_mapping(current))

    instructions = "\n".join([*before, *additional]).strip()
    return HeartbeatFile(instructions=instructions, tasks=tasks)


def is_effectively_empty_heartbeat(content: str) -> bool:
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if stripped.startswith("<!--") and stripped.endswith("-->"):
            continue
        return False
    return True


def parse_interval_seconds(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    match = re.match(r"^(\d+(?:\.\d+)?)\s*([a-z]*)$", text)
    if not match:
        raise ValueError(f"Invalid heartbeat interval: {value}")
    amount = float(match.group(1))
    unit = match.group(2) or "s"
    multipliers = {
        "s": 1,
        "sec": 1,
        "secs": 1,
        "second": 1,
        "seconds": 1,
        "m": 60,
        "min": 60,
        "mins": 60,
        "minute": 60,
        "minutes": 60,
        "h": 3600,
        "hr": 3600,
        "hrs": 3600,
        "hour": 3600,
        "hours": 3600,
        "d": 86400,
        "day": 86400,
        "days": 86400,
    }
    if unit not in multipliers:
        raise ValueError(f"Invalid heartbeat interval unit: {value}")
    return amount * multipliers[unit]


def within_active_hours(
    active_hours: HeartbeatActiveHours | dict[str, Any] | None,
    *,
    now: datetime | None = None,
) -> bool:
    if active_hours is None:
        return True
    hours = (
        active_hours
        if isinstance(active_hours, HeartbeatActiveHours)
        else HeartbeatActiveHours.model_validate(active_hours)
    )
    tz = ZoneInfo(hours.timezone) if hours.timezone else None
    current = now or datetime.now(tz or timezone.utc)
    if tz:
        current = current.astimezone(tz)
    start = _parse_clock(hours.start)
    end = _parse_clock(hours.end)
    current_time = current.time().replace(tzinfo=None)
    if start == end:
        return False
    if start < end:
        return start <= current_time < end
    return current_time >= start or current_time < end


def strip_heartbeat_ack(
    message: str,
    *,
    ack_token: str = "HEARTBEAT_OK",
    ack_max_chars: int = 300,
) -> tuple[bool, str]:
    text = message.strip()
    if not text:
        return False, message
    if text == ack_token:
        return True, ""
    if text.startswith(ack_token):
        remainder = text[len(ack_token) :].strip()
        if len(remainder) <= ack_max_chars:
            return True, remainder
    if text.endswith(ack_token):
        remainder = text[: -len(ack_token)].strip()
        if len(remainder) <= ack_max_chars:
            return True, remainder
    return False, message


class AutonomyLoop:
    def __init__(
        self,
        brain: Any,
        *,
        thread_id: str,
        persona: Persona | dict[str, Any] | None = None,
        config: HeartbeatConfig | None = None,
        actions: list[AutonomyAction | dict[str, Any]] | None = None,
        context_provider: ContextProvider | None = None,
    ):
        self.brain = brain
        self.thread_id = thread_id
        self.persona = persona
        self.config = config or HeartbeatConfig()
        self.actions = [
            action if isinstance(action, AutonomyAction) else AutonomyAction.model_validate(action)
            for action in actions or []
        ]
        self.context_provider = context_provider
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    async def run(self) -> AsyncIterator[BrainEvent]:
        tick = 0
        while self.config.enabled and not self._stop.is_set():
            if self.config.max_ticks is not None and tick >= self.config.max_ticks:
                return
            delay = self._next_delay()
            if delay > 0:
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=delay)
                    return
                except asyncio.TimeoutError:
                    pass
            tick += 1
            yield BrainEvent("heartbeat.tick", {"tick": tick, "thread_id": self.thread_id})
            if hasattr(self.brain, "heartbeat_tick"):
                result = await self.brain.heartbeat_tick(
                    thread_id=self.thread_id,
                    persona=self.persona,
                    actions=self.actions,
                    context=await self._context(),
                    config=self.config,
                )
                if result.skipped:
                    yield BrainEvent(
                        "heartbeat.skipped",
                        {
                            "tick": tick,
                            "thread_id": self.thread_id,
                            "reason": result.reason,
                        },
                    )
                    continue
                decision = result.decision or AutonomyDecision()
            else:
                decision = await self.brain.autonomy_tick(
                    thread_id=self.thread_id,
                    persona=self.persona,
                    actions=self.actions,
                    context=await self._context(),
                    config=self.config,
                )
            yield BrainEvent(
                "heartbeat.decision",
                {"tick": tick, "thread_id": self.thread_id, "decision": decision.model_dump()},
            )

    async def _context(self) -> dict[str, Any]:
        if self.context_provider is None:
            return {}
        return await maybe_await(self.context_provider())

    def _next_delay(self) -> float:
        jitter = self.config.jitter_seconds
        if jitter <= 0:
            return max(0.0, self.config.interval_seconds)
        return max(0.0, self.config.interval_seconds + random.uniform(-jitter, jitter))


def _default_heartbeat_path() -> Path:
    return Path(os.environ.get("AIBRAIN_HEARTBEAT_FILE", "HEARTBEAT.md"))


def _parse_scalar(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        return text[1:-1]
    return text


def _task_from_mapping(raw: dict[str, Any]) -> HeartbeatTask:
    return HeartbeatTask(
        name=str(raw.get("name", "")).strip(),
        prompt=str(raw.get("prompt", "")).strip(),
        interval_seconds=parse_interval_seconds(raw.get("interval")),
    )


def _task_is_due(task: HeartbeatTask, last_run_at: str | None, now: datetime) -> bool:
    if task.interval_seconds is None:
        return True
    if not last_run_at:
        return True
    try:
        last_run = datetime.fromisoformat(last_run_at)
    except ValueError:
        return True
    if last_run.tzinfo is None:
        last_run = last_run.replace(tzinfo=timezone.utc)
    return (now - last_run.astimezone(timezone.utc)).total_seconds() >= task.interval_seconds


def _parse_clock(value: str) -> time:
    if value == "24:00":
        return time(23, 59, 59, 999999)
    hour, minute = value.split(":", 1)
    return time(int(hour), int(minute))
