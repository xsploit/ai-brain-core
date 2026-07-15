from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


BRIDGE_REQUEST_SCHEMA = "neuro_codex_bridge.request.v1"
BRIDGE_RESULT_SCHEMA = "neuro_codex_bridge.result.v1"
DEFAULT_CODEX_THREAD_ID = "019e53da-7adc-7251-a203-e9da141553f7"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def bridge_file_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def safe_bridge_id(value: str | None = None) -> str:
    raw = value or str(uuid4())
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip(".-")
    return safe or str(uuid4())


@dataclass(slots=True)
class CodexBridgeStatus:
    enabled: bool
    paused: bool
    root: Path
    inbox_count: int
    outbox_count: int
    archive_count: int
    oldest_request: str | None = None
    last_result: str | None = None


class CodexBridgeQueue:
    def __init__(
        self,
        root: Path | str,
        *,
        enabled: bool = False,
        thread_id: str = DEFAULT_CODEX_THREAD_ID,
    ) -> None:
        self.root = Path(root)
        self.enabled = enabled
        self.thread_id = thread_id
        self.inbox = self.root / "inbox"
        self.outbox = self.root / "outbox"
        self.archive = self.root / "archive"
        self.state_path = self.root / "state.json"
        self.ensure_dirs()

    @classmethod
    def from_env(cls) -> "CodexBridgeQueue":
        root = Path(os.getenv("DISCORD_BRAIN_CODEX_BRIDGE_QUEUE", "codex_bridge"))
        return cls(
            root,
            enabled=_env_bool_local("DISCORD_BRAIN_CODEX_BRIDGE_ENABLED", False),
            thread_id=os.getenv("DISCORD_BRAIN_CODEX_THREAD_ID", DEFAULT_CODEX_THREAD_ID),
        )

    def ensure_dirs(self) -> None:
        self.inbox.mkdir(parents=True, exist_ok=True)
        self.outbox.mkdir(parents=True, exist_ok=True)
        self.archive.mkdir(parents=True, exist_ok=True)

    def state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"paused": False}
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"paused": False}
        return data if isinstance(data, dict) else {"paused": False}

    def set_paused(self, paused: bool, *, actor_id: str | int | None = None) -> None:
        state = self.state()
        state.update(
            {
                "paused": bool(paused),
                "updated_at": utc_now_iso(),
                "updated_by": str(actor_id) if actor_id is not None else None,
            }
        )
        self.state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def is_paused(self) -> bool:
        return bool(self.state().get("paused", False))

    def pending_files(self) -> list[Path]:
        return sorted(
            (path for path in self.inbox.glob("*.json") if path.is_file()),
            key=lambda path: (path.stat().st_mtime, path.name),
        )

    def result_files(self) -> list[Path]:
        return sorted(
            (path for path in self.outbox.glob("*.json") if path.is_file()),
            key=lambda path: (path.stat().st_mtime, path.name),
        )

    def archive_files(self) -> list[Path]:
        return sorted(
            (path for path in self.archive.rglob("*.json") if path.is_file()),
            key=lambda path: (path.stat().st_mtime, path.name),
        )

    def status(self) -> CodexBridgeStatus:
        pending = self.pending_files()
        results = self.result_files()
        archives = self.archive_files()
        return CodexBridgeStatus(
            enabled=self.enabled,
            paused=self.is_paused(),
            root=self.root,
            inbox_count=len(pending),
            outbox_count=len(results),
            archive_count=len(archives),
            oldest_request=pending[0].name if pending else None,
            last_result=results[-1].name if results else None,
        )

    def enqueue(
        self,
        *,
        requester_id: str | int,
        requester_name: str,
        prompt: str,
        intent: str = "ask_codex",
        priority: str = "normal",
        authority_mode: str = "manual_admin",
        authority_reason: str = "authorized Discord bridge command",
        delivery_mode: str = "thread_heartbeat",
        guild_id: str | int | None = None,
        channel_id: str | int | None = None,
        message_id: str | int | None = None,
        recent_messages: list[dict[str, Any]] | None = None,
        attachments: list[dict[str, Any]] | None = None,
        model_suggestion: str | None = None,
        harness_agent: str = "claude",
        harness_permission_profile: str = "inspect",
    ) -> Path:
        self.ensure_dirs()
        request_id = safe_bridge_id()
        payload = {
            "schema": BRIDGE_REQUEST_SCHEMA,
            "id": request_id,
            "created_at": utc_now_iso(),
            "source": "discord",
            "requester_id": str(requester_id),
            "requester_name": str(requester_name),
            "guild_id": str(guild_id) if guild_id is not None else None,
            "channel_id": str(channel_id) if channel_id is not None else None,
            "message_id": str(message_id) if message_id is not None else None,
            "intent": intent,
            "priority": priority,
            "prompt": prompt,
            "authority": {
                "mode": authority_mode,
                "authorized": True,
                "reason": authority_reason,
            },
            "delivery": {
                "mode": delivery_mode,
                "thread_id": self.thread_id,
            },
            "context": {
                "recent_messages": recent_messages or [],
                "attachments": attachments or [],
                "model_suggestion": model_suggestion,
            },
            "harness": {
                "agent": harness_agent,
                "permission_profile": harness_permission_profile,
            },
        }
        path = self.inbox / f"{bridge_file_timestamp()}-{request_id}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def clear_pending(self, *, actor_id: str | int | None = None) -> int:
        files = self.pending_files()
        if not files:
            return 0
        destination = self.archive / f"cleared-{bridge_file_timestamp()}"
        destination.mkdir(parents=True, exist_ok=True)
        for path in files:
            shutil.move(str(path), str(destination / path.name))
        state = self.state()
        state.update(
            {
                "last_clear_at": utc_now_iso(),
                "last_clear_by": str(actor_id) if actor_id is not None else None,
                "last_clear_count": len(files),
            }
        )
        self.state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return len(files)


def _env_bool_local(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
