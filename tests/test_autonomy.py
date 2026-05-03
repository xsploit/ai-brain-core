from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from aibrain import (
    AutonomyDecision,
    Brain,
    BrainConfig,
    HeartbeatConfig,
    NullTTS,
    TTSConfig,
)
from aibrain.autonomy import (
    load_heartbeat_file,
    parse_interval_seconds,
    strip_heartbeat_ack,
    within_active_hours,
)


class FakeConversations:
    async def create(self, **kwargs):
        return SimpleNamespace(id="conv_heartbeat")


class FakeResponses:
    def __init__(self, decision: AutonomyDecision):
        self.decision = decision
        self.calls = []

    async def parse(self, *, text_format, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            id="resp_heartbeat",
            output=[],
            output_text=self.decision.model_dump_json(),
            output_parsed=self.decision,
            conversation=kwargs.get("conversation"),
            usage=None,
        )


class FakeClient:
    def __init__(self, decision: AutonomyDecision):
        self.responses = FakeResponses(decision)
        self.conversations = FakeConversations()


def test_heartbeat_file_parses_due_tasks(tmp_path):
    path = tmp_path / "HEARTBEAT.md"
    path.write_text(
        """
# Heartbeat checklist
- Keep alerts short.

tasks:
- name: inbox
  interval: 30m
  prompt: "Check urgent messages."

# Extra
- Reply HEARTBEAT_OK when idle.
""".strip(),
        encoding="utf-8",
    )
    now = datetime.now(timezone.utc)
    heartbeat = load_heartbeat_file(
        path,
        task_state={"inbox": (now - timedelta(hours=1)).isoformat()},
        now=now,
    )

    assert heartbeat.skipped_reason is None
    assert heartbeat.due_tasks[0].name == "inbox"
    assert heartbeat.due_tasks[0].interval_seconds == 1800
    assert "Keep alerts short" in heartbeat.instructions
    assert "Reply HEARTBEAT_OK" in heartbeat.instructions


def test_heartbeat_file_skips_empty_or_not_due(tmp_path):
    empty_path = tmp_path / "empty.md"
    empty_path.write_text("# Heading\n\n", encoding="utf-8")
    assert load_heartbeat_file(empty_path).skipped_reason == "empty-heartbeat-file"

    tasks_path = tmp_path / "tasks.md"
    tasks_path.write_text(
        """
tasks:
- name: slow-check
  interval: 2h
  prompt: "Check slowly."
""".strip(),
        encoding="utf-8",
    )
    now = datetime.now(timezone.utc)
    heartbeat = load_heartbeat_file(
        tasks_path,
        task_state={"slow-check": now.isoformat()},
        now=now,
    )

    assert heartbeat.skipped_reason == "no-tasks-due"


def test_heartbeat_helpers():
    assert parse_interval_seconds("2h") == 7200
    assert parse_interval_seconds("30m") == 1800
    assert within_active_hours(
        {"start": "09:00", "end": "17:00", "timezone": "UTC"},
        now=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
    )
    assert not within_active_hours(
        {"start": "09:00", "end": "17:00", "timezone": "UTC"},
        now=datetime(2026, 1, 1, 21, 0, tzinfo=timezone.utc),
    )
    assert strip_heartbeat_ack("HEARTBEAT_OK") == (True, "")


@pytest.mark.asyncio
async def test_heartbeat_tick_reads_file_updates_task_state_and_strips_ack(tmp_path):
    path = tmp_path / "HEARTBEAT.md"
    path.write_text(
        """
tasks:
- name: memory-review
  interval: 1m
  prompt: "Review whether anything should be remembered."
""".strip(),
        encoding="utf-8",
    )
    client = FakeClient(
        AutonomyDecision(
            should_act=True,
            action="message",
            message="HEARTBEAT_OK",
            confidence=0.99,
        )
    )
    brain = Brain(
        BrainConfig(database_path=tmp_path / "brain.sqlite3"),
        client=client,
        tts_provider=NullTTS(TTSConfig(provider="null")),
    )

    result = await brain.heartbeat_tick(
        thread_id="discord:channel:123",
        config=HeartbeatConfig(
            heartbeat_path=path,
            interval_seconds=0,
            jitter_seconds=0,
            run_probability=1.0,
        ),
        use_memory=False,
    )

    assert not result.skipped
    assert result.decision is not None
    assert result.decision.should_act is False
    assert result.decision.action == "none"
    prompt = client.responses.calls[0]["input"][-1]["content"][0]["text"]
    assert "memory-review" in prompt

    state = brain.thread_store.get("discord:channel:123")
    assert state is not None
    assert "memory-review" in state.metadata["heartbeat_tasks"]


@pytest.mark.asyncio
async def test_heartbeat_tick_can_skip_by_probability(tmp_path):
    client = FakeClient(AutonomyDecision(should_act=True, action="message", confidence=1.0))
    brain = Brain(
        BrainConfig(database_path=tmp_path / "brain.sqlite3"),
        client=client,
        tts_provider=NullTTS(TTSConfig(provider="null")),
    )

    result = await brain.heartbeat_tick(
        thread_id="heartbeat:skip",
        config=HeartbeatConfig(run_probability=0.0, heartbeat_path=tmp_path / "missing.md"),
    )

    assert result.skipped
    assert result.reason == "random-chance"
    assert client.responses.calls == []
