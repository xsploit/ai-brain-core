import json

from aibrain.codex_bridge import BRIDGE_REQUEST_SCHEMA, CodexBridgeQueue


def test_codex_bridge_enqueue_writes_authorized_request(tmp_path):
    queue = CodexBridgeQueue(tmp_path / "bridge", enabled=True, thread_id="thread-123")

    path = queue.enqueue(
        requester_id=120418341775998976,
        requester_name="SUBSECT",
        guild_id=1,
        channel_id=2,
        message_id=3,
        prompt="build the bridge",
        recent_messages=[{"author": "Subby", "content": "context"}],
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == BRIDGE_REQUEST_SCHEMA
    assert payload["requester_id"] == "120418341775998976"
    assert payload["guild_id"] == "1"
    assert payload["channel_id"] == "2"
    assert payload["message_id"] == "3"
    assert payload["prompt"] == "build the bridge"
    assert payload["authority"]["authorized"] is True
    assert payload["delivery"] == {"mode": "thread_heartbeat", "thread_id": "thread-123"}
    assert payload["context"]["recent_messages"] == [{"author": "Subby", "content": "context"}]
    assert queue.status().inbox_count == 1


def test_codex_bridge_pause_and_clear_pending_archives_files(tmp_path):
    queue = CodexBridgeQueue(tmp_path / "bridge", enabled=True)
    queue.enqueue(requester_id=1, requester_name="admin", prompt="one")
    queue.enqueue(requester_id=1, requester_name="admin", prompt="two")

    queue.set_paused(True, actor_id=1)
    cleared = queue.clear_pending(actor_id=1)
    status = queue.status()

    assert cleared == 2
    assert status.paused is True
    assert status.inbox_count == 0
    assert status.archive_count == 2
