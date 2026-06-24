from __future__ import annotations

import pytest

from aibrain.treblo_song import (
    TrebloSongClient,
    TrebloSongConfig,
    TrebloSongQueue,
    TrebloSongRateLimited,
    extract_audio_url,
    extract_task_id,
    normalize_mode,
    normalize_status,
    stream_url_for_task,
)


@pytest.mark.asyncio
async def test_treblo_prompt_only_payload_is_prompt_without_tags_or_lyrics():
    client = TrebloSongClient(TrebloSongConfig(api_key="test", ai_gateway_api_key=None))

    payload = await client.payload("raw melodic hardcore with a huge chorus", mode="prompt_only")

    assert payload["prompt"] == "raw melodic hardcore with a huge chorus"
    assert payload["output_format"] == "ogg"
    assert payload["enable_streaming"] is True
    assert payload["align_lyrics"] is True
    assert payload["length_range"] == [30, 120]
    assert "tags" not in payload
    assert "lyrics" not in payload


@pytest.mark.asyncio
async def test_treblo_instrumental_payload_uses_prompt_modifier_not_fake_api_field():
    client = TrebloSongClient(TrebloSongConfig(api_key="test", ai_gateway_api_key=None))

    payload = await client.payload("glassy synthwave at midnight", mode="instrumental")

    assert payload["prompt"].startswith("Instrumental track with no vocals and no lyrics.")
    assert payload["align_lyrics"] is False
    assert "instrumental" not in payload
    assert "tags" not in payload
    assert "lyrics" not in payload


@pytest.mark.asyncio
async def test_treblo_auto_lyrics_payload_uses_gateway_lyrics(monkeypatch):
    client = TrebloSongClient(TrebloSongConfig(api_key="test", ai_gateway_api_key="gateway"))

    async def fake_generate_lyrics(prompt: str) -> str:
        return "[Chorus]\nI can hear the static bloom"

    monkeypatch.setattr(client, "generate_lyrics", fake_generate_lyrics)

    payload = await client.payload("noisy pop chorus", mode="auto_lyrics")

    assert payload["prompt"] == "noisy pop chorus"
    assert payload["lyrics"] == "[Chorus]\nI can hear the static bloom"
    assert "tags" not in payload


def test_treblo_extractors_match_ui_response_shapes():
    assert extract_task_id({"task_id": "task-1"}) == "task-1"
    assert extract_task_id({"data": {"taskId": "task-2"}}) == "task-2"
    assert extract_audio_url({"song_paths": ["https://cdn.example/song.ogg"]}) == "https://cdn.example/song.ogg"
    assert normalize_status({"status": "GENERATING_STREAMING_READY"}) == "GENERATING_STREAMING_READY"
    assert stream_url_for_task("abc") == "https://api-stream.treblo.com/stream/abc"
    assert normalize_mode("autolyrics") == "auto_lyrics"


@pytest.mark.asyncio
async def test_treblo_queue_rate_limits_non_owner():
    queue = TrebloSongQueue(
        TrebloSongConfig(api_key="test", user_cooldown_seconds=60),
        owner_user_ids={999},
    )

    await queue.submit(user_id=123, channel_id=1, prompt="first")

    with pytest.raises(TrebloSongRateLimited):
        await queue.submit(user_id=123, channel_id=1, prompt="second")

    owner_job = await queue.submit(user_id=999, channel_id=1, prompt="owner bypass")
    assert owner_job.prompt == "owner bypass"
