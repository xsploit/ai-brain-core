from __future__ import annotations

import pytest

from aibrain.treblo_song import (
    TrebloSongClient,
    TrebloSongConfig,
    TrebloSongJob,
    TrebloSongQueue,
    TrebloSongRateLimited,
    extract_audio_url,
    extract_task_id,
    normalize_duration_preset,
    normalize_mode,
    normalize_status,
    stream_url_for_task,
)


@pytest.mark.asyncio
async def test_treblo_prompt_only_payload_is_prompt_without_tags_or_lyrics():
    client = TrebloSongClient(TrebloSongConfig(api_key="test"))

    payload = await client.payload("raw melodic hardcore with a huge chorus", mode="prompt_only")

    assert payload["prompt"] == "raw melodic hardcore with a huge chorus"
    assert payload["output_format"] == "ogg"
    assert payload["enable_streaming"] is True
    assert payload["align_lyrics"] is True
    assert "length_range" not in payload
    assert "tags" not in payload
    assert "lyrics" not in payload


@pytest.mark.asyncio
async def test_treblo_instrumental_payload_uses_documented_api_field():
    client = TrebloSongClient(TrebloSongConfig(api_key="test"))

    payload = await client.payload("glassy synthwave at midnight", mode="instrumental")

    assert payload["prompt"] == "glassy synthwave at midnight"
    assert payload["instrumental"] is True
    assert payload["align_lyrics"] is False
    assert "tags" not in payload
    assert "lyrics" not in payload


@pytest.mark.asyncio
async def test_treblo_auto_lyrics_payload_uses_treblo_native_prompt_without_gateway():
    client = TrebloSongClient(TrebloSongConfig(api_key="test"))

    payload = await client.payload("noisy pop chorus", mode="auto_lyrics")

    assert payload["prompt"] == "Original vocal song with Treblo-generated lyrics. noisy pop chorus"
    assert payload["align_lyrics"] is True
    assert "tags" not in payload
    assert "lyrics" not in payload


@pytest.mark.asyncio
async def test_treblo_payload_includes_selected_duration_range():
    client = TrebloSongClient(TrebloSongConfig(api_key="test"))

    payload = await client.payload("long ambient song", mode="prompt_only", length_range=(270, 300))

    assert payload["length_range"] == [270, 300]


def test_treblo_extractors_match_ui_response_shapes():
    assert extract_task_id({"task_id": "task-1"}) == "task-1"
    assert extract_task_id({"data": {"taskId": "task-2"}}) == "task-2"
    assert extract_audio_url({"song_paths": ["https://cdn.example/song.ogg"]}) == "https://cdn.example/song.ogg"
    assert normalize_status({"status": "GENERATING_STREAMING_READY"}) == "GENERATING_STREAMING_READY"
    assert stream_url_for_task("abc") == "https://api-stream.treblo.com/stream/abc"
    assert normalize_mode("autolyrics") == "auto_lyrics"
    assert normalize_duration_preset("1m") == ("60", (30, 60))
    assert normalize_duration_preset("5m") == ("300", (270, 300))
    assert normalize_duration_preset("default") == ("default", None)


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
    owner_job_2 = await queue.submit(user_id=999, channel_id=1, prompt="owner bypass again")
    snapshot = queue.snapshot()

    assert owner_job.prompt == "owner bypass"
    assert owner_job_2.prompt == "owner bypass again"
    assert 999 in snapshot["owner_user_ids"]
    assert 999 not in queue.last_submit_at


@pytest.mark.asyncio
async def test_treblo_song_completion_downloads_archives_and_uploads_file_without_public_links(tmp_path):
    class FakeClient:
        def __init__(self):
            self.status_calls = 0

        async def generate(self, prompt, *, mode="prompt_only", length_range=None):
            assert length_range == (270, 300)
            return {"task_id": "task-1"}, None

        async def status(self, task_id):
            self.status_calls += 1
            if self.status_calls == 1:
                return {"status": "GENERATING_STREAMING_READY"}
            return {"status": "SUCCESS"}

        async def result(self, task_id):
            return {"audio_url": "https://treblo.example/final-song.ogg"}

        async def download_audio(self, url):
            return b"treblo-audio", "audio/ogg"

    class FakeChannel:
        def __init__(self):
            self.messages = []

        async def send(self, content=None, **kwargs):
            self.messages.append((content or "", kwargs))

    class FakeBot:
        def __init__(self):
            self.channel = FakeChannel()

        def get_channel(self, channel_id):
            return self.channel

    queue = TrebloSongQueue(TrebloSongConfig(api_key="test", poll_interval_seconds=0, download_dir=tmp_path))
    queue.client = FakeClient()
    bot = FakeBot()
    job = TrebloSongJob(
        job_id="song-1",
        user_id=123,
        channel_id=456,
        prompt="test song",
        duration_preset="300",
        length_range=(270, 300),
    )

    await queue._run_job(bot, job)

    messages = [message for message, _kwargs in bot.channel.messages]
    final_kwargs = bot.channel.messages[-1][1]

    assert any("stream ready. waiting for final file." in message for message in messages)
    assert "https://" not in "\n".join(messages)
    assert "archived `treblo-song-1.ogg`" in messages[-1]
    assert final_kwargs["file"].filename == "treblo-song-1.ogg"
    assert (tmp_path / "treblo-song-1.ogg").read_bytes() == b"treblo-audio"
    assert (tmp_path / "treblo-song-1.ogg.json").exists()
    assert job.status == "ready"


@pytest.mark.asyncio
async def test_treblo_song_completion_archives_too_large_audio_without_public_link(tmp_path):
    class FakeClient:
        async def result(self, task_id):
            return {"audio_url": "https://treblo.example/final-song.ogg"}

        async def download_audio(self, url):
            return b"treblo-audio-too-big", "audio/ogg"

    class FakeChannel:
        def __init__(self):
            self.messages = []

        async def send(self, content=None, **kwargs):
            self.messages.append((content or "", kwargs))

    class FakeBot:
        def __init__(self):
            self.channel = FakeChannel()

        def get_channel(self, channel_id):
            return self.channel

    queue = TrebloSongQueue(
        TrebloSongConfig(
            api_key="test",
            max_attachment_bytes=4,
            download_dir=tmp_path,
        )
    )
    queue.client = FakeClient()
    bot = FakeBot()
    job = TrebloSongJob(job_id="song-2", user_id=123, channel_id=456, prompt="test song", task_id="task-2")

    await queue._finish_success(bot, job)

    messages = [message for message, _kwargs in bot.channel.messages]
    assert (tmp_path / "treblo-song-2.ogg").read_bytes() == b"treblo-audio-too-big"
    assert "too large for Discord upload" in messages[-1]
    assert "https://" not in "\n".join(messages)
    assert "file" not in bot.channel.messages[-1][1]
