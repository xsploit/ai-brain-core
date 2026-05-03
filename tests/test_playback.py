import base64

import pytest

from aibrain.playback import AudioPlaybackWorker, BaseAudioPlayer
from aibrain.types import BrainEvent


class FakeAudioPlayer(BaseAudioPlayer):
    name = "fake"

    def __init__(self):
        self.played = []

    async def play(self, frame):
        self.played.append(frame)


@pytest.mark.asyncio
async def test_audio_playback_worker_plays_tts_audio_events():
    player = FakeAudioPlayer()
    worker = AudioPlaybackWorker(player)
    await worker.start()

    await worker.enqueue_event(
        BrainEvent(
            "tts.audio",
            {
                "audio": base64.b64encode(b"one").decode("ascii"),
                "sample_rate": 22050,
                "encoding": "pcm_s16le",
                "segment_index": 0,
            },
        )
    )
    await worker.enqueue_event(
        BrainEvent(
            "tts.audio",
            {
                "audio": base64.b64encode(b"two").decode("ascii"),
                "sample_rate": 22050,
                "encoding": "pcm_s16le",
                "segment_index": 0,
            },
        )
    )
    await worker.drain()
    await worker.close()

    assert b"".join(frame.audio for frame in player.played) == b"onetwo"
    assert worker.frames == 2
    assert worker.bytes == 6
