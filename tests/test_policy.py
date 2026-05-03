from aibrain import ThreadPolicy


def test_thread_policy_ids_are_stable():
    assert ThreadPolicy.discord_channel(1, 2) == "discord:guild:1:channel:2"
    assert ThreadPolicy.discord_dm(9) == "discord:dm:9"
    assert ThreadPolicy.discord_thread(1, 3) == "discord:guild:1:thread:3"
    assert ThreadPolicy.discord_voice(1, 4) == "discord:guild:1:voice:4"
    assert ThreadPolicy.twitch_channel("MyChan") == "twitch:channel:mychan"
    assert ThreadPolicy.vrm() == "vrm:local"
