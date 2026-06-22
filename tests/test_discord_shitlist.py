import json

import pytest

from aibrain.discord_shitlist import DiscordShitlistStore, format_shitlist_reply


def test_shitlist_store_persists_entries(tmp_path):
    path = tmp_path / "shitlist.json"
    store = DiscordShitlistStore(path, owner_user_ids={1})

    entry = store.add(2, reason="kept looping", spice_level=99)
    reloaded = DiscordShitlistStore(path, owner_user_ids={1}).get(2)

    assert entry.spice_level == 10
    assert reloaded is not None
    assert reloaded.reason == "kept looping"
    assert format_shitlist_reply(store.add(3, reason="minor bit", spice_level=1)) == "gfy"


def test_shitlist_store_hard_blocks_owner(tmp_path):
    store = DiscordShitlistStore(tmp_path / "shitlist.json", owner_user_ids={120418341775998976})

    with pytest.raises(ValueError, match="bot owner cannot be added"):
        store.add(120418341775998976, reason="impossible", spice_level=10)


def test_shitlist_store_filters_owner_from_existing_file(tmp_path):
    path = tmp_path / "shitlist.json"
    path.write_text(
        json.dumps(
            {
                "schema": "aibrain.discord_shitlist.v1",
                "entries": {
                    "1": {"user_id": 1, "reason": "owner", "spice_level": 10, "added_at": "now"},
                    "2": {"user_id": 2, "reason": "listed", "spice_level": 2, "added_at": "now"},
                },
            }
        ),
        encoding="utf-8",
    )
    store = DiscordShitlistStore(path, owner_user_ids={1})

    assert store.get(1) is None
    assert store.get(2) is not None
