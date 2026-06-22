import json
import sqlite3

from aibrain.discord_grillo_backfill import backfill_discord_grillo_memory
from aibrain.memory_stack import (
    GrilloCandidate,
    GrilloDiaryEntry,
    GrilloRelationshipProfile,
    GrilloSlot,
    GrilloTurn,
    SQLiteGrilloStore,
)


def test_discord_grillo_backfill_moves_channel_rows_to_server_user_scope(tmp_path):
    db_path = tmp_path / "brain.sqlite3"
    store = SQLiteGrilloStore(db_path)
    legacy_scope = "discord:guild:guild-1:channel:chan-1"
    target_scope = "discord:guild:guild-1:user:user-1:persona:neuro-sama"

    store._append_turn_sync(
        GrilloTurn(
            turn_id="turn-old-user",
            scope_key=legacy_scope,
            participant_key="user-1",
            role="user",
            content="Remember that my name is Subby.",
            author_name="Subby",
            channel_id=None,
            source="discord",
            metadata={"message_id": "m1"},
            created_at="2026-06-21T01:00:00+00:00",
        )
    )
    store._append_turn_sync(
        GrilloTurn(
            turn_id="turn-old-assistant",
            scope_key=legacy_scope,
            participant_key="user-1",
            role="assistant",
            content="Got it.",
            author_name="Neuro-sama",
            channel_id="chan-1",
            source="discord",
            metadata={},
            created_at="2026-06-21T01:00:01+00:00",
        )
    )
    store._append_candidate_sync(
        GrilloCandidate(
            candidate_id="candidate-old",
            scope_key=legacy_scope,
            participant_key="user-1",
            type="identity",
            content="The user's name is Subby.",
            summary="User prefers Subby.",
            confidence=0.92,
            tags=["identity"],
            source_turn_ids=["turn-old-user"],
            promoted=True,
            created_at="2026-06-21T01:00:02+00:00",
        )
    )
    store._append_diary_sync(
        GrilloDiaryEntry(
            diary_id="diary-old",
            scope_key=legacy_scope,
            participant_key="user-1",
            beat_type="relationship",
            summary="Subby corrected the bot name context.",
            personal_thought="I should remember that he wants to be called Subby.",
            tags=["relationship"],
            source_turn_ids=["turn-old-user", "turn-old-assistant"],
            created_at="2026-06-21T01:00:03+00:00",
        )
    )
    store._upsert_slot_sync(
        GrilloSlot(
            slot_id="slot-old",
            scope_key=legacy_scope,
            participant_key="user-1",
            slot_name="user_facts",
            items=["User is called Subby."],
            source_candidate_ids=["candidate-old"],
            updated_at="2026-06-21T01:00:04+00:00",
        )
    )
    store._upsert_relationship_profile_sync(
        GrilloRelationshipProfile(
            profile_id="profile-old",
            scope_key=legacy_scope,
            persona_id="neuro-sama",
            participant_keys=["user-1"],
            relationship_stage="familiar",
            mood="focused",
            trust=6,
            respect=7,
            turn_count=3,
            last_seen_at="2026-06-21T01:00:05+00:00",
            facts=["User prefers Subby."],
            diary_entry="I should not call him LO.",
            diary_history=["He corrected the name."],
            updated_at="2026-06-21T01:00:06+00:00",
        )
    )
    store.close()

    dry = backfill_discord_grillo_memory(db_path, participant_keys=["user-1"], dry_run=True)

    assert dry.turns == 2
    assert dry.candidates == 1
    assert dry.diary_entries == 1
    assert dry.slots == 1
    assert dry.profiles == 1
    assert dry.target_scopes == [target_scope]

    applied = backfill_discord_grillo_memory(db_path, participant_keys=["user-1"], dry_run=False, backup=False)

    assert applied.turns == 2
    assert applied.candidates == 1
    assert applied.diary_entries == 1
    assert applied.slots == 1
    assert applied.profiles == 1

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        turns = conn.execute(
            "SELECT * FROM grillo_turns WHERE scope_key = ? ORDER BY created_at",
            (target_scope,),
        ).fetchall()
        candidate = conn.execute("SELECT * FROM grillo_candidates WHERE scope_key = ?", (target_scope,)).fetchone()
        diary = conn.execute("SELECT * FROM grillo_diary_entries WHERE scope_key = ?", (target_scope,)).fetchone()
        slot = conn.execute("SELECT * FROM grillo_slots WHERE scope_key = ?", (target_scope,)).fetchone()
        profile = conn.execute("SELECT * FROM grillo_relationship_profiles WHERE scope_key = ?", (target_scope,)).fetchone()
    finally:
        conn.close()

    assert [turn["channel_id"] for turn in turns] == ["chan-1", "chan-1"]
    assert json.loads(turns[0]["metadata_json"])["legacy_scope_key"] == legacy_scope
    assert json.loads(candidate["tags_json"]) == ["identity", "legacy_channel:chan-1", "backfilled"]
    assert json.loads(candidate["source_turn_ids_json"])[0].startswith("backfill:turn:")
    assert json.loads(diary["source_turn_ids_json"])[0].startswith("backfill:turn:")
    assert json.loads(slot["items_json"]) == ["User is called Subby."]
    assert json.loads(slot["source_candidate_ids_json"])[0].startswith("backfill:candidate:")
    assert profile["relationship_stage"] == "familiar"
    assert json.loads(profile["facts_json"]) == ["User prefers Subby."]
    assert "I should not call him LO." in json.loads(profile["diary_history_json"])

    second = backfill_discord_grillo_memory(db_path, participant_keys=["user-1"], dry_run=False, backup=False)

    assert second.turns == 0
    assert second.candidates == 0
    assert second.diary_entries == 0
    assert second.slots == 0
    assert second.profiles == 0


def test_discord_grillo_backfill_backup_copies_memory_sidecars(tmp_path):
    db_path = tmp_path / "brain.sqlite3"
    store = SQLiteGrilloStore(db_path)
    legacy_scope = "discord:guild:guild-1:channel:chan-1"
    store._append_turn_sync(
        GrilloTurn(
            turn_id="turn-old-user",
            scope_key=legacy_scope,
            participant_key="user-1",
            role="user",
            content="Remember this before backup.",
            author_name="Subby",
            channel_id="chan-1",
            source="discord",
            metadata={},
            created_at="2026-06-21T01:00:00+00:00",
        )
    )
    store.close()
    (tmp_path / "brain.ladybug").write_text("ladybug", encoding="utf-8")
    (tmp_path / "brain.ladybug.wal").write_text("ladybug wal", encoding="utf-8")
    turbovec = tmp_path / "brain.turbovec"
    turbovec.mkdir()
    (turbovec / "index.bin").write_text("index", encoding="utf-8")
    (tmp_path / "brain.turbovec.sqlite3").write_text("metadata", encoding="utf-8")
    (tmp_path / "brain.turbovec.sqlite3-wal").write_text("metadata wal", encoding="utf-8")

    report = backfill_discord_grillo_memory(db_path, participant_keys=["user-1"], dry_run=False, backup=True)
    backup_path = tmp_path / report.backup_path

    assert backup_path.exists()
    assert backup_path.with_name(f"{backup_path.name}.brain.ladybug").read_text(encoding="utf-8") == "ladybug"
    assert backup_path.with_name(f"{backup_path.name}.brain.ladybug.wal").read_text(encoding="utf-8") == "ladybug wal"
    assert (backup_path.with_name(f"{backup_path.name}.brain.turbovec") / "index.bin").read_text(encoding="utf-8") == "index"
    assert backup_path.with_name(f"{backup_path.name}.brain.turbovec.sqlite3").read_text(encoding="utf-8") == "metadata"
    assert backup_path.with_name(f"{backup_path.name}.brain.turbovec.sqlite3-wal").read_text(encoding="utf-8") == "metadata wal"
