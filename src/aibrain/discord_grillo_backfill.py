from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


LEGACY_CHANNEL_SCOPE_RE = re.compile(r"^discord:guild:(?P<guild_id>[^:]+):channel:(?P<channel_id>[^:]+)$")


@dataclass(slots=True)
class DiscordGrilloBackfillReport:
    db_path: str
    dry_run: bool
    persona_id: str
    backup_path: str | None = None
    pairs_seen: int = 0
    pairs_selected: int = 0
    turns: int = 0
    candidates: int = 0
    diary_entries: int = 0
    slots: int = 0
    profiles: int = 0
    skipped_pairs: list[str] = field(default_factory=list)
    target_scopes: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def backfill_discord_grillo_memory(
    db_path: str | Path,
    *,
    persona_id: str = "neuro-sama",
    participant_keys: Iterable[str] | None = None,
    guild_ids: Iterable[str] | None = None,
    dry_run: bool = True,
    backup: bool = True,
) -> DiscordGrilloBackfillReport:
    path = Path(db_path)
    selected_participants = {str(item) for item in (participant_keys or []) if str(item).strip()}
    selected_guilds = {str(item) for item in (guild_ids or []) if str(item).strip()}
    report = DiscordGrilloBackfillReport(db_path=str(path), dry_run=dry_run, persona_id=persona_id)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        pairs = _legacy_scope_participants(conn)
        report.pairs_seen = len(pairs)
        selected: list[tuple[str, str, str, str, str]] = []
        for scope_key, participant_key in pairs:
            parsed = _parse_legacy_scope(scope_key)
            if parsed is None:
                report.skipped_pairs.append(f"{scope_key}:{participant_key}:unrecognized_scope")
                continue
            guild_id, channel_id = parsed
            if selected_participants and participant_key not in selected_participants:
                continue
            if selected_guilds and guild_id not in selected_guilds:
                continue
            target_scope = _target_scope(guild_id, participant_key, persona_id)
            selected.append((scope_key, participant_key, guild_id, channel_id, target_scope))
        report.pairs_selected = len(selected)
        report.target_scopes = sorted({item[4] for item in selected})
        if selected and not dry_run and backup:
            report.backup_path = _backup_sqlite_db(path)
        with conn:
            for legacy_scope, participant_key, _guild_id, channel_id, target_scope in selected:
                id_map = _backfill_turns(
                    conn,
                    legacy_scope=legacy_scope,
                    target_scope=target_scope,
                    participant_key=participant_key,
                    channel_id=channel_id,
                    dry_run=dry_run,
                )
                report.turns += len(id_map)
                candidate_id_map = _backfill_candidates(
                    conn,
                    legacy_scope=legacy_scope,
                    target_scope=target_scope,
                    participant_key=participant_key,
                    channel_id=channel_id,
                    turn_id_map=id_map,
                    dry_run=dry_run,
                )
                report.candidates += len(candidate_id_map)
                report.diary_entries += _backfill_diary(
                    conn,
                    legacy_scope=legacy_scope,
                    target_scope=target_scope,
                    participant_key=participant_key,
                    channel_id=channel_id,
                    turn_id_map=id_map,
                    dry_run=dry_run,
                )
                report.slots += _backfill_slots(
                    conn,
                    legacy_scope=legacy_scope,
                    target_scope=target_scope,
                    participant_key=participant_key,
                    candidate_id_map=candidate_id_map,
                    dry_run=dry_run,
                )
                report.profiles += _backfill_profile(
                    conn,
                    legacy_scope=legacy_scope,
                    target_scope=target_scope,
                    participant_key=participant_key,
                    persona_id=persona_id,
                    dry_run=dry_run,
                )
    finally:
        conn.close()
    return report


def _legacy_scope_participants(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for table in ("grillo_turns", "grillo_candidates", "grillo_diary_entries", "grillo_slots"):
        if not _table_exists(conn, table):
            continue
        rows = conn.execute(
            f"""
            SELECT DISTINCT scope_key, participant_key
            FROM {table}
            WHERE scope_key LIKE 'discord:guild:%:channel:%'
            """
        ).fetchall()
        pairs.update((str(row["scope_key"]), str(row["participant_key"])) for row in rows)
    if _table_exists(conn, "grillo_relationship_profiles"):
        rows = conn.execute(
            """
            SELECT scope_key, participant_keys_json
            FROM grillo_relationship_profiles
            WHERE scope_key LIKE 'discord:guild:%:channel:%'
            """
        ).fetchall()
        for row in rows:
            for participant_key in _json_list(row["participant_keys_json"]):
                pairs.add((str(row["scope_key"]), participant_key))
    return sorted(pairs)


def _backfill_turns(
    conn: sqlite3.Connection,
    *,
    legacy_scope: str,
    target_scope: str,
    participant_key: str,
    channel_id: str,
    dry_run: bool,
) -> dict[str, str]:
    id_map: dict[str, str] = {}
    rows = conn.execute(
        """
        SELECT * FROM grillo_turns
        WHERE scope_key = ? AND participant_key = ?
        ORDER BY created_at, turn_id
        """,
        (legacy_scope, participant_key),
    ).fetchall()
    for row in rows:
        old_id = str(row["turn_id"])
        new_id = _stable_id("turn", target_scope, old_id)
        if _exists(conn, "grillo_turns", "turn_id", new_id):
            continue
        id_map[old_id] = new_id
        if dry_run:
            continue
        metadata = _json_obj(row["metadata_json"])
        metadata.update(
            {
                "legacy_scope_key": legacy_scope,
                "legacy_turn_id": old_id,
                "legacy_channel_id": channel_id,
                "backfill": "discord_grillo_server_user_scope",
            }
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO grillo_turns (
                turn_id, scope_key, participant_key, role, author_name, channel_id,
                interface_path, source, content, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_id,
                target_scope,
                participant_key,
                row["role"],
                row["author_name"],
                row["channel_id"] or channel_id,
                row["interface_path"],
                row["source"],
                row["content"],
                json.dumps(metadata),
                row["created_at"],
            ),
        )
    return id_map


def _backfill_candidates(
    conn: sqlite3.Connection,
    *,
    legacy_scope: str,
    target_scope: str,
    participant_key: str,
    channel_id: str,
    turn_id_map: dict[str, str],
    dry_run: bool,
) -> dict[str, str]:
    id_map: dict[str, str] = {}
    rows = conn.execute(
        """
        SELECT * FROM grillo_candidates
        WHERE scope_key = ? AND participant_key = ?
        ORDER BY created_at, candidate_id
        """,
        (legacy_scope, participant_key),
    ).fetchall()
    for row in rows:
        old_id = str(row["candidate_id"])
        new_id = _stable_id("candidate", target_scope, old_id)
        if _exists(conn, "grillo_candidates", "candidate_id", new_id):
            continue
        id_map[old_id] = new_id
        if dry_run:
            continue
        tags = _merge_lists(_json_list(row["tags_json"]), [f"legacy_channel:{channel_id}", "backfilled"])
        source_turn_ids = [turn_id_map.get(item, _stable_id("turn", target_scope, item)) for item in _json_list(row["source_turn_ids_json"])]
        conn.execute(
            """
            INSERT OR IGNORE INTO grillo_candidates (
                candidate_id, scope_key, participant_key, type, content, summary,
                confidence, tags_json, source_turn_ids_json, promoted, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_id,
                target_scope,
                participant_key,
                row["type"],
                row["content"],
                row["summary"],
                row["confidence"],
                json.dumps(tags),
                json.dumps(source_turn_ids),
                row["promoted"],
                row["created_at"],
            ),
        )
    return id_map


def _backfill_diary(
    conn: sqlite3.Connection,
    *,
    legacy_scope: str,
    target_scope: str,
    participant_key: str,
    channel_id: str,
    turn_id_map: dict[str, str],
    dry_run: bool,
) -> int:
    inserted = 0
    rows = conn.execute(
        """
        SELECT * FROM grillo_diary_entries
        WHERE scope_key = ? AND participant_key = ?
        ORDER BY created_at, diary_id
        """,
        (legacy_scope, participant_key),
    ).fetchall()
    for row in rows:
        old_id = str(row["diary_id"])
        new_id = _stable_id("diary", target_scope, old_id)
        if _exists(conn, "grillo_diary_entries", "diary_id", new_id):
            continue
        inserted += 1
        if dry_run:
            continue
        tags = _merge_lists(_json_list(row["tags_json"]), [f"legacy_channel:{channel_id}", "backfilled"])
        source_turn_ids = [turn_id_map.get(item, _stable_id("turn", target_scope, item)) for item in _json_list(row["source_turn_ids_json"])]
        conn.execute(
            """
            INSERT OR IGNORE INTO grillo_diary_entries (
                diary_id, scope_key, participant_key, beat_type, summary,
                personal_thought, tags_json, source_turn_ids_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_id,
                target_scope,
                participant_key,
                row["beat_type"],
                row["summary"],
                row["personal_thought"],
                json.dumps(tags),
                json.dumps(source_turn_ids),
                row["created_at"],
            ),
        )
    return inserted


def _backfill_slots(
    conn: sqlite3.Connection,
    *,
    legacy_scope: str,
    target_scope: str,
    participant_key: str,
    candidate_id_map: dict[str, str],
    dry_run: bool,
) -> int:
    changed = 0
    rows = conn.execute(
        """
        SELECT * FROM grillo_slots
        WHERE scope_key = ? AND participant_key = ?
        ORDER BY slot_name
        """,
        (legacy_scope, participant_key),
    ).fetchall()
    for row in rows:
        existing = conn.execute(
            """
            SELECT * FROM grillo_slots
            WHERE scope_key = ? AND participant_key = ? AND slot_name = ?
            """,
            (target_scope, participant_key, row["slot_name"]),
        ).fetchone()
        items = _merge_lists(
            _json_list(existing["items_json"]) if existing else [],
            _json_list(row["items_json"]),
        )
        source_candidate_ids = _merge_lists(
            _json_list(existing["source_candidate_ids_json"]) if existing else [],
            [candidate_id_map.get(item, _stable_id("candidate", target_scope, item)) for item in _json_list(row["source_candidate_ids_json"])],
        )
        if existing and items == _json_list(existing["items_json"]) and source_candidate_ids == _json_list(existing["source_candidate_ids_json"]):
            continue
        changed += 1
        if dry_run:
            continue
        conn.execute(
            """
            INSERT INTO grillo_slots (
                slot_id, scope_key, participant_key, slot_name,
                items_json, source_candidate_ids_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(scope_key, participant_key, slot_name) DO UPDATE SET
                items_json = excluded.items_json,
                source_candidate_ids_json = excluded.source_candidate_ids_json,
                updated_at = excluded.updated_at
            """,
            (
                _stable_id("slot", target_scope, participant_key, row["slot_name"]),
                target_scope,
                participant_key,
                row["slot_name"],
                json.dumps(items),
                json.dumps(source_candidate_ids),
                _max_iso(existing["updated_at"] if existing else None, row["updated_at"]) or _now_iso(),
            ),
        )
    return changed


def _backfill_profile(
    conn: sqlite3.Connection,
    *,
    legacy_scope: str,
    target_scope: str,
    participant_key: str,
    persona_id: str,
    dry_run: bool,
) -> int:
    old_rows = [
        row
        for row in conn.execute(
            """
            SELECT * FROM grillo_relationship_profiles
            WHERE scope_key = ?
            ORDER BY updated_at
            """,
            (legacy_scope,),
        ).fetchall()
        if participant_key in _json_list(row["participant_keys_json"])
    ]
    if not old_rows:
        return 0
    existing = conn.execute("SELECT * FROM grillo_relationship_profiles WHERE scope_key = ?", (target_scope,)).fetchone()
    latest = old_rows[-1]
    merged = _profile_dict(existing or latest)
    merged["profile_id"] = merged["profile_id"] if existing else _stable_id("profile", target_scope)
    merged["scope_key"] = target_scope
    merged["persona_id"] = persona_id or merged["persona_id"]
    merged["participant_keys_json"] = json.dumps(_merge_lists(_json_list(merged["participant_keys_json"]), [participant_key]))
    merged["facts_json"] = json.dumps(_merge_profile_lists(existing, old_rows, "facts_json"))
    merged["diary_history_json"] = json.dumps(
        _merge_lists(
            _merge_profile_lists(existing, old_rows, "diary_history_json"),
            [str(row["diary_entry"]) for row in old_rows if str(row["diary_entry"]).strip()],
        )
    )
    merged["tone_preferences_json"] = json.dumps(_merge_profile_lists(existing, old_rows, "tone_preferences_json"))
    merged["interaction_style_json"] = json.dumps(_merge_profile_lists(existing, old_rows, "interaction_style_json"))
    merged["boundaries_json"] = json.dumps(_merge_profile_lists(existing, old_rows, "boundaries_json"))
    merged["active_threads_json"] = json.dumps(_merge_profile_lists(existing, old_rows, "active_threads_json"))
    merged["turn_count"] = max(int(merged["turn_count"] or 0), sum(int(row["turn_count"] or 0) for row in old_rows))
    merged["last_seen_at"] = _max_iso(merged["last_seen_at"], *[row["last_seen_at"] for row in old_rows])
    merged["last_diary_turn_count"] = max(
        int(merged["last_diary_turn_count"] or 0),
        max(int(row["last_diary_turn_count"] or 0) for row in old_rows),
    )
    merged["summary"] = merged["summary"] or latest["summary"]
    merged["diary_entry"] = merged["diary_entry"] or latest["diary_entry"]
    merged["updated_at"] = _max_iso(merged["updated_at"], *[row["updated_at"] for row in old_rows]) or _now_iso()
    if existing and _profile_dict(existing) == merged:
        return 0
    if dry_run:
        return 1
    conn.execute(
        """
        INSERT INTO grillo_relationship_profiles (
            profile_id, scope_key, persona_id, participant_keys_json,
            relationship_stage, mood, trust, attraction, respect, irritation,
            jealousy, guard, turn_count, last_seen_at, last_diary_turn_count,
            last_action_tag, facts_json, summary, diary_entry, diary_history_json,
            affect_state_json, tone_preferences_json, interaction_style_json,
            boundaries_json, active_threads_json, updated_at
        ) VALUES (
            :profile_id, :scope_key, :persona_id, :participant_keys_json,
            :relationship_stage, :mood, :trust, :attraction, :respect, :irritation,
            :jealousy, :guard, :turn_count, :last_seen_at, :last_diary_turn_count,
            :last_action_tag, :facts_json, :summary, :diary_entry, :diary_history_json,
            :affect_state_json, :tone_preferences_json, :interaction_style_json,
            :boundaries_json, :active_threads_json, :updated_at
        )
        ON CONFLICT(scope_key) DO UPDATE SET
            participant_keys_json = excluded.participant_keys_json,
            facts_json = excluded.facts_json,
            summary = excluded.summary,
            diary_entry = excluded.diary_entry,
            diary_history_json = excluded.diary_history_json,
            tone_preferences_json = excluded.tone_preferences_json,
            interaction_style_json = excluded.interaction_style_json,
            boundaries_json = excluded.boundaries_json,
            active_threads_json = excluded.active_threads_json,
            turn_count = excluded.turn_count,
            last_seen_at = excluded.last_seen_at,
            last_diary_turn_count = excluded.last_diary_turn_count,
            updated_at = excluded.updated_at
        """,
        merged,
    )
    return 1


def _parse_legacy_scope(scope_key: str) -> tuple[str, str] | None:
    match = LEGACY_CHANNEL_SCOPE_RE.match(scope_key)
    if not match:
        return None
    return match.group("guild_id"), match.group("channel_id")


def _target_scope(guild_id: str, participant_key: str, persona_id: str) -> str:
    return f"discord:guild:{guild_id}:user:{participant_key}:persona:{persona_id}"


def _stable_id(kind: str, *parts: Any) -> str:
    digest = hashlib.sha256("\x1f".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:32]
    return f"backfill:{kind}:{digest}"


def _exists(conn: sqlite3.Connection, table: str, column: str, value: str) -> bool:
    return conn.execute(f"SELECT 1 FROM {table} WHERE {column} = ? LIMIT 1", (value,)).fetchone() is not None


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)).fetchone() is not None


def _json_list(value: Any) -> list[str]:
    try:
        parsed = json.loads(value or "[]")
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if str(item).strip()]


def _json_obj(value: Any) -> dict[str, Any]:
    try:
        parsed = json.loads(value or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _merge_lists(*values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for items in values:
        for item in items:
            text = str(item).strip()
            if text and text not in seen:
                seen.add(text)
                merged.append(text)
    return merged


def _merge_profile_lists(existing: sqlite3.Row | None, old_rows: list[sqlite3.Row], field: str) -> list[str]:
    values: list[Iterable[str]] = []
    if existing is not None:
        values.append(_json_list(existing[field]))
    values.extend(_json_list(row[field]) for row in old_rows)
    return _merge_lists(*values)


def _profile_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def _max_iso(*values: Any) -> str | None:
    strings = [str(value) for value in values if str(value or "").strip()]
    return max(strings) if strings else None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _backup_sqlite_db(path: Path) -> str:
    backup_path = path.with_name(f"{path.name}.bak-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    source = sqlite3.connect(path)
    try:
        dest = sqlite3.connect(backup_path)
        try:
            source.backup(dest)
        finally:
            dest.close()
    finally:
        source.close()
    for suffix in ("-wal", "-shm"):
        extra = Path(f"{path}{suffix}")
        if extra.exists():
            shutil.copy2(extra, Path(f"{backup_path}{suffix}"))
    for sidecar in _derived_memory_sidecars(path):
        _copy_backup_sidecar(sidecar, backup_path.with_name(f"{backup_path.name}.{sidecar.name}"))
    return str(backup_path)


def _derived_memory_sidecars(path: Path) -> list[Path]:
    ladybug = path.with_suffix(".ladybug") if path.suffix else path / "memory.ladybug"
    turbovec = path.with_suffix(".turbovec") if path.suffix else path / "memory.turbovec"
    turbovec_metadata = path.with_suffix(".turbovec.sqlite3") if path.suffix else path / "memory.turbovec.sqlite3"
    return [
        ladybug,
        Path(f"{ladybug}.wal"),
        turbovec,
        turbovec_metadata,
        Path(f"{turbovec_metadata}-wal"),
        Path(f"{turbovec_metadata}-shm"),
    ]


def _copy_backup_sidecar(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    if source.is_dir():
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
        return
    shutil.copy2(source, destination)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill old Discord channel-scoped GRILLO rows into server-user scopes.")
    parser.add_argument("--db", default="discord_brain.sqlite3", help="Path to the Brain SQLite database.")
    parser.add_argument("--persona", default="neuro-sama", help="Persona id used in the target GRILLO scope.")
    parser.add_argument("--participant", action="append", default=[], help="Discord user id to migrate. Repeat for multiple users.")
    parser.add_argument("--guild", action="append", default=[], help="Discord guild id to migrate. Repeat for multiple guilds.")
    parser.add_argument("--apply", action="store_true", help="Write changes. Default is dry-run only.")
    parser.add_argument("--no-backup", action="store_true", help="Skip SQLite backup when applying.")
    args = parser.parse_args(argv)
    report = backfill_discord_grillo_memory(
        args.db,
        persona_id=args.persona,
        participant_keys=args.participant,
        guild_ids=args.guild,
        dry_run=not args.apply,
        backup=not args.no_backup,
    )
    print(report.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
