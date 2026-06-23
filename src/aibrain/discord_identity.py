import json
import re
import sqlite3
import threading
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MENTION_RE = re.compile(r"<@!?(\d+)>")
GUILD_USER_SCOPE_RE = re.compile(r"^discord:guild:(?P<guild_id>[^:]+):user:(?P<user_id>[^:]+):persona:")


@dataclass(slots=True)
class DiscordIdentityProfile:
    guild_id: str
    user_id: str
    username: str | None
    display_name: str | None
    global_name: str | None
    mention: str | None
    is_bot: bool
    first_seen_at: str
    last_seen_at: str
    message_count: int
    aliases: list[str]


@dataclass(slots=True)
class DiscordIdentityHit:
    profile: DiscordIdentityProfile
    alias: str
    kind: str
    count: int
    last_seen_at: str
    score: int


class DiscordIdentityStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.RLock()
        self._init()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA temp_store=MEMORY")
            self._conn = conn
        return self._conn

    def _init(self) -> None:
        with self._lock:
            self._connect().executescript(
                """
                CREATE TABLE IF NOT EXISTS discord_user_identities (
                    guild_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    username TEXT,
                    display_name TEXT,
                    global_name TEXT,
                    mention TEXT,
                    is_bot INTEGER NOT NULL DEFAULT 0,
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    message_count INTEGER NOT NULL DEFAULT 0,
                    aliases_json TEXT NOT NULL DEFAULT '[]',
                    PRIMARY KEY (guild_id, user_id)
                );
                CREATE TABLE IF NOT EXISTS discord_user_aliases (
                    guild_id TEXT NOT NULL,
                    alias_norm TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    alias TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    count INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY (guild_id, alias_norm, user_id)
                );
                CREATE INDEX IF NOT EXISTS idx_discord_user_aliases_guild_norm
                    ON discord_user_aliases(guild_id, alias_norm);
                CREATE INDEX IF NOT EXISTS idx_discord_user_aliases_user
                    ON discord_user_aliases(guild_id, user_id);
                """
            )

    def record_observation(
        self,
        *,
        guild_id: int | str | None,
        user_id: int | str | None,
        username: str | None = None,
        display_name: str | None = None,
        global_name: str | None = None,
        mention: str | None = None,
        is_bot: bool = False,
        seen_at: str | None = None,
        increment_message_count: bool = True,
    ) -> DiscordIdentityProfile | None:
        if guild_id is None or user_id is None:
            return None
        guild_key = str(guild_id)
        user_key = str(user_id)
        seen = seen_at or datetime.now(timezone.utc).isoformat()
        alias_pairs = _dedupe_alias_pairs(
            _identity_aliases(
                user_id=user_key,
                username=username,
                display_name=display_name,
                global_name=global_name,
                mention=mention,
            )
        )
        aliases = [alias for alias, _kind in alias_pairs]
        with self._lock:
            conn = self._connect()
            row = conn.execute(
                "SELECT * FROM discord_user_identities WHERE guild_id = ? AND user_id = ?",
                (guild_key, user_key),
            ).fetchone()
            old_aliases = _json_list(row["aliases_json"]) if row is not None else []
            merged_aliases = _dedupe([*aliases, *old_aliases])[:64]
            first_seen = min(_row_text(row, "first_seen_at") or seen, seen)
            existing_last_seen = _row_text(row, "last_seen_at")
            last_seen = max(existing_last_seen or seen, seen)
            replace_current_fields = row is None or seen >= (existing_last_seen or seen)
            current_username = (
                username
                if replace_current_fields and username is not None
                else _row_text(row, "username")
            )
            current_display_name = (
                display_name
                if replace_current_fields and display_name is not None
                else _row_text(row, "display_name")
            )
            current_global_name = (
                global_name
                if replace_current_fields and global_name is not None
                else _row_text(row, "global_name")
            )
            current_mention = (
                mention
                if replace_current_fields and mention is not None
                else _row_text(row, "mention")
            )
            current_is_bot = bool(is_bot) if replace_current_fields or row is None else bool(row["is_bot"])
            old_count = int(row["message_count"] or 0) if row is not None else 0
            count = old_count + (1 if increment_message_count else 0)
            conn.execute(
                """
                INSERT INTO discord_user_identities (
                    guild_id, user_id, username, display_name, global_name, mention,
                    is_bot, first_seen_at, last_seen_at, message_count, aliases_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(guild_id, user_id) DO UPDATE SET
                    username = excluded.username,
                    display_name = excluded.display_name,
                    global_name = excluded.global_name,
                    mention = excluded.mention,
                    is_bot = excluded.is_bot,
                    first_seen_at = excluded.first_seen_at,
                    last_seen_at = excluded.last_seen_at,
                    message_count = excluded.message_count,
                    aliases_json = excluded.aliases_json
                """,
                (
                    guild_key,
                    user_key,
                    current_username,
                    current_display_name,
                    current_global_name,
                    current_mention,
                    1 if current_is_bot else 0,
                    first_seen,
                    last_seen,
                    count,
                    json.dumps(merged_aliases),
                ),
            )
            for alias, kind in alias_pairs:
                norm = normalize_identity_alias(alias)
                if not norm:
                    continue
                alias_row = conn.execute(
                    """
                    SELECT count, first_seen_at, last_seen_at
                    FROM discord_user_aliases
                    WHERE guild_id = ? AND alias_norm = ? AND user_id = ?
                    """,
                    (guild_key, norm, user_key),
                ).fetchone()
                alias_count = int(alias_row["count"] or 0) if alias_row is not None else 0
                alias_first_seen = min(_row_text(alias_row, "first_seen_at") or seen, seen)
                alias_last_seen = max(_row_text(alias_row, "last_seen_at") or seen, seen)
                conn.execute(
                    """
                    INSERT INTO discord_user_aliases (
                        guild_id, alias_norm, user_id, alias, kind, first_seen_at, last_seen_at, count
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(guild_id, alias_norm, user_id) DO UPDATE SET
                        alias = excluded.alias,
                        kind = excluded.kind,
                        first_seen_at = excluded.first_seen_at,
                        last_seen_at = excluded.last_seen_at,
                        count = excluded.count
                    """,
                    (
                        guild_key,
                        norm,
                        user_key,
                        alias,
                        kind,
                        alias_first_seen,
                        alias_last_seen,
                        alias_count + (1 if increment_message_count else 0),
                    ),
                )
            return self.get_profile(guild_key, user_key)

    def get_profile(self, guild_id: int | str, user_id: int | str) -> DiscordIdentityProfile | None:
        with self._lock:
            row = self._connect().execute(
                "SELECT * FROM discord_user_identities WHERE guild_id = ? AND user_id = ?",
                (str(guild_id), str(user_id)),
            ).fetchone()
        return _profile_from_row(row) if row is not None else None

    def search(self, guild_id: int | str | None, query: str, *, limit: int = 5) -> list[DiscordIdentityHit]:
        if guild_id is None or not query.strip():
            return []
        guild_key = str(guild_id)
        normalized_query = normalize_identity_alias(query)
        mention_ids = MENTION_RE.findall(query)
        with self._lock:
            conn = self._connect()
            scored: dict[str, DiscordIdentityHit] = {}
            for user_id in mention_ids:
                profile = self.get_profile(guild_key, user_id)
                if profile is None:
                    continue
                scored[profile.user_id] = DiscordIdentityHit(
                    profile=profile,
                    alias=profile.mention or f"<@{profile.user_id}>",
                    kind="mention",
                    count=1,
                    last_seen_at=profile.last_seen_at,
                    score=1000,
                )
            rows = conn.execute(
                """
                SELECT a.*, i.username, i.display_name, i.global_name, i.mention, i.is_bot,
                       i.first_seen_at AS profile_first_seen_at,
                       i.last_seen_at AS profile_last_seen_at,
                       i.message_count, i.aliases_json
                FROM discord_user_aliases AS a
                JOIN discord_user_identities AS i
                    ON i.guild_id = a.guild_id AND i.user_id = a.user_id
                WHERE a.guild_id = ?
                ORDER BY a.last_seen_at DESC
                LIMIT 5000
                """,
                (guild_key,),
            ).fetchall()
        for row in rows:
            score = _alias_match_score(normalized_query, str(row["alias_norm"] or ""))
            if score <= 0:
                continue
            profile = DiscordIdentityProfile(
                guild_id=guild_key,
                user_id=str(row["user_id"]),
                username=row["username"],
                display_name=row["display_name"],
                global_name=row["global_name"],
                mention=row["mention"],
                is_bot=bool(row["is_bot"]),
                first_seen_at=row["profile_first_seen_at"],
                last_seen_at=row["profile_last_seen_at"],
                message_count=int(row["message_count"] or 0),
                aliases=_json_list(row["aliases_json"]),
            )
            hit = DiscordIdentityHit(
                profile=profile,
                alias=row["alias"],
                kind=row["kind"],
                count=int(row["count"] or 0),
                last_seen_at=row["last_seen_at"],
                score=score,
            )
            previous = scored.get(profile.user_id)
            if previous is None or (hit.score, hit.last_seen_at) > (previous.score, previous.last_seen_at):
                scored[profile.user_id] = hit
        return sorted(
            scored.values(),
            key=lambda hit: (hit.score, hit.last_seen_at, hit.count),
            reverse=True,
        )[: max(1, limit)]

    def backfill_from_grillo(self, grillo_path: str | Path, *, limit: int = 5000) -> int:
        source_path = Path(grillo_path)
        if not source_path.exists():
            return 0
        count = 0
        try:
            conn = sqlite3.connect(source_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT scope_key, participant_key, author_name, metadata_json, created_at
                FROM grillo_turns
                WHERE role = 'user'
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
        except sqlite3.Error:
            return 0
        finally:
            with suppress(UnboundLocalError):
                conn.close()
        for row in rows:
            metadata = _json_dict(row["metadata_json"])
            parsed = GUILD_USER_SCOPE_RE.match(str(row["scope_key"] or ""))
            guild_id = metadata.get("guild_id") or (parsed.group("guild_id") if parsed else None)
            user_id = metadata.get("author_id") or row["participant_key"] or (parsed.group("user_id") if parsed else None)
            if guild_id is None or user_id is None:
                continue
            profile = self.record_observation(
                guild_id=guild_id,
                user_id=user_id,
                username=_optional_str(metadata.get("author_username")),
                display_name=_optional_str(metadata.get("author_display_name") or row["author_name"]),
                global_name=_optional_str(metadata.get("author_global_name")),
                mention=_optional_str(metadata.get("author_mention")),
                is_bot=bool(metadata.get("author_is_bot", False)),
                seen_at=_optional_str(row["created_at"]),
                increment_message_count=False,
            )
            if profile is not None:
                count += 1
        return count


def format_identity_context(
    *,
    current_profile: DiscordIdentityProfile | None,
    query_hits: list[DiscordIdentityHit],
) -> list[str]:
    lines: list[str] = []
    if current_profile is not None:
        lines.append("Discord server identity memory:")
        lines.append(
            "- current speaker identity: "
            f"user_id={current_profile.user_id}, username={current_profile.username or 'unknown'}, "
            f"display_name={current_profile.display_name or 'unknown'}, "
            f"aliases_seen={_format_aliases(current_profile.aliases)}"
        )
    if query_hits:
        if not lines:
            lines.append("Discord server identity memory:")
        lines.append("- query alias matches:")
        for hit in query_hits[:5]:
            aliases = _format_aliases(hit.profile.aliases)
            lines.append(
                f"  - matched_alias={hit.alias!r}, user_id={hit.profile.user_id}, "
                f"username={hit.profile.username or 'unknown'}, "
                f"display_name={hit.profile.display_name or 'unknown'}, aliases_seen={aliases}, "
                f"last_seen_at={hit.profile.last_seen_at}"
            )
    if lines:
        lines.append(
            "Use this only for server-visible identity/name changes. "
            "It is not private relationship memory."
        )
    return lines


def normalize_identity_alias(value: str | None) -> str:
    if not value:
        return ""
    cleaned = MENTION_RE.sub(r"\1", str(value).casefold())
    cleaned = re.sub(r"[_\-]+", " ", cleaned)
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _identity_aliases(
    *,
    user_id: str,
    username: str | None,
    display_name: str | None,
    global_name: str | None,
    mention: str | None,
) -> list[tuple[str, str]]:
    aliases = [
        (user_id, "user_id"),
        (f"<@{user_id}>", "mention"),
    ]
    if mention:
        aliases.append((mention, "mention"))
    if username:
        aliases.append((username, "username"))
    if display_name:
        aliases.append((display_name, "display_name"))
    if global_name:
        aliases.append((global_name, "global_name"))
    return [(alias.strip(), kind) for alias, kind in aliases if alias and alias.strip()]


def _alias_match_score(query_norm: str, alias_norm: str) -> int:
    if not query_norm or not alias_norm:
        return 0
    padded_query = f" {query_norm} "
    padded_alias = f" {alias_norm} "
    if query_norm == alias_norm:
        return 900
    if padded_alias in padded_query:
        return 700
    if len(alias_norm) >= 3 and alias_norm in query_norm:
        return 500
    if len(query_norm) >= 3 and query_norm in alias_norm:
        return 300
    return 0


def _profile_from_row(row: sqlite3.Row) -> DiscordIdentityProfile:
    return DiscordIdentityProfile(
        guild_id=str(row["guild_id"]),
        user_id=str(row["user_id"]),
        username=row["username"],
        display_name=row["display_name"],
        global_name=row["global_name"],
        mention=row["mention"],
        is_bot=bool(row["is_bot"]),
        first_seen_at=row["first_seen_at"],
        last_seen_at=row["last_seen_at"],
        message_count=int(row["message_count"] or 0),
        aliases=_json_list(row["aliases_json"]),
    )


def _dedupe(items: list[Any]) -> list[Any]:
    seen: set[str] = set()
    result: list[Any] = []
    for item in items:
        key = json.dumps(item, sort_keys=True) if isinstance(item, tuple) else str(item)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _dedupe_alias_pairs(items: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[str] = set()
    result: list[tuple[str, str]] = []
    for alias, kind in items:
        norm = normalize_identity_alias(alias)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        result.append((alias, kind))
    return result


def _json_list(value: Any) -> list[str]:
    try:
        data = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [str(item) for item in data if str(item).strip()]


def _json_dict(value: Any) -> dict[str, Any]:
    try:
        data = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _row_text(row: sqlite3.Row | None, key: str) -> str | None:
    if row is None:
        return None
    value = row[key]
    return str(value) if value else None


def _optional_str(value: Any) -> str | None:
    return str(value) if value is not None and str(value).strip() else None


def _format_aliases(aliases: list[str]) -> str:
    shown = aliases[:8]
    suffix = f", +{len(aliases) - len(shown)} more" if len(aliases) > len(shown) else ""
    return "[" + ", ".join(repr(alias) for alias in shown) + suffix + "]"
