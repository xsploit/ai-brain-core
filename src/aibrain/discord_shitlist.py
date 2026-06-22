from __future__ import annotations

import json
import re
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


SHITLIST_SCHEMA = "aibrain.discord_shitlist.v1"


@dataclass(slots=True)
class DiscordShitlistEntry:
    user_id: int
    reason: str
    spice_level: int
    added_at: str


class DiscordShitlistStore:
    def __init__(self, path: Path | str, *, owner_user_ids: set[int] | None = None) -> None:
        self.path = Path(path)
        self.owner_user_ids = {int(user_id) for user_id in (owner_user_ids or set())}

    def list(self) -> list[DiscordShitlistEntry]:
        entries = self._load()
        return sorted(entries.values(), key=lambda entry: (entry.spice_level, entry.user_id), reverse=True)

    def get(self, user_id: int | str | None) -> DiscordShitlistEntry | None:
        if user_id is None:
            return None
        normalized = int(user_id)
        if normalized in self.owner_user_ids:
            return None
        return self._load().get(str(normalized))

    def add(self, user_id: int | str, *, reason: str, spice_level: int) -> DiscordShitlistEntry:
        normalized = int(user_id)
        if normalized in self.owner_user_ids:
            raise ValueError("bot owner cannot be added to the shitlist")
        entry = DiscordShitlistEntry(
            user_id=normalized,
            reason=_clean_reason(reason),
            spice_level=max(1, min(10, int(spice_level))),
            added_at=datetime.now(timezone.utc).isoformat(),
        )
        entries = self._load()
        entries[str(normalized)] = entry
        self._save(entries)
        return entry

    def remove(self, user_id: int | str) -> bool:
        normalized = int(user_id)
        entries = self._load()
        removed = entries.pop(str(normalized), None) is not None
        if removed:
            self._save(entries)
        return removed

    def _load(self) -> dict[str, DiscordShitlistEntry]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        raw_entries = data.get("entries") if isinstance(data, dict) else None
        if not isinstance(raw_entries, dict):
            return {}
        entries: dict[str, DiscordShitlistEntry] = {}
        for key, value in raw_entries.items():
            if not isinstance(value, dict):
                continue
            with suppress(ValueError):
                user_id = int(value.get("user_id", key))
                if user_id in self.owner_user_ids:
                    continue
                entries[str(user_id)] = DiscordShitlistEntry(
                    user_id=user_id,
                    reason=_clean_reason(str(value.get("reason") or "manual")),
                    spice_level=max(1, min(10, int(value.get("spice_level") or 1))),
                    added_at=str(value.get("added_at") or ""),
                )
        return entries

    def _save(self, entries: dict[str, DiscordShitlistEntry]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": SHITLIST_SCHEMA,
            "entries": {key: asdict(entry) for key, entry in sorted(entries.items())},
        }
        temp_path = self.path.with_name(f"{self.path.name}.tmp")
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp_path.replace(self.path)


def format_shitlist_reply(entry: DiscordShitlistEntry) -> str:
    reason = entry.reason or "listed"
    spice = entry.spice_level
    if spice <= 3:
        return "gfy"
    if spice <= 6:
        return f"gfy. you're on the list: {reason}."
    if spice <= 9:
        return f"nah. listed for {reason}. come back when the bit improves 💀"
    return f"absolutely not. spice 10 entry: {reason}. gfy 💀"


def _clean_reason(reason: str) -> str:
    cleaned = re.sub(r"\s+", " ", reason).strip()
    return cleaned[:240] if cleaned else "manual"
