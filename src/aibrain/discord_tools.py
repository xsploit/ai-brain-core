from __future__ import annotations

import contextlib
import io
import inspect
import os
import re
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import discord

from .codex_app_bridge import notify_codex_app_bridge
from .codex_bridge import CodexBridgeQueue
from .discord_shitlist import DiscordShitlistStore, format_shitlist_reply


DEFAULT_OWNER_USER_IDS = {120418341775998976}

OWNER_ONLY_TOOL_NAMES = [
    "discord_queue_codex_request",
    "discord_shitlist_add",
    "discord_shitlist_remove",
    "discord_shitlist_status",
    "discord_list_bot_guilds",
    "discord_get_application_info",
    "discord_delete_message",
    "discord_bulk_delete_messages",
    "discord_delete_recent_messages",
    "discord_create_text_channel",
    "discord_create_voice_channel",
    "discord_edit_channel",
    "discord_delete_channel",
    "discord_set_channel_overwrite",
    "discord_block_user_from_channel",
    "discord_unblock_user_from_channel",
    "discord_create_role",
    "discord_edit_role",
    "discord_delete_role",
    "discord_create_invite",
    "discord_create_webhook",
    "discord_send_webhook",
    "discord_delete_webhook",
]

DISCORD_AGENT_TOOL_NAMES = [
    "discord_get_capabilities",
    *OWNER_ONLY_TOOL_NAMES,
    "discord_get_guild",
    "discord_get_bot_user",
    "discord_list_channels",
    "discord_list_roles",
    "discord_list_members",
    "discord_search_members",
    "discord_get_member",
    "discord_get_current_context",
    "discord_get_permissions",
    "discord_can_do",
    "discord_audit_permissions",
    "discord_get_channel_overwrites",
    "discord_get_audit_log",
    "discord_read_channel_history",
    "discord_search_channel_messages",
    "discord_send_channel_message",
    "discord_send_rich_embed",
    "discord_send_file",
    "discord_send_dm",
    "discord_edit_own_message",
    "discord_fetch_message",
    "discord_pin_message",
    "discord_unpin_message",
    "discord_add_reaction",
    "discord_remove_reaction",
    "discord_create_thread",
    "discord_list_threads",
    "discord_archive_thread",
    "discord_lock_thread",
    "discord_timeout_member",
    "discord_remove_timeout",
    "discord_kick_member",
    "discord_ban_member",
    "discord_unban_user",
    "discord_add_member_role",
    "discord_remove_member_role",
    "discord_list_voice_states",
    "discord_move_member_voice",
    "discord_disconnect_member_voice",
    "discord_list_invites",
    "discord_delete_invite",
    "discord_create_poll",
    "discord_end_poll",
    "discord_list_emojis_stickers",
]

CHANNEL_BLOCK_TEXT_DENIES = [
    "send_messages",
    "add_reactions",
    "create_public_threads",
    "create_private_threads",
    "send_messages_in_threads",
]
CHANNEL_BLOCK_VOICE_DENIES = [
    "connect",
    "speak",
    "stream",
    "use_voice_activation",
]


@dataclass(slots=True)
class DiscordToolRuntime:
    bot: Any
    message: Any
    authority_mode: str = "discord_turn"


DISCORD_TOOL_CONTEXT: ContextVar[DiscordToolRuntime | None] = ContextVar("DISCORD_TOOL_CONTEXT", default=None)


class DiscordToolError(RuntimeError):
    pass


async def discord_get_capabilities() -> dict[str, Any]:
    """Return the current Discord tool gates, owner/admin status, and configured limits."""
    runtime = _runtime()
    guild = getattr(runtime.message, "guild", None)
    actor = await _actor_member(runtime)
    return {
        "actor": {
            "id": _id(actor),
            "name": _name(actor),
            "is_owner": _is_owner(runtime),
            "is_admin": _is_admin(actor),
        },
        "guild": _serialize_guild(guild) if guild is not None else None,
        "tool_channel_ids": sorted(_csv_ints("DISCORD_BRAIN_TOOL_CHANNEL_IDS")),
        "owner_user_ids": sorted(_owner_user_ids()),
        "owner_only_tools": OWNER_ONLY_TOOL_NAMES,
        "tools": DISCORD_AGENT_TOOL_NAMES,
        "moderation_enabled": _env_bool("DISCORD_BRAIN_MODERATION_TOOLS_ENABLED", True),
        "confirmation_required": _env_bool("DISCORD_BRAIN_MODERATION_REQUIRE_CONFIRM", True) and not _is_owner(runtime),
        "limits": {
            "history": _env_int("DISCORD_BRAIN_TOOL_HISTORY_LIMIT", 50),
            "search_history": _env_int("DISCORD_BRAIN_TOOL_SEARCH_HISTORY_LIMIT", 200),
            "send_chars": _env_int("DISCORD_BRAIN_TOOL_SEND_MAX_CHARS", 1900),
            "list_items": _env_int("DISCORD_BRAIN_TOOL_LIST_LIMIT", 100),
        },
    }


async def discord_queue_codex_request(prompt: str, route: str = "codex") -> dict[str, Any]:
    """Owner-only: queue a bounded Codex bridge request for Neuro self-upgrades, bugfixes, or reviews."""
    runtime = _runtime()
    _require_owner(runtime, "queue Codex bridge request")
    prompt = _bounded_text(prompt, _env_int("DISCORD_BRAIN_CODEX_BRIDGE_PROMPT_MAX_CHARS", 6000))
    if not prompt:
        raise DiscordToolError("prompt is required.")
    route = str(route or "codex").strip().lower()
    if route not in {"codex", "harness"}:
        raise DiscordToolError("route must be codex or harness.")
    queue = getattr(runtime.bot, "codex_bridge", None) or CodexBridgeQueue.from_env()
    if not queue.enabled:
        raise DiscordToolError("Codex bridge is disabled. Set DISCORD_BRAIN_CODEX_BRIDGE_ENABLED=true.")
    if queue.is_paused():
        raise DiscordToolError("Codex bridge queue is paused.")

    message = runtime.message
    author = getattr(message, "author", None)
    guild = getattr(message, "guild", None)
    channel = getattr(message, "channel", None)
    recent_messages: list[dict[str, Any]] = []
    context_builder = getattr(runtime.bot, "_context_for_message", None)
    if callable(context_builder):
        with contextlib.suppress(Exception):
            context = context_builder(message)
            recent = context.get("recent_messages") if isinstance(context, dict) else None
            if isinstance(recent, list):
                recent_messages = recent[-8:]

    delivery_mode = "harness_brain" if route == "harness" else "thread_heartbeat"
    path = queue.enqueue(
        requester_id=_id(author) or "unknown",
        requester_name=_name(author) or "unknown",
        guild_id=_id(guild),
        channel_id=_id(channel),
        message_id=_id(message),
        prompt=prompt,
        intent="harness" if route == "harness" else "ask_codex",
        authority_mode="manual_owner",
        authority_reason="owner Discord turn authorized Neuro Codex tool request",
        delivery_mode=delivery_mode,
        recent_messages=recent_messages,
        harness_agent="claude",
        harness_permission_profile="inspect",
    )
    notify = await notify_codex_app_bridge(path)
    return {
        "queued": True,
        "file": path.name,
        "delivery_mode": delivery_mode,
        "queue_root": str(queue.root),
        "bridge_notify": notify,
        "message": "Queued one Codex bridge request and notified the local bridge server when configured.",
    }


async def discord_shitlist_add(user_id: int | str, reason: str = "manual", spice_level: int = 3) -> dict[str, Any]:
    """Owner/autonomous Neuro: add or update a user in Neuro's persistent shitlist."""
    runtime = _runtime()
    _require_shitlist_authority(runtime, "edit shitlist")
    store = _shitlist_store(runtime)
    spice = int(spice_level)
    if _is_autonomous_neuro(runtime):
        spice = min(spice, _clamp(_env_int("DISCORD_BRAIN_SHITLIST_AUTONOMY_MAX_SPICE", 3), 1, 10))
        _require_not_bot_self(runtime, user_id)
    try:
        entry = store.add(user_id, reason=reason, spice_level=spice)
    except ValueError as exc:
        raise DiscordToolError(str(exc)) from exc
    return {
        "ok": True,
        "entry": _serialize_shitlist_entry(entry),
        "preview_reply": format_shitlist_reply(entry),
    }


async def discord_shitlist_remove(user_id: int | str) -> dict[str, Any]:
    """Owner/autonomous Neuro: remove a user from Neuro's persistent shitlist."""
    runtime = _runtime()
    _require_shitlist_authority(runtime, "edit shitlist")
    removed = _shitlist_store(runtime).remove(user_id)
    return {"ok": True, "removed": removed, "user_id": str(user_id)}


async def discord_shitlist_status() -> dict[str, Any]:
    """Owner/autonomous Neuro: list Neuro's persistent shitlist entries."""
    runtime = _runtime()
    _require_shitlist_authority(runtime, "view shitlist")
    entries = _shitlist_store(runtime).list()
    return {
        "ok": True,
        "count": len(entries),
        "entries": [_serialize_shitlist_entry(entry) for entry in entries],
    }


async def discord_list_bot_guilds(limit: int = 50) -> dict[str, Any]:
    """Owner-only: list guilds the bot is currently in."""
    runtime = _runtime()
    _require_owner(runtime, "list bot guilds")
    limit = _clamp(limit, 1, _env_int("DISCORD_BRAIN_TOOL_GUILD_LIMIT", 100))
    guilds = list(getattr(runtime.bot, "guilds", None) or [])
    current = getattr(runtime.message, "guild", None)
    if current is not None and all(_id(guild) != _id(current) for guild in guilds):
        guilds.append(current)
    return {"guilds": [_serialize_guild(guild) for guild in guilds[:limit]], "count": len(guilds)}


async def discord_get_guild(guild_id: int | None = None) -> dict[str, Any]:
    """Return metadata for the current guild, or another bot guild when called by the bot owner."""
    runtime = _runtime()
    guild = await _resolve_guild(runtime, guild_id)
    return {"guild": _serialize_guild(guild)}


async def discord_get_bot_user() -> dict[str, Any]:
    """Return the bot user and current guild member identity."""
    runtime = _runtime()
    bot_user = getattr(runtime.bot, "user", None)
    bot_member = await _bot_member(runtime)
    return {
        "bot_user": _serialize_user(bot_user),
        "bot_member": _serialize_member(bot_member) if bot_member is not None else None,
    }


async def discord_get_application_info() -> dict[str, Any]:
    """Owner-only: return safe application metadata without token or credential material."""
    runtime = _runtime()
    _require_owner(runtime, "get application info")
    method = getattr(runtime.bot, "application_info", None)
    if not callable(method):
        raise DiscordToolError("Bot runtime does not expose application_info().")
    info = await method()
    owner = getattr(info, "owner", None)
    team = getattr(info, "team", None)
    return {
        "application": {
            "id": _id(info),
            "name": _name(info),
            "description": getattr(info, "description", None),
            "bot_public": getattr(info, "bot_public", None),
            "bot_require_code_grant": getattr(info, "bot_require_code_grant", None),
            "owner": _serialize_user(owner) if owner is not None else None,
            "team": _name(team) if team is not None else None,
        }
    }


async def discord_list_channels(include_threads: bool = True, limit: int = 100) -> dict[str, Any]:
    """List visible guild channels and per-channel actions available to the current Discord user and bot."""
    runtime = _runtime()
    guild = _require_guild(runtime)
    actor = await _actor_member(runtime)
    bot_member = await _bot_member(runtime)
    limit = _clamp(limit, 1, _env_int("DISCORD_BRAIN_TOOL_LIST_LIMIT", 100))
    channels = []
    for channel in _iter_guild_channels(guild, include_threads=include_threads):
        if len(channels) >= limit:
            break
        if not _channel_is_allowed(channel):
            continue
        actor_perms = _permissions_for(channel, actor)
        bot_perms = _permissions_for(channel, bot_member)
        if not _perm(bot_perms, "view_channel"):
            continue
        if not (_is_owner(runtime) or _perm(actor_perms, "view_channel")):
            continue
        channels.append(
            {
                **_serialize_channel(channel),
                "can_read_history": (_is_owner(runtime) or _perm(actor_perms, "read_message_history"))
                and _perm(bot_perms, "read_message_history"),
                "can_send": (_is_owner(runtime) or _perm(actor_perms, "send_messages")) and _perm(bot_perms, "send_messages"),
                "can_manage_messages": (_is_owner(runtime) or _perm(actor_perms, "manage_messages"))
                and _perm(bot_perms, "manage_messages"),
                "can_manage_threads": (_is_owner(runtime) or _perm(actor_perms, "manage_threads"))
                and _perm(bot_perms, "manage_threads"),
            }
        )
    return {"guild_id": _id(guild), "channels": channels}


async def discord_list_roles(limit: int = 100) -> dict[str, Any]:
    """List roles in the current guild."""
    runtime = _runtime()
    guild = _require_guild(runtime)
    roles = sorted(list(getattr(guild, "roles", None) or []), key=lambda role: getattr(role, "position", 0), reverse=True)
    limit = _clamp(limit, 1, _env_int("DISCORD_BRAIN_TOOL_ROLE_LIMIT", 150))
    return {"guild_id": _id(guild), "roles": [_serialize_role(role) for role in roles[:limit]], "count": len(roles)}


async def discord_list_members(
    limit: int = 50,
    include_bots: bool = True,
    guild_id: int | None = None,
) -> dict[str, Any]:
    """Admin/owner: list cached guild members. Requires the Discord members intent for reliable output."""
    runtime = _runtime()
    _require_admin_or_owner(runtime, "list members")
    _require_member_intent(runtime)
    guild = await _resolve_guild(runtime, guild_id)
    members = [member for member in _iter_guild_members(guild) if include_bots or not getattr(member, "bot", False)]
    limit = _clamp(limit, 1, _env_int("DISCORD_BRAIN_TOOL_MEMBER_LIMIT", 100))
    return {"guild_id": _id(guild), "members": [_serialize_member(member) for member in members[:limit]], "count": len(members)}


async def discord_search_members(
    query: str,
    limit: int = 25,
    include_bots: bool = True,
    guild_id: int | None = None,
) -> dict[str, Any]:
    """Admin/owner: search guild members by id, username, global name, display name, or nick."""
    runtime = _runtime()
    _require_admin_or_owner(runtime, "search members")
    _require_member_intent(runtime)
    guild = await _resolve_guild(runtime, guild_id)
    query = query.strip().lower()
    limit = _clamp(limit, 1, _env_int("DISCORD_BRAIN_TOOL_MEMBER_SEARCH_LIMIT", 50))
    matched = []
    if query:
        for member in _iter_guild_members(guild):
            if not include_bots and getattr(member, "bot", False):
                continue
            haystack = " ".join(
                str(value or "")
                for value in (
                    _id(member),
                    _name(member),
                    getattr(member, "display_name", None),
                    getattr(member, "global_name", None),
                    getattr(member, "nick", None),
                )
            ).lower()
            if query in haystack:
                matched.append(member)
                if len(matched) >= limit:
                    break
    if not matched and hasattr(guild, "query_members") and query:
        with contextlib.suppress(Exception):
            matched = await guild.query_members(query, limit=limit)
            if not include_bots:
                matched = [member for member in matched if not getattr(member, "bot", False)]
    return {"guild_id": _id(guild), "members": [_serialize_member(member) for member in matched[:limit]], "count": len(matched)}


async def discord_get_member(user_id: int, guild_id: int | None = None) -> dict[str, Any]:
    """Return guild member details for the current guild, or another owner-authorized bot guild."""
    runtime = _runtime()
    guild = await _resolve_guild(runtime, guild_id)
    member = await _resolve_member(guild, user_id)
    return {"member": _serialize_member(member)}


async def discord_get_current_context() -> dict[str, Any]:
    """Return the active Discord message context, bot identity, and real permissions for this turn."""
    runtime = _runtime()
    message = runtime.message
    channel = getattr(message, "channel", None)
    guild = getattr(message, "guild", None) or _guild_for_channel(runtime, channel)
    actor = await _actor_member_for_guild(runtime, guild)
    bot_member = await _bot_member_for_guild(runtime, guild)
    actor_perms = _permissions_for(channel, actor) if channel is not None and actor is not None else getattr(actor, "guild_permissions", None)
    bot_perms = _permissions_for(channel, bot_member) if channel is not None and bot_member is not None else getattr(bot_member, "guild_permissions", None)
    return {
        "message_id": _id(message),
        "is_dm": getattr(message, "guild", None) is None,
        "guild": _serialize_guild(guild) if guild is not None else None,
        "channel": _serialize_channel(channel) if channel is not None else None,
        "actor": _serialize_member(actor) if hasattr(actor, "guild_permissions") else _serialize_user(actor),
        "bot": _serialize_member(bot_member) if hasattr(bot_member, "guild_permissions") else _serialize_user(bot_member),
        "actor_is_owner": _is_owner(runtime),
        "actor_is_admin": _is_admin(actor),
        "actor_permissions": _serialize_permissions(actor_perms),
        "bot_permissions": _serialize_permissions(bot_perms),
        "tool_channel_ids": sorted(_csv_ints("DISCORD_BRAIN_TOOL_CHANNEL_IDS")),
    }


async def discord_get_permissions(
    channel_id: int | None = None,
    user_id: int | None = None,
    guild_id: int | None = None,
) -> dict[str, Any]:
    """Return guild/channel permissions for the requester, bot, or a selected member."""
    runtime = _runtime()
    channel = await _resolve_channel(runtime, channel_id) if channel_id is not None else getattr(runtime.message, "channel", None)
    channel_guild = _guild_for_channel(runtime, channel) if channel is not None else None
    guild = await _resolve_guild(runtime, guild_id) if guild_id is not None else (channel_guild or _require_guild(runtime))
    if channel_guild is not None and _id(channel_guild) != _id(guild):
        raise DiscordToolError(f"Channel {_id(channel)} is not in guild {_id(guild)}.")
    actor = await _actor_member_for_guild(runtime, guild)
    if user_id is not None and (actor is None or int(user_id) != _id(actor)):
        _require_admin_or_owner(runtime, "inspect another member's permissions")
        member = await _resolve_member(guild, user_id)
    else:
        member = actor
    bot_member = await _bot_member_for_guild(runtime, guild)
    return {
        "guild": _serialize_guild(guild),
        "channel": _serialize_channel(channel) if channel is not None else None,
        "member": _serialize_member(member) if member is not None else None,
        "bot": _serialize_member(bot_member) if bot_member is not None else None,
        "member_guild_permissions": _serialize_permissions(getattr(member, "guild_permissions", None)),
        "bot_guild_permissions": _serialize_permissions(getattr(bot_member, "guild_permissions", None)),
        "member_permissions": _serialize_permissions(_permissions_for(channel, member)) if channel is not None and member is not None else {},
        "bot_permissions": _serialize_permissions(_permissions_for(channel, bot_member)) if channel is not None else {},
    }


async def discord_can_do(
    action: str,
    channel_id: int | None = None,
    user_id: int | None = None,
    guild_id: int | None = None,
) -> dict[str, Any]:
    """Dry-run whether the requester and bot can perform a Discord action; no state is changed."""
    runtime = _runtime()
    normalized = action.strip().lower().replace("-", "_").replace(" ", "_")
    requirements = _action_requirements(normalized)
    channel = await _resolve_channel(runtime, channel_id) if channel_id is not None else getattr(runtime.message, "channel", None)
    channel_guild = _guild_for_channel(runtime, channel) if channel is not None else None
    guild = await _resolve_guild(runtime, guild_id) if guild_id is not None else (channel_guild or getattr(runtime.message, "guild", None))
    actor = await _actor_member_for_guild(runtime, guild)
    bot_member = await _bot_member_for_guild(runtime, guild)
    target_member = await _resolve_member(guild, user_id) if guild is not None and user_id is not None else None
    actor_perms = _permissions_for(channel, actor) if channel is not None and actor is not None else getattr(actor, "guild_permissions", None)
    bot_perms = _permissions_for(channel, bot_member) if channel is not None and bot_member is not None else getattr(bot_member, "guild_permissions", None)
    owner = _is_owner(runtime)
    actor_missing = [] if owner else [permission for permission in requirements if not _perm(actor_perms, permission)]
    bot_missing = [permission for permission in requirements if not _perm(bot_perms, permission)]
    hierarchy_ok = True
    hierarchy_reason = None
    if target_member is not None and normalized in {"timeout_member", "kick_member", "ban_member", "add_member_role", "remove_member_role", "move_voice", "disconnect_voice"}:
        try:
            _require_member_hierarchy(runtime, target_member, require_actor=not owner)
        except DiscordToolError as exc:
            hierarchy_ok = False
            hierarchy_reason = str(exc)
    return {
        "action": normalized,
        "requirements": requirements,
        "guild": _serialize_guild(guild) if guild is not None else None,
        "channel": _serialize_channel(channel) if channel is not None else None,
        "target_member": _serialize_member(target_member) if target_member is not None else None,
        "actor_allowed": not actor_missing and hierarchy_ok,
        "bot_allowed": not bot_missing and hierarchy_ok,
        "allowed": not actor_missing and not bot_missing and hierarchy_ok,
        "actor_missing": actor_missing,
        "bot_missing": bot_missing,
        "hierarchy_reason": hierarchy_reason,
    }


async def discord_audit_permissions(
    guild_id: int | None = None,
    channel_id: int | None = None,
    include_channels: bool = True,
    limit: int = 100,
) -> dict[str, Any]:
    """Return actual bot guild permissions and channel-level capabilities; owner can use this from DMs."""
    runtime = _runtime()
    _require_admin_or_owner(runtime, "audit permissions")
    guild = await _resolve_guild(runtime, guild_id)
    bot_member = await _bot_member_for_guild(runtime, guild)
    limit = _clamp(limit, 1, _env_int("DISCORD_BRAIN_TOOL_LIST_LIMIT", 100))
    channels = []
    if include_channels:
        candidates = [await _resolve_channel(runtime, channel_id)] if channel_id is not None else _iter_guild_channels(guild, include_threads=True)
        for channel in candidates:
            if len(channels) >= limit:
                break
            if channel is None or not _channel_is_allowed(channel):
                continue
            if _id(_guild_for_channel(runtime, channel)) != _id(guild):
                continue
            perms = _permissions_for(channel, bot_member)
            channels.append(
                {
                    **_serialize_channel(channel),
                    "bot_permissions": _serialize_permissions(perms),
                    "can_view": _perm(perms, "view_channel"),
                    "can_read_history": _perm(perms, "view_channel") and _perm(perms, "read_message_history"),
                    "can_send": _perm(perms, "view_channel") and _perm(perms, "send_messages"),
                    "can_connect": _perm(perms, "connect"),
                    "can_speak": _perm(perms, "connect") and _perm(perms, "speak"),
                    "can_moderate": _perm(perms, "moderate_members"),
                    "can_manage_messages": _perm(perms, "manage_messages"),
                    "can_manage_channels": _perm(perms, "manage_channels"),
                    "can_manage_roles": _perm(perms, "manage_roles"),
                }
            )
    return {
        "guild": _serialize_guild(guild),
        "bot": _serialize_member(bot_member) if bot_member is not None else None,
        "bot_guild_permissions": _serialize_permissions(getattr(bot_member, "guild_permissions", None)),
        "channels": channels,
        "channel_count": len(channels),
    }


async def discord_get_channel_overwrites(channel_id: int) -> dict[str, Any]:
    """Admin/owner: read channel permission overwrites for debugging server sandboxing."""
    runtime = _runtime()
    _require_admin_or_owner(runtime, "read channel overwrites")
    channel = await _resolve_channel(runtime, channel_id)
    await _require_channel_permissions(runtime, channel, "read channel overwrites", "view_channel")
    overwrites = []
    for target, overwrite in getattr(channel, "overwrites", {}).items():
        overwrites.append(
            {
                "target": _serialize_overwrite_target(target),
                "allow": [name for name, value in overwrite if value is True],
                "deny": [name for name, value in overwrite if value is False],
            }
        )
    return {"channel": _serialize_channel(channel), "overwrites": overwrites, "count": len(overwrites)}


async def discord_get_audit_log(
    guild_id: int | None = None,
    limit: int = 10,
    action: str | None = None,
    user_id: int | None = None,
) -> dict[str, Any]:
    """Admin/owner: read recent guild audit-log entries when the bot has view_audit_log."""
    runtime = _runtime()
    _require_admin_or_owner(runtime, "read audit log")
    guild = await _resolve_guild(runtime, guild_id)
    await _require_bot_guild_permission_for_guild(runtime, guild, "view_audit_log")
    actor = await _actor_member_for_guild(runtime, guild)
    if not _is_owner(runtime) and not _perm(getattr(actor, "guild_permissions", None), "view_audit_log"):
        raise DiscordToolError("Current user lacks view_audit_log.")
    method = getattr(guild, "audit_logs", None)
    if not callable(method):
        raise DiscordToolError("Guild runtime does not expose audit_logs().")
    action_value = _audit_log_action(action) if action else None
    limit = _clamp(limit, 1, _env_int("DISCORD_BRAIN_AUDIT_LOG_LIMIT", 25))
    target_user = await _fetch_user(runtime, user_id) if user_id is not None else None
    entries = []
    async for entry in method(limit=limit, action=action_value, user=target_user):
        entries.append(_serialize_audit_log_entry(entry))
    return {"guild": _serialize_guild(guild), "entries": entries, "count": len(entries)}


async def discord_read_channel_history(
    channel_id: int | None = None,
    limit: int = 20,
    before_message_id: int | None = None,
    include_bot_messages: bool = True,
) -> dict[str, Any]:
    """Read recent message history from a channel the current user and bot can both view."""
    runtime = _runtime()
    channel = await _resolve_channel(runtime, channel_id)
    await _require_channel_permissions(runtime, channel, "read channel history", "view_channel", "read_message_history")
    limit = _clamp(limit, 1, _env_int("DISCORD_BRAIN_TOOL_HISTORY_LIMIT", 50))
    before = discord.Object(id=before_message_id) if before_message_id else None
    messages = []
    async for message in channel.history(limit=limit, before=before):
        author = getattr(message, "author", None)
        if not include_bot_messages and getattr(author, "bot", False):
            continue
        messages.append(_serialize_message(message))
    return {"channel": _serialize_channel(channel), "messages": messages}


async def discord_search_channel_messages(
    query: str,
    channel_id: int | None = None,
    history_limit: int = 200,
    max_results: int = 25,
) -> dict[str, Any]:
    """Search recent channel history by plain substring. Results are bounded to avoid broad scraping."""
    runtime = _runtime()
    channel = await _resolve_channel(runtime, channel_id)
    await _require_channel_permissions(runtime, channel, "search channel history", "view_channel", "read_message_history")
    needle = query.strip().lower()
    if not needle:
        raise DiscordToolError("query is required.")
    history_limit = _clamp(history_limit, 1, _env_int("DISCORD_BRAIN_TOOL_SEARCH_HISTORY_LIMIT", 200))
    max_results = _clamp(max_results, 1, _env_int("DISCORD_BRAIN_TOOL_SEARCH_RESULT_LIMIT", 50))
    messages = []
    async for message in channel.history(limit=history_limit):
        content = getattr(message, "clean_content", None) or getattr(message, "content", "") or ""
        if needle in content.lower():
            messages.append(_serialize_message(message))
            if len(messages) >= max_results:
                break
    return {"channel": _serialize_channel(channel), "query": query, "messages": messages}


async def discord_send_channel_message(
    content: str,
    channel_id: int | None = None,
    reply_to_message_id: int | None = None,
    allow_user_mentions: bool = False,
) -> dict[str, Any]:
    """Send a message to a channel the current user and bot can both send messages in."""
    runtime = _runtime()
    channel = await _resolve_channel(runtime, channel_id)
    await _require_channel_permissions(runtime, channel, "send messages", "view_channel", "send_messages")
    content = _bounded_text(content, _env_int("DISCORD_BRAIN_TOOL_SEND_MAX_CHARS", 1900))
    allowed_mentions = discord.AllowedMentions(users=allow_user_mentions, roles=False, everyone=False, replied_user=False)
    if reply_to_message_id:
        await _require_channel_permissions(runtime, channel, "read reply target", "read_message_history")
        target = await channel.fetch_message(reply_to_message_id)
        sent = await _call_discord(target.reply, content, mention_author=False, allowed_mentions=allowed_mentions)
    else:
        sent = await _call_discord(channel.send, content, allowed_mentions=allowed_mentions)
    await _audit_action(runtime, "discord_send_channel_message", _id(sent), {"channel_id": _id(channel)})
    return {"sent": True, "message": _serialize_message(sent)}


async def discord_send_dm(
    user_id: int,
    content: str,
    allow_user_mentions: bool = False,
) -> dict[str, Any]:
    """Owner/admin: send a direct message to a Discord user by ID."""
    runtime = _runtime()
    _require_admin_or_owner(runtime, "send direct messages")
    content = _bounded_text(content, _env_int("DISCORD_BRAIN_TOOL_DM_MAX_CHARS", 1900))
    if not content:
        raise DiscordToolError("content is required.")
    user = await _fetch_user(runtime, int(user_id))
    sender = getattr(user, "send", None)
    if not callable(sender):
        raise DiscordToolError(f"User {user_id} could not be fetched for DM.")
    allowed_mentions = discord.AllowedMentions(users=allow_user_mentions, roles=False, everyone=False)
    sent = await _call_discord(sender, content, allowed_mentions=allowed_mentions)
    await _audit_action(runtime, "discord_send_dm", _id(sent), {"user_id": int(user_id)})
    return {"sent": True, "user": _serialize_user(user), "message": _serialize_message(sent)}


async def discord_send_rich_embed(
    title: str | None = None,
    description: str | None = None,
    fields: list[dict[str, Any]] | None = None,
    channel_id: int | None = None,
    content: str | None = None,
    color: int | str | None = None,
    url: str | None = None,
    image_url: str | None = None,
    thumbnail_url: str | None = None,
    footer_text: str | None = None,
    author_name: str | None = None,
    author_url: str | None = None,
    reply_to_message_id: int | None = None,
    allow_user_mentions: bool = False,
) -> dict[str, Any]:
    """Send a structured Discord embed with optional title, fields, image, thumbnail, and footer."""
    runtime = _runtime()
    channel = await _resolve_channel(runtime, channel_id)
    await _require_channel_permissions(runtime, channel, "send rich embed", "view_channel", "send_messages")
    embed = _build_rich_embed(
        title=title,
        description=description,
        fields=fields,
        color=color,
        url=url,
        image_url=image_url,
        thumbnail_url=thumbnail_url,
        footer_text=footer_text,
        author_name=author_name,
        author_url=author_url,
    )
    allowed_mentions = discord.AllowedMentions(users=allow_user_mentions, roles=False, everyone=False, replied_user=False)
    send_content = _bounded_optional(content, _env_int("DISCORD_BRAIN_TOOL_SEND_MAX_CHARS", 1900))
    if reply_to_message_id:
        await _require_channel_permissions(runtime, channel, "read reply target", "read_message_history")
        target = await channel.fetch_message(reply_to_message_id)
        sent = await _call_discord(target.reply, send_content, mention_author=False, embed=embed, allowed_mentions=allowed_mentions)
    else:
        sent = await _call_discord(channel.send, content=send_content, embed=embed, allowed_mentions=allowed_mentions)
    await _audit_action(runtime, "discord_send_rich_embed", _id(sent), {"channel_id": _id(channel), "fields": len(embed.fields)})
    return {
        "sent": True,
        "message": _serialize_message(sent),
        "embed": _serialize_embed(embed),
    }


async def discord_send_file(
    filename: str,
    content: str,
    channel_id: int | None = None,
    message: str | None = None,
    spoiler: bool = False,
    allow_user_mentions: bool = False,
) -> dict[str, Any]:
    """Send a generated text file to a channel the requester and bot can both post attachments in."""
    runtime = _runtime()
    channel = await _resolve_channel(runtime, channel_id)
    await _require_channel_permissions(runtime, channel, "send file", "view_channel", "send_messages", "attach_files")
    max_bytes = _env_int("DISCORD_BRAIN_TOOL_FILE_MAX_BYTES", 512_000)
    body = content.encode("utf-8", errors="replace")[:max_bytes]
    clean_name = _safe_filename(filename, default="discord-export.txt")
    allowed_mentions = discord.AllowedMentions(users=allow_user_mentions, roles=False, everyone=False, replied_user=False)
    sent = await _call_discord(
        channel.send,
        content=_bounded_optional(message, _env_int("DISCORD_BRAIN_TOOL_SEND_MAX_CHARS", 1900)),
        file=discord.File(io.BytesIO(body), filename=clean_name, spoiler=bool(spoiler)),
        allowed_mentions=allowed_mentions,
    )
    await _audit_action(runtime, "discord_send_file", _id(sent), {"channel_id": _id(channel), "filename": clean_name, "bytes": len(body)})
    return {"sent": True, "message": _serialize_message(sent), "filename": clean_name, "bytes": len(body)}


async def discord_edit_own_message(channel_id: int, message_id: int, content: str) -> dict[str, Any]:
    """Edit a message authored by this bot."""
    runtime = _runtime()
    channel = await _resolve_channel(runtime, channel_id)
    await _require_channel_permissions(runtime, channel, "edit own message", "view_channel", "send_messages", "read_message_history")
    message = await channel.fetch_message(message_id)
    bot_user = getattr(runtime.bot, "user", None)
    if _id(getattr(message, "author", None)) != _id(bot_user):
        raise DiscordToolError("Refusing to edit a message not authored by this bot.")
    edited = await _call_discord(message.edit, content=_bounded_text(content, _env_int("DISCORD_BRAIN_TOOL_SEND_MAX_CHARS", 1900)))
    await _audit_action(runtime, "discord_edit_own_message", message_id, {"channel_id": _id(channel)})
    return {"edited": True, "message": _serialize_message(edited or message)}


async def discord_fetch_message(channel_id: int, message_id: int) -> dict[str, Any]:
    """Fetch one message from a channel the current user and bot can both read."""
    runtime = _runtime()
    channel = await _resolve_channel(runtime, channel_id)
    await _require_channel_permissions(runtime, channel, "fetch message", "view_channel", "read_message_history")
    message = await channel.fetch_message(message_id)
    return {"message": _serialize_message(message)}


async def discord_delete_message(
    channel_id: int,
    message_id: int,
    reason: str | None = None,
    confirm: bool = False,
) -> dict[str, Any]:
    """Owner-only: delete a message."""
    runtime = _runtime()
    _require_owner(runtime, "delete message")
    _require_confirm(runtime, confirm, "delete_message")
    channel = await _resolve_channel(runtime, channel_id)
    await _require_channel_permissions(runtime, channel, "manage messages", "view_channel", "read_message_history", "manage_messages")
    message = await channel.fetch_message(message_id)
    await _call_discord(message.delete, reason=_bounded_optional(reason, 512))
    await _audit_action(runtime, "discord_delete_message", message_id, {"channel_id": _id(channel), "reason": reason})
    return {"deleted": True, "channel_id": _id(channel), "message_id": message_id}


async def discord_bulk_delete_messages(
    channel_id: int,
    message_ids: list[int],
    reason: str | None = None,
    confirm: bool = False,
) -> dict[str, Any]:
    """Owner-only: bulk-delete specific message IDs."""
    runtime = _runtime()
    _require_owner(runtime, "bulk delete messages")
    _require_confirm(runtime, confirm, "bulk_delete_messages")
    channel = await _resolve_channel(runtime, channel_id)
    await _require_channel_permissions(runtime, channel, "bulk delete messages", "view_channel", "read_message_history", "manage_messages")
    limit = _env_int("DISCORD_BRAIN_TOOL_BULK_DELETE_LIMIT", 50)
    ids = [int(message_id) for message_id in message_ids[:limit]]
    messages = [await channel.fetch_message(message_id) for message_id in ids]
    await _delete_messages(channel, messages, reason)
    await _audit_action(runtime, "discord_bulk_delete_messages", _id(channel), {"message_ids": ids, "reason": reason})
    return {"deleted": len(messages), "channel_id": _id(channel), "message_ids": ids}


async def discord_delete_recent_messages(
    channel_id: int | None = None,
    limit: int = 25,
    user_id: int | None = None,
    contains: str | None = None,
    include_bot_messages: bool = True,
    include_pinned: bool = False,
    reason: str | None = None,
    confirm: bool = False,
) -> dict[str, Any]:
    """Owner-only: delete recent messages from a channel with simple filters."""
    runtime = _runtime()
    _require_owner(runtime, "delete recent messages")
    _require_confirm(runtime, confirm, "delete_recent_messages")
    channel = await _resolve_channel(runtime, channel_id)
    await _require_channel_permissions(runtime, channel, "delete recent messages", "view_channel", "read_message_history", "manage_messages")
    max_limit = _env_int("DISCORD_BRAIN_TOOL_BULK_DELETE_LIMIT", 50)
    limit = _clamp(limit, 1, max_limit)
    needle = contains.lower() if contains else None
    messages: list[Any] = []
    async for message in channel.history(limit=limit):
        if user_id is not None and _id(getattr(message, "author", None)) != int(user_id):
            continue
        if not include_bot_messages and bool(getattr(getattr(message, "author", None), "bot", False)):
            continue
        if not include_pinned and bool(getattr(message, "pinned", False)):
            continue
        if needle is not None:
            content = (getattr(message, "clean_content", None) or getattr(message, "content", "") or "").lower()
            if needle not in content:
                continue
        messages.append(message)
    await _delete_messages(channel, messages, reason)
    ids = [_id(message) for message in messages]
    await _audit_action(
        runtime,
        "discord_delete_recent_messages",
        _id(channel),
        {"message_ids": ids, "user_id": user_id, "contains": contains, "reason": reason},
    )
    return {"deleted": len(messages), "channel_id": _id(channel), "message_ids": ids}


async def discord_pin_message(channel_id: int, message_id: int, reason: str | None = None, confirm: bool = False) -> dict[str, Any]:
    """Pin a message after confirmation unless invoked by the configured bot owner."""
    runtime = _runtime()
    _require_confirm(runtime, confirm, "pin_message")
    channel = await _resolve_channel(runtime, channel_id)
    await _require_channel_permissions(runtime, channel, "pin message", "view_channel", "read_message_history", "manage_messages")
    message = await channel.fetch_message(message_id)
    await _call_discord(message.pin, reason=_bounded_optional(reason, 512))
    await _audit_action(runtime, "discord_pin_message", message_id, {"channel_id": _id(channel)})
    return {"pinned": True, "channel_id": _id(channel), "message_id": message_id}


async def discord_unpin_message(channel_id: int, message_id: int, reason: str | None = None, confirm: bool = False) -> dict[str, Any]:
    """Unpin a message after confirmation unless invoked by the configured bot owner."""
    runtime = _runtime()
    _require_confirm(runtime, confirm, "unpin_message")
    channel = await _resolve_channel(runtime, channel_id)
    await _require_channel_permissions(runtime, channel, "unpin message", "view_channel", "read_message_history", "manage_messages")
    message = await channel.fetch_message(message_id)
    await _call_discord(message.unpin, reason=_bounded_optional(reason, 512))
    await _audit_action(runtime, "discord_unpin_message", message_id, {"channel_id": _id(channel)})
    return {"unpinned": True, "channel_id": _id(channel), "message_id": message_id}


async def discord_add_reaction(channel_id: int, message_id: int, emoji: str) -> dict[str, Any]:
    """Add a reaction to a message when the current user and bot can both read and react in that channel."""
    runtime = _runtime()
    channel = await _resolve_channel(runtime, channel_id)
    await _require_channel_permissions(runtime, channel, "add reactions", "view_channel", "read_message_history", "add_reactions")
    message = await channel.fetch_message(message_id)
    await message.add_reaction(emoji[:80])
    await _audit_action(runtime, "discord_add_reaction", message_id, {"channel_id": _id(channel), "emoji": emoji[:80]})
    return {"reacted": True, "channel_id": _id(channel), "message_id": _id(message), "emoji": emoji[:80]}


async def discord_remove_reaction(channel_id: int, message_id: int, emoji: str, user_id: int | None = None) -> dict[str, Any]:
    """Remove this bot's reaction, or another user's reaction when the requester and bot can manage messages."""
    runtime = _runtime()
    channel = await _resolve_channel(runtime, channel_id)
    if user_id is None:
        await _require_channel_permissions(runtime, channel, "remove own reaction", "view_channel", "read_message_history")
        user = getattr(runtime.bot, "user", None)
    else:
        await _require_channel_permissions(runtime, channel, "remove reactions", "view_channel", "read_message_history", "manage_messages")
        user = await _fetch_user(runtime, user_id)
    message = await channel.fetch_message(message_id)
    await _call_discord(message.remove_reaction, emoji[:80], user)
    await _audit_action(runtime, "discord_remove_reaction", message_id, {"channel_id": _id(channel), "emoji": emoji[:80], "user_id": _id(user)})
    return {"reaction_removed": True, "channel_id": _id(channel), "message_id": message_id, "emoji": emoji[:80], "user_id": _id(user)}


async def discord_create_thread(
    channel_id: int,
    name: str,
    message_id: int | None = None,
    auto_archive_duration: int = 1440,
) -> dict[str, Any]:
    """Create a public thread in a channel when the current user and bot can both create threads there."""
    runtime = _runtime()
    channel = await _resolve_channel(runtime, channel_id)
    await _require_channel_permissions(runtime, channel, "create public threads", "view_channel", "create_public_threads")
    name = _bounded_text(name.strip(), 100)
    auto_archive_duration = _clamp(auto_archive_duration, 60, 10080)
    if message_id:
        target = await channel.fetch_message(message_id)
        thread = await _call_discord(target.create_thread, name=name, auto_archive_duration=auto_archive_duration)
    else:
        thread = await _call_discord(channel.create_thread, name=name, auto_archive_duration=auto_archive_duration)
    await _audit_action(runtime, "discord_create_thread", _id(thread), {"channel_id": _id(channel)})
    return {"created": True, "thread": _serialize_channel(thread)}


async def discord_list_threads(channel_id: int | None = None, limit: int = 50) -> dict[str, Any]:
    """List active threads in the current guild or under a specific channel when available."""
    runtime = _runtime()
    guild = _require_guild(runtime)
    threads = []
    if channel_id is not None:
        channel = await _resolve_channel(runtime, channel_id)
        await _require_channel_permissions(runtime, channel, "list threads", "view_channel")
        threads.extend(getattr(channel, "threads", None) or [])
    else:
        for thread in getattr(guild, "threads", None) or []:
            if _channel_is_allowed(thread):
                threads.append(thread)
    limit = _clamp(limit, 1, _env_int("DISCORD_BRAIN_TOOL_THREAD_LIMIT", 100))
    return {"guild_id": _id(guild), "threads": [_serialize_channel(thread) for thread in threads[:limit]], "count": len(threads)}


async def discord_archive_thread(thread_id: int, archived: bool = True, reason: str | None = None, confirm: bool = False) -> dict[str, Any]:
    """Archive or unarchive a thread after confirmation unless invoked by the configured bot owner."""
    runtime = _runtime()
    _require_confirm(runtime, confirm, "archive_thread")
    thread = await _resolve_channel(runtime, thread_id)
    await _require_channel_permissions(runtime, thread, "manage threads", "view_channel", "manage_threads")
    await _call_discord(thread.edit, archived=bool(archived), reason=_bounded_optional(reason, 512))
    await _audit_action(runtime, "discord_archive_thread", thread_id, {"archived": archived, "reason": reason})
    return {"edited": True, "thread_id": thread_id, "archived": bool(archived)}


async def discord_lock_thread(thread_id: int, locked: bool = True, reason: str | None = None, confirm: bool = False) -> dict[str, Any]:
    """Lock or unlock a thread after confirmation unless invoked by the configured bot owner."""
    runtime = _runtime()
    _require_confirm(runtime, confirm, "lock_thread")
    thread = await _resolve_channel(runtime, thread_id)
    await _require_channel_permissions(runtime, thread, "manage threads", "view_channel", "manage_threads")
    await _call_discord(thread.edit, locked=bool(locked), reason=_bounded_optional(reason, 512))
    await _audit_action(runtime, "discord_lock_thread", thread_id, {"locked": locked, "reason": reason})
    return {"edited": True, "thread_id": thread_id, "locked": bool(locked)}


async def discord_timeout_member(
    user_id: int,
    duration_seconds: int,
    reason: str | None = None,
    confirm: bool = False,
) -> dict[str, Any]:
    """Timeout a guild member after confirmation unless invoked by the configured bot owner."""
    runtime = _runtime()
    _require_confirm(runtime, confirm, "timeout_member")
    guild = _require_guild(runtime)
    await _require_moderation(runtime, "moderate_members", user_id)
    member = await _resolve_member(guild, user_id)
    _require_member_hierarchy(runtime, member, require_actor=not _is_owner(runtime))
    duration_seconds = _clamp(duration_seconds, 1, _env_int("DISCORD_BRAIN_MAX_TIMEOUT_SECONDS", 86_400))
    until = datetime.now(timezone.utc) + timedelta(seconds=duration_seconds)
    await member.timeout(until, reason=_bounded_optional(reason, 512))
    await _audit_action(runtime, "discord_timeout_member", user_id, {"until": until.isoformat(), "reason": reason})
    return {"timed_out": True, "user_id": user_id, "until": until.isoformat()}


async def discord_remove_timeout(user_id: int, reason: str | None = None, confirm: bool = False) -> dict[str, Any]:
    """Remove a guild member timeout after confirmation unless invoked by the configured bot owner."""
    runtime = _runtime()
    _require_confirm(runtime, confirm, "remove_timeout")
    guild = _require_guild(runtime)
    await _require_moderation(runtime, "moderate_members", user_id)
    member = await _resolve_member(guild, user_id)
    _require_member_hierarchy(runtime, member, require_actor=not _is_owner(runtime))
    await member.timeout(None, reason=_bounded_optional(reason, 512))
    await _audit_action(runtime, "discord_remove_timeout", user_id, {"reason": reason})
    return {"timeout_removed": True, "user_id": user_id}


async def discord_kick_member(user_id: int, reason: str | None = None, confirm: bool = False) -> dict[str, Any]:
    """Kick a guild member after confirmation unless invoked by the configured bot owner."""
    runtime = _runtime()
    _require_confirm(runtime, confirm, "kick_member")
    guild = _require_guild(runtime)
    await _require_moderation(runtime, "kick_members", user_id)
    member = await _resolve_member(guild, user_id)
    _require_member_hierarchy(runtime, member, require_actor=not _is_owner(runtime))
    await member.kick(reason=_bounded_optional(reason, 512))
    await _audit_action(runtime, "discord_kick_member", user_id, {"reason": reason})
    return {"kicked": True, "user_id": user_id}


async def discord_ban_member(
    user_id: int,
    delete_message_seconds: int = 0,
    reason: str | None = None,
    confirm: bool = False,
) -> dict[str, Any]:
    """Ban a guild member after confirmation unless invoked by the configured bot owner."""
    runtime = _runtime()
    _require_confirm(runtime, confirm, "ban_member")
    guild = _require_guild(runtime)
    await _require_moderation(runtime, "ban_members", user_id)
    member = await _resolve_member(guild, user_id)
    _require_member_hierarchy(runtime, member, require_actor=not _is_owner(runtime))
    delete_message_seconds = _clamp(delete_message_seconds, 0, _env_int("DISCORD_BRAIN_MAX_BAN_DELETE_SECONDS", 604_800))
    await _call_discord(guild.ban, member, reason=_bounded_optional(reason, 512), delete_message_seconds=delete_message_seconds)
    await _audit_action(runtime, "discord_ban_member", user_id, {"delete_message_seconds": delete_message_seconds, "reason": reason})
    return {"banned": True, "user_id": user_id, "delete_message_seconds": delete_message_seconds}


async def discord_unban_user(user_id: int, reason: str | None = None, confirm: bool = False) -> dict[str, Any]:
    """Unban a user by id after confirmation unless invoked by the configured bot owner."""
    runtime = _runtime()
    _require_confirm(runtime, confirm, "unban_user")
    guild = _require_guild(runtime)
    await _require_moderation(runtime, "ban_members", user_id, require_target_member=False)
    user = await _fetch_user(runtime, user_id)
    await guild.unban(user, reason=_bounded_optional(reason, 512))
    await _audit_action(runtime, "discord_unban_user", user_id, {"reason": reason})
    return {"unbanned": True, "user_id": user_id}


async def discord_add_member_role(user_id: int, role_id: int, reason: str | None = None, confirm: bool = False) -> dict[str, Any]:
    """Add a role to a member after confirmation unless invoked by the configured bot owner."""
    runtime = _runtime()
    _require_confirm(runtime, confirm, "add_member_role")
    guild = _require_guild(runtime)
    await _require_role_management(runtime, user_id, role_id)
    member = await _resolve_member(guild, user_id)
    role = _resolve_role(guild, role_id)
    await _call_discord(member.add_roles, role, reason=_bounded_optional(reason, 512))
    await _audit_action(runtime, "discord_add_member_role", user_id, {"role_id": role_id, "reason": reason})
    return {"role_added": True, "user_id": user_id, "role": _serialize_role(role)}


async def discord_remove_member_role(user_id: int, role_id: int, reason: str | None = None, confirm: bool = False) -> dict[str, Any]:
    """Remove a role from a member after confirmation unless invoked by the configured bot owner."""
    runtime = _runtime()
    _require_confirm(runtime, confirm, "remove_member_role")
    guild = _require_guild(runtime)
    await _require_role_management(runtime, user_id, role_id)
    member = await _resolve_member(guild, user_id)
    role = _resolve_role(guild, role_id)
    await _call_discord(member.remove_roles, role, reason=_bounded_optional(reason, 512))
    await _audit_action(runtime, "discord_remove_member_role", user_id, {"role_id": role_id, "reason": reason})
    return {"role_removed": True, "user_id": user_id, "role": _serialize_role(role)}


async def discord_list_voice_states(guild_id: int | None = None, channel_id: int | None = None, limit: int = 100) -> dict[str, Any]:
    """List members currently connected to voice channels the bot can inspect."""
    runtime = _runtime()
    guild = await _resolve_guild(runtime, guild_id)
    limit = _clamp(limit, 1, _env_int("DISCORD_BRAIN_TOOL_LIST_LIMIT", 100))
    voice_states = []
    for channel in _iter_guild_channels(guild, include_threads=False):
        if channel_id is not None and _id(channel) != int(channel_id):
            continue
        members = list(getattr(channel, "members", None) or [])
        if not members:
            continue
        perms = _permissions_for(channel, await _bot_member_for_guild(runtime, guild))
        if not _perm(perms, "view_channel"):
            continue
        for member in members:
            if len(voice_states) >= limit:
                break
            state = getattr(member, "voice", None)
            voice_states.append(
                {
                    "channel": _serialize_channel(channel),
                    "member": _serialize_member(member),
                    "mute": getattr(state, "mute", None),
                    "deaf": getattr(state, "deaf", None),
                    "self_mute": getattr(state, "self_mute", None),
                    "self_deaf": getattr(state, "self_deaf", None),
                }
            )
    return {"guild": _serialize_guild(guild), "voice_states": voice_states, "count": len(voice_states)}


async def discord_move_member_voice(user_id: int, channel_id: int, reason: str | None = None, confirm: bool = False) -> dict[str, Any]:
    """Move a member to another voice channel after confirmation."""
    runtime = _runtime()
    _require_confirm(runtime, confirm, "move_member_voice")
    guild = _require_guild(runtime)
    await _require_moderation(runtime, "move_members", user_id)
    member = await _resolve_member(guild, user_id)
    channel = await _resolve_channel(runtime, channel_id)
    await _require_channel_permissions(runtime, channel, "move voice member", "view_channel", "connect", "move_members")
    await _call_discord(member.move_to, channel, reason=_bounded_optional(reason, 512))
    await _audit_action(runtime, "discord_move_member_voice", user_id, {"channel_id": channel_id, "reason": reason})
    return {"moved": True, "user_id": user_id, "channel": _serialize_channel(channel)}


async def discord_disconnect_member_voice(user_id: int, reason: str | None = None, confirm: bool = False) -> dict[str, Any]:
    """Disconnect a member from voice after confirmation."""
    runtime = _runtime()
    _require_confirm(runtime, confirm, "disconnect_member_voice")
    guild = _require_guild(runtime)
    await _require_moderation(runtime, "move_members", user_id)
    member = await _resolve_member(guild, user_id)
    await _call_discord(member.move_to, None, reason=_bounded_optional(reason, 512))
    await _audit_action(runtime, "discord_disconnect_member_voice", user_id, {"reason": reason})
    return {"disconnected": True, "user_id": user_id}


async def discord_create_text_channel(
    name: str,
    category_id: int | None = None,
    topic: str | None = None,
    reason: str | None = None,
    confirm: bool = False,
) -> dict[str, Any]:
    """Owner-only: create a text channel in the current guild."""
    runtime = _runtime()
    _require_owner(runtime, "create text channel")
    _require_confirm(runtime, confirm, "create_text_channel")
    guild = _require_guild(runtime)
    await _require_bot_guild_permission(runtime, "manage_channels")
    category = _resolve_category(guild, category_id) if category_id is not None else None
    channel = await _call_discord(
        guild.create_text_channel,
        _bounded_text(name, 100),
        category=category,
        topic=_bounded_optional(topic, 1024),
        reason=_bounded_optional(reason, 512),
    )
    await _audit_action(runtime, "discord_create_text_channel", _id(channel), {"category_id": category_id, "reason": reason})
    return {"created": True, "channel": _serialize_channel(channel)}


async def discord_create_voice_channel(
    name: str,
    category_id: int | None = None,
    reason: str | None = None,
    confirm: bool = False,
) -> dict[str, Any]:
    """Owner-only: create a voice channel in the current guild."""
    runtime = _runtime()
    _require_owner(runtime, "create voice channel")
    _require_confirm(runtime, confirm, "create_voice_channel")
    guild = _require_guild(runtime)
    await _require_bot_guild_permission(runtime, "manage_channels")
    category = _resolve_category(guild, category_id) if category_id is not None else None
    channel = await _call_discord(
        guild.create_voice_channel,
        _bounded_text(name, 100),
        category=category,
        reason=_bounded_optional(reason, 512),
    )
    await _audit_action(runtime, "discord_create_voice_channel", _id(channel), {"category_id": category_id, "reason": reason})
    return {"created": True, "channel": _serialize_channel(channel)}


async def discord_edit_channel(
    channel_id: int,
    name: str | None = None,
    topic: str | None = None,
    nsfw: bool | None = None,
    slowmode_delay: int | None = None,
    reason: str | None = None,
    confirm: bool = False,
) -> dict[str, Any]:
    """Owner-only: edit basic channel settings."""
    runtime = _runtime()
    _require_owner(runtime, "edit channel")
    _require_confirm(runtime, confirm, "edit_channel")
    channel = await _resolve_channel(runtime, channel_id)
    await _require_bot_guild_permission(runtime, "manage_channels")
    updates: dict[str, Any] = {}
    if name is not None:
        updates["name"] = _bounded_text(name, 100)
    if topic is not None:
        updates["topic"] = _bounded_text(topic, 1024)
    if nsfw is not None:
        updates["nsfw"] = bool(nsfw)
    if slowmode_delay is not None:
        updates["slowmode_delay"] = _clamp(slowmode_delay, 0, 21600)
    if not updates:
        raise DiscordToolError("No editable channel fields were provided.")
    edited = await _call_discord(channel.edit, **updates, reason=_bounded_optional(reason, 512))
    await _audit_action(runtime, "discord_edit_channel", channel_id, {"fields": sorted(updates), "reason": reason})
    return {"edited": True, "channel": _serialize_channel(edited or channel)}


async def discord_delete_channel(channel_id: int, reason: str | None = None, confirm: bool = False) -> dict[str, Any]:
    """Owner-only: delete a channel."""
    runtime = _runtime()
    _require_owner(runtime, "delete channel")
    _require_confirm(runtime, confirm, "delete_channel")
    channel = await _resolve_channel(runtime, channel_id)
    await _require_bot_guild_permission(runtime, "manage_channels")
    await _call_discord(channel.delete, reason=_bounded_optional(reason, 512))
    await _audit_action(runtime, "discord_delete_channel", channel_id, {"reason": reason})
    return {"deleted": True, "channel_id": channel_id}


async def discord_set_channel_overwrite(
    channel_id: int,
    target_id: int,
    target_type: str,
    allow: list[str] | None = None,
    deny: list[str] | None = None,
    reason: str | None = None,
    confirm: bool = False,
) -> dict[str, Any]:
    """Owner-only: set a channel permission overwrite from allow and deny permission-name lists."""
    runtime = _runtime()
    _require_owner(runtime, "set channel overwrite")
    _require_confirm(runtime, confirm, "set_channel_overwrite")
    guild = _require_guild(runtime)
    channel = await _resolve_channel(runtime, channel_id)
    await _require_bot_guild_permission(runtime, "manage_channels")
    target = await _resolve_overwrite_target(guild, target_id, target_type)
    overwrite = _permission_overwrite_from_lists(allow or [], deny or [])
    await _call_discord(channel.set_permissions, target, overwrite=overwrite, reason=_bounded_optional(reason, 512))
    await _audit_action(runtime, "discord_set_channel_overwrite", channel_id, {"target_id": target_id, "target_type": target_type})
    return {
        "overwrite_set": True,
        "channel_id": channel_id,
        "target_id": target_id,
        "target_type": target_type,
        "allow": allow or [],
        "deny": deny or [],
    }


async def discord_block_user_from_channel(
    user_id: int,
    channel_id: int | None = None,
    hide_channel: bool = True,
    block_text: bool = True,
    block_voice: bool = True,
    reason: str | None = None,
    confirm: bool = False,
) -> dict[str, Any]:
    """Owner-only: deny a member from viewing, posting in, or joining a channel."""
    runtime = _runtime()
    _require_owner(runtime, "block user from channel")
    _require_confirm(runtime, confirm, "block_user_from_channel")
    channel = await _resolve_channel(runtime, channel_id)
    await _require_channel_permissions(runtime, channel, "block user from channel", "manage_channels")
    guild = _guild_for_channel(runtime, channel)
    if guild is None:
        raise DiscordToolError("Could not resolve the channel guild.")
    target = await _resolve_member(guild, user_id)
    deny_permissions: list[str] = []
    if hide_channel:
        deny_permissions.append("view_channel")
    if block_text:
        deny_permissions.extend(CHANNEL_BLOCK_TEXT_DENIES)
    if block_voice:
        deny_permissions.extend(CHANNEL_BLOCK_VOICE_DENIES)
    deny_permissions = _supported_overwrite_permissions(deny_permissions)
    if not deny_permissions:
        raise DiscordToolError("No supported channel block permissions are available in this discord.py build.")
    overwrite = _existing_overwrite(channel, target)
    for permission in deny_permissions:
        setattr(overwrite, permission, False)
    await _call_discord(channel.set_permissions, target, overwrite=overwrite, reason=_bounded_optional(reason, 512))
    await _audit_action(runtime, "discord_block_user_from_channel", _id(channel), {"user_id": user_id, "deny": deny_permissions, "reason": reason})
    return {
        "blocked": True,
        "channel_id": _id(channel),
        "user_id": int(user_id),
        "deny": deny_permissions,
    }


async def discord_unblock_user_from_channel(
    user_id: int,
    channel_id: int | None = None,
    restore_view: bool = True,
    restore_text: bool = True,
    restore_voice: bool = True,
    reason: str | None = None,
    confirm: bool = False,
) -> dict[str, Any]:
    """Owner-only: clear this bot's channel block denies for a member."""
    runtime = _runtime()
    _require_owner(runtime, "unblock user from channel")
    _require_confirm(runtime, confirm, "unblock_user_from_channel")
    channel = await _resolve_channel(runtime, channel_id)
    await _require_channel_permissions(runtime, channel, "unblock user from channel", "manage_channels")
    guild = _guild_for_channel(runtime, channel)
    if guild is None:
        raise DiscordToolError("Could not resolve the channel guild.")
    target = await _resolve_member(guild, user_id)
    clear_permissions: list[str] = []
    if restore_view:
        clear_permissions.append("view_channel")
    if restore_text:
        clear_permissions.extend(CHANNEL_BLOCK_TEXT_DENIES)
    if restore_voice:
        clear_permissions.extend(CHANNEL_BLOCK_VOICE_DENIES)
    clear_permissions = _supported_overwrite_permissions(clear_permissions)
    overwrite = _existing_overwrite(channel, target)
    for permission in clear_permissions:
        setattr(overwrite, permission, None)
    stored_overwrite = None if _overwrite_is_empty(overwrite) else overwrite
    await _call_discord(channel.set_permissions, target, overwrite=stored_overwrite, reason=_bounded_optional(reason, 512))
    await _audit_action(runtime, "discord_unblock_user_from_channel", _id(channel), {"user_id": user_id, "cleared": clear_permissions, "reason": reason})
    return {
        "unblocked": True,
        "channel_id": _id(channel),
        "user_id": int(user_id),
        "cleared": clear_permissions,
    }


async def discord_create_role(
    name: str,
    permissions: list[str] | None = None,
    color: int | None = None,
    hoist: bool = False,
    mentionable: bool = False,
    reason: str | None = None,
    confirm: bool = False,
) -> dict[str, Any]:
    """Owner-only: create a role with optional named Discord permissions."""
    runtime = _runtime()
    _require_owner(runtime, "create role")
    _require_confirm(runtime, confirm, "create_role")
    guild = _require_guild(runtime)
    await _require_bot_guild_permission(runtime, "manage_roles")
    role = await _call_discord(
        guild.create_role,
        name=_bounded_text(name, 100),
        permissions=_permissions_from_names(permissions or []),
        colour=discord.Colour(int(color)) if color is not None else None,
        hoist=bool(hoist),
        mentionable=bool(mentionable),
        reason=_bounded_optional(reason, 512),
    )
    await _audit_action(runtime, "discord_create_role", _id(role), {"permissions": permissions or [], "reason": reason})
    return {"created": True, "role": _serialize_role(role)}


async def discord_edit_role(
    role_id: int,
    name: str | None = None,
    permissions: list[str] | None = None,
    color: int | None = None,
    hoist: bool | None = None,
    mentionable: bool | None = None,
    reason: str | None = None,
    confirm: bool = False,
) -> dict[str, Any]:
    """Owner-only: edit a role."""
    runtime = _runtime()
    _require_owner(runtime, "edit role")
    _require_confirm(runtime, confirm, "edit_role")
    guild = _require_guild(runtime)
    await _require_bot_guild_permission(runtime, "manage_roles")
    role = _resolve_role(guild, role_id)
    _require_role_hierarchy(runtime, role, require_actor=False)
    updates: dict[str, Any] = {}
    if name is not None:
        updates["name"] = _bounded_text(name, 100)
    if permissions is not None:
        updates["permissions"] = _permissions_from_names(permissions)
    if color is not None:
        updates["colour"] = discord.Colour(int(color))
    if hoist is not None:
        updates["hoist"] = bool(hoist)
    if mentionable is not None:
        updates["mentionable"] = bool(mentionable)
    if not updates:
        raise DiscordToolError("No editable role fields were provided.")
    edited = await _call_discord(role.edit, **updates, reason=_bounded_optional(reason, 512))
    await _audit_action(runtime, "discord_edit_role", role_id, {"fields": sorted(updates), "reason": reason})
    return {"edited": True, "role": _serialize_role(edited or role)}


async def discord_delete_role(role_id: int, reason: str | None = None, confirm: bool = False) -> dict[str, Any]:
    """Owner-only: delete a role."""
    runtime = _runtime()
    _require_owner(runtime, "delete role")
    _require_confirm(runtime, confirm, "delete_role")
    guild = _require_guild(runtime)
    await _require_bot_guild_permission(runtime, "manage_roles")
    role = _resolve_role(guild, role_id)
    _require_role_hierarchy(runtime, role, require_actor=False)
    await _call_discord(role.delete, reason=_bounded_optional(reason, 512))
    await _audit_action(runtime, "discord_delete_role", role_id, {"reason": reason})
    return {"deleted": True, "role_id": role_id}


async def discord_create_invite(
    channel_id: int | None = None,
    max_age: int = 86400,
    max_uses: int = 1,
    temporary: bool = False,
    unique: bool = True,
    reason: str | None = None,
    confirm: bool = False,
) -> dict[str, Any]:
    """Owner-only: create an invite in a channel where the bot has create_instant_invite."""
    runtime = _runtime()
    _require_owner(runtime, "create invite")
    _require_confirm(runtime, confirm, "create_invite")
    channel = await _resolve_channel(runtime, channel_id)
    await _require_channel_permissions(runtime, channel, "create invite", "view_channel", "create_instant_invite")
    invite = await _call_discord(
        channel.create_invite,
        max_age=_clamp(max_age, 0, 604800),
        max_uses=_clamp(max_uses, 0, 100),
        temporary=bool(temporary),
        unique=bool(unique),
        reason=_bounded_optional(reason, 512),
    )
    await _audit_action(runtime, "discord_create_invite", _id(channel), {"max_age": max_age, "max_uses": max_uses, "reason": reason})
    return {
        "created": True,
        "channel_id": _id(channel),
        "code": getattr(invite, "code", None),
        "url": getattr(invite, "url", str(invite)),
        "max_age": max_age,
        "max_uses": max_uses,
    }


async def discord_list_invites(guild_id: int | None = None, channel_id: int | None = None, limit: int = 100) -> dict[str, Any]:
    """Admin/owner: list guild or channel invites when the bot has manage_guild/create_instant_invite visibility."""
    runtime = _runtime()
    _require_admin_or_owner(runtime, "list invites")
    limit = _clamp(limit, 1, _env_int("DISCORD_BRAIN_TOOL_INVITE_LIMIT", 100))
    if channel_id is not None:
        channel = await _resolve_channel(runtime, channel_id)
        await _require_channel_permissions(runtime, channel, "list invites", "view_channel", "create_instant_invite")
        method = getattr(channel, "invites", None)
        if not callable(method):
            raise DiscordToolError("Channel runtime does not expose invites().")
        invites = await method()
        guild = _guild_for_channel(runtime, channel)
    else:
        guild = await _resolve_guild(runtime, guild_id)
        await _require_bot_guild_permission_for_guild(runtime, guild, "manage_guild")
        method = getattr(guild, "invites", None)
        if not callable(method):
            raise DiscordToolError("Guild runtime does not expose invites().")
        invites = await method()
        channel = None
    return {
        "guild": _serialize_guild(guild) if guild is not None else None,
        "channel": _serialize_channel(channel) if channel is not None else None,
        "invites": [_serialize_invite(invite) for invite in list(invites)[:limit]],
        "count": len(invites),
    }


async def discord_delete_invite(invite_code: str, reason: str | None = None, confirm: bool = False) -> dict[str, Any]:
    """Admin/owner: delete an invite by code after confirmation."""
    runtime = _runtime()
    _require_admin_or_owner(runtime, "delete invite")
    _require_confirm(runtime, confirm, "delete_invite")
    code = invite_code.strip().rsplit("/", 1)[-1]
    if not code:
        raise DiscordToolError("invite_code is required.")
    fetcher = getattr(runtime.bot, "fetch_invite", None)
    if not callable(fetcher):
        raise DiscordToolError("Bot runtime does not expose fetch_invite().")
    invite = await fetcher(code)
    guild = getattr(invite, "guild", None)
    if guild is not None:
        await _require_bot_guild_permission_for_guild(runtime, guild, "manage_guild")
    await _call_discord(invite.delete, reason=_bounded_optional(reason, 512))
    await _audit_action(runtime, "discord_delete_invite", code, {"reason": reason})
    return {"deleted": True, "code": code, "guild": _serialize_guild(guild) if guild is not None else None}


async def discord_create_poll(
    question: str,
    answers: list[str],
    channel_id: int | None = None,
    duration_hours: int = 24,
    multiple: bool = False,
) -> dict[str, Any]:
    """Create a Discord poll in a channel the requester and bot can send messages in."""
    runtime = _runtime()
    channel = await _resolve_channel(runtime, channel_id)
    await _require_channel_permissions(runtime, channel, "create poll", "view_channel", "send_messages")
    if not hasattr(discord, "Poll"):
        raise DiscordToolError("This discord.py build does not expose Poll support.")
    cleaned_answers = [_bounded_text(answer.strip(), 80) for answer in answers if answer and answer.strip()]
    if len(cleaned_answers) < 2:
        raise DiscordToolError("Polls require at least two answers.")
    poll = discord.Poll(_bounded_text(question.strip(), 300), timedelta(hours=_clamp(duration_hours, 1, 168)), multiple=bool(multiple))
    for answer in cleaned_answers[:10]:
        poll.add_answer(text=answer)
    sent = await _call_discord(channel.send, poll=poll)
    await _audit_action(runtime, "discord_create_poll", _id(sent), {"channel_id": _id(channel), "answers": len(cleaned_answers[:10])})
    return {"created": True, "message": _serialize_message(sent), "poll": _serialize_poll(getattr(sent, "poll", poll))}


async def discord_end_poll(channel_id: int, message_id: int, confirm: bool = False) -> dict[str, Any]:
    """End a poll message after confirmation."""
    runtime = _runtime()
    _require_confirm(runtime, confirm, "end_poll")
    channel = await _resolve_channel(runtime, channel_id)
    await _require_channel_permissions(runtime, channel, "end poll", "view_channel", "send_messages", "read_message_history")
    message = await channel.fetch_message(message_id)
    poll = getattr(message, "poll", None)
    if poll is None:
        raise DiscordToolError("Message does not contain a poll.")
    if hasattr(message, "end_poll"):
        updated = await _call_discord(message.end_poll)
    else:
        poll.end()
        updated = await _call_discord(message.edit, poll=poll)
    await _audit_action(runtime, "discord_end_poll", message_id, {"channel_id": channel_id})
    return {"ended": True, "message": _serialize_message(updated or message), "poll": _serialize_poll(getattr(updated or message, "poll", poll))}


async def discord_list_emojis_stickers(guild_id: int | None = None, limit: int = 100) -> dict[str, Any]:
    """List guild emojis and stickers so reactions/content can use real server assets."""
    runtime = _runtime()
    guild = await _resolve_guild(runtime, guild_id)
    limit = _clamp(limit, 1, _env_int("DISCORD_BRAIN_TOOL_LIST_LIMIT", 100))
    emojis = list(getattr(guild, "emojis", None) or [])[:limit]
    stickers = list(getattr(guild, "stickers", None) or [])[:limit]
    return {
        "guild": _serialize_guild(guild),
        "emojis": [_serialize_emoji(emoji) for emoji in emojis],
        "stickers": [_serialize_sticker(sticker) for sticker in stickers],
        "emoji_count": len(getattr(guild, "emojis", None) or []),
        "sticker_count": len(getattr(guild, "stickers", None) or []),
    }


async def discord_create_webhook(channel_id: int, name: str, reason: str | None = None, confirm: bool = False) -> dict[str, Any]:
    """Owner-only: create a webhook. The returned payload does not include the webhook token."""
    runtime = _runtime()
    _require_owner(runtime, "create webhook")
    _require_confirm(runtime, confirm, "create_webhook")
    channel = await _resolve_channel(runtime, channel_id)
    await _require_channel_permissions(runtime, channel, "create webhook", "view_channel", "manage_webhooks")
    webhook = await _call_discord(channel.create_webhook, name=_bounded_text(name, 80), reason=_bounded_optional(reason, 512))
    await _audit_action(runtime, "discord_create_webhook", _id(webhook), {"channel_id": channel_id, "reason": reason})
    return {"created": True, "webhook": _serialize_webhook(webhook)}


async def discord_send_webhook(
    webhook_id: int,
    content: str,
    channel_id: int | None = None,
    username: str | None = None,
    avatar_url: str | None = None,
    confirm: bool = False,
) -> dict[str, Any]:
    """Owner-only: send through a bot-visible webhook without exposing webhook tokens."""
    runtime = _runtime()
    _require_owner(runtime, "send webhook")
    _require_confirm(runtime, confirm, "send_webhook")
    webhook = await _resolve_webhook(runtime, webhook_id, channel_id)
    message = await _call_discord(
        webhook.send,
        _bounded_text(content, _env_int("DISCORD_BRAIN_TOOL_SEND_MAX_CHARS", 1900)),
        username=_bounded_optional(username, 80),
        avatar_url=_bounded_optional(avatar_url, 512),
        wait=True,
        allowed_mentions=discord.AllowedMentions(users=False, roles=False, everyone=False),
    )
    await _audit_action(runtime, "discord_send_webhook", webhook_id, {"channel_id": channel_id})
    return {"sent": True, "webhook": _serialize_webhook(webhook), "message": _serialize_message(message) if message is not None else None}


async def discord_delete_webhook(webhook_id: int, channel_id: int | None = None, reason: str | None = None, confirm: bool = False) -> dict[str, Any]:
    """Owner-only: delete a webhook by id without exposing webhook tokens."""
    runtime = _runtime()
    _require_owner(runtime, "delete webhook")
    _require_confirm(runtime, confirm, "delete_webhook")
    webhook = await _resolve_webhook(runtime, webhook_id, channel_id)
    await _call_discord(webhook.delete, reason=_bounded_optional(reason, 512))
    await _audit_action(runtime, "discord_delete_webhook", webhook_id, {"channel_id": channel_id, "reason": reason})
    return {"deleted": True, "webhook_id": webhook_id}


def register_discord_tools(registry: Any) -> None:
    timeout = _env_int("DISCORD_BRAIN_TOOL_TIMEOUT", 20)
    for name in DISCORD_AGENT_TOOL_NAMES:
        registry.register(globals()[name], name=name, timeout_seconds=timeout)


def _runtime() -> DiscordToolRuntime:
    runtime = DISCORD_TOOL_CONTEXT.get()
    if runtime is None:
        raise DiscordToolError("Discord tools are only available during a Discord message turn.")
    return runtime


def _require_guild(runtime: DiscordToolRuntime) -> Any:
    guild = getattr(runtime.message, "guild", None)
    if guild is None:
        raise DiscordToolError("This Discord tool requires a guild message context.")
    return guild


async def _delete_messages(channel: Any, messages: list[Any], reason: str | None) -> None:
    if not messages:
        return
    if hasattr(channel, "delete_messages") and len(messages) > 1:
        await _call_discord(channel.delete_messages, messages, reason=_bounded_optional(reason, 512))
        return
    for message in messages:
        await _call_discord(message.delete, reason=_bounded_optional(reason, 512))


def _permission_overwrite_from_lists(allow: list[str], deny: list[str]) -> discord.PermissionOverwrite:
    values: dict[str, bool] = {}
    for name in allow:
        values[_normalize_overwrite_permission_name(name)] = True
    for name in deny:
        values[_normalize_overwrite_permission_name(name)] = False
    return discord.PermissionOverwrite(**values)


def _supported_overwrite_permissions(names: list[str]) -> list[str]:
    supported: list[str] = []
    seen: set[str] = set()
    probe = discord.PermissionOverwrite()
    for name in names:
        normalized = str(name).strip().lower()
        if not normalized or normalized in seen:
            continue
        if hasattr(probe, normalized):
            supported.append(normalized)
            seen.add(normalized)
    return supported


def _normalize_overwrite_permission_name(name: str) -> str:
    normalized = str(name).strip().lower()
    if not normalized:
        raise DiscordToolError("Permission names cannot be empty.")
    if not hasattr(discord.PermissionOverwrite(), normalized):
        raise DiscordToolError(f"Unknown channel overwrite permission: {normalized}")
    return normalized


def _existing_overwrite(channel: Any, target: Any) -> discord.PermissionOverwrite:
    overwrites_for = getattr(channel, "overwrites_for", None)
    if callable(overwrites_for):
        with contextlib.suppress(Exception):
            overwrite = overwrites_for(target)
            if overwrite is not None:
                return overwrite
    overwrites = getattr(channel, "overwrites", None) or {}
    with contextlib.suppress(Exception):
        overwrite = overwrites.get(target)
        if overwrite is not None:
            return overwrite
    return discord.PermissionOverwrite()


def _overwrite_is_empty(overwrite: Any) -> bool:
    checker = getattr(overwrite, "is_empty", None)
    if callable(checker):
        with contextlib.suppress(Exception):
            return bool(checker())
    with contextlib.suppress(Exception):
        return all(value is None for _, value in overwrite)
    return False


async def _resolve_guild(runtime: DiscordToolRuntime, guild_id: int | None) -> Any:
    current = getattr(runtime.message, "guild", None)
    if guild_id is None:
        if current is None:
            raise DiscordToolError("This Discord tool requires a guild message context.")
        return current
    guild_id = int(guild_id)
    if current is not None and _id(current) == guild_id:
        return current
    _require_owner(runtime, "inspect another guild")
    getter = getattr(runtime.bot, "get_guild", None)
    if callable(getter):
        guild = getter(guild_id)
        if guild is not None:
            return guild
    for guild in getattr(runtime.bot, "guilds", None) or []:
        if _id(guild) == guild_id:
            return guild
    fetcher = getattr(runtime.bot, "fetch_guild", None)
    if callable(fetcher):
        return await fetcher(guild_id)
    raise DiscordToolError(f"Guild {guild_id} was not found.")


async def _resolve_channel(runtime: DiscordToolRuntime, channel_id: int | None) -> Any:
    current = getattr(runtime.message, "channel", None)
    if channel_id is None:
        channel = current
    else:
        channel = await _lookup_channel(runtime, int(channel_id))
    if channel is None:
        raise DiscordToolError(f"Channel {channel_id} was not found.")
    if not _channel_is_allowed(channel):
        raise DiscordToolError(f"Channel {_id(channel)} is outside DISCORD_BRAIN_TOOL_CHANNEL_IDS.")
    return channel


async def _lookup_channel(runtime: DiscordToolRuntime, channel_id: int) -> Any:
    current = getattr(runtime.message, "channel", None)
    if current is not None and _id(current) == channel_id:
        return current
    guild = getattr(runtime.message, "guild", None)
    bot = getattr(runtime, "bot", None)
    owners = [guild, bot]
    owners.extend(getattr(bot, "guilds", None) or [])
    for owner in owners:
        if owner is None:
            continue
        for method_name in ("get_channel_or_thread", "get_thread", "get_channel"):
            method = getattr(owner, method_name, None)
            if callable(method):
                channel = method(channel_id)
                if channel is not None:
                    return channel
    for owner in [guild, *(getattr(bot, "guilds", None) or [])]:
        if owner is not None and hasattr(owner, "fetch_channel"):
            with contextlib.suppress(Exception):
                return await owner.fetch_channel(channel_id)
    if bot is not None and hasattr(bot, "fetch_channel"):
        with contextlib.suppress(Exception):
            return await bot.fetch_channel(channel_id)
    return None


def _iter_guild_channels(guild: Any, *, include_threads: bool) -> list[Any]:
    channels = list(getattr(guild, "channels", None) or [])
    if include_threads:
        channels.extend(getattr(guild, "threads", None) or [])
    return channels


def _iter_guild_members(guild: Any) -> list[Any]:
    members = getattr(guild, "members", None)
    if isinstance(members, dict):
        return list(members.values())
    return list(members or [])


async def _actor_member(runtime: DiscordToolRuntime) -> Any:
    guild = getattr(runtime.message, "guild", None)
    return await _actor_member_for_guild(runtime, guild)


async def _actor_member_for_guild(runtime: DiscordToolRuntime, guild: Any) -> Any:
    author = getattr(runtime.message, "author", None)
    if guild is None or author is None:
        return author
    message_guild = getattr(runtime.message, "guild", None)
    if hasattr(author, "guild_permissions") and _id(message_guild) == _id(guild):
        return author
    author_id = _id(author)
    if author_id is None:
        return author
    with contextlib.suppress(DiscordToolError, KeyError, LookupError):
        return await _resolve_member(guild, author_id)
    return author


async def _bot_member(runtime: DiscordToolRuntime) -> Any:
    guild = getattr(runtime.message, "guild", None)
    return await _bot_member_for_guild(runtime, guild)


async def _bot_member_for_guild(runtime: DiscordToolRuntime, guild: Any) -> Any:
    if guild is None:
        return getattr(runtime.bot, "user", None)
    member = getattr(guild, "me", None)
    if member is not None:
        return member
    bot_user = getattr(runtime.bot, "user", None)
    bot_id = _id(bot_user)
    if bot_id is not None:
        getter = getattr(guild, "get_member", None)
        if callable(getter):
            member = getter(bot_id)
            if member is not None:
                return member
        fetcher = getattr(guild, "fetch_member", None)
        if callable(fetcher):
            with contextlib.suppress(Exception):
                return await fetcher(bot_id)
    return bot_user


def _guild_for_channel(runtime: DiscordToolRuntime, channel: Any) -> Any:
    if channel is None:
        return None
    guild = getattr(channel, "guild", None)
    if guild is not None:
        return guild
    channel_id = _id(channel)
    current = getattr(runtime.message, "guild", None)
    if current is not None and any(_id(item) == channel_id for item in _iter_guild_channels(current, include_threads=True)):
        return current
    for item in getattr(getattr(runtime, "bot", None), "guilds", None) or []:
        if any(_id(guild_channel) == channel_id for guild_channel in _iter_guild_channels(item, include_threads=True)):
            return item
    guild_id = getattr(channel, "guild_id", None)
    if guild_id is not None:
        for item in [current, *(getattr(getattr(runtime, "bot", None), "guilds", None) or [])]:
            if _id(item) == int(guild_id):
                return item
    return None


async def _resolve_member(guild: Any, user_id: int) -> Any:
    getter = getattr(guild, "get_member", None)
    if callable(getter):
        member = getter(int(user_id))
        if member is not None:
            return member
    fetcher = getattr(guild, "fetch_member", None)
    if callable(fetcher):
        return await fetcher(int(user_id))
    raise DiscordToolError(f"Member {user_id} was not found in this guild.")


def _resolve_role(guild: Any, role_id: int) -> Any:
    getter = getattr(guild, "get_role", None)
    if callable(getter):
        role = getter(int(role_id))
        if role is not None:
            return role
    for role in getattr(guild, "roles", None) or []:
        if _id(role) == int(role_id):
            return role
    raise DiscordToolError(f"Role {role_id} was not found in this guild.")


def _resolve_category(guild: Any, category_id: int) -> Any:
    channel = None
    getter = getattr(guild, "get_channel", None)
    if callable(getter):
        channel = getter(int(category_id))
    if channel is None:
        for candidate in getattr(guild, "categories", None) or []:
            if _id(candidate) == int(category_id):
                channel = candidate
                break
    if channel is None:
        raise DiscordToolError(f"Category {category_id} was not found.")
    return channel


async def _resolve_overwrite_target(guild: Any, target_id: int, target_type: str) -> Any:
    normalized = target_type.strip().lower()
    if normalized in {"everyone", "@everyone"}:
        return getattr(guild, "default_role", None) or _resolve_role(guild, target_id)
    if normalized == "role":
        return _resolve_role(guild, target_id)
    if normalized in {"member", "user"}:
        return await _resolve_member(guild, target_id)
    raise DiscordToolError("target_type must be role, member, user, or everyone.")


async def _resolve_webhook(runtime: DiscordToolRuntime, webhook_id: int, channel_id: int | None) -> Any:
    if channel_id is not None:
        channel = await _resolve_channel(runtime, channel_id)
        await _require_channel_permissions(runtime, channel, "manage webhooks", "view_channel", "manage_webhooks")
        webhooks_method = getattr(channel, "webhooks", None)
        if callable(webhooks_method):
            for webhook in await webhooks_method():
                if _id(webhook) == int(webhook_id):
                    return webhook
    fetcher = getattr(runtime.bot, "fetch_webhook", None)
    if callable(fetcher):
        return await fetcher(int(webhook_id))
    raise DiscordToolError(f"Webhook {webhook_id} was not found.")


async def _fetch_user(runtime: DiscordToolRuntime, user_id: int) -> Any:
    fetcher = getattr(runtime.bot, "fetch_user", None)
    if callable(fetcher):
        with contextlib.suppress(Exception):
            return await fetcher(int(user_id))
    return discord.Object(id=int(user_id))


async def _require_channel_permissions(
    runtime: DiscordToolRuntime,
    channel: Any,
    action: str,
    *permissions: str,
) -> None:
    guild = _guild_for_channel(runtime, channel) or getattr(runtime.message, "guild", None)
    current = getattr(runtime.message, "channel", None)
    if getattr(runtime.message, "guild", None) is None and _id(current) != _id(channel):
        _require_owner(runtime, action)
        bot_member = await _bot_member_for_guild(runtime, guild)
        bot_perms = _permissions_for(channel, bot_member)
        for permission in permissions:
            if not _perm(bot_perms, permission):
                raise DiscordToolError(f"Bot lacks {permission} to {action} in #{_name(channel)}.")
        return
    if guild is None:
        return
    actor = await _actor_member_for_guild(runtime, guild)
    bot_member = await _bot_member_for_guild(runtime, guild)
    actor_perms = _permissions_for(channel, actor)
    bot_perms = _permissions_for(channel, bot_member)
    owner = _is_owner(runtime)
    for permission in permissions:
        if not owner and not _perm(actor_perms, permission):
            raise DiscordToolError(f"Current user lacks {permission} to {action} in #{_name(channel)}.")
        if not _perm(bot_perms, permission):
            raise DiscordToolError(f"Bot lacks {permission} to {action} in #{_name(channel)}.")


async def _require_bot_guild_permission(runtime: DiscordToolRuntime, permission: str) -> None:
    guild = getattr(runtime.message, "guild", None)
    await _require_bot_guild_permission_for_guild(runtime, guild, permission)


async def _require_bot_guild_permission_for_guild(runtime: DiscordToolRuntime, guild: Any, permission: str) -> None:
    bot_member = await _bot_member_for_guild(runtime, guild)
    bot_perms = getattr(bot_member, "guild_permissions", None)
    if not _perm(bot_perms, permission):
        raise DiscordToolError(f"Bot lacks {permission}.")


async def _require_moderation(
    runtime: DiscordToolRuntime,
    permission: str,
    target_user_id: int,
    *,
    require_target_member: bool = True,
) -> None:
    if _is_autonomous_neuro(runtime):
        raise DiscordToolError("moderation requires an explicit Discord admin or bot owner turn.")
    if not _env_bool("DISCORD_BRAIN_MODERATION_TOOLS_ENABLED", True):
        raise DiscordToolError("Discord moderation tools are disabled.")
    allowed_users = _csv_ints("DISCORD_BRAIN_MODERATION_ALLOWED_USER_IDS")
    actor = await _actor_member(runtime)
    owner = _is_owner(runtime)
    if allowed_users and _id(actor) not in allowed_users and not owner:
        raise DiscordToolError("Current user is not in DISCORD_BRAIN_MODERATION_ALLOWED_USER_IDS.")
    if require_target_member and _id(actor) == int(target_user_id):
        raise DiscordToolError("Refusing to moderate the invoking user.")
    bot_member = await _bot_member(runtime)
    actor_perms = getattr(actor, "guild_permissions", None)
    bot_perms = getattr(bot_member, "guild_permissions", None)
    if not owner and not _perm(actor_perms, permission):
        raise DiscordToolError(f"Current user lacks {permission}.")
    if not _perm(bot_perms, permission):
        raise DiscordToolError(f"Bot lacks {permission}.")


async def _require_role_management(runtime: DiscordToolRuntime, target_user_id: int, role_id: int) -> None:
    if _is_autonomous_neuro(runtime):
        raise DiscordToolError("role management requires an explicit Discord admin or bot owner turn.")
    guild = _require_guild(runtime)
    role = _resolve_role(guild, role_id)
    actor = await _actor_member(runtime)
    bot_member = await _bot_member(runtime)
    owner = _is_owner(runtime)
    if not owner and not _perm(getattr(actor, "guild_permissions", None), "manage_roles"):
        raise DiscordToolError("Current user lacks manage_roles.")
    if not _perm(getattr(bot_member, "guild_permissions", None), "manage_roles"):
        raise DiscordToolError("Bot lacks manage_roles.")
    _require_role_hierarchy(runtime, role, require_actor=not owner)
    target = await _resolve_member(guild, target_user_id)
    _require_member_hierarchy(runtime, target, require_actor=not owner)


def _require_owner(runtime: DiscordToolRuntime, action: str) -> None:
    if not _is_owner(runtime):
        raise DiscordToolError(f"{action} requires the configured bot owner.")


def _require_admin_or_owner(runtime: DiscordToolRuntime, action: str) -> None:
    if _is_autonomous_neuro(runtime):
        raise DiscordToolError(f"{action} requires an explicit Discord admin or bot owner turn.")
    actor = getattr(runtime.message, "author", None)
    if _is_owner(runtime) or _is_admin(actor):
        return
    raise DiscordToolError(f"{action} requires a Discord admin or bot owner.")


def _require_member_intent(runtime: DiscordToolRuntime) -> None:
    intents = getattr(runtime.bot, "intents", None)
    if intents is not None and not bool(getattr(intents, "members", False)):
        raise DiscordToolError("Member list/search tools require DISCORD_BRAIN_MEMBERS_INTENT=true and the Discord Developer Portal Members Intent.")


def _require_member_hierarchy(runtime: DiscordToolRuntime, target: Any, *, require_actor: bool) -> None:
    guild = getattr(runtime.message, "guild", None)
    actor = getattr(runtime.message, "author", None)
    bot_member = getattr(guild, "me", None) if guild is not None else None
    owner_id = getattr(guild, "owner_id", None)
    target_id = _id(target)
    if owner_id is not None and target_id == owner_id:
        raise DiscordToolError("Refusing to moderate the guild owner.")
    if require_actor and actor is not None and owner_id is not None and _id(actor) != owner_id and not _role_higher(actor, target):
        raise DiscordToolError("Current user's top role is not higher than the target member.")
    if bot_member is not None and owner_id is not None and _id(bot_member) != owner_id and not _role_higher(bot_member, target):
        raise DiscordToolError("Bot's top role is not higher than the target member.")


def _require_role_hierarchy(runtime: DiscordToolRuntime, role: Any, *, require_actor: bool) -> None:
    guild = getattr(runtime.message, "guild", None)
    actor = getattr(runtime.message, "author", None)
    bot_member = getattr(guild, "me", None) if guild is not None else None
    if require_actor and actor is not None and not _role_higher(actor, role):
        raise DiscordToolError("Current user's top role is not higher than the target role.")
    if bot_member is not None and not _role_higher(bot_member, role):
        raise DiscordToolError("Bot's top role is not higher than the target role.")


def _role_higher(actor: Any, target: Any) -> bool:
    actor_role = getattr(actor, "top_role", actor)
    target_role = getattr(target, "top_role", target)
    if actor_role is None or target_role is None:
        return True
    with contextlib.suppress(Exception):
        return actor_role > target_role
    actor_pos = getattr(actor_role, "position", 0)
    target_pos = getattr(target_role, "position", 0)
    return actor_pos > target_pos


def _require_confirm(runtime: DiscordToolRuntime, confirm: bool, action: str) -> None:
    if _is_owner(runtime):
        return
    if _env_bool("DISCORD_BRAIN_MODERATION_REQUIRE_CONFIRM", True) and not confirm:
        raise DiscordToolError(f"{action} requires confirm=true.")


def _is_owner(runtime: DiscordToolRuntime) -> bool:
    if _is_autonomous_neuro(runtime):
        return False
    actor = getattr(runtime.message, "author", None)
    actor_id = _id(actor)
    return actor_id is not None and int(actor_id) in _owner_user_ids()


def _is_autonomous_neuro(runtime: DiscordToolRuntime) -> bool:
    return str(getattr(runtime, "authority_mode", "") or "").strip().lower() == "autonomous_neuro"


def _require_shitlist_authority(runtime: DiscordToolRuntime, action: str) -> None:
    if _is_owner(runtime):
        return
    if _is_autonomous_neuro(runtime):
        if _env_bool("DISCORD_BRAIN_SHITLIST_AUTONOMY_ENABLED", True):
            return
        raise DiscordToolError(f"{action} by autonomous Neuro is disabled.")
    raise DiscordToolError(f"{action} requires the configured bot owner or autonomous Neuro authority.")


def _require_not_bot_self(runtime: DiscordToolRuntime, user_id: int | str) -> None:
    target_id = int(user_id)
    bot_user_id = _id(getattr(runtime.bot, "user", None))
    if bot_user_id is not None and target_id == int(bot_user_id):
        raise DiscordToolError("Refusing to add the bot itself to the shitlist.")


def _is_admin(actor: Any) -> bool:
    permissions = getattr(actor, "guild_permissions", None)
    return bool(getattr(permissions, "administrator", False))


def _owner_user_ids() -> set[int]:
    return _csv_ints("DISCORD_BRAIN_V2_OWNER_USER_IDS") | _csv_ints("DISCORD_BRAIN_OWNER_USER_IDS") | set(DEFAULT_OWNER_USER_IDS)


async def _audit_action(runtime: DiscordToolRuntime, tool: str, target_id: Any, details: dict[str, Any] | None = None) -> None:
    channel_id = _env_int("DISCORD_BRAIN_TOOL_AUDIT_CHANNEL_ID", 0)
    if channel_id <= 0:
        return
    channel = await _lookup_channel(runtime, channel_id)
    if channel is None:
        return
    actor = getattr(runtime.message, "author", None)
    content = _bounded_text(
        f"[discord-tool] {tool} actor={_id(actor)} target={target_id} details={_sanitize_audit_details(details or {})}",
        1800,
    )
    with contextlib.suppress(Exception):
        await channel.send(content, allowed_mentions=discord.AllowedMentions(users=False, roles=False, everyone=False))


def _sanitize_audit_details(details: dict[str, Any]) -> dict[str, Any]:
    blocked = {"token", "authorization", "webhook_token", "password", "secret", "api_key"}
    clean: dict[str, Any] = {}
    for key, value in details.items():
        if key.lower() in blocked:
            clean[key] = "[redacted]"
        elif isinstance(value, str):
            clean[key] = _bounded_text(value, 200)
        else:
            clean[key] = value
    return clean


def _permissions_for(channel: Any, member: Any) -> Any:
    permissions_for = getattr(channel, "permissions_for", None)
    if callable(permissions_for):
        with contextlib.suppress(Exception):
            return permissions_for(member)
    return getattr(member, "guild_permissions", None)


def _action_requirements(action: str) -> list[str]:
    requirements = {
        "read": ["view_channel", "read_message_history"],
        "read_history": ["view_channel", "read_message_history"],
        "send": ["view_channel", "send_messages"],
        "send_message": ["view_channel", "send_messages"],
        "send_file": ["view_channel", "send_messages", "attach_files"],
        "react": ["view_channel", "read_message_history", "add_reactions"],
        "create_thread": ["view_channel", "create_public_threads"],
        "manage_messages": ["view_channel", "read_message_history", "manage_messages"],
        "manage_threads": ["view_channel", "manage_threads"],
        "manage_channels": ["manage_channels"],
        "manage_roles": ["manage_roles"],
        "create_invite": ["view_channel", "create_instant_invite"],
        "manage_webhooks": ["view_channel", "manage_webhooks"],
        "audit_log": ["view_audit_log"],
        "list_invites": ["manage_guild"],
        "timeout_member": ["moderate_members"],
        "kick_member": ["kick_members"],
        "ban_member": ["ban_members"],
        "move_voice": ["move_members"],
        "disconnect_voice": ["move_members"],
    }
    if action in requirements:
        return requirements[action]
    if not hasattr(discord.Permissions.none(), action):
        raise DiscordToolError(f"Unknown Discord action or permission: {action}")
    return [action]


def _audit_log_action(action: str) -> Any:
    normalized = action.strip().lower()
    audit_log_action = getattr(discord, "AuditLogAction", None)
    if audit_log_action is None:
        raise DiscordToolError("This discord.py build does not expose AuditLogAction.")
    for name, value in inspect.getmembers(audit_log_action):
        if name.lower() == normalized:
            return value
    raise DiscordToolError(f"Unknown audit log action: {action}")


def _perm(permissions: Any, name: str) -> bool:
    if permissions is None:
        return False
    return bool(getattr(permissions, name, False))


def _channel_is_allowed(channel: Any) -> bool:
    allowed = _csv_ints("DISCORD_BRAIN_TOOL_CHANNEL_IDS")
    return not allowed or _id(channel) in allowed


def _safe_filename(filename: str, *, default: str) -> str:
    value = os.path.basename((filename or default).strip()) or default
    value = re.sub(r"[^A-Za-z0-9._ -]", "_", value).strip(" .")
    if not value:
        value = default
    return value[:120]


def _serialize_guild(guild: Any) -> dict[str, Any]:
    return {
        "id": _id(guild),
        "name": _name(guild),
        "owner_id": getattr(guild, "owner_id", None),
        "member_count": getattr(guild, "member_count", None),
        "channel_count": len(getattr(guild, "channels", None) or []),
        "thread_count": len(getattr(guild, "threads", None) or []),
        "role_count": len(getattr(guild, "roles", None) or []),
        "features": list(getattr(guild, "features", None) or [])[:30],
    }


def _serialize_channel(channel: Any) -> dict[str, Any]:
    parent = getattr(channel, "category", None) or getattr(channel, "parent", None)
    return {
        "id": _id(channel),
        "name": _name(channel),
        "type": type(channel).__name__,
        "mention": getattr(channel, "mention", None),
        "parent_id": _id(parent),
        "parent_name": _name(parent) if parent is not None else None,
        "position": getattr(channel, "position", None),
        "topic": getattr(channel, "topic", None),
        "nsfw": getattr(channel, "nsfw", None),
    }


def _serialize_message(message: Any) -> dict[str, Any]:
    author = getattr(message, "author", None)
    channel = getattr(message, "channel", None)
    attachments = []
    for attachment in list(getattr(message, "attachments", None) or [])[:10]:
        attachments.append(
            {
                "filename": getattr(attachment, "filename", None),
                "content_type": getattr(attachment, "content_type", None),
                "size": getattr(attachment, "size", None),
                "url": getattr(attachment, "url", None),
            }
        )
    created_at = getattr(message, "created_at", None)
    return {
        "id": _id(message),
        "channel_id": _id(channel) or getattr(message, "channel_id", None),
        "author_id": _id(author),
        "author": _name(author) or str(author) if author is not None else None,
        "author_bot": bool(getattr(author, "bot", False)),
        "content": _bounded_text(getattr(message, "clean_content", None) or getattr(message, "content", "") or "", 4000),
        "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else None,
        "jump_url": getattr(message, "jump_url", None),
        "attachments": attachments,
    }


def _serialize_embed(embed: discord.Embed) -> dict[str, Any]:
    return {
        "title": embed.title,
        "description": embed.description,
        "url": embed.url,
        "color": embed.color.value if embed.color is not None else None,
        "fields": [{"name": field.name, "value": field.value, "inline": field.inline} for field in embed.fields],
        "footer": getattr(embed.footer, "text", None),
        "author": getattr(embed.author, "name", None),
        "image_url": getattr(embed.image, "url", None),
        "thumbnail_url": getattr(embed.thumbnail, "url", None),
    }


def _serialize_user(user: Any) -> dict[str, Any] | None:
    if user is None:
        return None
    return {
        "id": _id(user),
        "name": _name(user) or str(user),
        "bot": bool(getattr(user, "bot", False)),
        "mention": getattr(user, "mention", None),
    }


def _serialize_member(member: Any) -> dict[str, Any]:
    roles = [_serialize_role(role) for role in list(getattr(member, "roles", None) or [])[:50]]
    joined_at = getattr(member, "joined_at", None)
    return {
        "id": _id(member),
        "name": _name(member) or str(member),
        "display_name": getattr(member, "display_name", None),
        "bot": bool(getattr(member, "bot", False)),
        "roles": roles,
        "joined_at": joined_at.isoformat() if hasattr(joined_at, "isoformat") else None,
        "mention": getattr(member, "mention", None),
    }


def _serialize_role(role: Any) -> dict[str, Any]:
    color = getattr(role, "color", None) or getattr(role, "colour", None)
    return {
        "id": _id(role),
        "name": _name(role),
        "position": getattr(role, "position", None),
        "managed": getattr(role, "managed", None),
        "mentionable": getattr(role, "mentionable", None),
        "hoist": getattr(role, "hoist", None),
        "color": getattr(color, "value", color),
        "permissions": _serialize_permissions(getattr(role, "permissions", None)),
    }


def _serialize_webhook(webhook: Any) -> dict[str, Any]:
    return {
        "id": _id(webhook),
        "name": _name(webhook),
        "channel_id": getattr(webhook, "channel_id", None),
        "guild_id": getattr(webhook, "guild_id", None),
        "user": _serialize_user(getattr(webhook, "user", None)),
    }


def _serialize_invite(invite: Any) -> dict[str, Any]:
    channel = getattr(invite, "channel", None)
    guild = getattr(invite, "guild", None)
    inviter = getattr(invite, "inviter", None)
    expires_at = getattr(invite, "expires_at", None)
    created_at = getattr(invite, "created_at", None)
    return {
        "code": getattr(invite, "code", None),
        "url": getattr(invite, "url", None),
        "guild_id": _id(guild) or getattr(invite, "guild_id", None),
        "channel_id": _id(channel) or getattr(invite, "channel_id", None),
        "channel_name": _name(channel),
        "inviter": _serialize_user(inviter),
        "uses": getattr(invite, "uses", None),
        "max_uses": getattr(invite, "max_uses", None),
        "max_age": getattr(invite, "max_age", None),
        "temporary": getattr(invite, "temporary", None),
        "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else None,
        "expires_at": expires_at.isoformat() if hasattr(expires_at, "isoformat") else None,
    }


def _serialize_audit_log_entry(entry: Any) -> dict[str, Any]:
    created_at = getattr(entry, "created_at", None)
    action = getattr(entry, "action", None)
    return {
        "id": _id(entry),
        "action": getattr(action, "name", str(action)),
        "user": _serialize_user(getattr(entry, "user", None)),
        "target_id": _id(getattr(entry, "target", None)),
        "target": _name(getattr(entry, "target", None)) or str(getattr(entry, "target", ""))[:120],
        "reason": getattr(entry, "reason", None),
        "changes": [_serialize_audit_change(change) for change in list(getattr(entry, "changes", None) or [])[:20]],
        "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else None,
    }


def _serialize_audit_change(change: Any) -> dict[str, Any]:
    return {
        "attribute": getattr(change, "attribute", None),
        "before": _bounded_text(str(getattr(change, "before", None)), 300),
        "after": _bounded_text(str(getattr(change, "after", None)), 300),
    }


def _serialize_overwrite_target(target: Any) -> dict[str, Any]:
    kind = "role" if hasattr(target, "permissions") else "member" if hasattr(target, "guild_permissions") else type(target).__name__
    return {"id": _id(target), "name": _name(target) or str(target), "type": kind}


def _serialize_emoji(emoji: Any) -> dict[str, Any]:
    return {
        "id": _id(emoji),
        "name": _name(emoji),
        "animated": getattr(emoji, "animated", None),
        "available": getattr(emoji, "available", None),
        "managed": getattr(emoji, "managed", None),
        "require_colons": getattr(emoji, "require_colons", None),
        "text": str(emoji),
    }


def _serialize_sticker(sticker: Any) -> dict[str, Any]:
    return {
        "id": _id(sticker),
        "name": _name(sticker),
        "description": getattr(sticker, "description", None),
        "format": str(getattr(sticker, "format", "")),
        "url": getattr(sticker, "url", None),
    }


def _serialize_poll(poll: Any) -> dict[str, Any] | None:
    if poll is None:
        return None
    question = getattr(poll, "question", None)
    answers = []
    for answer in list(getattr(poll, "answers", None) or [])[:20]:
        answers.append(
            {
                "id": getattr(answer, "id", None),
                "text": getattr(getattr(answer, "media", None), "text", None) or getattr(answer, "text", None),
                "votes": getattr(answer, "vote_count", None),
            }
        )
    expires_at = getattr(poll, "expires_at", None)
    finalized = getattr(poll, "is_finalized", None)
    if callable(finalized):
        finalized = finalized()
    return {
        "question": getattr(question, "text", None) or str(question),
        "multiple": getattr(poll, "multiple", None),
        "answers": answers,
        "total_votes": getattr(poll, "total_votes", None),
        "expires_at": expires_at.isoformat() if hasattr(expires_at, "isoformat") else None,
        "finalized": bool(finalized),
    }


def _serialize_permissions(permissions: Any) -> dict[str, bool]:
    if permissions is None:
        return {}
    if hasattr(permissions, "__iter__"):
        with contextlib.suppress(Exception):
            return {str(name): bool(value) for name, value in permissions}
    names = [
        "administrator",
        "view_channel",
        "read_message_history",
        "send_messages",
        "attach_files",
        "manage_messages",
        "manage_channels",
        "manage_roles",
        "manage_guild",
        "manage_threads",
        "manage_webhooks",
        "view_audit_log",
        "create_instant_invite",
        "add_reactions",
        "create_public_threads",
        "moderate_members",
        "kick_members",
        "ban_members",
        "move_members",
        "connect",
        "speak",
        "use_voice_activation",
    ]
    return {name: bool(getattr(permissions, name, False)) for name in names}


def _permissions_from_names(names: list[str]) -> discord.Permissions:
    permissions = discord.Permissions.none()
    for name in names:
        normalized = name.strip()
        if not normalized:
            continue
        if not hasattr(permissions, normalized):
            raise DiscordToolError(f"Unknown Discord permission: {normalized}")
        setattr(permissions, normalized, True)
    return permissions


async def _call_discord(func: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return await func(*args, **kwargs)
    except TypeError:
        filtered = _filter_callable_kwargs(func, kwargs)
        try:
            return await func(*args, **filtered)
        except TypeError:
            return await func(*args)


def _filter_callable_kwargs(func: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    with contextlib.suppress(TypeError, ValueError):
        signature = inspect.signature(func)
        if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
            return kwargs
        return {key: value for key, value in kwargs.items() if key in signature.parameters}
    common = {"reason", "name", "auto_archive_duration", "content", "allowed_mentions", "embed"}
    return {key: value for key, value in kwargs.items() if key in common}


def _build_rich_embed(
    *,
    title: str | None,
    description: str | None,
    fields: list[dict[str, Any]] | None,
    color: int | str | None,
    url: str | None,
    image_url: str | None,
    thumbnail_url: str | None,
    footer_text: str | None,
    author_name: str | None,
    author_url: str | None,
) -> discord.Embed:
    if not any(str(item or "").strip() for item in (title, description, url, image_url, thumbnail_url, footer_text, author_name)) and not fields:
        raise DiscordToolError("At least one embed title, description, field, image, thumbnail, footer, or author is required.")
    embed = discord.Embed(
        title=_bounded_optional(title, 256),
        description=_bounded_optional(description, 4096),
        url=_clean_url(url),
        color=_parse_embed_color(color),
    )
    if author_name:
        embed.set_author(name=_bounded_text(author_name, 256), url=_clean_url(author_url))
    if footer_text:
        embed.set_footer(text=_bounded_text(footer_text, 2048))
    if image_url:
        embed.set_image(url=_clean_url(image_url))
    if thumbnail_url:
        embed.set_thumbnail(url=_clean_url(thumbnail_url))
    for field in list(fields or [])[:25]:
        if not isinstance(field, dict):
            continue
        name = _bounded_text(str(field.get("name") or ""), 256)
        value = _bounded_text(str(field.get("value") or ""), 1024)
        if not name or not value:
            continue
        embed.add_field(name=name, value=value, inline=bool(field.get("inline", False)))
    if len(embed) > 6000:
        raise DiscordToolError("Embed is too large after Discord limits; shorten the description or fields.")
    return embed


def _parse_embed_color(color: int | str | None) -> discord.Color | None:
    if color is None or color == "":
        return None
    if isinstance(color, str):
        raw = color.strip().lower().removeprefix("#").removeprefix("0x")
        try:
            value = int(raw, 16)
        except ValueError as exc:
            raise DiscordToolError(f"Invalid embed color: {color}") from exc
    else:
        value = int(color)
    if not 0 <= value <= 0xFFFFFF:
        raise DiscordToolError("Embed color must be between 0x000000 and 0xFFFFFF.")
    return discord.Color(value)


def _clean_url(value: str | None) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if not (text.startswith("http://") or text.startswith("https://")):
        raise DiscordToolError("Embed URLs must start with http:// or https://.")
    return text


def _bounded_text(text: str, limit: int) -> str:
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 14)].rstrip() + "\n[truncated]"


def _bounded_optional(text: str | None, limit: int) -> str | None:
    if text is None:
        return None
    return _bounded_text(text, limit)


def _shitlist_store(runtime: DiscordToolRuntime) -> DiscordShitlistStore:
    store = getattr(getattr(runtime, "bot", None), "shitlist_store", None)
    if isinstance(store, DiscordShitlistStore):
        return store
    path = Path(os.getenv("DISCORD_BRAIN_SHITLIST_FILE", "discord_brain.shitlist.json"))
    return DiscordShitlistStore(path, owner_user_ids=_owner_user_ids())


def _serialize_shitlist_entry(entry: Any) -> dict[str, Any]:
    return {
        "user_id": int(entry.user_id),
        "reason": str(entry.reason),
        "spice_level": int(entry.spice_level),
        "added_at": str(entry.added_at),
    }


def _id(value: Any) -> int | None:
    raw = getattr(value, "id", None)
    if raw is None:
        return None
    with contextlib.suppress(TypeError, ValueError):
        return int(raw)
    return raw


def _name(value: Any) -> str | None:
    if value is None:
        return None
    return (
        getattr(value, "display_name", None)
        or getattr(value, "name", None)
        or getattr(value, "global_name", None)
        or getattr(value, "username", None)
    )


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    with contextlib.suppress(ValueError):
        return int(value)
    return default


def _csv_ints(name: str) -> set[int]:
    values: set[int] = set()
    for chunk in os.getenv(name, "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        with contextlib.suppress(ValueError):
            values.add(int(chunk))
    return values


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(value)))
