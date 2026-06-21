from __future__ import annotations

import contextlib
import inspect
import os
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import discord


DEFAULT_OWNER_USER_IDS = {120418341775998976}

OWNER_ONLY_TOOL_NAMES = [
    "discord_list_bot_guilds",
    "discord_get_application_info",
    "discord_create_text_channel",
    "discord_create_voice_channel",
    "discord_edit_channel",
    "discord_delete_channel",
    "discord_set_channel_overwrite",
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
    "discord_list_bot_guilds",
    "discord_get_guild",
    "discord_get_bot_user",
    "discord_get_application_info",
    "discord_list_channels",
    "discord_list_roles",
    "discord_list_members",
    "discord_search_members",
    "discord_get_member",
    "discord_get_permissions",
    "discord_audit_permissions",
    "discord_read_channel_history",
    "discord_search_channel_messages",
    "discord_send_channel_message",
    "discord_edit_own_message",
    "discord_fetch_message",
    "discord_delete_message",
    "discord_bulk_delete_messages",
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
    *OWNER_ONLY_TOOL_NAMES[2:],
]


@dataclass(slots=True)
class DiscordToolRuntime:
    bot: Any
    message: Any


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


async def discord_list_members(limit: int = 50, include_bots: bool = True) -> dict[str, Any]:
    """Admin/owner: list cached guild members. Requires the Discord members intent for reliable output."""
    runtime = _runtime()
    _require_admin_or_owner(runtime, "list members")
    _require_member_intent(runtime)
    guild = _require_guild(runtime)
    members = [member for member in _iter_guild_members(guild) if include_bots or not getattr(member, "bot", False)]
    limit = _clamp(limit, 1, _env_int("DISCORD_BRAIN_TOOL_MEMBER_LIMIT", 100))
    return {"guild_id": _id(guild), "members": [_serialize_member(member) for member in members[:limit]], "count": len(members)}


async def discord_search_members(query: str, limit: int = 25, include_bots: bool = True) -> dict[str, Any]:
    """Admin/owner: search guild members by id, username, global name, display name, or nick."""
    runtime = _runtime()
    _require_admin_or_owner(runtime, "search members")
    _require_member_intent(runtime)
    guild = _require_guild(runtime)
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


async def discord_get_member(user_id: int) -> dict[str, Any]:
    """Return guild member details for the current guild."""
    runtime = _runtime()
    guild = _require_guild(runtime)
    member = await _resolve_member(guild, user_id)
    return {"member": _serialize_member(member)}


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
    """Delete a message after confirmation unless invoked by the configured bot owner."""
    runtime = _runtime()
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
    """Bulk-delete specific messages after confirmation unless invoked by the configured bot owner."""
    runtime = _runtime()
    _require_confirm(runtime, confirm, "bulk_delete_messages")
    channel = await _resolve_channel(runtime, channel_id)
    await _require_channel_permissions(runtime, channel, "bulk delete messages", "view_channel", "read_message_history", "manage_messages")
    limit = _env_int("DISCORD_BRAIN_TOOL_BULK_DELETE_LIMIT", 50)
    ids = [int(message_id) for message_id in message_ids[:limit]]
    messages = [await channel.fetch_message(message_id) for message_id in ids]
    if hasattr(channel, "delete_messages") and len(messages) > 1:
        await _call_discord(channel.delete_messages, messages, reason=_bounded_optional(reason, 512))
    else:
        for message in messages:
            await _call_discord(message.delete, reason=_bounded_optional(reason, 512))
    await _audit_action(runtime, "discord_bulk_delete_messages", _id(channel), {"message_ids": ids, "reason": reason})
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
    overwrite_values = {name: True for name in allow or []}
    overwrite_values.update({name: False for name in deny or []})
    overwrite = discord.PermissionOverwrite(**overwrite_values)
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
    guild = getattr(runtime.message, "guild", None)
    if guild is None:
        current = getattr(runtime.message, "channel", None)
        if _id(current) != _id(channel):
            raise DiscordToolError("DM context can only use the current DM channel.")
        return
    actor = await _actor_member(runtime)
    bot_member = await _bot_member(runtime)
    actor_perms = _permissions_for(channel, actor)
    bot_perms = _permissions_for(channel, bot_member)
    owner = _is_owner(runtime)
    for permission in permissions:
        if not owner and not _perm(actor_perms, permission):
            raise DiscordToolError(f"Current user lacks {permission} to {action} in #{_name(channel)}.")
        if not _perm(bot_perms, permission):
            raise DiscordToolError(f"Bot lacks {permission} to {action} in #{_name(channel)}.")


async def _require_bot_guild_permission(runtime: DiscordToolRuntime, permission: str) -> None:
    bot_member = await _bot_member(runtime)
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
    actor = getattr(runtime.message, "author", None)
    actor_id = _id(actor)
    return actor_id is not None and int(actor_id) in _owner_user_ids()


def _is_admin(actor: Any) -> bool:
    permissions = getattr(actor, "guild_permissions", None)
    return bool(getattr(permissions, "administrator", False))


def _owner_user_ids() -> set[int]:
    explicit = _csv_ints("DISCORD_BRAIN_OWNER_USER_IDS")
    if explicit:
        return explicit
    allowed = _csv_ints("DISCORD_BRAIN_ALLOWED_USER_IDS")
    return allowed or set(DEFAULT_OWNER_USER_IDS)


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


def _perm(permissions: Any, name: str) -> bool:
    if permissions is None:
        return False
    return bool(getattr(permissions, name, False))


def _channel_is_allowed(channel: Any) -> bool:
    allowed = _csv_ints("DISCORD_BRAIN_TOOL_CHANNEL_IDS")
    return not allowed or _id(channel) in allowed


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
        "manage_messages",
        "manage_channels",
        "manage_roles",
        "manage_threads",
        "manage_webhooks",
        "create_instant_invite",
        "add_reactions",
        "create_public_threads",
        "moderate_members",
        "kick_members",
        "ban_members",
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
    common = {"reason", "name", "auto_archive_duration", "content", "allowed_mentions"}
    return {key: value for key, value in kwargs.items() if key in common}


def _bounded_text(text: str, limit: int) -> str:
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 14)].rstrip() + "\n[truncated]"


def _bounded_optional(text: str | None, limit: int) -> str | None:
    if text is None:
        return None
    return _bounded_text(text, limit)


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
