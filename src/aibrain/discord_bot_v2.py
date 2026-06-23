from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from contextlib import suppress
import io
import logging
import os
from pathlib import Path
from typing import Any

import discord
from discord.ext import commands
from grillo_v2 import GrilloMemoryDocument, GrilloV2Worker, GrilloV2WorkerConfig

from .brain_v2 import BrainV2, BrainV2Config
from .discord_bot import DEFAULT_DISCORD_TOOL_NAMES, DISCORD_CONTEXT, build_brain, build_persona
from .discord_tools import DISCORD_TOOL_CONTEXT, DiscordToolRuntime
from .env import load_env_file
from .policy import MemoryPolicy
from .tavily_tools import TavilyConfigError, tavily_search


DEFAULT_DISCORD_V2_PREFIX = "!n2"
logger = logging.getLogger("aibrain.discord_v2")


class DiscordBrainV2Bot(commands.Bot):
    def __init__(self, *, brain: BrainV2):
        intents = discord.Intents.default()
        intents.message_content = _env_bool("DISCORD_BRAIN_V2_MESSAGE_CONTENT_INTENT", True)
        intents.guilds = True
        intents.messages = True
        intents.members = _env_bool("DISCORD_BRAIN_V2_MEMBERS_INTENT", False)
        self.command_prefix_text = os.getenv("DISCORD_BRAIN_V2_COMMAND_PREFIX", DEFAULT_DISCORD_V2_PREFIX).strip() or DEFAULT_DISCORD_V2_PREFIX
        super().__init__(
            command_prefix=_build_command_prefix(self.command_prefix_text),
            help_command=None,
            intents=intents,
        )
        self.brain_v2 = brain
        self.allowed_guilds = _csv_ints("DISCORD_BRAIN_V2_ALLOWED_GUILD_IDS")
        self.allowed_users = _csv_ints("DISCORD_BRAIN_V2_ALLOWED_USER_IDS")
        self.owner_users = _csv_ints("DISCORD_BRAIN_V2_OWNER_USER_IDS") or _csv_ints("DISCORD_BRAIN_OWNER_USER_IDS")
        self.respond_to_dms = _env_bool("DISCORD_BRAIN_V2_RESPOND_TO_DMS", True)
        self.respond_to_mentions = _env_bool("DISCORD_BRAIN_V2_RESPOND_TO_MENTIONS", True)
        self.require_mention_in_guilds = _env_bool("DISCORD_BRAIN_V2_REQUIRE_MENTION_IN_GUILDS", True)
        self.max_reply_chars = _env_int("DISCORD_BRAIN_V2_MAX_REPLY_CHARS", 1900)
        self.rolling_context_messages = max(1, _env_int("DISCORD_BRAIN_V2_ROLLING_CONTEXT_MESSAGES", 15))
        self.recent_by_scope: dict[str, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=max(self.rolling_context_messages * 4, 32))
        )
        self.worker_enabled = _env_bool("DISCORD_BRAIN_V2_WORKER_ENABLED", True)
        self.worker = GrilloV2Worker(
            runtime=self.brain_v2.grillo,
            config=GrilloV2WorkerConfig(
                interval_seconds=_env_float("DISCORD_BRAIN_V2_WORKER_INTERVAL_SECONDS", 60.0),
                initial_delay_seconds=_env_float("DISCORD_BRAIN_V2_WORKER_INITIAL_DELAY_SECONDS", 15.0),
                scope_limit=_env_int("DISCORD_BRAIN_V2_WORKER_SCOPE_LIMIT", 10),
                batch_size=_env_int("DISCORD_BRAIN_V2_WORKER_BATCH_SIZE", 12),
                max_batches=_env_int("DISCORD_BRAIN_V2_WORKER_MAX_BATCHES", 3),
                max_consecutive_errors=_env_int("DISCORD_BRAIN_V2_WORKER_MAX_ERRORS", 3),
            ),
            on_tick=self._on_worker_tick,
            on_error=self._on_worker_error,
        )
        self.worker_task: asyncio.Task | None = None
        self.add_command(_help_command(self))
        self.add_command(_status_command(self))
        self.add_command(_context_command(self))
        self.add_command(_remember_command(self))
        self.add_command(_summary_command(self))
        self.add_command(_search_command(self))
        self.add_command(_ping_command(self))
        self.add_command(_reflect_command(self))
        self.add_command(_worker_command(self))
        self.add_command(_backfill_command(self))

    async def setup_hook(self) -> None:
        if self.worker_enabled and self.worker_task is None:
            self.worker_task = asyncio.create_task(
                self.worker.run_forever(),
                name="discord-brain-v2-grillo-worker",
            )

    async def on_ready(self) -> None:
        logger.info("Discord Brain v2 bot logged in as %s/%s", self.user.id if self.user else "unknown", self.user)

    async def close(self) -> None:
        if self.worker_task is not None:
            self.worker_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.worker_task
            self.worker_task = None
        await super().close()

    async def _on_worker_tick(self, result) -> None:
        logger.info(
            "GRILLO v2 worker tick scopes=%s batches=%s episodes=%s facts=%s memory_docs=%s notes=%s",
            result.scopes,
            result.batches,
            result.episodes,
            result.facts,
            result.memory_docs,
            result.notes,
        )

    async def _on_worker_error(self, error: Exception) -> None:
        logger.error("GRILLO v2 worker error", exc_info=(type(error), error, error.__traceback__))

    async def on_message(self, message: discord.Message) -> None:
        if self.user is not None and message.author.id == self.user.id:
            return
        ctx = await self.get_context(message)
        if ctx.command is not None:
            await self.invoke(ctx)
            return
        if not self._allowed(message):
            return
        recorded_episode = self._record_discord_message(message)
        if not self._should_respond(message):
            return
        rolling_context = self._recent_messages(message)
        context_token = DISCORD_CONTEXT.set(_discord_context_for_message(message, rolling_context))
        tool_token = DISCORD_TOOL_CONTEXT.set(DiscordToolRuntime(bot=self, message=message))
        try:
            async with message.channel.typing():
                response = await self.brain_v2.respond(
                    scope_key=_scope_for_message(message),
                    actor_id=_actor_id(message.author),
                    user_text=_message_text(message),
                    source="discord",
                    channel_id=str(message.channel.id),
                    metadata=_discord_metadata(message),
                    rolling_context=rolling_context,
                    record_user_episode=recorded_episode is None,
                    reply_to_episode_id=getattr(recorded_episode, "episode_id", None),
                    tool_names=DEFAULT_DISCORD_TOOL_NAMES,
                    use_memory=MemoryPolicy(top_k=_env_int("DISCORD_BRAIN_MEMORY_TOP_K", 8)),
                )
        finally:
            DISCORD_TOOL_CONTEXT.reset(tool_token)
            DISCORD_CONTEXT.reset(context_token)
        if response:
            await message.reply(response[: self.max_reply_chars], mention_author=False)

    def _allowed(self, message: discord.Message) -> bool:
        guild = message.guild
        if self.allowed_guilds and (guild is None or guild.id not in self.allowed_guilds):
            return False
        if self.allowed_users and message.author.id not in self.allowed_users:
            return False
        return True

    def _should_respond(self, message: discord.Message) -> bool:
        if message.guild is None:
            return self.respond_to_dms
        if not self.require_mention_in_guilds:
            return True
        if self.respond_to_mentions and self.user is not None and self.user in message.mentions:
            return True
        reference = getattr(message, "reference", None)
        resolved = getattr(reference, "resolved", None) if reference is not None else None
        cached = getattr(reference, "cached_message", None) if reference is not None else None
        target = resolved or cached
        return bool(target is not None and self.user is not None and getattr(getattr(target, "author", None), "id", None) == self.user.id)

    def _record_discord_message(self, message: discord.Message):
        text = _message_text(message)
        if not text:
            return None
        scope_key = _scope_for_message(message)
        item = _recent_message_item(message)
        self.recent_by_scope[scope_key].append(item)
        return self.brain_v2.record_message(
            scope_key=scope_key,
            actor_id=_actor_id(message.author),
            user_text=text,
            source="discord",
            channel_id=str(message.channel.id),
            metadata=_discord_metadata(message),
        )

    def _recent_messages(self, message: discord.Message) -> list[dict[str, Any]]:
        return list(self.recent_by_scope.get(_scope_for_message(message), []))[-self.rolling_context_messages :]


def build_brain_v2() -> BrainV2:
    use_v1_response_path = _env_bool("DISCORD_BRAIN_V2_USE_V1_RESPONSE_PATH", True)
    response_brain = build_brain() if use_v1_response_path else None
    response_persona = build_persona() if use_v1_response_path else None
    return BrainV2(
        BrainV2Config(
            database_path=Path(os.getenv("DISCORD_BRAIN_V2_DATABASE_PATH", "discord_brain_v2.sqlite3")),
            model=os.getenv("DISCORD_BRAIN_V2_MODEL", os.getenv("DISCORD_BRAIN_MODEL", "deepseek/deepseek-v4-flash")),
            provider="vercel",
            persona_id=os.getenv("DISCORD_BRAIN_V2_PERSONA_ID", "neuro-sama-v2"),
            persona_name=os.getenv("DISCORD_BRAIN_V2_PERSONA_NAME", "Neuro-sama"),
            persona_prompt=_env_text(
                value_name="DISCORD_BRAIN_V2_PERSONA_PROMPT",
                path_name="DISCORD_BRAIN_V2_PERSONA_PROMPT_PATH",
            ),
        ),
        response_brain=response_brain,
        response_persona=response_persona,
        response_tool_names=DEFAULT_DISCORD_TOOL_NAMES if use_v1_response_path else None,
        response_memory_policy=(
            MemoryPolicy(top_k=_env_int("DISCORD_BRAIN_MEMORY_TOP_K", 8), save_response_summary=True)
            if use_v1_response_path
            else None
        ),
    )


def _help_command(bot: DiscordBrainV2Bot):
    @commands.command(name="help")
    async def help_command(ctx: commands.Context) -> None:
        prefix = bot.command_prefix_text
        await ctx.reply(
            "\n".join(
                [
                    "**Brain v2 commands**",
                    "`!help` - show this menu",
                    "`!summary [limit]` - summarize recent channel messages",
                    "`!search <query>` - explicit Tavily web search",
                    "`!remember <text>` - pin a manual GRILLO v2 memory for you",
                    "`!ping @user` - tag someone with a short hello",
                    f"`{prefix} status` - show v2 model and GRILLO counts",
                    f"`{prefix} context [query]` - owner-only context packet export",
                    f"`{prefix} reflect [limit]` - owner-only reflection pass",
                    f"`{prefix} worker [batch_size] [max_batches]` - owner-only worker tick",
                    f"`{prefix} backfill [limit]` - owner-only v1 to v2 backfill",
                ]
            ),
            mention_author=False,
        )

    return help_command


def _status_command(bot: DiscordBrainV2Bot):
    @commands.command(name="status")
    async def status(ctx: commands.Context) -> None:
        await ctx.reply(
            "\n".join([_format_status(bot.brain_v2.status()), _format_worker_loop_status(bot)]),
            mention_author=False,
        )

    return status


def _context_command(bot: DiscordBrainV2Bot):
    @commands.command(name="context")
    async def context(ctx: commands.Context, *, query: str = "") -> None:
        if not _is_owner(bot, ctx):
            await ctx.reply("owner only", mention_author=False)
            return
        packet = bot.brain_v2.build_context_packet(
            scope_key=_scope_for_message(ctx.message),
            actor_id=_actor_id(ctx.author),
            query=query or _message_text(ctx.message),
            channel_id=str(ctx.channel.id),
        )
        content = packet.as_prompt_text()
        if len(content) <= 1800:
            await ctx.reply(f"```xml\n{content}\n```", mention_author=False)
            return
        data = io.BytesIO(content.encode("utf-8", errors="replace"))
        await ctx.reply(
            "GRILLO v2 context packet",
            file=discord.File(data, filename="grillo-v2-context.xml"),
            mention_author=False,
        )

    return context


def _remember_command(bot: DiscordBrainV2Bot):
    @commands.command(name="remember")
    async def remember(ctx: commands.Context, *, content: str = "") -> None:
        content = content.strip()
        if not content:
            await ctx.reply("usage: `!remember <text>`", mention_author=False)
            return
        document = GrilloMemoryDocument.create(
            scope_key=_scope_for_message(ctx.message),
            document_type="manual_memory",
            subject_id=_actor_id(ctx.author),
            title=f"Manual memory from {_display_name(ctx.author)}",
            body=content,
            importance=0.9,
            metadata=_discord_metadata(ctx.message),
        )
        bot.brain_v2.store.upsert_memory_document(document)
        await ctx.reply(f"remembered `{document.memory_id}`", mention_author=False)

    return remember


def _summary_command(bot: DiscordBrainV2Bot):
    @commands.command(name="summary", aliases=["summarize"])
    async def summary(ctx: commands.Context, limit: int = 50) -> None:
        limit = max(1, min(100, int(limit)))
        messages: list[str] = []
        async for item in ctx.channel.history(limit=limit):
            text = _message_text(item)
            if not text:
                continue
            author = _display_name(item.author)
            marker = " (bot)" if getattr(item.author, "bot", False) else ""
            created_at = getattr(item, "created_at", None)
            stamp = created_at.isoformat() if created_at else "unknown time"
            messages.append(f"[{stamp}] {author}{marker}: {text[:700]}")
        messages.reverse()
        if not messages:
            await ctx.reply("no readable recent messages.", mention_author=False)
            return
        async with ctx.channel.typing():
            text = await bot.brain_v2.json_client.complete_text(
                instructions="Summarize these Discord channel messages. Keep names, decisions, unresolved questions, and notable context. Do not invent facts.",
                prompt="\n".join(messages),
                store=False,
            )
        await ctx.reply((text.strip() or "no summary generated.")[: bot.max_reply_chars], mention_author=False)

    return summary


def _search_command(bot: DiscordBrainV2Bot):
    @commands.command(name="search")
    async def search(ctx: commands.Context, *, query: str = "") -> None:
        query = query.strip()
        if not query:
            await ctx.reply("usage: `!search <query>`", mention_author=False)
            return
        try:
            result = await tavily_search(query, max_results=_env_int("DISCORD_BRAIN_V2_SEARCH_RESULTS", 5))
        except TavilyConfigError as exc:
            await ctx.reply(str(exc), mention_author=False)
            return
        except Exception as exc:
            await ctx.reply(f"search failed: {exc}", mention_author=False)
            return
        await ctx.reply(_format_tavily_search_result(result)[: bot.max_reply_chars], mention_author=False)

    return search


def _ping_command(bot: DiscordBrainV2Bot):
    @commands.command(name="ping")
    async def ping(ctx: commands.Context, *, target: str = "") -> None:
        target = target.strip()
        if ctx.message.mentions:
            target = ctx.message.mentions[0].mention
        if not target:
            await ctx.reply("usage: `!ping @user`", mention_author=False)
            return
        await ctx.send(
            f"{target} yo, what up fam",
            allowed_mentions=discord.AllowedMentions(users=True, roles=False, everyone=False),
        )

    return ping


def _reflect_command(bot: DiscordBrainV2Bot):
    @commands.command(name="reflect")
    async def reflect(ctx: commands.Context, limit: int = 12) -> None:
        if not _is_owner(bot, ctx):
            await ctx.reply("owner only", mention_author=False)
            return
        async with ctx.channel.typing():
            result = await bot.brain_v2.reflect_recent(
                scope_key=_scope_for_message(ctx.message),
                actor_id=_actor_id(ctx.author),
                channel_id=str(ctx.channel.id),
                limit=max(1, min(50, int(limit))),
            )
        await ctx.reply(
            "GRILLO v2 reflection: "
            f"episodes=`{result.episodes}` evidence=`{result.evidence}` facts=`{result.facts}` "
            f"opinions=`{result.opinions}` memory_docs=`{result.memory_docs}` "
            f"tool_calls=`{result.tool_calls}` ignored_tools=`{result.ignored_tool_calls}` "
            f"invalidated=`{result.invalidated_facts}` notes=`{result.notes}`",
            mention_author=False,
        )

    return reflect


def _worker_command(bot: DiscordBrainV2Bot):
    @commands.command(name="worker")
    async def worker(ctx: commands.Context, batch_size: int = 12, max_batches: int = 1) -> None:
        if not _is_owner(bot, ctx):
            await ctx.reply("owner only", mention_author=False)
            return
        async with ctx.channel.typing():
            result = await bot.brain_v2.worker_tick(
                scope_key=_scope_for_message(ctx.message),
                batch_size=max(1, min(50, int(batch_size))),
                max_batches=max(1, min(10, int(max_batches))),
            )
        await ctx.reply(_format_worker_result(result), mention_author=False)

    return worker


def _backfill_command(bot: DiscordBrainV2Bot):
    @commands.command(name="backfill")
    async def backfill(ctx: commands.Context, limit: int = 10_000) -> None:
        if not _is_owner(bot, ctx):
            await ctx.reply("owner only", mention_author=False)
            return
        grillo_path = Path(
            os.getenv("DISCORD_BRAIN_V2_BACKFILL_GRILLO_PATH")
            or os.getenv("DISCORD_BRAIN_DATABASE_PATH")
            or "discord_brain.sqlite3"
        )
        identity_path = Path(
            os.getenv("DISCORD_BRAIN_V2_BACKFILL_IDENTITY_PATH")
            or os.getenv("DISCORD_BRAIN_IDENTITY_DATABASE_PATH")
            or grillo_path.with_suffix(".discord-identity.sqlite3")
        )
        async with ctx.channel.typing():
            results = await asyncio.to_thread(
                bot.brain_v2.backfill_from_v1,
                grillo_path=grillo_path,
                identity_path=identity_path,
                limit=max(1, min(100_000, int(limit))),
            )
        await ctx.reply(_format_backfill_results(results), mention_author=False)

    return backfill


def _is_owner(bot: DiscordBrainV2Bot, ctx: commands.Context) -> bool:
    return getattr(ctx.author, "id", None) in bot.owner_users


def _format_status(status: dict[str, Any]) -> str:
    counts = status.get("counts") if isinstance(status.get("counts"), dict) else {}
    return (
        "Brain v2 online. "
        f"model=`{status.get('model')}` provider=`{status.get('provider')}` grillo=`v2` "
        f"entities=`{counts.get('entities', 0)}` episodes=`{counts.get('episodes', 0)}` "
        f"facts=`{counts.get('active_facts', 0)}` opinions=`{counts.get('active_opinion_edges', 0)}` "
        f"memory_docs=`{counts.get('memory_docs', 0)}`"
    )


def _format_backfill_results(results: dict[str, Any]) -> str:
    if not results:
        return "GRILLO v2 backfill: no sources provided."
    parts = ["GRILLO v2 backfill complete."]
    for name, result in results.items():
        parts.append(
            f"{name}: episodes=`{getattr(result, 'episodes', 0)}` entities=`{getattr(result, 'entities', 0)}` "
            f"evidence=`{getattr(result, 'evidence', 0)}` facts=`{getattr(result, 'facts', 0)}` "
            f"skipped=`{getattr(result, 'skipped', 0)}`"
        )
    return "\n".join(parts)


def _format_worker_result(result: Any) -> str:
    notes = ", ".join(getattr(result, "notes", []) or [])
    return (
        "GRILLO v2 worker: "
        f"scopes=`{getattr(result, 'scopes', 0)}` batches=`{getattr(result, 'batches', 0)}` "
        f"episodes=`{getattr(result, 'episodes', 0)}` evidence=`{getattr(result, 'evidence', 0)}` "
        f"facts=`{getattr(result, 'facts', 0)}` opinions=`{getattr(result, 'opinions', 0)}` "
        f"memory_docs=`{getattr(result, 'memory_docs', 0)}` tool_calls=`{getattr(result, 'tool_calls', 0)}` "
        f"ignored_tools=`{getattr(result, 'ignored_tool_calls', 0)}` "
        f"invalidated=`{getattr(result, 'invalidated_facts', 0)}` notes=`{notes}`"
    )


def _format_worker_loop_status(bot: DiscordBrainV2Bot) -> str:
    task = bot.worker_task
    running = bool(task is not None and not task.done())
    last = bot.worker.last_result
    last_notes = ", ".join(last.notes or []) if last is not None else "none"
    return (
        "GRILLO v2 worker loop: "
        f"enabled=`{bot.worker_enabled}` running=`{running}` ticks=`{bot.worker.ticks}` "
        f"errors=`{bot.worker.consecutive_errors}` last_batches=`{getattr(last, 'batches', 0)}` "
        f"last_episodes=`{getattr(last, 'episodes', 0)}` notes=`{last_notes}`"
    )


def _build_command_prefix(command_prefix_text: str):
    direct_prefix_commands = (
        "help",
        "summary",
        "summarize",
        "search",
        "remember",
        "ping",
    )

    def command_prefix(bot: commands.Bot, message: discord.Message):
        stripped = message.content.strip()
        if any(stripped == f"!{name}" or stripped.startswith(f"!{name} ") for name in direct_prefix_commands):
            return "!"
        return f"{command_prefix_text} "

    return command_prefix


def _scope_for_message(message: discord.Message) -> str:
    if message.guild is None:
        return f"discord:dm:{message.author.id}:persona:v2"
    return f"discord:guild:{message.guild.id}:persona:v2"


def _actor_id(user: discord.abc.User) -> str:
    return f"discord_user:{user.id}"


def _message_text(message: discord.Message) -> str:
    return (getattr(message, "clean_content", None) or getattr(message, "content", "") or "").strip()


def _display_name(user: Any) -> str:
    return (
        getattr(user, "display_name", None)
        or getattr(user, "global_name", None)
        or getattr(user, "name", None)
        or str(getattr(user, "id", "unknown"))
    )


def _discord_metadata(message: discord.Message) -> dict[str, Any]:
    guild = message.guild
    channel = message.channel
    author = message.author
    metadata = {
        "message_id": str(getattr(message, "id", "")),
        "guild_id": str(guild.id) if guild else None,
        "guild_name": guild.name if guild else None,
        "channel_id": str(getattr(channel, "id", "")),
        "channel_name": getattr(channel, "name", "dm"),
        "author_id": str(author.id),
        "author_username": getattr(author, "name", None),
        "author_display_name": getattr(author, "display_name", None),
        "author_global_name": getattr(author, "global_name", None),
        "author_is_bot": bool(getattr(author, "bot", False)),
        "jump_url": getattr(message, "jump_url", None),
    }
    reply_target = _reply_target_context(message)
    if reply_target is not None:
        metadata["reply_target"] = reply_target
    return metadata


def _discord_context_for_message(message: discord.Message, recent_messages: list[dict[str, Any]]) -> dict[str, Any]:
    metadata = _discord_metadata(message)
    return {
        "scope": _scope_for_message(message),
        "guild": metadata.get("guild_name"),
        "guild_id": metadata.get("guild_id"),
        "channel": metadata.get("channel_name"),
        "channel_id": metadata.get("channel_id"),
        "author": metadata.get("author_display_name") or metadata.get("author_global_name") or metadata.get("author_username"),
        "author_id": metadata.get("author_id"),
        "author_username": metadata.get("author_username"),
        "author_display_name": metadata.get("author_display_name"),
        "author_global_name": metadata.get("author_global_name"),
        "author_is_bot": metadata.get("author_is_bot"),
        "message_id": metadata.get("message_id"),
        "jump_url": metadata.get("jump_url"),
        "reply_target": metadata.get("reply_target"),
        "recent_messages": recent_messages,
        "discord_metadata": metadata,
    }


def _reply_target_context(message: discord.Message) -> dict[str, Any] | None:
    reference = getattr(message, "reference", None)
    if reference is None:
        return None
    resolved = None
    for attr in ("resolved", "cached_message"):
        candidate = getattr(reference, attr, None)
        if candidate is not None:
            resolved = candidate
            break
    message_id = getattr(reference, "message_id", None)
    if resolved is None:
        return {"message_id": str(message_id)} if message_id is not None else None
    author = getattr(resolved, "author", None)
    return {
        "message_id": str(getattr(resolved, "id", None) or message_id or ""),
        "author": _display_name(author) if author is not None else "unknown",
        "author_id": str(getattr(author, "id", "")) if author is not None else None,
        "author_is_bot": bool(getattr(author, "bot", False)),
        "content": _message_text(resolved)[:1000],
        "jump_url": getattr(resolved, "jump_url", None),
    }


def _recent_message_item(message: discord.Message) -> dict[str, Any]:
    reply_target = _reply_target_context(message) or {}
    created_at = getattr(message, "created_at", None)
    return {
        "message_id": str(getattr(message, "id", "")),
        "author": _display_name(message.author),
        "author_id": str(getattr(message.author, "id", "")),
        "author_is_bot": bool(getattr(message.author, "bot", False)),
        "content": _message_text(message),
        "created_at": created_at.isoformat() if created_at else None,
        "reply_to_message_id": reply_target.get("message_id"),
        "reply_to_author": reply_target.get("author"),
        "reply_to_author_id": reply_target.get("author_id"),
        "reply_to_author_is_bot": reply_target.get("author_is_bot"),
    }


def _format_tavily_search_result(result: dict[str, Any]) -> str:
    lines = ["search results:"]
    answer = result.get("answer")
    if answer:
        lines.append(str(answer).strip())
    for index, item in enumerate(result.get("results") or [], start=1):
        title = item.get("title") or item.get("url") or f"result {index}"
        url = item.get("url") or ""
        content = " ".join(str(item.get("content") or "").split())
        line = f"{index}. {title}"
        if url:
            line += f" - {url}"
        if content:
            line += f"\n{content[:240]}"
        lines.append(line)
    return "\n".join(lines)


def _csv_ints(name: str) -> set[int]:
    raw = os.getenv(name, "")
    values: set[int] = set()
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            values.add(int(chunk))
        except ValueError:
            logger.warning("Ignoring invalid integer in %s: %s", name, chunk)
    return values


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_text(*, value_name: str, path_name: str) -> str:
    raw = os.getenv(value_name)
    if raw:
        return raw
    raw_path = os.getenv(path_name)
    if not raw_path:
        return ""
    path = Path(raw_path).expanduser()
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not read %s=%s: %s", path_name, path, exc)
        return ""


def main() -> None:
    load_env_file(Path(os.getenv("DISCORD_BRAIN_V2_ENV_FILE", ".env")))
    logging.basicConfig(
        level=os.getenv("DISCORD_BRAIN_V2_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    token = os.getenv("DISCORD_BRAIN_V2_BOT_TOKEN")
    if not token:
        raise RuntimeError("DISCORD_BRAIN_V2_BOT_TOKEN is required for the v2 Discord bot")
    bot = DiscordBrainV2Bot(brain=build_brain_v2())
    asyncio.run(bot.start(token))


if __name__ == "__main__":
    main()
