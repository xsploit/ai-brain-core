from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from contextlib import suppress
from dataclasses import asdict, is_dataclass
import io
import json
import logging
import os
from pathlib import Path
from typing import Any

import discord
from discord.ext import commands
from grillo_v2 import GrilloMemoryDocument, GrilloV2Worker, GrilloV2WorkerConfig

from .brain_v2 import BrainV2, BrainV2Config
from .discord_bot import (
    DEFAULT_DISCORD_TOOL_NAMES,
    DISCORD_CONTEXT,
    JBPromptAddView,
    ModelSelectView,
    SummaryActionView,
    _build_jb_persona,
    _image_inputs as _v1_image_inputs,
    _jb_prompt_cache_key,
    _load_jb_prompt,
    _match_piper_voice,
    _ordered_model_choices as _ordered_model_choices_for_v2,
    _split_discord_text,
    _text_attachment_context as _v1_text_attachment_context,
    _tts_spoken_text,
    build_brain,
    build_discord_voice_clip,
    build_persona,
    discover_piper_voices,
    send_discord_voice_message,
)
from .discord_tools import DISCORD_TOOL_CONTEXT, DiscordToolRuntime
from .env import load_env_file
from .model_catalog import ModelChoice, list_model_choices
from .policy import MemoryPolicy
from .tavily_tools import TavilyConfigError, tavily_search


DEFAULT_DISCORD_V2_PREFIX = "!n2"
logger = logging.getLogger("aibrain.discord_v2")


class V2RelationshipGraphView(discord.ui.View):
    def __init__(self, bot: Any, owner_id: int, scope: str, actor_id: str):
        super().__init__(timeout=_env_int("DISCORD_BRAIN_V2_RELATIONSHIP_VIEW_TIMEOUT_SECONDS", 300))
        self.bot_ref = bot
        self.owner_id = owner_id
        self.scope = scope
        self.actor_id = actor_id
        self.page = "overview"
        self.snapshot = _relationship_v2_snapshot(bot, scope, actor_id)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user and interaction.user.id == self.owner_id:
            return True
        await interaction.response.send_message("Only the requester can use this relationship panel.", ephemeral=True)
        return False

    def embed(self) -> discord.Embed:
        return _relationship_v2_embed(self.snapshot, page=self.page)

    def refresh(self) -> None:
        self.snapshot = _relationship_v2_snapshot(self.bot_ref, self.scope, self.actor_id)

    async def edit_page(self, interaction: discord.Interaction, page: str, *, refresh: bool = False) -> None:
        self.page = page
        if refresh:
            self.refresh()
        await interaction.response.edit_message(embed=self.embed(), view=self)

    @discord.ui.button(label="Overview", style=discord.ButtonStyle.primary, row=0)
    async def overview_page(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await self.edit_page(interaction, "overview", refresh=True)

    @discord.ui.button(label="Facts", style=discord.ButtonStyle.secondary, row=0)
    async def facts_page(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await self.edit_page(interaction, "facts")

    @discord.ui.button(label="Memory", style=discord.ButtonStyle.secondary, row=0)
    async def memory_page(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await self.edit_page(interaction, "memory")

    @discord.ui.button(label="Export", style=discord.ButtonStyle.secondary, row=1)
    async def export_relationships(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        self.refresh()
        data = io.BytesIO(json.dumps(self.snapshot, indent=2, sort_keys=True, default=str).encode("utf-8"))
        await interaction.response.send_message(
            "relationship graph export",
            file=discord.File(data, filename="ladybug-v2-relationship-graph.json"),
            ephemeral=True,
        )


class DiscordBrainV2Bot(commands.Bot):
    def __init__(self, *, brain: BrainV2, discord_token: str | None = None):
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
        self.discord_token = (
            discord_token
            or os.getenv("DISCORD_BRAIN_V2_BOT_TOKEN")
            or os.getenv("DISCORD_BRAIN_BOT_TOKEN")
            or os.getenv("DISCORD_BOT_TOKEN")
            or os.getenv("DISCORD_TOKEN")
        )
        self.allowed_guilds = _csv_ints("DISCORD_BRAIN_V2_ALLOWED_GUILD_IDS")
        self.allowed_users = _csv_ints("DISCORD_BRAIN_V2_ALLOWED_USER_IDS")
        self.owner_users = _csv_ints("DISCORD_BRAIN_V2_OWNER_USER_IDS") or _csv_ints("DISCORD_BRAIN_OWNER_USER_IDS")
        self.respond_to_dms = _env_bool("DISCORD_BRAIN_V2_RESPOND_TO_DMS", True)
        self.respond_to_mentions = _env_bool("DISCORD_BRAIN_V2_RESPOND_TO_MENTIONS", True)
        self.require_mention_in_guilds = _env_bool("DISCORD_BRAIN_V2_REQUIRE_MENTION_IN_GUILDS", True)
        self.ignore_bots = _env_bool("DISCORD_BRAIN_V2_IGNORE_BOTS", _env_bool("DISCORD_BRAIN_IGNORE_BOTS", True))
        self.respond_to_bots = _env_bool("DISCORD_BRAIN_V2_RESPOND_TO_BOTS", _env_bool("DISCORD_BRAIN_RESPOND_TO_BOTS", False))
        self.paused = _env_bool("DISCORD_BRAIN_V2_PAUSED", _env_bool("DISCORD_BRAIN_PAUSED", False))
        self.max_reply_chars = _env_int("DISCORD_BRAIN_V2_MAX_REPLY_CHARS", 1900)
        self.tts_voice = os.getenv("DISCORD_BRAIN_V2_TTS_VOICE") or os.getenv("DISCORD_BRAIN_TTS_VOICE") or os.getenv("AIBRAIN_TTS_VOICE") or os.getenv("PIPER_VOICE")
        self.send_tts_replies = _env_bool("DISCORD_BRAIN_V2_TTS_REPLIES", _env_bool("DISCORD_BRAIN_TTS_REPLIES", False))
        self.model_cache: dict[str, Any] = {"expires_at": 0.0, "models": None}
        self.model_cache_lock = asyncio.Lock()
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
        self.add_command(_model_control_group(self))
        self.add_command(_jb_command(self))
        self.add_command(_say_command(self))
        self.add_command(_tts_control_group(self))
        self.add_command(_pause_command(self))
        self.add_command(_resume_command(self))
        self.add_command(_bot_control_group(self))
        self.add_command(_grillo_control_group(self))
        self.add_command(_ladybug_control_group(self))
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
        user_text = _message_text(message)
        attachment_text = await _v1_text_attachment_context(message)
        prompt_text = _append_readable_attachment_context(user_text, attachment_text)
        images = _v1_image_inputs(message)
        context_token = DISCORD_CONTEXT.set(_discord_context_for_message(message, rolling_context))
        tool_token = DISCORD_TOOL_CONTEXT.set(DiscordToolRuntime(bot=self, message=message))
        try:
            async with message.channel.typing():
                response = await self.brain_v2.respond(
                    scope_key=_scope_for_message(message),
                    actor_id=_actor_id(message.author),
                    user_text=prompt_text,
                    source="discord",
                    channel_id=str(message.channel.id),
                    metadata=_discord_metadata(message),
                    rolling_context=rolling_context,
                    record_user_episode=recorded_episode is None,
                    reply_to_episode_id=getattr(recorded_episode, "episode_id", None),
                    images=images,
                    tool_names=DEFAULT_DISCORD_TOOL_NAMES,
                    use_memory=MemoryPolicy(top_k=_env_int("DISCORD_BRAIN_MEMORY_TOP_K", 8)),
                )
        finally:
            DISCORD_TOOL_CONTEXT.reset(tool_token)
            DISCORD_CONTEXT.reset(context_token)
        if response:
            await self._send_final_reply(message, response)
            await self._maybe_send_tts_reply(message, response)

    def _allowed(self, message: discord.Message) -> bool:
        guild = message.guild
        if self.allowed_guilds and (guild is None or guild.id not in self.allowed_guilds):
            return False
        if self.allowed_users and message.author.id not in self.allowed_users:
            return False
        return True

    def _should_respond(self, message: discord.Message) -> bool:
        if getattr(self, "paused", False):
            return False
        author_is_bot = bool(getattr(message.author, "bot", False))
        directed = self._is_directed_at_self(message)
        if author_is_bot:
            return bool(self._bot_interactions_enabled() and directed)
        if message.guild is None:
            return self.respond_to_dms
        return directed

    def _is_directed_at_self(self, message: discord.Message) -> bool:
        if not self.respond_to_mentions or self.user is None:
            return False
        user_id = getattr(self.user, "id", None)
        mentioned = any(getattr(user, "id", None) == user_id for user in getattr(message, "mentions", []))
        return bool(mentioned or self._is_reply_to_self(message))

    def _is_reply_to_self(self, message: discord.Message) -> bool:
        if self.user is None:
            return False
        reference = getattr(message, "reference", None)
        resolved = getattr(reference, "resolved", None) if reference is not None else None
        cached = getattr(reference, "cached_message", None) if reference is not None else None
        target = resolved or cached
        return bool(target is not None and self.user is not None and getattr(getattr(target, "author", None), "id", None) == self.user.id)

    def _bot_interactions_enabled(self) -> bool:
        return bool(not self.ignore_bots and self.respond_to_bots)

    async def _remember_text(self, scope: str, author_id: int, content: str, *, source: str):
        document = GrilloMemoryDocument.create(
            scope_key=scope,
            document_type="manual_memory",
            subject_id=f"discord_user:{author_id}",
            title=f"Manual memory from {source}",
            body=content,
            importance=0.85,
            metadata={"source": source, "author_id": author_id},
        )
        self.brain_v2.store.upsert_memory_document(document)
        return type("RememberedDocument", (), {"id": document.memory_id})()

    def _current_model(self) -> str:
        response_persona = getattr(self.brain_v2, "response_persona", None)
        response_model = getattr(response_persona, "model", None)
        if response_model:
            return str(response_model)
        response_brain = getattr(self.brain_v2, "response_brain", None)
        response_config = getattr(response_brain, "config", None)
        default_model = getattr(response_config, "default_model", None)
        if default_model:
            return str(default_model)
        return str(self.brain_v2.config.model)

    def _set_runtime_model(self, model_id: str) -> None:
        model_id = model_id.strip()
        self.brain_v2.config.model = model_id
        if hasattr(self.brain_v2.json_client, "model"):
            self.brain_v2.json_client.model = model_id
        response_brain = getattr(self.brain_v2, "response_brain", None)
        if response_brain is not None and hasattr(getattr(response_brain, "config", None), "default_model"):
            response_brain.config.default_model = model_id
        response_persona = getattr(self.brain_v2, "response_persona", None)
        if response_persona is not None:
            if hasattr(response_persona, "model_copy"):
                self.brain_v2.response_persona = response_persona.model_copy(update={"model": model_id})
            else:
                setattr(response_persona, "model", model_id)

    async def _load_model_choices(self, *, refresh: bool = False) -> list[ModelChoice]:
        response_brain = getattr(self.brain_v2, "response_brain", None)
        if response_brain is None:
            response_brain = _JSONClientModelCatalogAdapter(self.brain_v2.json_client)
        ttl = int(getattr(getattr(response_brain, "config", None), "models_cache_ttl_seconds", 300) or 300)
        choices = await list_model_choices(
            response_brain,
            cache=self.model_cache,
            cache_lock=self.model_cache_lock,
            ttl_seconds=ttl,
            refresh=refresh,
            default_models=(self._current_model(), self.brain_v2.config.model),
            log=logger,
        )
        return _ordered_model_choices_for_v2(choices, self._current_model())

    async def _maybe_send_tts_reply(self, message: discord.Message, text: str) -> None:
        if not self.send_tts_replies or not self.discord_token or not text.strip():
            return
        brain = getattr(self.brain_v2, "response_brain", None)
        if brain is None:
            logger.warning("TTS reply skipped because V2 response_brain is not configured")
            return
        try:
            spoken = _tts_spoken_text(text)[: _env_int("DISCORD_BRAIN_TTS_MAX_CHARS", 1200)]
            if not spoken.strip():
                return
            clip = await build_discord_voice_clip(brain, spoken, voice=self.tts_voice)
            await send_discord_voice_message(message.channel.id, self.discord_token, clip)
        except Exception:
            logger.exception("Failed to send Discord Brain v2 TTS reply")

    async def _send_final_reply(self, message: discord.Message, text: str) -> discord.Message | None:
        chunks = _split_discord_text(text, min(self.max_reply_chars, 1900))
        first_sent = None
        for index, chunk in enumerate(chunks):
            if index == 0:
                first_sent = await message.reply(chunk, mention_author=False)
            else:
                await message.channel.send(chunk)
        return first_sent

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
                    "`!model` / `!model set/info/export/refresh` - admin/owner model control",
                    "`!jb <message>` - separate one-shot JB path with no memory/tools",
                    "`!say <text>` - send a Piper Discord voice clip",
                    "`!tts` / `!tts toggle` / `!tts voices` / `!tts voice <id>` - voice clip controls",
                    "`!pause` / `!resume` - admin/owner normal reply control",
                    "`!bot toggle` - admin/owner bot-to-bot reply control",
                    "`!grillo` / `!grillo facts/memory/relationships/export` - owner memory diagnostics",
                    "`!ladybug search/relationships/export` - owner graph diagnostics",
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
        summary_text = (text.strip() or "no summary generated.")[: bot.max_reply_chars]
        source_label = getattr(ctx.channel, "name", None) or str(getattr(ctx.channel, "id", "channel"))
        view = SummaryActionView(bot, ctx.author.id, _scope_for_message(ctx.message), summary_text, source_label)
        await ctx.reply(summary_text, mention_author=False, view=view)

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


def _model_control_group(bot: DiscordBrainV2Bot):
    @commands.group(name="model", invoke_without_command=True)
    async def model_control(ctx: commands.Context) -> None:
        await _send_model_picker(bot, ctx)

    @model_control.command(name="refresh")
    async def model_refresh(ctx: commands.Context) -> None:
        await _send_model_picker(bot, ctx, refresh=True)

    @model_control.command(name="set")
    async def model_set(ctx: commands.Context, *, model_id: str) -> None:
        if not await _require_admin_or_owner(bot, ctx, "model control"):
            return
        model_id = model_id.strip()
        if not model_id:
            await ctx.reply("usage: `!model set <model-id>`", mention_author=False)
            return
        choices = await bot._load_model_choices()
        known = {choice.id for choice in choices}
        bot._set_runtime_model(model_id)
        note = "" if model_id in known else " (manual id; not in cached model metadata)"
        await ctx.reply(f"model set to `{model_id}`{note}", mention_author=False)

    @model_control.command(name="info")
    async def model_info(ctx: commands.Context, *, model_id: str = "") -> None:
        await _send_model_info(bot, ctx, model_id=model_id or None)

    @model_control.command(name="export")
    async def model_export(ctx: commands.Context) -> None:
        await _send_model_metadata_export(bot, ctx)

    return model_control


async def _send_model_picker(bot: DiscordBrainV2Bot, ctx: commands.Context, *, refresh: bool = False) -> None:
    if not await _require_admin_or_owner(bot, ctx, "model control"):
        return
    choices = await bot._load_model_choices(refresh=refresh)
    if not choices:
        await ctx.reply("no model choices are available.", mention_author=False)
        return
    view = ModelSelectView(bot, ctx.author.id, choices)
    await ctx.reply(view.message_text(), view=view, mention_author=False)


async def _send_model_metadata_export(bot: DiscordBrainV2Bot, ctx: commands.Context, *, refresh: bool = False) -> None:
    if not await _require_admin_or_owner(bot, ctx, "model metadata export"):
        return
    choices = await bot._load_model_choices(refresh=refresh)
    payload = [
        {
            "id": choice.id,
            "label": choice.label,
            "owned_by": choice.owned_by,
            "created": choice.created,
            "metadata": choice.metadata or {},
        }
        for choice in choices
    ]
    data = io.BytesIO(json.dumps(payload, indent=2, sort_keys=True, default=str).encode("utf-8"))
    await ctx.reply(
        "model metadata export",
        file=discord.File(data, filename="discord-models.json"),
        mention_author=False,
    )


async def _send_model_info(bot: DiscordBrainV2Bot, ctx: commands.Context, *, model_id: str | None = None) -> None:
    if not await _require_admin_or_owner(bot, ctx, "model info"):
        return
    target = (model_id or bot._current_model()).strip()
    choices = await bot._load_model_choices()
    choice = next((item for item in choices if item.id == target), None)
    if choice is None:
        await ctx.reply(f"`{target}` is not in cached model metadata.", mention_author=False)
        return
    metadata = choice.metadata or {}
    lines = [
        f"id: `{choice.id}`",
        f"owned_by: `{choice.owned_by or metadata.get('owned_by') or 'unknown'}`",
        f"created: `{choice.created or metadata.get('created') or 'unknown'}`",
    ]
    for key in ("provider", "context_window", "max_output_tokens", "input_modalities", "output_modalities"):
        if key in metadata:
            lines.append(f"{key}: `{metadata[key]}`")
    await ctx.reply("\n".join(lines)[: bot.max_reply_chars], mention_author=False)


def _jb_command(bot: DiscordBrainV2Bot):
    @commands.group(name="jb", invoke_without_command=True)
    async def jb(ctx: commands.Context, *, content: str = "") -> None:
        if ctx.invoked_subcommand is not None:
            return
        content = content.strip()
        if not content:
            await ctx.reply("usage: `!jb <message>`", mention_author=False)
            return
        one_shot_prompt = _load_jb_prompt()
        if not one_shot_prompt:
            await ctx.reply("JB prompt file is not configured or could not be read.", mention_author=False)
            return
        async with ctx.channel.typing():
            text = await _complete_jb_turn(
                bot,
                content=content,
                message_id=getattr(ctx.message, "id", "latest"),
                one_shot_prompt=one_shot_prompt,
            )
        await ctx.reply((text.strip() or "JB returned no text.")[: bot.max_reply_chars], mention_author=False)

    @jb.command(name="add")
    async def jb_add(ctx: commands.Context) -> None:
        if not await _require_admin_or_owner(bot, ctx, "JB prompt editing"):
            return
        view = JBPromptAddView(ctx.author.id)
        await ctx.reply(
            "Click the button to paste JB prompt text. Future `!jb` turns will include it.",
            mention_author=False,
            view=view,
        )

    return jb


def _say_command(bot: DiscordBrainV2Bot):
    @commands.command(name="say")
    async def say(ctx: commands.Context, *, content: str = "") -> None:
        await _send_tts_voice_message(bot, ctx, content)

    return say


def _tts_control_group(bot: DiscordBrainV2Bot):
    @commands.group(name="tts", invoke_without_command=True)
    async def tts_control(ctx: commands.Context) -> None:
        brain = getattr(bot.brain_v2, "response_brain", None)
        provider = type(getattr(brain, "tts", None)).__name__ if brain is not None else "(none)"
        voices = await asyncio.to_thread(discover_piper_voices)
        current_voice = bot.tts_voice or "(default)"
        await ctx.reply(
            "\n".join(
                [
                    f"provider: `{provider}`",
                    f"voice: `{current_voice}`",
                    f"voice clips on replies: `{bot.send_tts_replies}`",
                    f"voices discovered: `{len(voices)}`",
                ]
            ),
            mention_author=False,
        )

    @tts_control.command(name="toggle")
    async def tts_toggle(ctx: commands.Context) -> None:
        bot.send_tts_replies = not bot.send_tts_replies
        state = "enabled" if bot.send_tts_replies else "disabled"
        await ctx.reply(f"TTS voice clips on normal replies: `{state}`", mention_author=False)

    @tts_control.command(name="voices")
    async def tts_voices(ctx: commands.Context) -> None:
        voices = await asyncio.to_thread(discover_piper_voices)
        if not voices:
            await ctx.reply("no Piper voices discovered.", mention_author=False)
            return
        lines = [f"- `{voice.slug}`: {voice.label}" for voice in voices[: _env_int("DISCORD_BRAIN_TTS_VOICE_LIST_LIMIT", 25)]]
        if len(voices) > len(lines):
            lines.append(f"... {len(voices) - len(lines)} more")
        await ctx.reply("\n".join(lines)[: bot.max_reply_chars], mention_author=False)

    @tts_control.command(name="voice")
    async def tts_voice(ctx: commands.Context, *, voice_id: str) -> None:
        voice_id = voice_id.strip()
        voices = await asyncio.to_thread(discover_piper_voices)
        matched = _match_piper_voice(voices, voice_id)
        if matched is None:
            await ctx.reply(f"unknown Piper voice `{voice_id}`. Use `!tts voices`.", mention_author=False)
            return
        bot.tts_voice = matched.slug
        await ctx.reply(f"Piper voice set to `{matched.slug}` ({matched.label})", mention_author=False)

    return tts_control


def _grillo_control_group(bot: DiscordBrainV2Bot):
    @commands.group(name="grillo", invoke_without_command=True)
    async def grillo(ctx: commands.Context) -> None:
        if not _is_owner(bot, ctx):
            await ctx.reply("owner only", mention_author=False)
            return
        await ctx.reply(
            "\n".join([_format_status(bot.brain_v2.status()), _format_worker_loop_status(bot)]),
            mention_author=False,
        )

    @grillo.command(name="context", aliases=["ctx", "debug"])
    async def grillo_context(ctx: commands.Context, *, query: str = "") -> None:
        if not _is_owner(bot, ctx):
            await ctx.reply("owner only", mention_author=False)
            return
        await _send_grillo_context_packet(bot, ctx, query=query)

    @grillo.command(name="facts")
    async def grillo_facts(ctx: commands.Context, *, query: str = "") -> None:
        if not _is_owner(bot, ctx):
            await ctx.reply("owner only", mention_author=False)
            return
        facts = bot.brain_v2.store.list_active_facts(
            _scope_for_message(ctx.message),
            subject_id=_actor_id(ctx.author),
            query=query,
            limit=_env_int("DISCORD_BRAIN_V2_GRILLO_FACT_LIMIT", 12),
        )
        await ctx.reply(_format_grillo_v2_facts(facts), mention_author=False)

    @grillo.command(name="memory", aliases=["diary", "docs", "slots"])
    async def grillo_memory(ctx: commands.Context, *, query: str = "") -> None:
        if not _is_owner(bot, ctx):
            await ctx.reply("owner only", mention_author=False)
            return
        documents = bot.brain_v2.store.list_memory_documents(
            _scope_for_message(ctx.message),
            subject_id=_actor_id(ctx.author),
            query=query,
            limit=_env_int("DISCORD_BRAIN_V2_GRILLO_MEMORY_LIMIT", 12),
        )
        await ctx.reply(_format_grillo_v2_memory_documents(documents), mention_author=False)

    @grillo.command(name="relationships", aliases=["relationship", "opinions"])
    async def grillo_relationships(ctx: commands.Context) -> None:
        if not _is_owner(bot, ctx):
            await ctx.reply("owner only", mention_author=False)
            return
        await _send_relationship_v2_panel(bot, ctx)

    @grillo.command(name="export")
    async def grillo_export(ctx: commands.Context, *, query: str = "") -> None:
        if not _is_owner(bot, ctx):
            await ctx.reply("owner only", mention_author=False)
            return
        await _send_grillo_v2_export(bot, ctx, query=query)

    return grillo


def _ladybug_control_group(bot: DiscordBrainV2Bot):
    @commands.group(name="ladybug", invoke_without_command=True)
    async def ladybug(ctx: commands.Context) -> None:
        if not _is_owner(bot, ctx):
            await ctx.reply("owner only", mention_author=False)
            return
        await ctx.reply(
            "Ladybug-compatible V2 graph view. Use `!ladybug search <query>`, `!ladybug relationships`, or `!ladybug export`.",
            mention_author=False,
        )

    @ladybug.command(name="search")
    async def ladybug_search(ctx: commands.Context, *, query: str = "") -> None:
        if not _is_owner(bot, ctx):
            await ctx.reply("owner only", mention_author=False)
            return
        query = query.strip()
        if not query:
            await ctx.reply("usage: `!ladybug search <query>`", mention_author=False)
            return
        facts = bot.brain_v2.store.search_facts(
            _scope_for_message(ctx.message),
            query,
            limit=_env_int("DISCORD_BRAIN_V2_LADYBUG_SEARCH_LIMIT", 12),
        )
        await ctx.reply(_format_grillo_v2_facts(facts), mention_author=False)

    @ladybug.command(name="relationships", aliases=["relationship", "rel", "profile"])
    async def ladybug_relationships(ctx: commands.Context) -> None:
        if not _is_owner(bot, ctx):
            await ctx.reply("owner only", mention_author=False)
            return
        await _send_relationship_v2_panel(bot, ctx)

    @ladybug.command(name="export")
    async def ladybug_export(ctx: commands.Context, *, query: str = "") -> None:
        if not _is_owner(bot, ctx):
            await ctx.reply("owner only", mention_author=False)
            return
        await _send_grillo_v2_export(bot, ctx, query=query, filename="ladybug-v2-graph-export.json")

    return ladybug


async def _send_tts_voice_message(bot: DiscordBrainV2Bot, ctx: commands.Context, text: str) -> None:
    text = text.strip()
    if not text:
        await ctx.reply("usage: `!say <text>`", mention_author=False)
        return
    if not bot.discord_token:
        await ctx.reply("Discord bot token is not available for voice clip upload.", mention_author=False)
        return
    brain = getattr(bot.brain_v2, "response_brain", None)
    if brain is None:
        await ctx.reply("TTS backend is not configured.", mention_author=False)
        return
    max_chars = _env_int("DISCORD_BRAIN_TTS_MAX_CHARS", 1200)
    text = _tts_spoken_text(text)[:max_chars]
    if not text.strip():
        await ctx.reply("nothing speakable after TTS cleanup.", mention_author=False)
        return
    async with ctx.typing():
        clip = await build_discord_voice_clip(brain, text, voice=bot.tts_voice)
        await send_discord_voice_message(ctx.channel.id, bot.discord_token, clip)
    await ctx.reply("sent voice clip.", mention_author=False)


async def _send_grillo_context_packet(bot: DiscordBrainV2Bot, ctx: commands.Context, *, query: str = "") -> None:
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


async def _send_grillo_v2_export(
    bot: DiscordBrainV2Bot,
    ctx: commands.Context,
    *,
    query: str = "",
    filename: str = "grillo-v2-export.json",
) -> None:
    scope = _scope_for_message(ctx.message)
    actor = _actor_id(ctx.author)
    packet = bot.brain_v2.build_context_packet(
        scope_key=scope,
        actor_id=actor,
        query=query or _message_text(ctx.message),
        channel_id=str(ctx.channel.id),
    )
    payload = {
        "scope_key": scope,
        "actor_id": actor,
        "counts": bot.brain_v2.store.counts(),
        "context_packet": _object_payload(packet),
        "active_facts": [_object_payload(fact) for fact in bot.brain_v2.store.list_active_facts(scope, subject_id=actor, query=query, limit=50)],
        "memory_documents": [_object_payload(document) for document in bot.brain_v2.store.list_memory_documents(scope, subject_id=actor, query=query, limit=50)],
        "opinion_edges": [
            _object_payload(edge)
            for edge in bot.brain_v2.store.list_opinion_edges(scope, source_id=bot.brain_v2.config.persona_id, target_id=actor, limit=50)
        ],
        "recent_episodes": [_object_payload(episode) for episode in bot.brain_v2.store.list_recent_episodes(scope, actor_id=actor, limit=50)],
    }
    data = io.BytesIO(json.dumps(payload, indent=2, sort_keys=True, default=str).encode("utf-8"))
    await ctx.reply(
        "GRILLO v2 graph export",
        file=discord.File(data, filename=filename),
        mention_author=False,
    )


async def _send_relationship_v2_panel(bot: DiscordBrainV2Bot, ctx: commands.Context) -> None:
    view = V2RelationshipGraphView(bot, ctx.author.id, _scope_for_message(ctx.message), _actor_id(ctx.author))
    await ctx.reply(embed=view.embed(), view=view, mention_author=False)


def _pause_command(bot: DiscordBrainV2Bot):
    @commands.command(name="pause")
    async def pause(ctx: commands.Context) -> None:
        if not await _require_admin_or_owner(bot, ctx, "pause control"):
            return
        bot.paused = True
        await ctx.reply("normal replies paused. commands still work. use `!resume` to resume.", mention_author=False)

    return pause


def _resume_command(bot: DiscordBrainV2Bot):
    @commands.command(name="resume", aliases=["unpause"])
    async def resume(ctx: commands.Context) -> None:
        if not await _require_admin_or_owner(bot, ctx, "pause control"):
            return
        bot.paused = False
        await ctx.reply("normal replies resumed.", mention_author=False)

    return resume


def _bot_control_group(bot: DiscordBrainV2Bot):
    @commands.group(name="bot", invoke_without_command=True)
    async def bot_control(ctx: commands.Context) -> None:
        state = "enabled" if bot._bot_interactions_enabled() else "stopped"
        await ctx.reply(f"bot-to-bot auto replies are `{state}`.", mention_author=False)

    @bot_control.command(name="toggle")
    async def bot_toggle(ctx: commands.Context) -> None:
        if not await _require_admin_or_owner(bot, ctx, "bot interaction control"):
            return
        if bot._bot_interactions_enabled():
            bot.respond_to_bots = False
            state = "stopped"
        else:
            bot.ignore_bots = False
            bot.respond_to_bots = True
            state = "enabled"
        await ctx.reply(f"bot-to-bot auto replies are now `{state}`.", mention_author=False)

    return bot_control


async def _complete_jb_turn(
    bot: DiscordBrainV2Bot,
    *,
    content: str,
    message_id: Any,
    one_shot_prompt: str,
) -> str:
    fallback_model = getattr(getattr(bot.brain_v2, "response_brain", None), "config", None)
    fallback_model_id = str(getattr(fallback_model, "default_model", None) or bot.brain_v2.config.model)
    persona = _build_jb_persona(one_shot_prompt, fallback_model=fallback_model_id)
    response_brain = getattr(bot.brain_v2, "response_brain", None)
    if response_brain is None:
        return (
            await bot.brain_v2.json_client.complete_text(
                instructions=persona.instructions,
                prompt=content,
                store=False,
            )
        ).strip()
    response_options: dict[str, Any] = {"stateless": True}
    cache_key = _jb_prompt_cache_key(one_shot_prompt)
    if cache_key:
        response_options["prompt_cache_key"] = cache_key
        response_options["prompt_cache_retention"] = os.getenv("DISCORD_BRAIN_JB_PROMPT_CACHE_RETENTION", "24h")
    buffer = ""
    async for event in response_brain.stream(
        content,
        thread_id=f"discord:jb:{message_id}",
        persona=persona,
        use_memory=False,
        tool_names=[],
        **response_options,
    ):
        if event.type == "text.delta":
            buffer += str(event.data.get("text", ""))
        elif event.type == "error":
            raise RuntimeError(event.data.get("message", "JB brain stream failed"))
    return buffer.strip()


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


def _is_admin_or_owner(bot: DiscordBrainV2Bot, ctx: commands.Context) -> bool:
    if _is_owner(bot, ctx):
        return True
    permissions = getattr(ctx.author, "guild_permissions", None)
    return bool(getattr(permissions, "administrator", False))


async def _require_admin_or_owner(bot: DiscordBrainV2Bot, ctx: commands.Context, action: str) -> bool:
    if _is_admin_or_owner(bot, ctx):
        return True
    await ctx.reply(f"{action} requires a Discord admin or bot owner.", mention_author=False)
    return False


class _JSONClientModelCatalogAdapter:
    def __init__(self, json_client: Any):
        self.client = getattr(json_client, "client", None)
        self.config = type("ModelCatalogConfig", (), {"models_cache_ttl_seconds": 300})()


def _format_status(status: dict[str, Any]) -> str:
    counts = status.get("counts") if isinstance(status.get("counts"), dict) else {}
    return (
        "Brain v2 online. "
        f"model=`{status.get('model')}` provider=`{status.get('provider')}` grillo=`v2` "
        f"entities=`{counts.get('entities', 0)}` episodes=`{counts.get('episodes', 0)}` "
        f"facts=`{counts.get('active_facts', 0)}` opinions=`{counts.get('active_opinion_edges', 0)}` "
        f"memory_docs=`{counts.get('memory_docs', 0)}` response_backend=`{status.get('response_backend')}`"
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


def _format_grillo_v2_facts(facts: list[Any]) -> str:
    if not facts:
        return "no active GRILLO v2 facts."
    lines = ["GRILLO v2 active facts:"]
    for fact in facts:
        lines.append(
            f"- `{getattr(fact, 'confidence', 0.0):.2f}` "
            f"{getattr(fact, 'subject_id', 'unknown')} --{getattr(fact, 'predicate', 'related_to')}-> "
            f"{getattr(fact, 'object_value', '')}: {getattr(fact, 'claim', '')}"
        )
    return "\n".join(lines)[:1900]


def _format_grillo_v2_memory_documents(documents: list[Any]) -> str:
    if not documents:
        return "no GRILLO v2 memory documents."
    lines = ["GRILLO v2 memory documents:"]
    for document in documents:
        body = " ".join(str(getattr(document, "body", "")).split())
        lines.append(
            f"- `{getattr(document, 'document_type', 'memory')}` "
            f"{getattr(document, 'title', getattr(document, 'memory_id', 'memory'))}: {body[:240]}"
        )
    return "\n".join(lines)[:1900]


def _format_grillo_v2_opinions(edges: list[Any]) -> str:
    if not edges:
        return "no GRILLO v2 opinion/relationship edges."
    lines = ["GRILLO v2 opinion edges:"]
    for edge in edges:
        lines.append(
            f"- `{getattr(edge, 'score', 0.0):+.2f}` "
            f"{getattr(edge, 'source_id', 'unknown')} --{getattr(edge, 'relation', 'related_to')}-> "
            f"{getattr(edge, 'target_id', 'unknown')}: {getattr(edge, 'rationale', '')}"
        )
    return "\n".join(lines)[:1900]


def _relationship_v2_snapshot(bot: Any, scope: str, actor_id: str) -> dict[str, Any]:
    store = bot.brain_v2.store
    persona_id = bot.brain_v2.config.persona_id
    return {
        "scope_key": scope,
        "participant": actor_id,
        "persona_id": persona_id,
        "counts": store.counts(),
        "facts": [_object_payload(fact) for fact in store.list_active_facts(scope, subject_id=actor_id, limit=12)],
        "memory_documents": [_object_payload(document) for document in store.list_memory_documents(scope, subject_id=actor_id, limit=12)],
        "opinion_edges": [
            _object_payload(edge)
            for edge in store.list_opinion_edges(
                scope,
                source_id=persona_id,
                target_id=actor_id,
                limit=_env_int("DISCORD_BRAIN_V2_GRILLO_OPINION_LIMIT", 12),
            )
        ],
    }


def _relationship_v2_embed(snapshot: dict[str, Any], *, page: str = "overview") -> discord.Embed:
    embed = discord.Embed(
        title="Ladybug / GRILLO v2 Relationship Graph",
        description=f"scope `{snapshot.get('scope_key')}`\nparticipant `{snapshot.get('participant')}`",
        color=0x5865F2,
    )
    counts = snapshot.get("counts") if isinstance(snapshot.get("counts"), dict) else {}
    embed.add_field(
        name="Counts",
        value=(
            f"facts `{counts.get('active_facts', 0)}`\n"
            f"opinions `{counts.get('active_opinion_edges', 0)}`\n"
            f"memory docs `{counts.get('memory_docs', 0)}`"
        ),
        inline=True,
    )
    if page == "facts":
        _relationship_embed_list(embed, "Facts", snapshot.get("facts") or [], _relationship_fact_line)
    elif page == "memory":
        _relationship_embed_list(embed, "Memory", snapshot.get("memory_documents") or [], _relationship_memory_line)
    else:
        _relationship_embed_list(embed, "Opinion Edges", snapshot.get("opinion_edges") or [], _relationship_opinion_line)
        if not snapshot.get("opinion_edges"):
            _relationship_embed_list(embed, "Facts", snapshot.get("facts") or [], _relationship_fact_line)
    embed.set_footer(text=f"page: {page}")
    return embed


def _relationship_embed_list(embed: discord.Embed, name: str, items: list[Any], formatter: Any) -> None:
    if not items:
        embed.add_field(name=name, value="(none)", inline=False)
        return
    lines = [formatter(item) for item in items[:6]]
    if len(items) > len(lines):
        lines.append(f"... {len(items) - len(lines)} more")
    embed.add_field(name=name, value="\n".join(lines)[:1024] or "(none)", inline=False)


def _relationship_fact_line(item: dict[str, Any]) -> str:
    return f"`{float(item.get('confidence') or 0.0):.2f}` {item.get('predicate')}: {item.get('claim')}"


def _relationship_memory_line(item: dict[str, Any]) -> str:
    body = " ".join(str(item.get("body") or "").split())
    return f"`{item.get('document_type')}` {item.get('title')}: {body[:140]}"


def _relationship_opinion_line(item: dict[str, Any]) -> str:
    return f"`{float(item.get('score') or 0.0):+.2f}` {item.get('relation')}: {item.get('rationale')}"


def _object_payload(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    return str(value)


def _build_command_prefix(command_prefix_text: str):
    direct_prefix_commands = (
        "help",
        "summary",
        "summarize",
        "search",
        "remember",
        "ping",
        "model",
        "jb",
        "say",
        "tts",
        "pause",
        "resume",
        "unpause",
        "bot",
        "grillo",
        "ladybug",
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


def _append_readable_attachment_context(user_text: str, attachment_text: str) -> str:
    user_text = (user_text or "").strip()
    attachment_text = (attachment_text or "").strip()
    if not attachment_text:
        return user_text
    return f"{user_text}\n\n[Readable attachments]\n{attachment_text}".strip()


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
