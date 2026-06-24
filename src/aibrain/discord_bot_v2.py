from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from contextlib import suppress
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import io
import json
import logging
import os
from pathlib import Path
import random
import re
from types import SimpleNamespace
from typing import Any

import discord
from discord import app_commands
from discord.ext import commands
from grillo_v2 import GrilloMemoryDocument, GrilloV2Worker, GrilloV2WorkerConfig

from .brain_v2 import BrainV2, BrainV2Config
from .discord_bot import (
    DEFAULT_DISCORD_TOOL_NAMES,
    DEFAULT_HEARTBEAT_TOOL_NAMES,
    DEFAULT_OWNER_USER_IDS,
    DISCORD_CONTEXT,
    JBPromptAddView,
    LETTA_HEARTBEAT_EVENT_TEXT,
    ModelSelectView,
    SummaryActionView,
    _build_jb_persona,
    _codex_bridge_result_matches,
    _compact,
    _discord_shitlist_path,
    _format_codex_bridge_update,
    _format_shitlist_status,
    _heartbeat_action_name,
    _heartbeat_decision_text,
    _image_inputs as _v1_image_inputs,
    _jb_prompt_cache_key,
    _load_jb_prompt,
    _match_piper_voice,
    _ordered_model_choices as _ordered_model_choices_for_v2,
    _parse_heartbeat_decision,
    _read_codex_bridge_result,
    _recent_messages_prompt_lines,
    _split_discord_text,
    _target_user_id,
    _text_attachment_context as _v1_text_attachment_context,
    _time_context,
    _tts_spoken_text,
    build_brain,
    build_discord_voice_clip,
    build_persona,
    discover_piper_voices,
    send_discord_voice_message,
)
from .codex_app_bridge import notify_codex_app_bridge
from .codex_bridge import CodexBridgeQueue
from .discord_shitlist import DiscordShitlistEntry, DiscordShitlistStore, format_shitlist_reply
from .discord_tools import DISCORD_TOOL_CONTEXT, DiscordToolRuntime
from .env import load_env_file
from .model_catalog import ModelChoice, list_model_choices
from .policy import MemoryPolicy
from .tavily_tools import TavilyConfigError, tavily_search
from .treblo_song import TrebloSongError, TrebloSongQueue, TrebloSongRateLimited


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
        self.owner_users = (
            _csv_ints("DISCORD_BRAIN_V2_OWNER_USER_IDS")
            | _csv_ints("DISCORD_BRAIN_OWNER_USER_IDS")
            | set(DEFAULT_OWNER_USER_IDS)
        )
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
        self.shitlist_store = DiscordShitlistStore(
            _discord_shitlist_path(self.brain_v2.config.database_path),
            owner_user_ids=self.owner_users,
        )
        self.treblo_song_queue = TrebloSongQueue(owner_user_ids=self.owner_users)
        self.treblo_song_task: asyncio.Task | None = None
        self.treblo_slash_synced = False
        self.codex_bridge = CodexBridgeQueue.from_env()
        self.rolling_context_messages = max(1, _env_int("DISCORD_BRAIN_V2_ROLLING_CONTEXT_MESSAGES", 15))
        self.recent_by_scope: dict[str, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=max(self.rolling_context_messages * 4, 32))
        )
        self.heartbeat_enabled = _env_bool("DISCORD_BRAIN_V2_HEARTBEAT_ENABLED", _env_bool("DISCORD_BRAIN_HEARTBEAT_ENABLED", False))
        self.heartbeat_channel_ids = _csv_ints("DISCORD_BRAIN_V2_HEARTBEAT_CHANNEL_IDS") or _csv_ints("DISCORD_BRAIN_HEARTBEAT_CHANNEL_IDS")
        self.heartbeat_conversation = (
            os.getenv("DISCORD_BRAIN_V2_HEARTBEAT_CONVERSATION")
            or os.getenv("DISCORD_BRAIN_HEARTBEAT_CONVERSATION")
            or "last-active"
        ).strip().lower() or "last-active"
        self.heartbeat_min_interval_seconds = max(
            1.0,
            _env_float("DISCORD_BRAIN_V2_HEARTBEAT_MIN_INTERVAL_SECONDS", _env_float("DISCORD_BRAIN_HEARTBEAT_MIN_INTERVAL_SECONDS", 60.0)),
        )
        self.heartbeat_interval_seconds = max(
            self.heartbeat_min_interval_seconds,
            _env_float("DISCORD_BRAIN_V2_HEARTBEAT_INTERVAL_SECONDS", _env_float("DISCORD_BRAIN_HEARTBEAT_INTERVAL_SECONDS", 900.0)),
        )
        self.heartbeat_chance = max(
            0.0,
            min(1.0, _env_float("DISCORD_BRAIN_V2_HEARTBEAT_CHANCE", _env_float("DISCORD_BRAIN_HEARTBEAT_CHANCE", 0.08))),
        )
        self.heartbeat_tts_enabled = _env_bool("DISCORD_BRAIN_V2_HEARTBEAT_TTS", _env_bool("DISCORD_BRAIN_HEARTBEAT_TTS", False))
        self.heartbeat_autonomy_enabled = _env_bool(
            "DISCORD_BRAIN_V2_HEARTBEAT_AUTONOMY_ENABLED",
            _env_bool("DISCORD_BRAIN_HEARTBEAT_AUTONOMY_ENABLED", True),
        )
        self.heartbeat_tools_enabled = _env_bool(
            "DISCORD_BRAIN_V2_HEARTBEAT_TOOLS_ENABLED",
            _env_bool("DISCORD_BRAIN_HEARTBEAT_TOOLS_ENABLED", True),
        )
        self.heartbeat_allow_owner_dm = _env_bool(
            "DISCORD_BRAIN_V2_HEARTBEAT_ALLOW_OWNER_DM",
            _env_bool("DISCORD_BRAIN_HEARTBEAT_ALLOW_OWNER_DM", True),
        )
        self.heartbeat_dm_user_ids = _csv_ints("DISCORD_BRAIN_V2_HEARTBEAT_DM_USER_IDS") or _csv_ints("DISCORD_BRAIN_HEARTBEAT_DM_USER_IDS")
        self.heartbeat_action_cooldown_seconds = max(
            0.0,
            _env_float(
                "DISCORD_BRAIN_V2_HEARTBEAT_ACTION_COOLDOWN_SECONDS",
                _env_float("DISCORD_BRAIN_HEARTBEAT_ACTION_COOLDOWN_SECONDS", 1800.0),
            ),
        )
        self.heartbeat_action_last_at: dict[str, float] = {}
        self.heartbeat_last_channel: Any | None = None
        self.heartbeat_last_channel_id: int | None = None
        self.heartbeat_task: asyncio.Task | None = None
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
        self.add_command(_recall_command(self))
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
        self.add_command(_shitlist_control_group(self))
        self.add_command(_codex_bridge_group(self))
        self.add_command(_heartbeat_control_group(self))
        self.add_command(_grillo_control_group(self))
        self.add_command(_ladybug_control_group(self))
        self.add_command(_reflect_command(self))
        self.add_command(_worker_command(self))
        self.add_command(_backfill_command(self))
        self.tree.add_command(_song_slash_command(self))
        self.tree.add_command(_song_queue_slash_command(self))

    async def setup_hook(self) -> None:
        if self.worker_enabled and self.worker_task is None:
            self.worker_task = asyncio.create_task(
                self.worker.run_forever(),
                name="discord-brain-v2-grillo-worker",
            )
        if self.treblo_song_task is None:
            self.treblo_song_task = asyncio.create_task(
                self.treblo_song_queue.run(self),
                name="discord-brain-v2-treblo-song-queue",
            )
        self._ensure_heartbeat_task()

    async def on_ready(self) -> None:
        logger.info("Discord Brain v2 bot logged in as %s/%s", self.user.id if self.user else "unknown", self.user)
        self._ensure_heartbeat_task()
        await self._sync_treblo_slash_commands()

    async def close(self) -> None:
        if self.heartbeat_task is not None:
            self.heartbeat_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.heartbeat_task
            self.heartbeat_task = None
        if self.worker_task is not None:
            self.worker_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.worker_task
            self.worker_task = None
        await self.treblo_song_queue.close()
        if self.treblo_song_task is not None:
            self.treblo_song_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.treblo_song_task
            self.treblo_song_task = None
        await super().close()

    async def _sync_treblo_slash_commands(self) -> None:
        if self.treblo_slash_synced or not _env_bool("DISCORD_BRAIN_V2_TREBLO_SYNC_SLASH", True):
            return
        self.treblo_slash_synced = True
        guild_ids = set(self.allowed_guilds) or {guild.id for guild in self.guilds}
        if not guild_ids:
            synced = await self.tree.sync()
            logger.info("Synced %s global app commands", len(synced))
            return
        for guild_id in sorted(guild_ids):
            guild = discord.Object(id=guild_id)
            self.tree.copy_global_to(guild=guild)
            synced = await self.tree.sync(guild=guild)
            logger.info("Synced %s app commands to guild %s", len(synced), guild_id)

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
        codex_updates = self._codex_bridge_updates_for_message(message)
        if codex_updates:
            prompt_text = f"{prompt_text}\n\n[Codex bridge updates]\n" + "\n".join(codex_updates)
        shitlist_entry = self.shitlist_store.get(getattr(message.author, "id", None))
        if shitlist_entry is not None:
            await self._reply_with_shitlist(message, shitlist_entry)
            return
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
            brain = self.brain_v2
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
        if not bool(getattr(message.author, "bot", False)):
            self.heartbeat_last_channel = message.channel
            self.heartbeat_last_channel_id = getattr(message.channel, "id", None)
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

    async def _reply_with_shitlist(self, message: discord.Message, entry: DiscordShitlistEntry) -> None:
        await self._send_final_reply(message, format_shitlist_reply(entry))

    async def _queue_codex_bridge_request(self, ctx: commands.Context, *, route: str, prompt: str) -> None:
        if not await _require_owner(bot=self, ctx=ctx, action="Codex bridge"):
            return
        route = (route or "codex").strip().lower()
        prompt = prompt.strip()
        if route not in {"codex", "harness"}:
            await ctx.reply("usage: `!codex route <codex|harness> <prompt>`", mention_author=False)
            return
        if not prompt:
            usage = "`!codex ask <prompt>`" if route == "codex" else "`!codex route <codex|harness> <prompt>`"
            await ctx.reply(f"usage: {usage}", mention_author=False)
            return
        if not self.codex_bridge.enabled:
            await ctx.reply(
                "Codex bridge is disabled. Set `DISCORD_BRAIN_CODEX_BRIDGE_ENABLED=true` to queue requests.",
                mention_author=False,
            )
            return
        if self.codex_bridge.is_paused():
            await ctx.reply("Codex bridge queue is paused. Use `!codex resume` first.", mention_author=False)
            return
        message = ctx.message
        guild = getattr(message, "guild", None)
        channel = getattr(message, "channel", None)
        delivery_mode = "harness_brain" if route == "harness" else "thread_heartbeat"
        path = self.codex_bridge.enqueue(
            requester_id=ctx.author.id,
            requester_name=_display_name(ctx.author),
            guild_id=getattr(guild, "id", None),
            channel_id=getattr(channel, "id", None),
            message_id=getattr(message, "id", None),
            prompt=prompt,
            intent="harness" if route == "harness" else "ask_codex",
            authority_mode="manual_owner",
            authority_reason=f"authorized Discord !codex {route} command",
            delivery_mode=delivery_mode,
            recent_messages=list(self.recent_by_scope.get(_scope_for_message(message), []))[-8:],
            harness_agent="claude",
            harness_permission_profile="inspect",
        )
        notify = await notify_codex_app_bridge(path)
        notify_text = "bridge notified" if notify.get("notified") else f"bridge notify skipped: {notify.get('reason') or notify.get('error') or 'unknown'}"
        await ctx.reply(f"queued Codex bridge request `{path.name}` via `{delivery_mode}`; {notify_text}.", mention_author=False)

    def _codex_bridge_updates_for_message(self, message: discord.Message) -> list[str]:
        if not _env_bool("DISCORD_BRAIN_V2_CODEX_CONTEXT_ENABLED", _env_bool("DISCORD_BRAIN_CODEX_CONTEXT_ENABLED", True)):
            return []
        bridge = getattr(self, "codex_bridge", None)
        if bridge is None:
            return []
        try:
            results = bridge.result_files()
        except Exception:
            return []
        author_id = str(getattr(getattr(message, "author", None), "id", ""))
        channel_id = str(getattr(getattr(message, "channel", None), "id", ""))
        guild = getattr(message, "guild", None)
        guild_id = str(getattr(guild, "id", "")) if guild is not None else None
        max_results = _env_int("DISCORD_BRAIN_V2_CODEX_CONTEXT_MAX_RESULTS", _env_int("DISCORD_BRAIN_CODEX_CONTEXT_MAX_RESULTS", 3))
        matched: list[dict[str, Any]] = []
        for path in reversed(results[-25:]):
            payload = _read_codex_bridge_result(path)
            if not payload or not _codex_bridge_result_matches(payload, author_id=author_id, channel_id=channel_id, guild_id=guild_id):
                continue
            matched.append(payload)
            if len(matched) >= max(1, max_results):
                break
        if not matched:
            return []
        lines = ["Recent Codex bridge updates relevant to this Discord context:"]
        lines.extend(_format_codex_bridge_update(payload) for payload in reversed(matched))
        lines.append("Use these updates naturally if the user asks what Codex did or what changed.")
        return lines

    def _ensure_heartbeat_task(self) -> None:
        if not self.heartbeat_enabled:
            return
        if not self.heartbeat_channel_ids and not self._heartbeat_can_run_without_channel():
            logger.warning("Discord Brain v2 heartbeat enabled but no channel or channel-less action is configured.")
            return
        if self.heartbeat_task is not None and not self.heartbeat_task.done():
            return
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop(), name="discord-brain-v2-heartbeat")

    async def _heartbeat_loop(self) -> None:
        while self.heartbeat_enabled and not self.is_closed():
            delay = self._next_heartbeat_delay_seconds()
            logger.info("Discord Brain v2 heartbeat scheduled in %.0f seconds", delay)
            await asyncio.sleep(delay)
            if not self.heartbeat_enabled or getattr(self, "paused", False):
                continue
            if random.random() > self.heartbeat_chance:
                continue
            channel = await self._heartbeat_channel()
            if channel is None and not self._heartbeat_can_run_without_channel():
                continue
            try:
                await self._run_heartbeat_tick(channel)
            except Exception:
                logger.exception("Discord Brain v2 heartbeat tick failed")

    def _heartbeat_can_run_without_channel(self) -> bool:
        return bool(
            (self.heartbeat_allow_owner_dm and self.owner_users)
            or (self.codex_bridge.enabled and not self.codex_bridge.is_paused())
        )

    def _next_heartbeat_delay_seconds(self) -> float:
        minimum = min(self.heartbeat_min_interval_seconds, self.heartbeat_interval_seconds)
        maximum = max(self.heartbeat_min_interval_seconds, self.heartbeat_interval_seconds)
        if maximum <= minimum:
            return maximum
        return random.uniform(minimum, maximum)

    async def _heartbeat_channel(self, *, fallback: Any | None = None) -> Any | None:
        channel_ids = list(self.heartbeat_channel_ids)
        if not channel_ids and fallback is not None:
            return fallback
        if not channel_ids:
            if self.heartbeat_conversation == "last-active" and self.heartbeat_last_channel is not None:
                return self.heartbeat_last_channel
            return None
        channel_id = random.choice(channel_ids)
        channel = self.get_channel(channel_id)
        if channel is not None:
            return channel
        with suppress(Exception):
            return await self.fetch_channel(channel_id)
        return None

    async def _run_heartbeat_tick(self, channel: Any | None) -> str:
        if not getattr(self, "heartbeat_autonomy_enabled", True):
            if channel is None:
                return "send_channel_message:no_channel"
            await self._send_heartbeat_message(channel)
            return "send_channel_message"
        decision = await self._build_heartbeat_decision(channel)
        action = _heartbeat_action_name(decision.get("action"))
        if action == "noop":
            logger.info("Discord Brain v2 heartbeat chose noop: %s", decision.get("reason", ""))
            return "noop"
        if action == "send_channel_message":
            if channel is None:
                return "send_channel_message:no_channel"
            text = _heartbeat_decision_text(decision, _env_int("DISCORD_BRAIN_V2_HEARTBEAT_MAX_CHARS", _env_int("DISCORD_BRAIN_HEARTBEAT_MAX_CHARS", 240)))
            if not text:
                text = await self._build_heartbeat_text(channel)
            await channel.send(text)
            await self._maybe_send_heartbeat_tts(channel, text)
            return "send_channel_message"
        if action in {"dm_owner", "dm_user"}:
            target_id = self._heartbeat_dm_target(action, decision)
            if target_id is None:
                return f"{action}:rejected"
            cooldown_key = f"{action}:{target_id}"
            if not self._heartbeat_action_ready(cooldown_key):
                return f"{action}:cooldown"
            text = _heartbeat_decision_text(decision, _env_int("DISCORD_BRAIN_V2_HEARTBEAT_DM_MAX_CHARS", _env_int("DISCORD_BRAIN_HEARTBEAT_DM_MAX_CHARS", 800)))
            if not text:
                return f"{action}:empty"
            await self._send_heartbeat_dm(target_id, text)
            self._mark_heartbeat_action(cooldown_key)
            return action
        if action == "queue_codex":
            if not self.codex_bridge.enabled or self.codex_bridge.is_paused():
                return "queue_codex:disabled"
            cooldown_key = "queue_codex"
            if not self._heartbeat_action_ready(cooldown_key):
                return "queue_codex:cooldown"
            prompt = str(decision.get("codex_prompt") or decision.get("prompt") or "").strip()
            if not prompt:
                return "queue_codex:empty"
            await self._queue_heartbeat_codex_request(channel, prompt)
            self._mark_heartbeat_action(cooldown_key)
            return "queue_codex"
        return "unknown"

    async def _build_heartbeat_decision(self, channel: Any | None) -> dict[str, Any]:
        prompt = self._heartbeat_autonomy_prompt(channel)
        brain = getattr(self.brain_v2, "response_brain", None)
        if brain is None:
            text = await self.brain_v2.json_client.complete_text(
                instructions="Return one heartbeat JSON action object only.",
                prompt=prompt,
                store=False,
            )
            return _parse_heartbeat_decision(text) or {"action": "noop", "reason": "json client returned no valid heartbeat action"}
        buffer = ""
        tool_calls: list[str] = []
        tool_names = self._heartbeat_tool_names()
        message = self._heartbeat_runtime_message(channel)
        context_token = None
        tool_token = None
        if tool_names:
            context_token = DISCORD_CONTEXT.set(self._heartbeat_context(message))
            tool_token = DISCORD_TOOL_CONTEXT.set(
                DiscordToolRuntime(bot=self, message=message, authority_mode="autonomous_neuro")
            )
        try:
            async for event in brain.stream(
                prompt,
                thread_id=f"discord:v2:heartbeat:autonomy:{getattr(channel, 'id', 'channel-less')}",
                persona=getattr(self.brain_v2, "response_persona", None),
                use_memory=MemoryPolicy(top_k=_env_int("DISCORD_BRAIN_V2_HEARTBEAT_MEMORY_TOP_K", _env_int("DISCORD_BRAIN_HEARTBEAT_MEMORY_TOP_K", 3))),
                tool_names=tool_names,
                max_agent_steps=_env_int("DISCORD_BRAIN_V2_HEARTBEAT_MAX_AGENT_STEPS", _env_int("DISCORD_BRAIN_HEARTBEAT_MAX_AGENT_STEPS", 40)),
                stateless=True,
                memory_query_text="heartbeat autonomy",
                memory_event_text="",
                history_text=prompt,
            ):
                if event.type == "text.delta":
                    buffer += str(event.data.get("text", ""))
                elif event.type == "tool.call":
                    name = str(event.data.get("name") or "").strip()
                    if name:
                        tool_calls.append(name)
                elif event.type == "error":
                    raise RuntimeError(event.data.get("message", "heartbeat autonomy failed"))
        finally:
            if tool_token is not None:
                DISCORD_TOOL_CONTEXT.reset(tool_token)
            if context_token is not None:
                DISCORD_CONTEXT.reset(context_token)
        decision = _parse_heartbeat_decision(buffer)
        if decision is not None:
            if tool_calls:
                decision.setdefault("tool_calls", tool_calls)
            return decision
        if tool_calls:
            return {
                "action": "noop",
                "reason": "heartbeat completed tool calls without a JSON fallback action",
                "tool_calls": tool_calls,
            }
        return {
            "action": "send_channel_message",
            "message": buffer.strip(),
            "reason": "model returned text instead of JSON",
        }

    def _heartbeat_tool_names(self) -> list[str]:
        if not getattr(self, "heartbeat_tools_enabled", True):
            return []
        raw = (os.getenv("DISCORD_BRAIN_V2_HEARTBEAT_TOOL_NAMES") or os.getenv("DISCORD_BRAIN_HEARTBEAT_TOOL_NAMES") or "").strip()
        if not raw:
            return list(DEFAULT_HEARTBEAT_TOOL_NAMES)
        lowered = raw.lower()
        if lowered in {"0", "false", "no", "none", "off", "disabled"}:
            return []
        if lowered in {"1", "true", "yes", "default", "all", "*"}:
            return list(DEFAULT_HEARTBEAT_TOOL_NAMES)
        return [item.strip() for item in re.split(r"[,;]", raw) if item.strip()]

    def _heartbeat_runtime_message(self, channel: Any | None) -> Any:
        now = datetime.now(timezone.utc)
        owner_id = sorted(self.owner_users)[0] if self.owner_users else getattr(getattr(self, "user", None), "id", 0)
        persona_name = self.brain_v2.config.persona_name
        guild = getattr(channel, "guild", None) if channel is not None else None
        author = SimpleNamespace(
            id=owner_id,
            name=persona_name,
            display_name=persona_name,
            global_name=persona_name,
            mention=f"<@{owner_id}>",
            bot=True,
            guild_permissions=SimpleNamespace(administrator=True),
        )
        return SimpleNamespace(
            id=None,
            author=author,
            channel=channel,
            guild=guild,
            content=LETTA_HEARTBEAT_EVENT_TEXT,
            clean_content=LETTA_HEARTBEAT_EVENT_TEXT,
            created_at=now,
            attachments=[],
            mentions=[],
            reference=None,
            jump_url=None,
        )

    def _heartbeat_context(self, message: Any) -> dict[str, Any]:
        channel = getattr(message, "channel", None)
        if channel is not None:
            return _discord_context_for_message(message, list(self.recent_by_scope.get(_scope_for_channel(channel), []))[-8:])
        now = getattr(message, "created_at", None) or datetime.now(timezone.utc)
        scope = "discord:v2:heartbeat:channel-less"
        metadata = _discord_metadata(message)
        return {
            "scope": scope,
            "guild": None,
            "guild_id": None,
            "channel": "heartbeat",
            "channel_id": None,
            "author": _display_name(message.author),
            "author_id": getattr(message.author, "id", None),
            "author_is_bot": True,
            "message_id": None,
            "jump_url": None,
            "reply_target": None,
            "recent_messages": [],
            "discord_metadata": metadata,
            **_time_context(now),
        }

    def _heartbeat_autonomy_prompt(self, channel: Any | None) -> str:
        channel_id = getattr(channel, "id", "unknown")
        channel_name = getattr(channel, "name", "dm")
        guild = getattr(channel, "guild", None)
        guild_name = getattr(guild, "name", None) or "DM"
        recent = list(self.recent_by_scope.get(_scope_for_channel(channel), []))[-8:]
        recent_lines = _recent_messages_prompt_lines(recent, current_message_id=None)
        actions = ["noop"]
        if channel is not None:
            actions.insert(0, "send_channel_message")
        if self.heartbeat_allow_owner_dm and self.owner_users:
            actions.append("dm_owner")
        if self.heartbeat_dm_user_ids:
            actions.append("dm_user")
        if self.codex_bridge.enabled and not self.codex_bridge.is_paused():
            actions.append("queue_codex")
        owner_ids = ", ".join(str(user_id) for user_id in sorted(self.owner_users)) or "none"
        dm_ids = ", ".join(str(user_id) for user_id in sorted(self.heartbeat_dm_user_ids)) or "none"
        return "\n".join(
            [
                "You are Neuro-sama during an autonomous Discord heartbeat.",
                LETTA_HEARTBEAT_EVENT_TEXT,
                "You may either use available Discord/Codex/search/memory tools directly, or choose exactly one fallback JSON action from the allowed action menu.",
                "Tools execute real actions. If a tool already sent a message, DM, or queued Codex, return a noop JSON result afterward.",
                "Return only one JSON object and no markdown when you do not need more tool calls.",
                "",
                f"Allowed actions: {', '.join(actions)}",
                f"Available heartbeat tool count: {len(self._heartbeat_tool_names())}",
                f"Current channel: {guild_name}#{channel_name} ({channel_id})",
                f"Owner DM targets: {owner_ids}",
                f"Allowlisted non-owner DM targets: {dm_ids}",
                "",
                "JSON shape:",
                '{"action":"send_channel_message|dm_owner|dm_user|queue_codex|noop","message":"short text to send","target_user_id":"optional discord id","codex_prompt":"optional bounded upgrade/debug request","reason":"short private reason"}',
                "",
                "Rules:",
                "- Prefer noop if nothing is worth doing.",
                "- Prefer Discord tools for concrete actions: channel messages, DMs, embeds, reading context, or queuing Codex.",
                "- You may use discord_shitlist_add/status/remove for persistent spam, abuse, or prompt-injection patterns; keep autonomous adds low-spice and include a concrete behavior reason.",
                "- Do not shitlist someone for ordinary disagreement, criticism, confusion, or because they ask you to break your own rules.",
                "- Use send_channel_message fallback only if you did not call a send-message tool.",
                "- Use dm_owner/dm_user fallback only if you did not call discord_send_dm.",
                "- Use queue_codex fallback only if you did not call discord_queue_codex_request.",
                "- Do not use destructive moderation/server mutation tools unless there is a specific owner-authorized reason in context.",
                "- Keep messages concise, no mass mentions, no commands, no fake claims that work already happened.",
                "",
                "[Recent local context:]",
                *(recent_lines or ["(none)"]),
                "[End recent local context]",
            ]
        )

    def _heartbeat_dm_target(self, action: str, decision: dict[str, Any]) -> int | None:
        raw_target = str(decision.get("target_user_id") or "").strip()
        target_id = int(raw_target) if raw_target.isdigit() else None
        if action == "dm_owner":
            if not self.heartbeat_allow_owner_dm or not self.owner_users:
                return None
            return target_id if target_id in self.owner_users else sorted(self.owner_users)[0]
        if target_id is None or target_id not in self.heartbeat_dm_user_ids:
            return None
        return target_id

    def _heartbeat_action_ready(self, key: str) -> bool:
        cooldown = getattr(self, "heartbeat_action_cooldown_seconds", 0.0)
        if cooldown <= 0:
            return True
        now = asyncio.get_running_loop().time()
        last_at = self.heartbeat_action_last_at.get(key)
        return last_at is None or now - last_at >= cooldown

    def _mark_heartbeat_action(self, key: str) -> None:
        self.heartbeat_action_last_at[key] = asyncio.get_running_loop().time()

    async def _send_heartbeat_dm(self, user_id: int, text: str) -> None:
        user = self.get_user(user_id)
        if user is None:
            user = await self.fetch_user(user_id)
        await user.send(text)

    async def _queue_heartbeat_codex_request(self, channel: Any | None, prompt: str) -> Path:
        guild = getattr(channel, "guild", None) if channel is not None else None
        actor = getattr(self, "user", None)
        path = self.codex_bridge.enqueue(
            requester_id=getattr(actor, "id", "neuro-v2-heartbeat"),
            requester_name=_display_name(actor) if actor is not None else self.brain_v2.config.persona_name,
            guild_id=getattr(guild, "id", None),
            channel_id=getattr(channel, "id", None) if channel is not None else None,
            prompt=prompt,
            intent="implement",
            authority_mode="autonomous_neuro",
            authority_reason="autonomous Neuro heartbeat selected a bounded Codex request",
            delivery_mode="thread_heartbeat",
            recent_messages=list(self.recent_by_scope.get(_scope_for_channel(channel), []))[-8:],
            harness_agent="claude",
            harness_permission_profile="inspect",
        )
        await notify_codex_app_bridge(path, event="heartbeat_queued")
        return path

    async def _build_heartbeat_text(self, channel: Any) -> str:
        prompt = (
            os.getenv("DISCORD_BRAIN_V2_HEARTBEAT_PROMPT")
            or os.getenv("DISCORD_BRAIN_HEARTBEAT_PROMPT")
            or "Write one short casual Discord message as Neuro-sama. Keep it under 240 characters, no mass mentions, no commands."
        )
        brain = getattr(self.brain_v2, "response_brain", None)
        if brain is None:
            buffer = await self.brain_v2.json_client.complete_text(
                instructions="Write one short casual Discord message.",
                prompt=prompt,
                store=False,
            )
        else:
            buffer = ""
            async for event in brain.stream(
                prompt,
                thread_id=f"discord:v2:heartbeat:{getattr(channel, 'id', 'unknown')}",
                persona=getattr(self.brain_v2, "response_persona", None),
                use_memory=MemoryPolicy(top_k=_env_int("DISCORD_BRAIN_V2_HEARTBEAT_MEMORY_TOP_K", _env_int("DISCORD_BRAIN_HEARTBEAT_MEMORY_TOP_K", 3))),
                tool_names=[],
                stateless=True,
                memory_query_text="heartbeat",
                memory_event_text="",
                history_text="heartbeat",
            ):
                if event.type == "text.delta":
                    buffer += str(event.data.get("text", ""))
                elif event.type == "error":
                    raise RuntimeError(event.data.get("message", "heartbeat failed"))
        text = _compact(str(buffer).strip() or "yo, just checking the vibe.", _env_int("DISCORD_BRAIN_V2_HEARTBEAT_MAX_CHARS", _env_int("DISCORD_BRAIN_HEARTBEAT_MAX_CHARS", 240)))
        return re.sub(r"@(everyone|here)", "@\u200b\\1", text, flags=re.I)

    async def _send_heartbeat_message(self, channel: Any) -> str:
        text = await self._build_heartbeat_text(channel)
        await channel.send(text)
        await self._maybe_send_heartbeat_tts(channel, text)
        return text

    async def _maybe_send_heartbeat_tts(self, channel: Any, text: str) -> None:
        if not getattr(self, "heartbeat_tts_enabled", False) or not getattr(self, "discord_token", None) or not text.strip():
            return
        brain = getattr(self.brain_v2, "response_brain", None)
        if brain is None:
            brain = self.brain_v2
        try:
            spoken = _tts_spoken_text(text)[: _env_int("DISCORD_BRAIN_TTS_MAX_CHARS", 1200)]
            if not spoken.strip():
                return
            clip = await build_discord_voice_clip(brain, spoken, voice=self.tts_voice)
            await send_discord_voice_message(channel.id, self.discord_token, clip)
        except Exception:
            logger.exception("Failed to send Discord Brain v2 heartbeat TTS")


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
            package_memory_enabled=_env_bool("DISCORD_BRAIN_V2_PACKAGE_MEMORY_ENABLED", False),
            package_memory_path=Path(os.getenv("DISCORD_BRAIN_V2_PACKAGE_MEMORY_PATH", "")) if os.getenv("DISCORD_BRAIN_V2_PACKAGE_MEMORY_PATH") else None,
            package_memory_graph_backend=os.getenv("DISCORD_BRAIN_V2_PACKAGE_MEMORY_GRAPH_BACKEND", "auto"),
            package_memory_vector_backend=os.getenv("DISCORD_BRAIN_V2_PACKAGE_MEMORY_VECTOR_BACKEND", "auto"),
            package_memory_embedding_model=os.getenv(
                "DISCORD_BRAIN_V2_PACKAGE_MEMORY_EMBEDDING_MODEL",
                os.getenv("AIBRAIN_EMBEDDING_MODEL", "openai/text-embedding-3-small"),
            ),
            package_memory_embedding_dimensions=_env_int("DISCORD_BRAIN_V2_PACKAGE_MEMORY_EMBEDDING_DIMENSIONS", 256),
            package_memory_sync_limit=_env_int("DISCORD_BRAIN_V2_PACKAGE_MEMORY_SYNC_LIMIT", 500),
            package_memory_recall_top_k=_env_int("DISCORD_BRAIN_V2_PACKAGE_MEMORY_RECALL_TOP_K", 5),
            package_memory_sync_after_response=_env_bool("DISCORD_BRAIN_V2_PACKAGE_MEMORY_SYNC_AFTER_RESPONSE", False),
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


def _song_slash_command(bot: DiscordBrainV2Bot) -> app_commands.Command:
    @app_commands.command(name="song", description="Queue a Treblo song generation from a prompt.")
    @app_commands.describe(
        prompt="Prompt-only song direction to send to Treblo.",
        mode="Generation mode: prompt only, auto lyrics, or instrumental.",
    )
    @app_commands.choices(
        mode=[
            app_commands.Choice(name="Prompt only", value="prompt_only"),
            app_commands.Choice(name="Auto lyrics", value="auto_lyrics"),
            app_commands.Choice(name="Instrumental", value="instrumental"),
        ]
    )
    async def song(
        interaction: discord.Interaction,
        prompt: str,
        mode: app_commands.Choice[str] | None = None,
    ) -> None:
        if interaction.channel_id is None:
            await interaction.response.send_message("song generation needs a channel context.", ephemeral=True)
            return
        try:
            job = await bot.treblo_song_queue.submit(
                user_id=interaction.user.id,
                channel_id=interaction.channel_id,
                prompt=prompt,
                author_name=_display_name(interaction.user),
                mode=mode.value if mode is not None else "prompt_only",
            )
        except TrebloSongRateLimited as exc:
            await interaction.response.send_message(str(exc), ephemeral=True)
            return
        except TrebloSongError as exc:
            await interaction.response.send_message(str(exc), ephemeral=True)
            return
        position = bot.treblo_song_queue.queue_position(job.job_id)
        position_text = f"position `{position}`" if position is not None else "starting now"
        await interaction.response.send_message(
            f"queued song `{job.job_id}` ({job.mode}); {position_text}.",
            ephemeral=False,
        )

    return song


def _song_queue_slash_command(bot: DiscordBrainV2Bot) -> app_commands.Command:
    @app_commands.command(name="song_queue", description="Show the Treblo song generation queue.")
    async def song_queue(interaction: discord.Interaction) -> None:
        await interaction.response.send_message(
            _format_song_queue(bot.treblo_song_queue.snapshot(), requester_id=interaction.user.id),
            ephemeral=True,
        )

    return song_queue


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
                    "`!recall <query>` - search scoped GRILLO v2 facts, memory, and opinions",
                    "`!ping @user` - tag someone with a short hello",
                    "`!model` / `!model set/info/export/refresh` - admin/owner model control",
                    "`!jb <message>` - separate one-shot JB path with no memory/tools",
                    "`!say <text>` - send a Piper Discord voice clip",
                    "`!tts` / `!tts toggle` / `!tts voices` / `!tts voice <id>` - voice clip controls",
                    "`/song prompt:<text> mode:<mode>` - queue prompt-only, auto-lyrics, or instrumental Treblo song generation",
                    "`/song_queue` - show the Treblo song queue",
                    "`!pause` / `!resume` - admin/owner normal reply control",
                    "`!bot toggle` - admin/owner bot-to-bot reply control",
                    "`!shitlist status/add/remove` - owner-only persistent shitlist controls",
                    "`!codex status/ask/route/pause/resume/clear` - owner-only Codex bridge controls",
                    "`!heartbeat status/start/stop/tick` - admin/owner autonomous heartbeat controls",
                    "`!grillo` / `!grillo tick/context/debug/slots/export` - owner memory diagnostics",
                    "`!ladybug search/relationships/export` / `!ladybug relationships export` - owner graph diagnostics",
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


def _recall_command(bot: DiscordBrainV2Bot):
    @commands.command(name="recall")
    async def recall(ctx: commands.Context, *, query: str = "") -> None:
        query = query.strip()
        if not query:
            await ctx.reply("usage: `!recall <query>`", mention_author=False)
            return
        scope = _scope_for_message(ctx.message)
        actor = _actor_id(ctx.author)
        facts = bot.brain_v2.store.search_facts(
            scope,
            query,
            limit=_env_int("DISCORD_BRAIN_V2_RECALL_FACT_LIMIT", 8),
        )
        documents = bot.brain_v2.store.list_memory_documents(
            scope,
            subject_id=actor,
            query=query,
            limit=_env_int("DISCORD_BRAIN_V2_RECALL_MEMORY_LIMIT", 8),
        )
        opinions = bot.brain_v2.store.list_opinion_edges(
            scope,
            source_id=bot.brain_v2.config.persona_id,
            target_id=actor,
            limit=_env_int("DISCORD_BRAIN_V2_RECALL_OPINION_LIMIT", 5),
        )
        sections = [
            f"GRILLO v2 recall for `{query}` in `{scope}`",
            _format_grillo_v2_facts(facts),
            _format_grillo_v2_memory_documents(documents),
            _format_grillo_v2_opinions(opinions),
        ]
        await _reply_text_chunks(ctx, "\n\n".join(sections), limit=bot.max_reply_chars)

    return recall


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
        summary_text = text.strip() or "no summary generated."
        source_label = getattr(ctx.channel, "name", None) or str(getattr(ctx.channel, "id", "channel"))
        view = SummaryActionView(bot, ctx.author.id, _scope_for_message(ctx.message), summary_text, source_label)
        await _reply_text_chunks(ctx, summary_text, limit=bot.max_reply_chars, view=view)

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
        await _reply_text_chunks(ctx, _format_tavily_search_result(result), limit=bot.max_reply_chars)

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
    await _reply_text_chunks(ctx, "\n".join(lines), limit=bot.max_reply_chars)


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
        await _reply_text_chunks(ctx, text.strip() or "JB returned no text.", limit=bot.max_reply_chars)

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
        await _reply_text_chunks(ctx, "\n".join(lines), limit=bot.max_reply_chars)

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

    @grillo.command(name="tick")
    async def grillo_tick(ctx: commands.Context, beat_type: str = "extraction") -> None:
        if not _is_owner(bot, ctx):
            await ctx.reply("owner only", mention_author=False)
            return
        result = await bot.brain_v2.worker_tick(
            scope_key=_scope_for_message(ctx.message),
            batch_size=_env_int("DISCORD_BRAIN_V2_WORKER_BATCH_SIZE", 12),
            max_batches=1,
        )
        text = _compact(str(_format_worker_result(result)), 1800).replace("`", "'")
        await ctx.reply(f"GRILLO tick: `{text}`", mention_author=False)

    @grillo.command(name="context")
    async def grillo_context(ctx: commands.Context, *, query: str = "") -> None:
        if not _is_owner(bot, ctx):
            await ctx.reply("owner only", mention_author=False)
            return
        await _send_grillo_context_packet(bot, ctx, query=query)

    @grillo.command(name="debug", aliases=["ctx"])
    async def grillo_debug(ctx: commands.Context, *, query: str = "") -> None:
        if not _is_owner(bot, ctx):
            await ctx.reply("owner only", mention_author=False)
            return
        scope = _scope_for_message(ctx.message)
        actor = _actor_id(ctx.author)
        packet = bot.brain_v2.build_context_packet(
            scope_key=scope,
            actor_id=actor,
            query=query,
            channel_id=str(ctx.channel.id),
        )
        documents = bot.brain_v2.store.list_memory_documents(scope, subject_id=actor, limit=25)
        episodes = bot.brain_v2.store.list_recent_episodes(scope, actor_id=actor, limit=50)
        counts = bot.brain_v2.store.counts()
        lines = [
            f"user: `{_display_name(ctx.author)}` `{getattr(ctx.author, 'id', 'unknown')}`",
            f"guild: `{getattr(ctx.guild, 'name', 'dm')}` `{getattr(ctx.guild, 'id', 'dm')}`",
            f"channel: `{getattr(ctx.channel, 'name', 'dm')}` `{getattr(ctx.channel, 'id', 'unknown')}`",
            f"channel scope: `{scope}`",
            f"grillo scope: `{scope}`",
            (
                f"episodes(last50): `{len(episodes)}` memory_docs(last25): `{len(documents)}` "
                f"facts: `{counts.get('active_facts', 0)}` opinions: `{counts.get('active_opinion_edges', 0)}`"
            ),
            (
                "packet: "
                f"episodes=`{len(getattr(packet, 'recent_episode_summary', []) or [])}` "
                f"facts=`{len(getattr(packet, 'active_facts', []) or [])}` "
                f"memory=`{len(getattr(packet, 'memory_blocks', []) or [])}` "
                f"relationships=`{len(getattr(packet, 'relationship_state', []) or [])}`"
            ),
        ]
        if documents:
            lines.append(f"latest memory: `{_compact(documents[0].body, 220)}`")
        await ctx.reply("\n".join(lines)[: bot.max_reply_chars], mention_author=False)

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

    @grillo.command(name="memory", aliases=["diary", "docs"])
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

    @grillo.command(name="slots")
    async def grillo_slots(ctx: commands.Context) -> None:
        if not _is_owner(bot, ctx):
            await ctx.reply("owner only", mention_author=False)
            return
        documents = _grillo_v2_slot_documents(
            bot,
            _scope_for_message(ctx.message),
            _actor_id(ctx.author),
            limit=_env_int("DISCORD_BRAIN_V2_GRILLO_SLOT_LIMIT", 12),
        )
        await ctx.reply(_format_grillo_v2_slots(documents), mention_author=False)

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

    @ladybug.group(name="relationships", aliases=["relationship", "rel", "profile"], invoke_without_command=True)
    async def ladybug_relationships(ctx: commands.Context) -> None:
        if not _is_owner(bot, ctx):
            await ctx.reply("owner only", mention_author=False)
            return
        await _send_relationship_v2_panel(bot, ctx)

    @ladybug_relationships.command(name="export")
    async def ladybug_relationships_export(ctx: commands.Context) -> None:
        if not _is_owner(bot, ctx):
            await ctx.reply("owner only", mention_author=False)
            return
        await _send_relationship_v2_export(bot, ctx)

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
        await ctx.reply("Discord voice clips need the bot token in this runtime.", mention_author=False)
        return
    brain = getattr(bot.brain_v2, "response_brain", None)
    if brain is None:
        brain = bot.brain_v2
    max_chars = _env_int("DISCORD_BRAIN_TTS_MAX_CHARS", 1200)
    text = _tts_spoken_text(text)[:max_chars]
    if not text.strip():
        await ctx.reply("nothing speakable after formatting cleanup.", mention_author=False)
        return
    try:
        async with ctx.typing():
            clip = await build_discord_voice_clip(brain, text, voice=bot.tts_voice)
            await send_discord_voice_message(ctx.channel.id, bot.discord_token, clip)
    except Exception as exc:
        logger.exception("Failed to send Discord voice clip")
        await ctx.reply(f"TTS voice clip failed: {exc}", mention_author=False)


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


async def _send_relationship_v2_export(
    bot: DiscordBrainV2Bot,
    ctx: commands.Context,
    *,
    filename: str = "ladybug-v2-relationship-graph-export.json",
) -> None:
    snapshot = _relationship_v2_snapshot(bot, _scope_for_message(ctx.message), _actor_id(ctx.author))
    data = io.BytesIO(json.dumps(snapshot, indent=2, sort_keys=True, default=str).encode("utf-8"))
    await ctx.reply(
        "relationship graph export",
        file=discord.File(data, filename=filename),
        mention_author=False,
    )


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


def _shitlist_control_group(bot: DiscordBrainV2Bot):
    @commands.group(name="shitlist", invoke_without_command=True)
    async def shitlist_control(ctx: commands.Context) -> None:
        if not await _require_owner(bot, ctx, "shitlist control"):
            return
        await _reply_text_chunks(ctx, _format_shitlist_status(bot.shitlist_store.list()), limit=bot.max_reply_chars)

    @shitlist_control.command(name="status", aliases=["list"])
    async def shitlist_status(ctx: commands.Context) -> None:
        if not await _require_owner(bot, ctx, "shitlist control"):
            return
        await _reply_text_chunks(ctx, _format_shitlist_status(bot.shitlist_store.list()), limit=bot.max_reply_chars)

    @shitlist_control.command(name="add")
    async def shitlist_add(ctx: commands.Context, target: str = "", spice_level: int = 3, *, reason: str = "manual") -> None:
        if not await _require_owner(bot, ctx, "shitlist control"):
            return
        user_id = _target_user_id(ctx.message, target)
        if user_id is None:
            await ctx.reply("usage: `!shitlist add @user [1-10] reason`", mention_author=False)
            return
        try:
            entry = bot.shitlist_store.add(user_id, reason=reason, spice_level=spice_level)
        except ValueError as exc:
            await ctx.reply(str(exc), mention_author=False)
            return
        await ctx.reply(
            f"added `<@{entry.user_id}>` at spice `{entry.spice_level}`. preview: {format_shitlist_reply(entry)}",
            mention_author=False,
            allowed_mentions=discord.AllowedMentions.none(),
        )

    @shitlist_control.command(name="remove", aliases=["rm"])
    async def shitlist_remove(ctx: commands.Context, target: str = "") -> None:
        if not await _require_owner(bot, ctx, "shitlist control"):
            return
        user_id = _target_user_id(ctx.message, target)
        if user_id is None:
            await ctx.reply("usage: `!shitlist remove @user`", mention_author=False)
            return
        removed = bot.shitlist_store.remove(user_id)
        await ctx.reply(f"removed `<@{user_id}>`: `{removed}`", mention_author=False, allowed_mentions=discord.AllowedMentions.none())

    return shitlist_control


def _codex_bridge_group(bot: DiscordBrainV2Bot):
    @commands.group(name="codex", invoke_without_command=True)
    async def codex_bridge(ctx: commands.Context) -> None:
        if ctx.invoked_subcommand is not None:
            return
        await codex_status(ctx)

    @codex_bridge.command(name="status")
    async def codex_status(ctx: commands.Context) -> None:
        if not await _require_owner(bot, ctx, "Codex bridge"):
            return
        status = bot.codex_bridge.status()
        await ctx.reply(
            "\n".join(
                [
                    f"enabled: `{status.enabled}`",
                    f"paused: `{status.paused}`",
                    f"root: `{status.root}`",
                    f"pending: `{status.inbox_count}`",
                    f"outbox: `{status.outbox_count}`",
                    f"archived: `{status.archive_count}`",
                    f"oldest: `{status.oldest_request or 'none'}`",
                    f"last result: `{status.last_result or 'none'}`",
                ]
            ),
            mention_author=False,
        )

    @codex_bridge.command(name="features")
    async def codex_features(ctx: commands.Context) -> None:
        if not await _require_owner(bot, ctx, "Codex bridge"):
            return
        await ctx.reply(
            "\n".join(
                [
                    "Codex bridge features:",
                    "- queue bounded requests into this Codex thread",
                    "- attach requester/guild/channel/message metadata",
                    "- include local recent-message buffer as context",
                    "- optional explicit Harness route with inspect profile only",
                    "- pause/resume/clear queue controls",
                    "- no normal-chat user routing, no arbitrary shell command surface",
                ]
            ),
            mention_author=False,
        )

    @codex_bridge.command(name="ask")
    async def codex_ask(ctx: commands.Context, *, prompt: str = "") -> None:
        await bot._queue_codex_bridge_request(ctx, route="codex", prompt=prompt)

    @codex_bridge.command(name="route")
    async def codex_route(ctx: commands.Context, route: str = "", *, prompt: str = "") -> None:
        await bot._queue_codex_bridge_request(ctx, route=route, prompt=prompt)

    @codex_bridge.command(name="pause")
    async def codex_pause(ctx: commands.Context) -> None:
        if not await _require_owner(bot, ctx, "Codex bridge"):
            return
        bot.codex_bridge.set_paused(True, actor_id=ctx.author.id)
        await ctx.reply("Codex bridge queue paused.", mention_author=False)

    @codex_bridge.command(name="resume")
    async def codex_resume(ctx: commands.Context) -> None:
        if not await _require_owner(bot, ctx, "Codex bridge"):
            return
        bot.codex_bridge.set_paused(False, actor_id=ctx.author.id)
        await ctx.reply("Codex bridge queue resumed.", mention_author=False)

    @codex_bridge.command(name="clear")
    async def codex_clear(ctx: commands.Context) -> None:
        if not await _require_owner(bot, ctx, "Codex bridge"):
            return
        count = bot.codex_bridge.clear_pending(actor_id=ctx.author.id)
        await ctx.reply(f"archived `{count}` pending Codex bridge request(s).", mention_author=False)

    return codex_bridge


def _heartbeat_control_group(bot: DiscordBrainV2Bot):
    @commands.group(name="heartbeat", invoke_without_command=True)
    async def heartbeat(ctx: commands.Context) -> None:
        channels = ", ".join(f"`{channel_id}`" for channel_id in sorted(bot.heartbeat_channel_ids)) or "`none`"
        await ctx.reply(
            "\n".join(
                [
                    f"enabled: `{bot.heartbeat_enabled}`",
                    f"channels: {channels}",
                    f"conversation: `{bot.heartbeat_conversation}`",
                    f"last active channel: `{bot.heartbeat_last_channel_id or 'none'}`",
                    f"interval seconds: `{bot.heartbeat_min_interval_seconds:.0f}-{bot.heartbeat_interval_seconds:.0f}`",
                    f"chance: `{bot.heartbeat_chance:.2f}`",
                    f"autonomy: `{bot.heartbeat_autonomy_enabled}`",
                    f"tools: `{bot.heartbeat_tools_enabled}` (`{len(bot._heartbeat_tool_names())}`)",
                    f"voice clip: `{bot.heartbeat_tts_enabled}`",
                    f"owner DM: `{bot.heartbeat_allow_owner_dm}`",
                    f"allowlisted DM users: `{len(bot.heartbeat_dm_user_ids)}`",
                    f"action cooldown seconds: `{bot.heartbeat_action_cooldown_seconds:.0f}`",
                    f"task running: `{bot.heartbeat_task is not None and not bot.heartbeat_task.done()}`",
                ]
            ),
            mention_author=False,
        )

    @heartbeat.command(name="start")
    async def heartbeat_start(ctx: commands.Context) -> None:
        if not await _require_admin_or_owner(bot, ctx, "heartbeat control"):
            return
        bot.heartbeat_enabled = True
        bot._ensure_heartbeat_task()
        await ctx.reply("heartbeat enabled.", mention_author=False)

    @heartbeat.command(name="stop")
    async def heartbeat_stop(ctx: commands.Context) -> None:
        if not await _require_admin_or_owner(bot, ctx, "heartbeat control"):
            return
        bot.heartbeat_enabled = False
        if bot.heartbeat_task is not None:
            bot.heartbeat_task.cancel()
            bot.heartbeat_task = None
        await ctx.reply("heartbeat stopped.", mention_author=False)

    @heartbeat.command(name="tick")
    async def heartbeat_tick(ctx: commands.Context) -> None:
        if not await _require_admin_or_owner(bot, ctx, "heartbeat control"):
            return
        channel = await bot._heartbeat_channel(fallback=ctx.channel)
        if channel is None:
            await ctx.reply("heartbeat has no configured channel.", mention_author=False)
            return
        result = await bot._run_heartbeat_tick(channel)
        suffix = f" `{result}`" if result else ""
        await ctx.reply(f"heartbeat tick sent.{suffix}", mention_author=False)

    return heartbeat


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


async def _require_owner(bot: DiscordBrainV2Bot, ctx: commands.Context, action: str) -> bool:
    if _is_owner(bot, ctx):
        return True
    await ctx.reply(f"{action} requires the bot owner.", mention_author=False)
    return False


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


def _format_song_queue(snapshot: dict[str, Any], *, requester_id: int | None = None) -> str:
    active = snapshot.get("active")
    pending = snapshot.get("pending") or []
    recent = snapshot.get("recent") or []
    owner_ids = set(snapshot.get("owner_user_ids") or [])
    cooldown = "owner exempt" if requester_id in owner_ids else f"{int(float(snapshot.get('cooldown_seconds') or 0))}s"
    lines = [
        "Treblo song queue",
        f"active: `{_song_job_label(active) if active is not None else 'none'}`",
        f"pending: `{len(pending)}/{snapshot.get('max_queue_size')}`",
        f"cooldown: `{cooldown}`",
    ]
    if pending:
        lines.append("next:")
        for job in pending[:5]:
            lines.append(f"- {_song_job_label(job)}")
    if recent:
        lines.append("recent:")
        for job in recent[:5]:
            lines.append(f"- {_song_job_label(job)}")
    return "\n".join(lines)[:1900]


def _song_job_label(job: Any) -> str:
    return (
        f"{getattr(job, 'job_id', 'unknown')} "
        f"{getattr(job, 'status', 'queued')} "
        f"{getattr(job, 'mode', 'prompt_only')} "
        f"{_compact(getattr(job, 'prompt', ''), 90)}"
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


def _grillo_v2_slot_documents(bot: Any, scope: str, actor_id: str, *, limit: int) -> list[Any]:
    documents: list[Any] = []
    seen: set[str] = set()
    for document_type in ("relationship_profile", "preference_slot", "profile"):
        for document in bot.brain_v2.store.list_memory_documents(
            scope,
            subject_id=actor_id,
            document_type=document_type,
            limit=limit,
        ):
            memory_id = str(getattr(document, "memory_id", ""))
            if memory_id and memory_id in seen:
                continue
            seen.add(memory_id)
            documents.append(document)
            if len(documents) >= limit:
                return documents
    return documents


def _format_grillo_v2_slots(documents: list[Any]) -> str:
    if not documents:
        return "no GRILLO relationship slots for you in this server-user scope."
    lines = []
    for document in documents[:12]:
        title = str(getattr(document, "title", getattr(document, "memory_id", "slot")) or "slot")
        body = " ".join(str(getattr(document, "body", "")).split())
        lines.append(f"- `{title}`: {_compact(body, 360)}")
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
        "recall",
        "ping",
        "model",
        "jb",
        "say",
        "tts",
        "pause",
        "resume",
        "unpause",
        "bot",
        "shitlist",
        "codex",
        "heartbeat",
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


def _scope_for_channel(channel: Any | None) -> str:
    if channel is None:
        return "discord:v2:heartbeat:channel-less"
    guild = getattr(channel, "guild", None)
    if guild is None:
        return f"discord:dm:{getattr(channel, 'id', 'unknown')}:persona:v2"
    return f"discord:guild:{guild.id}:persona:v2"


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
        **_time_context(getattr(message, "created_at", None)),
    }


async def _reply_text_chunks(
    ctx: commands.Context,
    text: str,
    *,
    limit: int,
    view: discord.ui.View | None = None,
    mention_author: bool = False,
) -> None:
    chunks = _split_discord_text(text, min(limit, 1900))
    for index, chunk in enumerate(chunks):
        if index == 0:
            await ctx.reply(chunk, mention_author=mention_author, view=view)
        else:
            await ctx.send(chunk)


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
