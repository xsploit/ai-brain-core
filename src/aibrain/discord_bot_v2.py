from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

import discord
from discord.ext import commands

from .brain_v2 import BrainV2, BrainV2Config
from .env import load_env_file


DEFAULT_DISCORD_V2_PREFIX = "!n2"
logger = logging.getLogger("aibrain.discord_v2")


class DiscordBrainV2Bot(commands.Bot):
    def __init__(self, *, brain: BrainV2):
        intents = discord.Intents.default()
        intents.message_content = _env_bool("DISCORD_BRAIN_V2_MESSAGE_CONTENT_INTENT", True)
        intents.guilds = True
        intents.messages = True
        intents.members = _env_bool("DISCORD_BRAIN_V2_MEMBERS_INTENT", False)
        super().__init__(
            command_prefix=os.getenv("DISCORD_BRAIN_V2_COMMAND_PREFIX", DEFAULT_DISCORD_V2_PREFIX),
            help_command=None,
            intents=intents,
        )
        self.brain_v2 = brain
        self.allowed_guilds = _csv_ints("DISCORD_BRAIN_V2_ALLOWED_GUILD_IDS")
        self.allowed_users = _csv_ints("DISCORD_BRAIN_V2_ALLOWED_USER_IDS")
        self.respond_to_dms = _env_bool("DISCORD_BRAIN_V2_RESPOND_TO_DMS", True)
        self.respond_to_mentions = _env_bool("DISCORD_BRAIN_V2_RESPOND_TO_MENTIONS", True)
        self.require_mention_in_guilds = _env_bool("DISCORD_BRAIN_V2_REQUIRE_MENTION_IN_GUILDS", True)
        self.max_reply_chars = _env_int("DISCORD_BRAIN_V2_MAX_REPLY_CHARS", 1900)
        self.add_command(_status_command(self))

    async def on_ready(self) -> None:
        logger.info("Discord Brain v2 bot logged in as %s/%s", self.user.id if self.user else "unknown", self.user)

    async def on_message(self, message: discord.Message) -> None:
        if self.user is not None and message.author.id == self.user.id:
            return
        if message.content.startswith(str(self.command_prefix)):
            await self.process_commands(message)
            return
        if not self._allowed(message) or not self._should_respond(message):
            return
        async with message.channel.typing():
            response = await self.brain_v2.respond(
                scope_key=_scope_for_message(message),
                actor_id=_actor_id(message.author),
                user_text=_message_text(message),
                source="discord",
                channel_id=str(message.channel.id),
                metadata=_discord_metadata(message),
            )
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
        return bool(resolved is not None and self.user is not None and getattr(getattr(resolved, "author", None), "id", None) == self.user.id)


def build_brain_v2() -> BrainV2:
    return BrainV2(
        BrainV2Config(
            database_path=Path(os.getenv("DISCORD_BRAIN_V2_DATABASE_PATH", "discord_brain_v2.sqlite3")),
            model=os.getenv("DISCORD_BRAIN_V2_MODEL", os.getenv("DISCORD_BRAIN_MODEL", "deepseek/deepseek-v4-flash")),
            provider="vercel",
            persona_id=os.getenv("DISCORD_BRAIN_V2_PERSONA_ID", "neuro-sama-v2"),
            persona_name=os.getenv("DISCORD_BRAIN_V2_PERSONA_NAME", "Neuro-sama"),
        )
    )


def _status_command(bot: DiscordBrainV2Bot):
    @commands.command(name="status")
    async def status(ctx: commands.Context) -> None:
        await ctx.reply(
            f"Brain v2 online. model=`{bot.brain_v2.config.model}` grillo=`v2`",
            mention_author=False,
        )

    return status


def _scope_for_message(message: discord.Message) -> str:
    if message.guild is None:
        return f"discord:dm:{message.author.id}:persona:v2"
    return f"discord:guild:{message.guild.id}:persona:v2"


def _actor_id(user: discord.abc.User) -> str:
    return f"discord_user:{user.id}"


def _message_text(message: discord.Message) -> str:
    return (getattr(message, "clean_content", None) or getattr(message, "content", "") or "").strip()


def _discord_metadata(message: discord.Message) -> dict[str, Any]:
    guild = message.guild
    channel = message.channel
    author = message.author
    return {
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
