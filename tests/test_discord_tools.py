import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from aibrain.discord_bot import DEFAULT_DISCORD_TOOL_NAMES, build_brain, build_persona
from aibrain.discord_tools import (
    DISCORD_AGENT_TOOL_NAMES,
    DISCORD_TOOL_CONTEXT,
    DiscordToolError,
    DiscordToolRuntime,
    discord_audit_permissions,
    discord_can_do,
    discord_get_capabilities,
    discord_get_channel_overwrites,
    discord_get_current_context,
    discord_get_member,
    discord_get_permissions,
    discord_list_bot_guilds,
    discord_list_members,
    discord_list_voice_states,
    discord_queue_codex_request,
    discord_read_channel_history,
    discord_search_members,
    discord_send_channel_message,
    discord_send_dm,
    discord_send_file,
    discord_send_rich_embed,
    discord_timeout_member,
    register_discord_tools,
)
from aibrain.codex_bridge import CodexBridgeQueue
from aibrain.tools import ToolRegistry


class _Perms:
    def __init__(self, **values):
        for name in (
            "administrator",
            "view_channel",
            "read_message_history",
            "send_messages",
            "manage_messages",
            "add_reactions",
            "create_public_threads",
            "manage_threads",
            "manage_channels",
            "manage_roles",
            "manage_webhooks",
            "create_instant_invite",
            "moderate_members",
            "kick_members",
            "ban_members",
            "attach_files",
            "manage_guild",
            "view_audit_log",
            "move_members",
            "connect",
            "speak",
            "use_voice_activation",
        ):
            setattr(self, name, values.get(name, False))


class _Role:
    def __init__(self, position: int):
        self.id = position
        self.name = f"role-{position}"
        self.position = position

    def __gt__(self, other):
        return self.position > other.position


class _Member:
    bot = False

    def __init__(self, member_id: int, *, name: str, perms: _Perms, role_position: int = 1, bot: bool = False):
        self.id = member_id
        self.name = name
        self.display_name = name
        self.guild_permissions = perms
        self.top_role = _Role(role_position)
        self.roles = [self.top_role]
        self.bot = bot
        self.timeout_until = None

    async def timeout(self, until, *, reason=None):
        self.timeout_until = until
        self.timeout_reason = reason

    async def add_roles(self, role, *, reason=None):
        self.roles.append(role)
        self.add_role_reason = reason

    async def remove_roles(self, role, *, reason=None):
        self.roles = [item for item in self.roles if item.id != role.id]
        self.remove_role_reason = reason

    async def move_to(self, channel, *, reason=None):
        self.voice_channel = channel
        self.move_reason = reason


class _Message:
    attachments = []
    jump_url = None

    def __init__(self, message_id: int, channel, author, content: str):
        self.id = message_id
        self.channel = channel
        self.author = author
        self.content = content
        self.clean_content = content
        self.created_at = datetime(2026, 6, 20, 12, 0, tzinfo=timezone.utc)
        self.reactions = []

    async def add_reaction(self, emoji):
        self.reactions.append(emoji)

    async def delete(self, *, reason=None):
        self.deleted = reason or True

    async def reply(self, content, **kwargs):
        return await self.channel.send(content, **kwargs)


class _DMUser:
    def __init__(self, user_id: int = 777):
        self.id = user_id
        self.name = f"user-{user_id}"
        self.display_name = self.name
        self.sent = []

    async def send(self, content, **kwargs):
        message = SimpleNamespace(
            id=42,
            author=SimpleNamespace(id=999, name="bot", bot=True),
            channel=SimpleNamespace(id=None),
            content=content,
            clean_content=content,
            created_at=datetime(2026, 6, 20, 12, 0, tzinfo=timezone.utc),
            attachments=[],
            jump_url=None,
        )
        self.sent.append((content, kwargs))
        return message


class _Channel:
    def __init__(self, channel_id: int, name: str):
        self.id = channel_id
        self.name = name
        self.mention = f"<#{channel_id}>"
        self.permission_map = {}
        self.messages = []
        self.sent = []
        self.overwrites = {}
        self.members = []

    def permissions_for(self, member):
        return self.permission_map.get(member.id, _Perms())

    async def send(self, content, **kwargs):
        message = _Message(10_000 + len(self.sent), self, kwargs.get("author", SimpleNamespace(id=999, name="bot")), content)
        for key, value in kwargs.items():
            setattr(message, key, value)
        self.sent.append((content, kwargs))
        return message

    async def fetch_message(self, message_id):
        for message in self.messages:
            if message.id == message_id:
                return message
        raise LookupError(message_id)

    def history(self, *, limit, before=None):
        async def gen():
            before_id = getattr(before, "id", None)
            yielded = 0
            for message in self.messages:
                if before_id is not None and message.id >= before_id:
                    continue
                yield message
                yielded += 1
                if yielded >= limit:
                    break

        return gen()


class _Guild:
    def __init__(self, *, actor: _Member, bot: _Member, target: _Member, channel: _Channel):
        self.id = 55
        self.name = "guild"
        self.owner_id = 999_999
        self.me = bot
        self.channels = [channel]
        self.threads = []
        self.roles = [_Role(0), _Role(5), actor.top_role, bot.top_role, target.top_role]
        self.members = {actor.id: actor, bot.id: bot, target.id: target}
        for item in self.channels:
            item.guild = self

    def get_channel(self, channel_id):
        return next((channel for channel in self.channels if channel.id == channel_id), None)

    def get_channel_or_thread(self, channel_id):
        return self.get_channel(channel_id)

    def get_member(self, member_id):
        return self.members.get(member_id)

    def get_role(self, role_id):
        return next((role for role in self.roles if role.id == role_id), None)

    async def fetch_member(self, member_id):
        return self.members[member_id]


@contextmanager
def _tool_context(*, actor_perms=None, bot_perms=None, target_perms=None, actor_role=10, bot_role=9, target_role=1):
    channel = _Channel(123, "bot-chat")
    actor = _Member(1, name="actor", perms=actor_perms or _Perms(), role_position=actor_role)
    bot_member = _Member(2, name="bot", perms=bot_perms or _Perms(), role_position=bot_role, bot=True)
    target = _Member(3, name="target", perms=target_perms or _Perms(), role_position=target_role)
    channel.permission_map = {
        actor.id: actor.guild_permissions,
        bot_member.id: bot_member.guild_permissions,
        target.id: target.guild_permissions,
    }
    guild = _Guild(actor=actor, bot=bot_member, target=target, channel=channel)
    message = _Message(500, channel, actor, "!brain test")
    message.guild = guild
    runtime_bot = SimpleNamespace(user=bot_member, guilds=[guild], intents=SimpleNamespace(members=True))
    token = DISCORD_TOOL_CONTEXT.set(DiscordToolRuntime(bot=runtime_bot, message=message))
    try:
        yield SimpleNamespace(
            guild=guild,
            channel=channel,
            actor=actor,
            bot=bot_member,
            runtime_bot=runtime_bot,
            target=target,
            message=message,
        )
    finally:
        DISCORD_TOOL_CONTEXT.reset(token)


def test_register_discord_tools_exposes_agentic_suite():
    registry = ToolRegistry()

    register_discord_tools(registry)

    assert set(DISCORD_AGENT_TOOL_NAMES).issubset(registry._tools)
    assert "discord_queue_codex_request" in registry._tools
    assert "discord_list_bot_guilds" in registry._tools
    assert "discord_send_dm" in registry._tools
    assert "discord_create_text_channel" in registry._tools
    assert "discord_get_capabilities" in registry._tools


def test_discord_brain_exposes_discord_schemas_to_model():
    brain = build_brain()
    persona = build_persona()

    schemas = brain._tool_schemas(persona, DEFAULT_DISCORD_TOOL_NAMES)

    assert set(DISCORD_AGENT_TOOL_NAMES).issubset({schema["name"] for schema in schemas})


def test_capabilities_reports_owner_gate(monkeypatch):
    monkeypatch.setenv("DISCORD_BRAIN_OWNER_USER_IDS", "1")
    with _tool_context(actor_perms=_Perms(administrator=True)) as ctx:
        result = asyncio.run(discord_get_capabilities())

        assert result["actor"]["id"] == ctx.actor.id
        assert result["actor"]["is_owner"] is True
        assert "discord_create_text_channel" in result["owner_only_tools"]


def test_allowed_user_is_not_implicitly_owner(monkeypatch):
    monkeypatch.setenv("DISCORD_BRAIN_ALLOWED_USER_IDS", "1")
    monkeypatch.delenv("DISCORD_BRAIN_OWNER_USER_IDS", raising=False)
    with _tool_context(actor_perms=_Perms(administrator=True)) as ctx:
        result = asyncio.run(discord_get_capabilities())

        assert result["actor"]["id"] == ctx.actor.id
        assert result["actor"]["is_owner"] is False


def test_get_permissions_works_from_dm_with_guild_and_channel(monkeypatch):
    monkeypatch.setenv("DISCORD_BRAIN_OWNER_USER_IDS", "1")
    with _tool_context(
        actor_perms=_Perms(administrator=True, view_channel=True),
        bot_perms=_Perms(view_channel=True, send_messages=True, read_message_history=True),
    ) as ctx:
        ctx.message.guild = None
        ctx.message.channel = SimpleNamespace(id=9_999, name="dm")

        result = asyncio.run(discord_get_permissions(guild_id=ctx.guild.id, channel_id=ctx.channel.id))

        assert result["guild"]["id"] == ctx.guild.id
        assert result["channel"]["id"] == ctx.channel.id
        assert result["member"]["id"] == ctx.actor.id
        assert result["bot_permissions"]["send_messages"] is True


def test_audit_permissions_reports_channel_capabilities_from_dm(monkeypatch):
    monkeypatch.setenv("DISCORD_BRAIN_OWNER_USER_IDS", "1")
    with _tool_context(
        bot_perms=_Perms(
            view_channel=True,
            send_messages=True,
            read_message_history=True,
            connect=True,
            speak=True,
        )
    ) as ctx:
        ctx.message.guild = None

        result = asyncio.run(discord_audit_permissions(guild_id=ctx.guild.id))

        assert result["guild"]["id"] == ctx.guild.id
        assert result["channels"][0]["can_send"] is True
        assert result["channels"][0]["can_speak"] is True


def test_member_lookup_can_target_owner_authorized_guild_from_dm(monkeypatch):
    monkeypatch.setenv("DISCORD_BRAIN_OWNER_USER_IDS", "1")
    with _tool_context(actor_perms=_Perms(administrator=True)) as ctx:
        ctx.message.guild = None
        ctx.message.channel = SimpleNamespace(id=9_999, name="dm")

        listed = asyncio.run(discord_list_members(guild_id=ctx.guild.id))
        searched = asyncio.run(discord_search_members("target", guild_id=ctx.guild.id))
        fetched = asyncio.run(discord_get_member(ctx.target.id, guild_id=ctx.guild.id))

        assert listed["guild_id"] == ctx.guild.id
        assert ctx.target.id in {member["id"] for member in listed["members"]}
        assert searched["members"][0]["id"] == ctx.target.id
        assert fetched["member"]["id"] == ctx.target.id


def test_context_and_can_do_report_cross_guild_permissions_from_dm(monkeypatch):
    monkeypatch.setenv("DISCORD_BRAIN_OWNER_USER_IDS", "1")
    with _tool_context(
        actor_perms=_Perms(administrator=True),
        bot_perms=_Perms(view_channel=True, send_messages=True, attach_files=True),
    ) as ctx:
        ctx.message.guild = None
        ctx.message.channel = SimpleNamespace(id=9_999, name="dm")

        current = asyncio.run(discord_get_current_context())
        result = asyncio.run(discord_can_do("send_file", guild_id=ctx.guild.id, channel_id=ctx.channel.id))

        assert current["is_dm"] is True
        assert result["allowed"] is True
        assert result["requirements"] == ["view_channel", "send_messages", "attach_files"]


def test_send_file_from_dm_owner_to_guild_channel(monkeypatch):
    monkeypatch.setenv("DISCORD_BRAIN_OWNER_USER_IDS", "1")
    with _tool_context(
        bot_perms=_Perms(view_channel=True, send_messages=True, attach_files=True),
    ) as ctx:
        ctx.message.guild = None
        ctx.message.channel = SimpleNamespace(id=9_999, name="dm")

        result = asyncio.run(discord_send_file("report?.txt", "hello", channel_id=ctx.channel.id, message="sent"))

        assert result["sent"] is True
        assert result["filename"] == "report_.txt"
        assert ctx.channel.sent[0][0] == "sent"
        assert ctx.channel.sent[0][1]["file"].filename == "report_.txt"


def test_get_channel_overwrites_serializes_allow_and_deny(monkeypatch):
    class _Overwrite:
        def __iter__(self):
            return iter([("send_messages", True), ("view_channel", False), ("attach_files", None)])

    monkeypatch.setenv("DISCORD_BRAIN_OWNER_USER_IDS", "1")
    with _tool_context(actor_perms=_Perms(administrator=True), bot_perms=_Perms(view_channel=True)) as ctx:
        ctx.channel.overwrites = {ctx.bot: _Overwrite()}

        result = asyncio.run(discord_get_channel_overwrites(ctx.channel.id))

        assert result["overwrites"][0]["target"]["id"] == ctx.bot.id
        assert result["overwrites"][0]["allow"] == ["send_messages"]
        assert result["overwrites"][0]["deny"] == ["view_channel"]


def test_list_voice_states_reports_connected_members(monkeypatch):
    monkeypatch.setenv("DISCORD_BRAIN_OWNER_USER_IDS", "1")
    with _tool_context(bot_perms=_Perms(view_channel=True)) as ctx:
        ctx.actor.voice = SimpleNamespace(mute=False, deaf=False, self_mute=True, self_deaf=False)
        ctx.channel.members = [ctx.actor]

        result = asyncio.run(discord_list_voice_states(guild_id=ctx.guild.id))

        assert result["count"] == 1
        assert result["voice_states"][0]["member"]["id"] == ctx.actor.id
        assert result["voice_states"][0]["self_mute"] is True


def test_bot_guild_list_is_owner_only(monkeypatch):
    monkeypatch.setenv("DISCORD_BRAIN_OWNER_USER_IDS", "999")
    with _tool_context():
        with pytest.raises(DiscordToolError, match="requires the configured bot owner"):
            asyncio.run(discord_list_bot_guilds())

    monkeypatch.setenv("DISCORD_BRAIN_OWNER_USER_IDS", "1")
    with _tool_context() as ctx:
        result = asyncio.run(discord_list_bot_guilds())

        assert result["guilds"][0]["id"] == ctx.guild.id


def test_codex_queue_request_is_owner_only(monkeypatch, tmp_path):
    monkeypatch.setenv("DISCORD_BRAIN_OWNER_USER_IDS", "999")
    with _tool_context() as ctx:
        ctx.runtime_bot.codex_bridge = CodexBridgeQueue(tmp_path / "bridge", enabled=True)

        with pytest.raises(DiscordToolError, match="requires the configured bot owner"):
            asyncio.run(discord_queue_codex_request("add a feature"))


def test_codex_queue_request_writes_bridge_file(monkeypatch, tmp_path):
    monkeypatch.setenv("DISCORD_BRAIN_OWNER_USER_IDS", "1")
    with _tool_context() as ctx:
        ctx.runtime_bot.codex_bridge = CodexBridgeQueue(tmp_path / "bridge", enabled=True, thread_id="thread-123")
        ctx.runtime_bot._context_for_message = lambda message: {
            "recent_messages": [{"author": "actor", "content": "please upgrade yourself"}]
        }

        result = asyncio.run(discord_queue_codex_request("add a better status panel", route="codex"))

        files = ctx.runtime_bot.codex_bridge.pending_files()
        assert result["queued"] is True
        assert result["delivery_mode"] == "thread_heartbeat"
        assert result["file"] == files[0].name
        payload = files[0].read_text(encoding="utf-8")
        assert '"authority"' in payload
        assert '"mode": "manual_owner"' in payload
        assert "add a better status panel" in payload
        assert "please upgrade yourself" in payload


def test_list_members_requires_admin_or_owner_and_members_intent():
    with _tool_context(actor_perms=_Perms(administrator=True)) as ctx:
        ctx.runtime_bot.intents.members = False
        with pytest.raises(DiscordToolError, match="Members Intent"):
            asyncio.run(discord_list_members())

        ctx.runtime_bot.intents.members = True
        result = asyncio.run(discord_list_members())

        assert {member["id"] for member in result["members"]} >= {ctx.actor.id, ctx.target.id}


def test_send_channel_message_requires_requester_and_bot_send_permission():
    with _tool_context(
        actor_perms=_Perms(view_channel=True),
        bot_perms=_Perms(view_channel=True, send_messages=True),
    ):
        with pytest.raises(DiscordToolError, match="Current user lacks send_messages"):
            asyncio.run(discord_send_channel_message("hello", channel_id=123))

    with _tool_context(
        actor_perms=_Perms(view_channel=True, send_messages=True),
        bot_perms=_Perms(view_channel=True, send_messages=True),
    ) as ctx:
        result = asyncio.run(discord_send_channel_message("hello @everyone", channel_id=123))

        assert result["sent"] is True
        assert ctx.channel.sent[0][0] == "hello @everyone"
        assert ctx.channel.sent[0][1]["allowed_mentions"].everyone is False


def test_send_dm_requires_admin_or_owner_and_blocks_everyone_mentions():
    with _tool_context() as ctx:
        dm_user = _DMUser(777)
        ctx.runtime_bot.fetch_user = lambda user_id: dm_user

        with pytest.raises(DiscordToolError, match="requires a Discord admin or bot owner"):
            asyncio.run(discord_send_dm(777, "hello"))

    with _tool_context(actor_perms=_Perms(administrator=True)) as ctx:
        dm_user = _DMUser(777)

        async def fetch_user(user_id):
            assert user_id == 777
            return dm_user

        ctx.runtime_bot.fetch_user = fetch_user
        result = asyncio.run(discord_send_dm(777, "hello @everyone"))

        assert result["sent"] is True
        assert result["user"]["id"] == 777
        assert dm_user.sent[0][0] == "hello @everyone"
        assert dm_user.sent[0][1]["allowed_mentions"].everyone is False


def test_send_rich_embed_builds_structured_embed_and_blocks_everyone_mentions():
    with _tool_context(
        actor_perms=_Perms(view_channel=True, send_messages=True),
        bot_perms=_Perms(view_channel=True, send_messages=True),
    ) as ctx:
        result = asyncio.run(
            discord_send_rich_embed(
                title="Memory Report",
                description="Here is the clean version.",
                fields=[
                    {"name": "Scope", "value": "server-user", "inline": True},
                    {"name": "Status", "value": "ready", "inline": True},
                ],
                color="#55AAFF",
                image_url="https://example.com/image.png",
                footer_text="Neuro memory tools",
                content="@everyone summary",
                channel_id=123,
            )
        )

        sent_content, kwargs = ctx.channel.sent[0]
        embed = kwargs["embed"]

        assert result["sent"] is True
        assert sent_content == "@everyone summary"
        assert kwargs["allowed_mentions"].everyone is False
        assert embed.title == "Memory Report"
        assert embed.description == "Here is the clean version."
        assert embed.color.value == 0x55AAFF
        assert embed.image.url == "https://example.com/image.png"
        assert [field.name for field in embed.fields] == ["Scope", "Status"]
        assert result["embed"]["fields"][0]["inline"] is True


def test_read_channel_history_requires_history_permission_and_serializes_messages():
    actor_perms = _Perms(view_channel=True, read_message_history=True)
    bot_perms = _Perms(view_channel=True, read_message_history=True)
    with _tool_context(actor_perms=actor_perms, bot_perms=bot_perms) as ctx:
        ctx.channel.messages = [
            _Message(1, ctx.channel, ctx.actor, "first"),
            _Message(2, ctx.channel, ctx.bot, "bot message"),
        ]

        result = asyncio.run(discord_read_channel_history(channel_id=123, limit=5, include_bot_messages=False))

        assert result["channel"]["id"] == 123
        assert [message["content"] for message in result["messages"]] == ["first"]


def test_timeout_member_requires_confirm_and_moderation_permissions(monkeypatch):
    actor_perms = _Perms(moderate_members=True)
    bot_perms = _Perms(moderate_members=True)
    monkeypatch.setenv("DISCORD_BRAIN_MAX_TIMEOUT_SECONDS", "60")
    with _tool_context(actor_perms=actor_perms, bot_perms=bot_perms) as ctx:
        with pytest.raises(DiscordToolError, match="requires confirm=true"):
            asyncio.run(discord_timeout_member(ctx.target.id, 120, confirm=False))

        result = asyncio.run(discord_timeout_member(ctx.target.id, 120, reason="test", confirm=True))

        assert result["timed_out"] is True
        assert result["user_id"] == ctx.target.id
        assert ctx.target.timeout_until is not None
        assert ctx.target.timeout_reason == "test"
