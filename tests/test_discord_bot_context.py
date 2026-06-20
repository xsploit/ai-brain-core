import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

from aibrain.discord_bot import (
    DiscordBrainBot,
    ModelSelectView,
    _build_jb_persona,
    _format_grillo_export,
    _ordered_model_choices,
    _scoped_ladybug_facts,
    _time_context,
)
from aibrain.model_catalog import ModelChoice, is_chat_model_id


class _TypingContext:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeChannel:
    id = 456
    name = "bot-chat"

    def __init__(self):
        self.sent = []

    def typing(self):
        return _TypingContext()

    async def send(self, content):
        self.sent.append(content)


class _FakeReply:
    def __init__(self, channel):
        self.channel = channel
        self.edits = []

    async def edit(self, *, content):
        self.edits.append(content)


class _FakeAttachment:
    filename = "notes.txt"
    content_type = "text/plain"
    url = "https://cdn.example.invalid/notes.txt"

    def __init__(self, data: bytes):
        self._data = data
        self.size = len(data)

    async def read(self, *, use_cached=True):
        return self._data


class _FakeBrain:
    memory_stack = None

    def __init__(self):
        self.prompt = None
        self.kwargs = None

    async def stream(self, prompt, **kwargs):
        self.prompt = prompt
        self.kwargs = kwargs
        yield SimpleNamespace(type="text.delta", data={"text": "ok"})
        yield SimpleNamespace(type="response.done", data={})


class _FakeGraphStore:
    async def search_facts(self, query):
        return [
            SimpleNamespace(id="in-scope", source_event_id="event-1"),
            SimpleNamespace(id="out-of-scope", source_event_id="event-2"),
        ]


class _FakeRawLog:
    async def list_thread_events(self, thread_id, limit=100):
        assert thread_id == "discord:dm:123"
        return [SimpleNamespace(id="event-1")]


class _FakeModelBot:
    def __init__(self, current_model: str):
        self.current_model = current_model
        self.selected_model = None

    def _current_model(self):
        return self.current_model

    def _set_runtime_model(self, model_id: str):
        self.selected_model = model_id
        self.current_model = model_id


def _fake_message(created_at: datetime):
    author = SimpleNamespace(id=123, display_name="Subsect", global_name=None, bot=False)
    channel = _FakeChannel()
    reply = _FakeReply(channel)

    async def _reply(content, *, mention_author=False):
        reply.edits.append(content)
        return reply

    return SimpleNamespace(
        guild=None,
        id=789,
        author=author,
        channel=channel,
        created_at=created_at,
        clean_content="what day is it?",
        content="what day is it?",
        attachments=[],
        reply=_reply,
        _fake_reply=reply,
    )


def test_time_context_defaults_to_los_angeles(monkeypatch):
    monkeypatch.delenv("DISCORD_BRAIN_TIMEZONE", raising=False)
    now = datetime(2026, 6, 19, 16, 30, tzinfo=timezone.utc)
    message_created_at = datetime(2026, 6, 19, 16, 15, tzinfo=timezone.utc)

    context = _time_context(message_created_at, now=now)

    assert context["local_timezone"] == "America/Los_Angeles"
    assert context["local_date"] == "2026-06-19"
    assert context["local_time"] == "09:30:00"
    assert context["local_now"] == "2026-06-19T09:30:00-07:00"
    assert context["message_local_created_at"] == "2026-06-19T09:15:00-07:00"
    assert context["utc_now"] == "2026-06-19T16:30:00+00:00"


def test_discord_message_context_and_prompt_include_local_time(monkeypatch):
    monkeypatch.setenv("DISCORD_BRAIN_TIMEZONE", "America/Los_Angeles")
    message = _fake_message(datetime(2026, 6, 19, 16, 15, tzinfo=timezone.utc))
    bot = DiscordBrainBot.__new__(DiscordBrainBot)
    bot.recent_by_scope = {}
    bot.brain = SimpleNamespace(memory_stack=None)

    context = bot._context_for_message(message)
    prompt = asyncio.run(bot._build_prompt_for_message(message, context["scope"], "what day is it?"))

    assert context["local_timezone"] == "America/Los_Angeles"
    assert context["local_date"]
    assert context["local_now"]
    assert context["message_local_created_at"] == "2026-06-19T09:15:00-07:00"
    assert "Local date/time (America/Los_Angeles):" in prompt
    assert "Message sent at: 2026-06-19T09:15:00-07:00" in prompt


def test_jb_persona_is_separate_from_normal_prompt(monkeypatch):
    monkeypatch.setenv("DISCORD_BRAIN_TIMEZONE", "America/Los_Angeles")
    message = _fake_message(datetime(2026, 6, 19, 16, 15, tzinfo=timezone.utc))
    bot = DiscordBrainBot.__new__(DiscordBrainBot)
    bot.recent_by_scope = {}
    bot.brain = SimpleNamespace(memory_stack=None)

    normal = asyncio.run(bot._build_prompt_for_message(message, "discord:dm:123", "write normally"))
    base_persona = SimpleNamespace(
        id="neuro-sama",
        name="Neuro-sama",
        tools=["discord_context", "remember", "search_memory", "current_time", "brain_context"],
    )
    jb_persona = _build_jb_persona("DOC PROMPT", fallback_model="deepseek/test", base_persona=base_persona)

    assert "DOC PROMPT" not in normal
    assert "DOC PROMPT" in jb_persona.instructions
    assert jb_persona.id == "neuro-sama"
    assert jb_persona.name == "JB"
    assert jb_persona.model == "deepseek/test"
    assert "search_memory" in jb_persona.tools
    assert "remember" in jb_persona.tools
    assert "Neuro-sama" not in jb_persona.name


def test_jb_reply_uses_jb_persona_with_memory_enabled(monkeypatch):
    monkeypatch.setenv("DISCORD_BRAIN_TIMEZONE", "America/Los_Angeles")
    message = _fake_message(datetime(2026, 6, 19, 16, 15, tzinfo=timezone.utc))
    brain = _FakeBrain()
    bot = DiscordBrainBot.__new__(DiscordBrainBot)
    bot.recent_by_scope = {}
    bot.brain = brain
    bot.persona = SimpleNamespace(
        id="neuro-sama",
        name="Neuro-sama",
        tools=["discord_context", "remember", "search_memory", "current_time", "brain_context"],
    )
    bot.max_reply_chars = 1900
    bot.edit_interval_seconds = 0.25
    jb_persona = _build_jb_persona("DOC PROMPT", fallback_model="deepseek/test", base_persona=bot.persona)

    asyncio.run(
        bot._reply_with_brain(
            message,
            user_text_override="write this once",
            persona_override=jb_persona,
            prompt_cache_key="discord-brain:jb:test",
            prompt_cache_retention="24h",
        )
    )

    assert "DOC PROMPT" not in brain.prompt
    assert brain.kwargs["persona"] is jb_persona
    assert brain.kwargs["thread_id"] == "discord:dm:123"
    assert brain.kwargs["use_memory"].enabled is True
    assert brain.kwargs["tool_names"] == ["discord_context", "remember", "search_memory", "current_time", "brain_context"]
    assert brain.kwargs["prompt_cache_key"] == "discord-brain:jb:test"
    assert brain.kwargs["prompt_cache_retention"] == "24h"
    assert message._fake_reply.edits == ["ok"]


def test_reply_includes_text_file_attachments(monkeypatch):
    monkeypatch.setenv("DISCORD_BRAIN_TIMEZONE", "America/Los_Angeles")
    message = _fake_message(datetime(2026, 6, 19, 16, 15, tzinfo=timezone.utc))
    message.attachments = [_FakeAttachment(b"user uploaded notes\nimportant line")]
    brain = _FakeBrain()
    bot = DiscordBrainBot.__new__(DiscordBrainBot)
    bot.recent_by_scope = {}
    bot.brain = brain
    bot.persona = SimpleNamespace(id="neuro-sama", name="Neuro-sama", tools=[])
    bot.max_reply_chars = 1900
    bot.edit_interval_seconds = 0.25

    asyncio.run(bot._reply_with_brain(message))

    assert "[Text file attachments]" in brain.prompt
    assert "--- notes.txt (text/plain," in brain.prompt
    assert "important line" in brain.prompt
    assert message._fake_reply.edits == ["ok"]


def test_bot_message_ignore_toggle_keeps_self_guard():
    bot = DiscordBrainBot.__new__(DiscordBrainBot)
    bot._connection = SimpleNamespace(user=SimpleNamespace(id=999))
    bot.ignore_bots = True
    other_bot_message = SimpleNamespace(author=SimpleNamespace(id=111, bot=True))
    self_message = SimpleNamespace(author=SimpleNamespace(id=999, bot=True))

    assert bot._is_ignored_bot_message(other_bot_message) is True

    bot.ignore_bots = False

    assert bot._is_ignored_bot_message(other_bot_message) is False
    assert bot._is_ignored_bot_message(self_message) is True


def test_unignored_bot_messages_bypass_human_allow_list():
    bot = DiscordBrainBot.__new__(DiscordBrainBot)
    bot.allowed_users = {123}
    bot.allowed_guilds = set()
    bot.ignore_bots = False
    bot_message = SimpleNamespace(author=SimpleNamespace(id=456, bot=True), guild=None)
    human_message = SimpleNamespace(author=SimpleNamespace(id=456, bot=False), guild=None)

    assert bot._allowed(bot_message) is True
    assert bot._allowed(human_message) is False


def test_grillo_export_uses_current_record_fields():
    content = _format_grillo_export(
        scope="discord:dm:123",
        participant="123",
        query="prefs",
        packet_text="<grillo_context_packet />",
        turns=[
            SimpleNamespace(
                created_at="2026-06-20T12:00:00+00:00",
                role="user",
                author_name="Subsect",
                content="likes graph commands",
            )
        ],
        slots=[SimpleNamespace(slot_name="preferences", items=["prefers exports"])],
        diary=[
            SimpleNamespace(
                created_at="2026-06-20T12:01:00+00:00",
                beat_type="manual",
                personal_thought="remember command needs",
                summary="User wants memory visibility.",
            )
        ],
        candidates=[
            SimpleNamespace(
                created_at="2026-06-20T12:02:00+00:00",
                type="preference",
                confidence=0.91,
                promoted=True,
                summary="User prefers scoped memory exports.",
                tags=["discord", "memory"],
            )
        ],
    )

    assert "[preference]" in content
    assert "[user] Subsect" in content
    assert "prefers exports" in content


def test_scoped_ladybug_facts_filter_by_raw_event_scope(monkeypatch):
    monkeypatch.delenv("DISCORD_BRAIN_LADYBUG_GLOBAL_COMMANDS", raising=False)
    stack = SimpleNamespace(graph_store=_FakeGraphStore(), raw_log=_FakeRawLog())

    facts = asyncio.run(
        _scoped_ladybug_facts(stack, "discord:dm:123", "scope", top_k=10, include_expired=True)
    )

    assert [fact.id for fact in facts] == ["in-scope"]


def test_gateway_model_ids_are_chat_models_and_embeddings_are_not():
    assert is_chat_model_id("deepseek/deepseek-v4-flash") is True
    assert is_chat_model_id("anthropic/claude-3-5-sonnet") is True
    assert is_chat_model_id("text-embedding-3-small") is False


def test_runtime_model_set_updates_persona_and_default_config():
    bot = DiscordBrainBot.__new__(DiscordBrainBot)
    bot.persona = SimpleNamespace(model="old-model")
    bot.brain = SimpleNamespace(config=SimpleNamespace(default_model="old-model"))

    bot._set_runtime_model("deepseek/deepseek-v4-flash")

    assert bot.persona.model == "deepseek/deepseek-v4-flash"
    assert bot.brain.config.default_model == "deepseek/deepseek-v4-flash"


def test_model_choices_order_current_model_first():
    choices = [
        ModelChoice(id="gpt-5", label="gpt-5"),
        ModelChoice(id="deepseek/deepseek-v4-flash", label="deepseek/deepseek-v4-flash"),
    ]

    ordered = _ordered_model_choices(choices, "deepseek/deepseek-v4-flash")

    assert ordered[0].id == "deepseek/deepseek-v4-flash"


def test_model_select_view_paginates_large_model_list():
    choices = [ModelChoice(id=f"provider/model-{index:02d}", label=f"provider/model-{index:02d}") for index in range(30)]
    view = ModelSelectView(_FakeModelBot("provider/model-00"), owner_id=123, choices=choices)

    assert view.total_pages == 2
    assert len(view.children) == 3
    assert [getattr(child, "label", None) for child in view.children[1:]] == ["Prev", "Next"]
