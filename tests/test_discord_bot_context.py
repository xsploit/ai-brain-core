import asyncio
import base64
from array import array
from datetime import datetime, timezone
from types import SimpleNamespace

import aibrain.discord_bot as discord_bot_module
from aibrain.discord_bot import (
    DEFAULT_IGNORE_BOTS,
    DEFAULT_RESPOND_TO_BOTS,
    DEFAULT_REQUIRE_MENTION_IN_GUILDS,
    DEFAULT_TTS_REPLIES,
    DiscordBrainBot,
    DiscordVoiceClip,
    ModelSelectView,
    RelationshipGraphView,
    _build_jb_persona,
    _format_summary_transcript,
    _format_tavily_search_result,
    _format_grillo_export,
    _format_relationship_graph_export,
    _format_relationship_graph_status,
    _relationship_graph_embed,
    _grillo_scope_for_message,
    _message_text,
    _model_choice_description,
    _ordered_model_choices,
    _ping_reply,
    _ping_target_mention,
    _read_attachment_bytes,
    _scoped_ladybug_facts,
    _summary_limit,
    _text_attachment_context,
    _time_context,
    _tts_spoken_text,
    _voice_opus_bitrate,
    build_discord_voice_clip,
    limit_pcm_s16le_peak,
    waveform_base64_from_pcm_s16le,
)
from aibrain.model_catalog import ModelChoice, is_chat_model_id
from aibrain.tts import TTSAudio, TTSConfig


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
    def __init__(self, data: bytes, *, filename: str = "notes.txt", content_type: str = "text/plain"):
        self._data = data
        self.filename = filename
        self.content_type = content_type
        self.url = f"https://cdn.example.invalid/{filename}"
        self.size = len(data)

    async def read(self, *, use_cached=True):
        return self._data


class _FakeVoiceAttachment:
    filename = "voice-message.ogg"
    content_type = "audio/ogg"
    duration = 1.25
    waveform = bytes([0, 128, 255])

    def is_voice_message(self):
        return True


class _CachedFailsAttachment:
    def __init__(self):
        self.calls = []

    async def read(self, *, use_cached=True):
        self.calls.append(use_cached)
        if use_cached:
            raise RuntimeError("415 Unsupported Media Type: failed to get asset")
        return b"direct-url-bytes"


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


class _FakeGrilloPacket:
    def __init__(self, text: str):
        self.text = text

    def as_prompt_text(self):
        return self.text


class _FakeGrilloRuntime:
    def __init__(self, text: str = "<grillo_context />", error: Exception | None = None):
        self.text = text
        self.error = error
        self.calls = []
        self.ingests = []

    async def build_context_packet(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return _FakeGrilloPacket(self.text)

    async def ingest_turn_pair(self, **kwargs):
        self.ingests.append(kwargs)


class _FakeTTSBrain:
    async def speak(self, text, **tts_options):
        self.text = text
        self.tts_options = tts_options
        return TTSAudio(audio=(b"\x00\x00\xff\x7f\x00\x00\x01\x80" * 100), sample_rate=16000)


class _RecordingTTSBrain:
    def __init__(self):
        self.texts = []

    async def speak(self, text, **tts_options):
        self.texts.append(text)
        return TTSAudio(audio=(b"\x00\x00\xff\x7f" * 100), sample_rate=16000)


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


class _FakeMatchingRawLog:
    def __init__(self, event_id: str = "event-1"):
        self.event_id = event_id
        self.calls = []

    async def list_thread_events_matching(
        self,
        *,
        thread_ids,
        thread_like_patterns,
        persona_id,
        limit=100,
    ):
        self.calls.append(
            {
                "thread_ids": thread_ids,
                "thread_like_patterns": thread_like_patterns,
                "persona_id": persona_id,
                "limit": limit,
            }
        )
        return [SimpleNamespace(id=self.event_id)]


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
    events = []

    async def _reply(content, *, mention_author=False):
        reply.edits.append(content)
        events.append(("text", content))
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
        _events=events,
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


def test_discord_grillo_scope_is_server_user_persona_scoped():
    message = _fake_message(datetime(2026, 6, 19, 16, 15, tzinfo=timezone.utc))
    message.guild = SimpleNamespace(id=222, name="Test Guild")
    message.channel = _FakeChannel()
    message.channel.id = 456
    message.author = SimpleNamespace(id=123, display_name="Subby", global_name=None, bot=False)

    assert _grillo_scope_for_message(message, "neuro-sama") == "discord:guild:222:user:123:persona:neuro-sama"


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


def test_discord_prompt_includes_recent_channel_context(monkeypatch):
    monkeypatch.setenv("DISCORD_BRAIN_TIMEZONE", "America/Los_Angeles")
    channel = _FakeChannel()
    side_message = _fake_message(datetime(2026, 6, 19, 16, 10, tzinfo=timezone.utc))
    side_message.id = 555
    side_message.guild = SimpleNamespace(id=222, name="Test Guild")
    side_message.channel = channel
    side_message.author = SimpleNamespace(id=456, display_name="Karah", global_name=None, bot=False)
    side_message.content = "oh wait technically my bot can already do all of them"
    side_message.clean_content = side_message.content
    message = _fake_message(datetime(2026, 6, 19, 16, 15, tzinfo=timezone.utc))
    message.id = 789
    message.guild = SimpleNamespace(id=222, name="Test Guild")
    message.channel = channel
    bot = DiscordBrainBot.__new__(DiscordBrainBot)
    bot.recent_by_scope = {}
    bot.brain = SimpleNamespace(memory_stack=None)
    bot.persona = SimpleNamespace(name="Neuro-sama")
    bot._connection = SimpleNamespace(user=SimpleNamespace(id=999, display_name="Neuro-sama", global_name=None, bot=True))
    bot._record_recent(side_message)
    bot._record_recent_assistant(side_message, "oh yeah, your bot can already do the toolbox stuff")

    prompt = asyncio.run(bot._build_prompt_for_message(message, "discord:guild:222:channel:456", "yeah that was wild"))

    assert "Recent channel context before this message:" in prompt
    assert "Karah" in prompt
    assert "oh wait technically my bot can already do all of them" in prompt
    assert "Neuro-sama (bot)" in prompt
    assert "oh yeah, your bot can already do the toolbox stuff" in prompt


def test_discord_prompt_and_grillo_ingest_include_author_metadata(monkeypatch):
    monkeypatch.setenv("DISCORD_BRAIN_TIMEZONE", "America/Los_Angeles")
    message = _fake_message(datetime(2026, 6, 19, 16, 15, tzinfo=timezone.utc))
    message.guild = SimpleNamespace(id=222, name="Test Guild")
    message.channel = _FakeChannel()
    message.channel.id = 456
    message.channel.name = "bot-chat"
    message.author = SimpleNamespace(
        id=123,
        name="subsect",
        display_name="SUBSECT",
        global_name="LO",
        mention="<@123>",
        bot=False,
    )
    grillo = _FakeGrilloRuntime()
    brain = _FakeBrain()
    brain.memory_stack = SimpleNamespace(grillo=grillo)
    bot = DiscordBrainBot.__new__(DiscordBrainBot)
    bot.paused = False
    bot.recent_by_scope = {}
    bot.brain = brain
    bot.persona = SimpleNamespace(id="neuro-sama", name="Neuro-sama", tools=[])
    bot.max_reply_chars = 1900
    bot.edit_interval_seconds = 0.25
    bot.send_tts_replies = False

    asyncio.run(bot._reply_with_brain(message))

    assert brain.kwargs["thread_id"] == "discord:guild:222:channel:456:user:123"
    assert grillo.calls[0]["scope_key"] == "discord:guild:222:user:123:persona:neuro-sama"
    assert grillo.calls[0]["participant_key"] == "123"
    assert grillo.calls[0]["channel_id"] == "456"
    assert "author_id: 123" in brain.prompt
    assert "author_username: subsect" in brain.prompt
    assert "author_display_name: SUBSECT" in brain.prompt
    metadata = grillo.ingests[0]["metadata"]
    assert metadata["author_id"] == 123
    assert metadata["author_username"] == "subsect"
    assert metadata["author_display_name"] == "SUBSECT"
    assert metadata["guild_id"] == 222
    assert metadata["channel_id"] == 456
    assert grillo.ingests[0]["scope_key"] == "discord:guild:222:user:123:persona:neuro-sama"


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
    assert jb_persona.id == "jb-one-shot"
    assert jb_persona.name == "JB"
    assert jb_persona.model == "deepseek/test"
    assert jb_persona.tools == []
    assert "Neuro-sama" not in jb_persona.name


def test_jb_reply_uses_isolated_jb_path(monkeypatch):
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
            thread_id_override=f"discord:jb:{message.id}",
            use_memory=False,
            tool_names=[],
            include_grillo_context=False,
            record_grillo=False,
            stateless=True,
            prompt_cache_key="discord-brain:jb:test",
            prompt_cache_retention="24h",
        )
    )

    assert "DOC PROMPT" not in brain.prompt
    assert brain.kwargs["persona"] is jb_persona
    assert brain.kwargs["thread_id"] == "discord:jb:789"
    assert brain.kwargs["use_memory"] is False
    assert brain.kwargs["tool_names"] == []
    assert brain.kwargs["stateless"] is True
    assert brain.kwargs["prompt_cache_key"] == "discord-brain:jb:test"
    assert brain.kwargs["prompt_cache_retention"] == "24h"
    assert message._fake_reply.edits == ["ok"]


def test_grillo_query_is_bounded_without_truncating_prompt(monkeypatch):
    monkeypatch.setenv("DISCORD_BRAIN_MEMORY_QUERY_MAX_CHARS", "12")
    message = _fake_message(datetime(2026, 6, 19, 16, 15, tzinfo=timezone.utc))
    runtime = _FakeGrilloRuntime("<grillo>ok</grillo>")
    bot = DiscordBrainBot.__new__(DiscordBrainBot)
    bot.recent_by_scope = {}
    bot.persona = SimpleNamespace(name="Neuro-sama")
    bot.brain = SimpleNamespace(memory_stack=SimpleNamespace(grillo=runtime))

    prompt = asyncio.run(bot._build_prompt_for_message(message, "discord:dm:123", "x" * 40))

    assert len(runtime.calls[0]["query"]) == 12
    assert len(runtime.calls[0]["current_turn_text"]) == 12
    assert "x" * 40 in prompt
    assert "<grillo>ok</grillo>" in prompt


def test_grillo_failure_keeps_plain_prompt(monkeypatch):
    message = _fake_message(datetime(2026, 6, 19, 16, 15, tzinfo=timezone.utc))
    runtime = _FakeGrilloRuntime(error=RuntimeError("embedding limit"))
    bot = DiscordBrainBot.__new__(DiscordBrainBot)
    bot.recent_by_scope = {}
    bot.persona = SimpleNamespace(name="Neuro-sama")
    bot.brain = SimpleNamespace(memory_stack=SimpleNamespace(grillo=runtime))

    prompt = asyncio.run(bot._build_prompt_for_message(message, "discord:dm:123", "plain text"))

    assert "plain text" in prompt
    assert "<grillo" not in prompt


def test_reply_includes_text_file_attachments(monkeypatch):
    monkeypatch.setenv("DISCORD_BRAIN_TIMEZONE", "America/Los_Angeles")
    message = _fake_message(datetime(2026, 6, 19, 16, 15, tzinfo=timezone.utc))
    message.attachments = [_FakeAttachment(b"user uploaded notes\nimportant line")]
    brain = _FakeBrain()
    runtime = _FakeGrilloRuntime("<grillo>ok</grillo>")
    brain.memory_stack = SimpleNamespace(grillo=runtime)
    bot = DiscordBrainBot.__new__(DiscordBrainBot)
    bot.recent_by_scope = {}
    bot.brain = brain
    bot.persona = SimpleNamespace(id="neuro-sama", name="Neuro-sama", tools=[])
    bot.max_reply_chars = 1900
    bot.edit_interval_seconds = 0.25

    asyncio.run(bot._reply_with_brain(message))

    assert "[Readable attachments]" in brain.prompt
    assert "--- notes.txt (text/plain," in brain.prompt
    assert "important line" in brain.prompt
    assert runtime.calls[0]["query"] == "what day is it?"
    assert "important line" not in runtime.calls[0]["query"]
    assert runtime.ingests[0]["user_text"] == "what day is it?"
    assert brain.kwargs["memory_query_text"] == "what day is it?"
    assert brain.kwargs["memory_event_text"] == "what day is it?"
    assert brain.kwargs["history_text"] == "what day is it?"
    assert message._fake_reply.edits == ["ok"]


def test_reply_includes_pdf_attachment_text(monkeypatch):
    monkeypatch.setenv("DISCORD_BRAIN_TIMEZONE", "America/Los_Angeles")

    def fake_pdf_text(raw, *, max_pages):
        assert raw == b"%PDF fake"
        assert max_pages == 16
        return "[page 1]\nPDF important line", 3

    monkeypatch.setattr(discord_bot_module, "_pdf_text_from_bytes", fake_pdf_text)
    message = _fake_message(datetime(2026, 6, 19, 16, 15, tzinfo=timezone.utc))
    message.attachments = [_FakeAttachment(b"%PDF fake", filename="paper.pdf", content_type="application/pdf")]
    brain = _FakeBrain()
    bot = DiscordBrainBot.__new__(DiscordBrainBot)
    bot.recent_by_scope = {}
    bot.brain = brain
    bot.persona = SimpleNamespace(id="neuro-sama", name="Neuro-sama", tools=[])
    bot.max_reply_chars = 1900
    bot.edit_interval_seconds = 0.25

    asyncio.run(bot._reply_with_brain(message))

    assert "[Readable attachments]" in brain.prompt
    assert "--- paper.pdf (application/pdf, 3 pages," in brain.prompt
    assert "PDF important line" in brain.prompt
    assert brain.kwargs["memory_query_text"] == "what day is it?"
    assert "PDF important line" not in brain.kwargs["memory_event_text"]
    assert "PDF important line" not in brain.kwargs["history_text"]
    assert message._fake_reply.edits == ["ok"]


def test_pdf_without_extractable_text_is_reported(monkeypatch):
    def fake_pdf_text(raw, *, max_pages):
        return "", 2

    monkeypatch.setattr(discord_bot_module, "_pdf_text_from_bytes", fake_pdf_text)
    message = _fake_message(datetime(2026, 6, 19, 16, 15, tzinfo=timezone.utc))
    message.attachments = [_FakeAttachment(b"%PDF scanned", filename="scan.pdf", content_type="application/pdf")]

    context = asyncio.run(_text_attachment_context(message))

    assert "[scan.pdf skipped: application/pdf, 2 pages, no extractable text]" in context


def test_text_attachment_defaults_are_larger(monkeypatch):
    monkeypatch.delenv("DISCORD_BRAIN_TEXT_ATTACHMENT_MAX_BYTES", raising=False)
    message = _fake_message(datetime(2026, 6, 19, 16, 15, tzinfo=timezone.utc))
    message.attachments = [_FakeAttachment(b"a" * 200_000)]

    context = asyncio.run(_text_attachment_context(message))

    assert "notes.txt" in context
    assert "exceeds" not in context


def test_attachment_read_uses_direct_url_before_cached_proxy():
    attachment = _CachedFailsAttachment()

    raw = asyncio.run(_read_attachment_bytes(attachment))

    assert raw == b"direct-url-bytes"
    assert attachment.calls == [False]


def test_tts_replies_default_on():
    assert DEFAULT_TTS_REPLIES is True


def test_tts_spoken_text_removes_markdown_formatting():
    text = "# **Big** update\n- *first* item\n- `code` and [docs](https://example.com)\nplain *asterisks*"

    assert _tts_spoken_text(text) == "Big update\nfirst item\ncode and docs\nplain asterisks"


def test_tts_replies_skip_voice_without_constructor_attrs(monkeypatch):
    async def fail_voice(*args, **kwargs):
        raise AssertionError("voice should not be sent without constructor TTS attrs")

    monkeypatch.setattr(discord_bot_module, "send_discord_voice_message", fail_voice)
    message = _fake_message(datetime(2026, 6, 19, 16, 15, tzinfo=timezone.utc))
    brain = _FakeBrain()
    bot = DiscordBrainBot.__new__(DiscordBrainBot)
    bot.recent_by_scope = {}
    bot.brain = brain
    bot.persona = SimpleNamespace(id="neuro-sama", name="Neuro-sama", tools=[])
    bot.max_reply_chars = 1900

    asyncio.run(bot._reply_with_brain(message))

    assert message._events == [("text", "ok")]


def test_tts_reply_sends_text_before_voice(monkeypatch):
    async def fake_clip(brain, text, *, voice=None):
        assert text == "ok"
        return DiscordVoiceClip(ogg=b"ogg", duration_secs=0.1, waveform="AA==")

    async def fake_voice(channel_id, token, clip):
        message._events.append(("voice", clip.ogg))

    monkeypatch.setattr(discord_bot_module, "build_discord_voice_clip", fake_clip)
    monkeypatch.setattr(discord_bot_module, "send_discord_voice_message", fake_voice)
    message = _fake_message(datetime(2026, 6, 19, 16, 15, tzinfo=timezone.utc))
    brain = _FakeBrain()
    bot = DiscordBrainBot.__new__(DiscordBrainBot)
    bot.recent_by_scope = {}
    bot.brain = brain
    bot.persona = SimpleNamespace(id="neuro-sama", name="Neuro-sama", tools=[])
    bot.max_reply_chars = 1900
    bot.send_tts_replies = True
    bot.discord_token = "token"
    bot.tts_voice = "neuro-sama"
    bot.logger = SimpleNamespace(exception=lambda *args, **kwargs: None)

    asyncio.run(bot._reply_with_brain(message))

    assert message._events == [("text", "ok"), ("voice", b"ogg")]


def test_message_text_reads_voice_message_waveform():
    message = _fake_message(datetime(2026, 6, 19, 16, 15, tzinfo=timezone.utc))
    message.attachments = [_FakeVoiceAttachment()]

    text = _message_text(message)

    assert "voice message" in text
    assert "duration=1.25s" in text
    assert "waveform_points=3" in text
    assert "waveform_peak=255" in text


def test_summary_limit_clamps_to_safe_channel_history_window():
    assert _summary_limit(1) == 5
    assert _summary_limit(75) == 75
    assert _summary_limit(500) == 200


def test_format_summary_transcript_includes_author_bot_marker_and_content():
    user = SimpleNamespace(display_name="Subby", global_name=None, bot=False)
    bot_user = SimpleNamespace(display_name="Neuro-sama", global_name=None, bot=True)
    messages = [
        SimpleNamespace(
            author=user,
            clean_content="hello",
            content="hello",
            attachments=[],
            created_at=datetime(2026, 6, 21, 8, 0, tzinfo=timezone.utc),
        ),
        SimpleNamespace(
            author=bot_user,
            clean_content="yo",
            content="yo",
            attachments=[],
            created_at=datetime(2026, 6, 21, 8, 1, tzinfo=timezone.utc),
        ),
    ]

    transcript = _format_summary_transcript(messages)

    assert "Subby: hello" in transcript
    assert "Neuro-sama bot: yo" in transcript


def test_format_tavily_search_result_lists_answer_and_sources():
    text = _format_tavily_search_result(
        {
            "query": "discord components v2",
            "answer": "Components v2 adds layout components.",
            "results": [
                {
                    "title": "Component Reference",
                    "url": "https://docs.discord.com/developers/components/reference",
                    "content": "Layout, content, and interactive components.",
                    "score": 0.91,
                }
            ],
        }
    )

    assert "Components v2 adds layout components." in text
    assert "Component Reference" in text
    assert "https://docs.discord.com/developers/components/reference" in text


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


def test_bot_messages_are_not_ignored_by_default():
    assert DEFAULT_IGNORE_BOTS is False
    assert DEFAULT_RESPOND_TO_BOTS is True
    assert DEFAULT_REQUIRE_MENTION_IN_GUILDS is True


def test_bot_messages_do_not_trigger_without_mention():
    bot = DiscordBrainBot.__new__(DiscordBrainBot)
    bot.respond_to_all = False
    bot.respond_to_dms = True
    bot.respond_to_mentions = True
    bot.respond_to_bots = True
    bot.ignore_bots = False
    bot._connection = SimpleNamespace(user=SimpleNamespace(id=999))
    message = SimpleNamespace(
        author=SimpleNamespace(id=111, bot=True),
        guild=SimpleNamespace(id=222),
        mentions=[],
    )

    assert bot._should_respond(message) is False


def test_bot_mentions_trigger_when_bot_interactions_are_enabled():
    bot = DiscordBrainBot.__new__(DiscordBrainBot)
    bot.paused = False
    bot.respond_to_all = False
    bot.respond_to_mentions = True
    bot.respond_to_bots = True
    bot.ignore_bots = False
    bot._connection = SimpleNamespace(user=SimpleNamespace(id=999))
    message = SimpleNamespace(
        author=SimpleNamespace(id=111, bot=True),
        guild=SimpleNamespace(id=222),
        mentions=[bot.user],
    )

    assert bot._should_respond(message) is True

    bot.respond_to_bots = False

    assert bot._should_respond(message) is False


def test_bot_direct_replies_trigger_when_bot_interactions_are_enabled():
    bot = DiscordBrainBot.__new__(DiscordBrainBot)
    bot.paused = False
    bot.respond_to_all = True
    bot.respond_to_mentions = True
    bot.respond_to_bots = True
    bot.ignore_bots = False
    bot._connection = SimpleNamespace(user=SimpleNamespace(id=999))
    message = SimpleNamespace(
        author=SimpleNamespace(id=111, bot=True),
        guild=SimpleNamespace(id=222),
        mentions=[],
        reference=SimpleNamespace(resolved=SimpleNamespace(author=bot.user)),
    )

    assert bot._should_respond(message) is True

    bot.respond_to_bots = False

    assert bot._should_respond(message) is False


def test_bot_mentions_do_not_bypass_bot_interaction_toggle():
    bot = DiscordBrainBot.__new__(DiscordBrainBot)
    bot.paused = False
    bot.respond_to_all = True
    bot.respond_to_mentions = True
    bot.respond_to_bots = False
    bot.ignore_bots = False
    bot._connection = SimpleNamespace(user=SimpleNamespace(id=999))
    message = SimpleNamespace(
        author=SimpleNamespace(id=111, bot=True),
        guild=SimpleNamespace(id=222),
        mentions=[SimpleNamespace(id=999)],
    )

    assert bot._should_respond(message) is False


def test_human_guild_messages_do_not_trigger_without_direct_target_by_default():
    bot = DiscordBrainBot.__new__(DiscordBrainBot)
    bot.paused = False
    bot.respond_to_all = True
    bot.respond_to_dms = True
    bot.respond_to_mentions = True
    bot.respond_to_bots = True
    bot.ignore_bots = False
    bot._connection = SimpleNamespace(user=SimpleNamespace(id=999))
    message = SimpleNamespace(
        author=SimpleNamespace(id=111, bot=False),
        guild=SimpleNamespace(id=222),
        mentions=[],
    )

    assert bot._should_respond(message) is False


def test_human_guild_direct_replies_trigger_by_default():
    bot = DiscordBrainBot.__new__(DiscordBrainBot)
    bot.paused = False
    bot.respond_to_all = False
    bot.respond_to_dms = True
    bot.respond_to_mentions = True
    bot.respond_to_bots = True
    bot.ignore_bots = False
    bot._connection = SimpleNamespace(user=SimpleNamespace(id=999))
    message = SimpleNamespace(
        author=SimpleNamespace(id=111, bot=False),
        guild=SimpleNamespace(id=222),
        mentions=[],
    )

    assert bot._should_respond(message) is False

    message.reference = SimpleNamespace(resolved=SimpleNamespace(author=bot.user))

    assert bot._should_respond(message) is True


def test_human_guild_messages_do_not_use_ambient_mode():
    bot = DiscordBrainBot.__new__(DiscordBrainBot)
    bot.paused = False
    bot.respond_to_all = True
    bot.respond_to_dms = True
    bot.respond_to_mentions = True
    bot.respond_to_bots = True
    bot.ignore_bots = False
    bot.require_mention_in_guilds = False
    bot._connection = SimpleNamespace(user=SimpleNamespace(id=999))
    message = SimpleNamespace(
        author=SimpleNamespace(id=111, bot=False),
        guild=SimpleNamespace(id=222),
        mentions=[],
    )

    assert bot._should_respond(message) is False


def test_human_mentions_still_trigger_when_bot_interactions_are_stopped():
    bot = DiscordBrainBot.__new__(DiscordBrainBot)
    bot.paused = False
    bot.respond_to_all = False
    bot.respond_to_dms = True
    bot.respond_to_mentions = True
    bot.respond_to_bots = False
    bot.ignore_bots = False
    bot._connection = SimpleNamespace(user=SimpleNamespace(id=999))
    message = SimpleNamespace(
        author=SimpleNamespace(id=111, bot=False),
        guild=SimpleNamespace(id=222),
        mentions=[bot.user],
    )

    assert bot._should_respond(message) is True


def test_bot_dm_does_not_bypass_bot_interaction_toggle():
    bot = DiscordBrainBot.__new__(DiscordBrainBot)
    bot.paused = False
    bot.respond_to_dms = True
    bot.respond_to_mentions = True
    bot.respond_to_bots = False
    bot.ignore_bots = False
    bot._connection = SimpleNamespace(user=SimpleNamespace(id=999))
    message = SimpleNamespace(
        author=SimpleNamespace(id=111, bot=True),
        guild=None,
        mentions=[],
        reference=SimpleNamespace(resolved=SimpleNamespace(author=bot.user)),
    )

    assert bot._should_respond(message) is False


def test_pause_blocks_all_normal_responses():
    bot = DiscordBrainBot.__new__(DiscordBrainBot)
    bot.paused = True
    bot.respond_to_all = True
    bot.respond_to_dms = True
    bot.respond_to_mentions = True
    bot.respond_to_bots = True
    bot.ignore_bots = False
    bot._connection = SimpleNamespace(user=SimpleNamespace(id=999))
    message = SimpleNamespace(
        author=SimpleNamespace(id=111, bot=True),
        guild=SimpleNamespace(id=222),
        mentions=[SimpleNamespace(id=999)],
    )

    assert bot._should_respond(message) is False


def test_pause_command_messages_are_still_commands():
    bot = DiscordBrainBot.__new__(DiscordBrainBot)
    bot.command_prefix_text = "!brain"

    assert bot._is_command_message(SimpleNamespace(content="!pause")) is True
    assert bot._is_command_message(SimpleNamespace(content="!resume")) is True
    assert bot._is_command_message(SimpleNamespace(content="!unpause")) is True
    assert bot._is_command_message(SimpleNamespace(content="!grillo debug")) is True
    assert bot._is_command_message(SimpleNamespace(content="!ladybug search Subby")) is True
    assert bot._is_command_message(SimpleNamespace(content="!summary 25")) is True
    assert bot._is_command_message(SimpleNamespace(content="!search discord ui")) is True
    assert bot._is_command_message(SimpleNamespace(content="!heartbeat tick")) is True


def test_bot_interactions_enabled_requires_not_ignored_and_responding():
    bot = DiscordBrainBot.__new__(DiscordBrainBot)
    bot.ignore_bots = False
    bot.respond_to_bots = True

    assert bot._bot_interactions_enabled() is True

    bot.respond_to_bots = False

    assert bot._bot_interactions_enabled() is False

    bot.respond_to_bots = True
    bot.ignore_bots = True

    assert bot._bot_interactions_enabled() is False


def test_paused_reply_with_brain_does_not_send_final_reply(monkeypatch):
    monkeypatch.setenv("DISCORD_BRAIN_TIMEZONE", "America/Los_Angeles")
    message = _fake_message(datetime(2026, 6, 19, 16, 15, tzinfo=timezone.utc))
    brain = _FakeBrain()
    bot = DiscordBrainBot.__new__(DiscordBrainBot)
    bot.paused = True
    bot.recent_by_scope = {}
    bot.brain = brain
    bot.persona = SimpleNamespace(id="neuro-sama", name="Neuro-sama", tools=[])
    bot.max_reply_chars = 1900
    bot.edit_interval_seconds = 0.25

    asyncio.run(bot._reply_with_brain(message))

    assert message._fake_reply.edits == []


def test_blank_visible_message_uses_nonempty_memory_query(monkeypatch):
    monkeypatch.setenv("DISCORD_BRAIN_TIMEZONE", "America/Los_Angeles")
    message = _fake_message(datetime(2026, 6, 19, 16, 15, tzinfo=timezone.utc))
    message.content = ""
    message.clean_content = ""
    brain = _FakeBrain()
    bot = DiscordBrainBot.__new__(DiscordBrainBot)
    bot.paused = False
    bot.recent_by_scope = {}
    bot.brain = brain
    bot.persona = SimpleNamespace(id="neuro-sama", name="Neuro-sama", tools=[])
    bot.max_reply_chars = 1900
    bot.edit_interval_seconds = 0.25
    bot.send_tts_replies = False
    bot.discord_token = None
    bot.logger = SimpleNamespace(debug=lambda *args, **kwargs: None, exception=lambda *args, **kwargs: None)

    asyncio.run(bot._reply_with_brain(message))

    assert brain.kwargs["memory_query_text"].strip()
    assert brain.kwargs["history_text"].strip()
    assert brain.kwargs["memory_event_text"] == ""


def test_unignored_bot_commands_are_invoked_without_process_commands():
    bot = DiscordBrainBot.__new__(DiscordBrainBot)
    events = []
    command = object()
    message = SimpleNamespace(author=SimpleNamespace(id=111, bot=True), content="!ping <@123>")

    async def fake_process_commands(_message):
        events.append("process_commands")

    async def fake_get_context(_message):
        events.append("get_context")
        return SimpleNamespace(command=command)

    async def fake_invoke(ctx):
        events.append(("invoke", ctx.command))

    bot.process_commands = fake_process_commands
    bot.get_context = fake_get_context
    bot.invoke = fake_invoke

    asyncio.run(bot._process_commands_including_unignored_bots(message))

    assert events == ["get_context", ("invoke", command)]


def test_human_commands_still_use_process_commands():
    bot = DiscordBrainBot.__new__(DiscordBrainBot)
    events = []
    message = SimpleNamespace(author=SimpleNamespace(id=111, bot=False), content="!ping <@123>")

    async def fake_process_commands(_message):
        events.append("process_commands")

    bot.process_commands = fake_process_commands

    asyncio.run(bot._process_commands_including_unignored_bots(message))

    assert events == ["process_commands"]


def test_ping_command_targets_user_mentions():
    mentioned = SimpleNamespace(id=123, mention="<@123>")
    message = SimpleNamespace(mentions=[mentioned])

    assert _ping_target_mention(message, "") == "<@123>"
    assert _ping_target_mention(SimpleNamespace(mentions=[]), "<@!456>") == "<@456>"
    assert _ping_target_mention(SimpleNamespace(mentions=[]), "789") == "<@789>"
    assert _ping_reply("<@123>") == "yo, what up, fam <@123>"


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


def test_relationship_graph_status_and_export_include_grillo_and_ladybug_sections():
    snapshot = {
        "graph": {
            "profile": {"relationship_stage": "familiar"},
            "relationship_facts": [{"text": "Subsect likes graph visibility."}],
            "participants": [{"id": "123", "login": "subsect"}],
            "error": "",
        },
        "profile": SimpleNamespace(
            profile_id="relationship:123",
            relationship_stage="familiar",
            mood="focused",
            trust=7,
            attraction=2,
            respect=8,
            irritation=1,
            jealousy=0,
            guard=9,
            turn_count=42,
            last_seen_at="2026-06-21T12:00:00+00:00",
            summary="Subsect wants memory receipts.",
            diary_entry="I should show my receipts.",
            facts=["Subsect wants relationship graph export."],
            tone_preferences=["direct"],
            interaction_style=["fast"],
            boundaries=[],
            active_threads=["memory debugging"],
        ),
        "slots": [SimpleNamespace(slot_name="relationship_state", items=["trust is high"])],
        "diary": [
            SimpleNamespace(
                created_at="2026-06-21T12:01:00+00:00",
                beat_type="relationship",
                summary="Memory visibility improved.",
                personal_thought="This makes the graph inspectable.",
            )
        ],
        "candidates": [
            SimpleNamespace(
                created_at="2026-06-21T12:02:00+00:00",
                type="preference",
                confidence=0.9,
                promoted=True,
                summary="User wants graph receipts.",
            )
        ],
        "emotion": {"intensities": {"focus": 0.8}, "updated_at": "2026-06-21T12:03:00+00:00"},
        "archival": [{"created_at": "2026-06-21T12:04:00+00:00", "text": "archive note"}],
    }

    status = _format_relationship_graph_status(
        scope="discord:dm:123:persona:neuro-sama",
        graph_backend="LadybugGraphMemoryStore",
        snapshot=snapshot,
    )
    export = _format_relationship_graph_export(
        scope="discord:dm:123:persona:neuro-sama",
        participant="123",
        graph_backend="LadybugGraphMemoryStore",
        snapshot=snapshot,
    )

    assert "relationship_facts=`1`" in status
    assert "emotion: `focus=0.8`" in status
    assert "== Ladybug Mirror ==" in export
    assert "== GRILLO Relationship Profile ==" in export
    assert "Subsect wants relationship graph export." in export
    assert "trust is high" in export


def test_relationship_graph_embed_and_view_render_dashboard_controls():
    snapshot = {
        "graph": {
            "profile": {"relationship_stage": "familiar"},
            "relationship_facts": [{"text": "Subsect likes graph visibility."}],
            "participants": [{"id": "123", "login": "subsect"}],
            "error": "",
        },
        "profile": SimpleNamespace(
            relationship_stage="familiar",
            mood="focused",
            trust=7,
            respect=8,
            guard=9,
            turn_count=42,
            summary="Subsect wants memory receipts.",
        ),
        "slots": [SimpleNamespace(slot_name="relationship_state", items=["trust is high"])],
        "diary": [
            SimpleNamespace(
                created_at="2026-06-21T12:01:00+00:00",
                beat_type="relationship",
                summary="Memory visibility improved.",
            )
        ],
        "candidates": [
            SimpleNamespace(type="preference", confidence=0.9, promoted=True, summary="User wants graph receipts.")
        ],
        "emotion": {"intensities": {"focus": 0.8}},
        "archival": [],
    }

    overview = _relationship_graph_embed(
        scope="discord:dm:123:persona:neuro-sama",
        participant="123",
        graph_backend="LadybugGraphMemoryStore",
        snapshot=snapshot,
        page="overview",
        tick_result={"ok": True, "mode": "worker_loop", "writes": 1, "tool_calls": 2},
    )
    slots = _relationship_graph_embed(
        scope="discord:dm:123:persona:neuro-sama",
        participant="123",
        graph_backend="LadybugGraphMemoryStore",
        snapshot=snapshot,
        page="slots",
    )
    view = RelationshipGraphView(
        SimpleNamespace(brain=SimpleNamespace(memory_stack=None)),
        owner_id=123,
        scope="discord:dm:123:persona:neuro-sama",
        participant="123",
        graph_backend="LadybugGraphMemoryStore",
        snapshot=snapshot,
    )

    assert overview.title == "Ladybug Relationship Graph"
    assert any(field.name == "Last Tick" for field in overview.fields)
    assert any(field.name == "Slots" and "trust is high" in field.value for field in slots.fields)
    assert [getattr(child, "label", None) for child in view.children] == [
        "Overview",
        "Slots",
        "Diary",
        "Emotion",
        "Tick",
        "Export",
    ]


def test_scoped_ladybug_facts_filter_by_raw_event_scope(monkeypatch):
    monkeypatch.delenv("DISCORD_BRAIN_LADYBUG_GLOBAL_COMMANDS", raising=False)
    stack = SimpleNamespace(graph_store=_FakeGraphStore(), raw_log=_FakeRawLog())

    facts = asyncio.run(
        _scoped_ladybug_facts(stack, "discord:dm:123", "scope", top_k=10, include_expired=True)
    )

    assert [fact.id for fact in facts] == ["in-scope"]


def test_scoped_ladybug_facts_include_dm_persona_raw_thread(monkeypatch):
    monkeypatch.delenv("DISCORD_BRAIN_LADYBUG_GLOBAL_COMMANDS", raising=False)
    raw_log = _FakeMatchingRawLog()
    stack = SimpleNamespace(graph_store=_FakeGraphStore(), raw_log=raw_log)

    facts = asyncio.run(
        _scoped_ladybug_facts(
            stack,
            "discord:dm:123:persona:neuro-sama",
            "scope",
            top_k=10,
            include_expired=True,
        )
    )

    assert [fact.id for fact in facts] == ["in-scope"]
    assert raw_log.calls == [
        {
            "thread_ids": ["discord:dm:123:persona:neuro-sama", "discord:dm:123"],
            "thread_like_patterns": [],
            "persona_id": "neuro-sama",
            "limit": 1000,
        }
    ]


def test_scoped_ladybug_facts_include_guild_user_raw_threads(monkeypatch):
    monkeypatch.delenv("DISCORD_BRAIN_LADYBUG_GLOBAL_COMMANDS", raising=False)
    raw_log = _FakeMatchingRawLog()
    stack = SimpleNamespace(graph_store=_FakeGraphStore(), raw_log=raw_log)

    facts = asyncio.run(
        _scoped_ladybug_facts(
            stack,
            "discord:guild:222:user:123:persona:neuro-sama",
            "scope",
            top_k=10,
            include_expired=True,
        )
    )

    assert [fact.id for fact in facts] == ["in-scope"]
    assert raw_log.calls == [
        {
            "thread_ids": ["discord:guild:222:user:123:persona:neuro-sama"],
            "thread_like_patterns": ["discord:guild:222:channel:%:user:123"],
            "persona_id": "neuro-sama",
            "limit": 1000,
        }
    ]


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


def test_model_choice_description_includes_gateway_metadata():
    choice = ModelChoice(
        id="provider/model",
        label="provider/model",
        owned_by="team",
        metadata={"provider": "gateway", "context_window": 128000},
    )

    description = _model_choice_description(choice, True)

    assert "current" in description
    assert "gateway" in description
    assert "ctx 128000" in description


def test_waveform_base64_from_pcm_s16le_is_bounded():
    pcm = (b"\x00\x00\xff\x7f\x00\x00\x01\x80" * 800)

    waveform = base64.b64decode(waveform_base64_from_pcm_s16le(pcm, 16000))

    assert 1 <= len(waveform) <= 256
    assert max(waveform) == 255


def test_limit_pcm_s16le_peak_reduces_full_scale_audio():
    pcm = b"\xff\x7f\x00\x80\x00\x00" * 12

    limited = limit_pcm_s16le_peak(pcm, target_peak=0.5)
    samples = array("h")
    samples.frombytes(limited)

    assert max(abs(sample) for sample in samples) <= 16384
    assert len(limited) == len(pcm)


def test_build_discord_voice_clip_encodes_waveform_and_ogg(monkeypatch):
    async def fake_encode(pcm, sample_rate):
        assert sample_rate == 16000
        assert pcm
        return b"ogg-data"

    monkeypatch.setattr(discord_bot_module, "encode_pcm_s16le_to_ogg_opus", fake_encode)

    clip = asyncio.run(build_discord_voice_clip(_FakeTTSBrain(), "hello", voice="neuro-sama"))

    assert clip == DiscordVoiceClip(ogg=b"ogg-data", duration_secs=0.025, waveform=clip.waveform)
    assert base64.b64decode(clip.waveform)


def test_discord_voice_clip_uses_current_text_only(monkeypatch):
    async def fake_encode(pcm, sample_rate):
        return b"ogg-data"

    monkeypatch.setattr(discord_bot_module, "encode_pcm_s16le_to_ogg_opus", fake_encode)
    brain = _RecordingTTSBrain()

    asyncio.run(build_discord_voice_clip(brain, "first reply"))
    asyncio.run(build_discord_voice_clip(brain, "second reply"))

    assert brain.texts == ["first reply", "second reply"]


def test_discord_voice_clip_uses_brain_speak_by_default(monkeypatch):
    async def fake_encode(pcm, sample_rate):
        return b"ogg-data"

    class FailIsolatedPiper:
        def __init__(self, config):
            raise AssertionError("isolated Piper must be opt-in")

    monkeypatch.delenv("DISCORD_BRAIN_TTS_ISOLATE_PROCESS", raising=False)
    monkeypatch.setattr(discord_bot_module, "encode_pcm_s16le_to_ogg_opus", fake_encode)
    monkeypatch.setattr(discord_bot_module, "PiperExecutableTTS", FailIsolatedPiper)
    brain = _RecordingTTSBrain()
    brain.tts = SimpleNamespace(config=TTSConfig(provider="piper_process"))

    asyncio.run(build_discord_voice_clip(brain, "default reply", voice="neuro-sama"))

    assert brain.texts == ["default reply"]


def test_discord_voice_clip_isolates_piper_process_provider(monkeypatch):
    async def fake_encode(pcm, sample_rate):
        return b"ogg-data"

    class FakeIsolatedPiper:
        instances = []

        def __init__(self, config):
            self.config = config
            self.calls = []
            self.instances.append(self)

        async def synthesize(self, text, **tts_options):
            self.calls.append((text, tts_options))
            return TTSAudio(audio=(b"\x00\x00\xff\x7f" * 100), sample_rate=16000)

    async def fail_speak(*args, **kwargs):
        raise AssertionError("persistent brain.speak should not be used for Discord voice clips")

    monkeypatch.setattr(discord_bot_module, "encode_pcm_s16le_to_ogg_opus", fake_encode)
    monkeypatch.setattr(discord_bot_module, "PiperExecutableTTS", FakeIsolatedPiper)
    monkeypatch.setenv("DISCORD_BRAIN_TTS_ISOLATE_PROCESS", "true")
    brain = SimpleNamespace(tts=SimpleNamespace(config=TTSConfig(provider="piper_process")), speak=fail_speak)

    asyncio.run(build_discord_voice_clip(brain, "isolated reply", voice="neuro-sama"))

    assert FakeIsolatedPiper.instances[0].calls == [("isolated reply", {"voice": "neuro-sama"})]


def test_discord_voice_opus_bitrate_defaults_to_discord_client_shape(monkeypatch):
    monkeypatch.delenv("DISCORD_BRAIN_VOICE_OPUS_BITRATE", raising=False)

    assert _voice_opus_bitrate() == "32k"
