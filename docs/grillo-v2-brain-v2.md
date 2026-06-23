# GRILLO v2, Brain v2, and Discord v2

This stack is the v2 memory/runtime path for a persistent Discord persona bot.

## Pieces

- `grillo_v2` is the standalone memory package.
- `aibrain.brain_v2` wraps GRILLO v2 with Vercel AI Gateway response/reflection calls.
- `aibrain.discord_bot_v2` is the Discord host for Brain v2.

GRILLO v2 does not depend on Discord. Discord messages are only one source of episodes.

## Memory Model

GRILLO v2 stores:

- `episodes`: raw events/messages with scope, actor, participants, channel, timestamp, and metadata.
- `evidence`: quoted support tied to episodes.
- `temporal_facts`: evidence-backed facts with validity windows and contradiction links.
- `opinion_edges`: persona-to-entity relationship state, not objective truth.
- `memory_docs`: durable diary/profile/slot/procedural blocks that can be pinned into prompts.
- `entities`: users, bots, channels, or other named actors with aliases and metadata.
- `reflection_cursors`: worker progress markers per scope/persona.

The context packet is XML-wrapped with JSON-lines sections:

- `current_actor`
- `relationship_state`
- `memory_blocks`
- `active_facts`
- `evidence_gaps`
- `recent_episode_summary`
- `retrieval_notes`
- `instructions`

## Worker Flow

The worker path is:

1. Append episodes with `GrilloV2Runtime.append_episode`.
2. Run `GrilloV2Runtime.worker_tick` or `GrilloV2Worker.run_forever`.
3. The worker loads unprocessed episodes after its cursor.
4. The reflection model receives the context packet, recent episodes, and available memory tools.
5. The model returns memory tool calls.
6. GRILLO applies the tool calls and advances the cursor.

The memory tools are:

- `record_evidence`
- `upsert_fact`
- `upsert_opinion_edge`
- `upsert_memory_document`
- `invalidate_fact`

Legacy reflection arrays are still accepted, but `tool_calls` is the preferred protocol.

## Brain v2

Brain v2 uses Vercel AI Gateway through `VercelAIGatewayJSONClient`.

Minimal construction:

```python
from pathlib import Path

from aibrain.brain_v2 import BrainV2, BrainV2Config

brain = BrainV2(
    BrainV2Config(
        database_path=Path("discord_brain_v2.sqlite3"),
        model="deepseek/deepseek-v4-flash",
        persona_id="neuro-sama-v2",
        persona_name="Neuro-sama",
        persona_prompt=Path("local/neuro-sama.persona.txt").read_text(encoding="utf-8"),
    )
)
```

Useful methods:

- `respond(...)`: append user episode, build context, call the response model, append assistant episode.
- `reflect_recent(...)`: manually reflect recent episodes for diagnostics.
- `worker_tick(...)`: process unreflected episodes using the cursor-based worker protocol.
- `backfill_from_v1(...)`: migrate old GRILLO/identity databases into GRILLO v2.
- `status()`: inspect model, provider, database path, and memory counts.

## Package-Backed Memory Index

Current decision: GRILLO v2 keeps `SQLiteGrilloV2Store` as the canonical source of truth and can optionally mirror that state into the existing package-backed memory adapters.

This is deliberate:

- SQLite remains the write-ahead source for episodes, evidence, facts, opinion edges, memory docs, entities, and cursors.
- `aibrain.grillo_v2_index.GrilloV2PackageIndex` mirrors active temporal facts and opinion edges into the existing Ladybug graph adapter.
- The same index mirrors facts, opinion edges, and memory docs into the existing TurboVec/vector recall adapter.
- Brain v2 syncs the current scope before response context assembly when package memory is enabled.
- Package recall augments the GRILLO context packet with additional facts, relationship state, memory blocks, and retrieval notes.
- Hash embeddings are used for this package index by default, so enabling it does not require OpenAI embeddings or extra gateway calls.

Enable deliberately:

```powershell
$env:DISCORD_BRAIN_V2_PACKAGE_MEMORY_ENABLED = "true"
$env:DISCORD_BRAIN_V2_PACKAGE_MEMORY_GRAPH_BACKEND = "auto"     # auto/ladybug/sqlite
$env:DISCORD_BRAIN_V2_PACKAGE_MEMORY_VECTOR_BACKEND = "auto"    # auto/turbovec/sqlite
$env:DISCORD_BRAIN_V2_PACKAGE_MEMORY_EMBEDDING_DIMENSIONS = "256"
$env:DISCORD_BRAIN_V2_PACKAGE_MEMORY_SYNC_LIMIT = "500"
$env:DISCORD_BRAIN_V2_PACKAGE_MEMORY_RECALL_TOP_K = "5"
```

Why not replace GRILLO v2 with Graphiti/Mem0/Letta immediately:

- Ladybug is already installed locally and its Python API supports embedded on-disk graph use through `ladybug.Database`, `ladybug.Connection`, node tables, relationship tables, and Cypher queries. See <https://docs.ladybugdb.com/client-apis/python/>.
- Graphiti is a strong design reference for temporal context graphs, provenance, incremental updates, and hybrid retrieval, but it is not installed in this repo and its current setup expects external graph backends plus LLM/embedding provider configuration. See <https://github.com/getzep/graphiti>.
- Mem0 is a general memory layer and Letta has useful stateful-agent/memory-block concepts, but adopting either as the core would be a framework migration, not a grounded incremental fix. See <https://docs.mem0.ai/introduction> and <https://docs.letta.com/guides/core-concepts/stateful-agents/>.

Near-term direction:

- Keep GRILLO as the reflection/controller contract.
- Keep SQLite as canonical until package-backed retrieval is stable under live Discord load.
- Use Ladybug and TurboVec as real local indexes first.
- Later, evaluate Graphiti as a full temporal graph engine only if the project accepts its storage/runtime dependencies.

## Discord v2

Install with the Discord extra:

```powershell
uv sync --extra discord
```

Required environment:

```powershell
$env:DISCORD_BRAIN_V2_BOT_TOKEN = "..."
$env:AI_GATEWAY_API_KEY = "..."
```

Common optional environment:

```powershell
$env:DISCORD_BRAIN_V2_DATABASE_PATH = "discord_brain_v2.sqlite3"
$env:DISCORD_BRAIN_V2_MODEL = "deepseek/deepseek-v4-flash"
$env:DISCORD_BRAIN_V2_PERSONA_ID = "neuro-sama-v2"
$env:DISCORD_BRAIN_V2_PERSONA_NAME = "Neuro-sama"
$env:DISCORD_BRAIN_V2_PERSONA_PROMPT_PATH = "local/neuro-sama.persona.txt"
$env:DISCORD_BRAIN_V2_ROLLING_CONTEXT_MESSAGES = "15"
$env:DISCORD_BRAIN_V2_COMMAND_PREFIX = "!n2"
$env:DISCORD_BRAIN_V2_REQUIRE_MENTION_IN_GUILDS = "true"
$env:DISCORD_BRAIN_V2_WORKER_ENABLED = "true"
$env:DISCORD_BRAIN_V2_WORKER_INTERVAL_SECONDS = "60"
$env:DISCORD_BRAIN_V2_WORKER_BATCH_SIZE = "12"
$env:DISCORD_BRAIN_V2_WORKER_MAX_BATCHES = "3"
```

Run:

```powershell
uv run aibrain-discord-v2
```

Owner-only commands:

- `!n2 status`: Brain v2 and worker-loop status.
- `!n2 context [query]`: export the current GRILLO v2 context packet.
- `!n2 reflect [limit]`: manually reflect recent messages in the current scope.
- `!n2 worker [batch_size] [max_batches]`: manually run a bounded worker tick in the current scope.
- `!n2 backfill [limit]`: migrate v1 GRILLO and Discord identity data.

## Standalone Embedding

Use `GrilloV2Runtime` directly when hosting outside Discord:

```python
from pathlib import Path

from grillo_v2 import GrilloEpisode, GrilloV2Runtime, GrilloV2Worker, GrilloV2WorkerConfig, SQLiteGrilloV2Store

store = SQLiteGrilloV2Store(Path("grillo_v2.sqlite3"))
runtime = GrilloV2Runtime(store=store, completion=my_reflection_completion, persona_id="assistant")

runtime.append_episode(
    GrilloEpisode.create(
        scope_key="app:demo:persona:v2",
        source="demo",
        actor_id="user:123",
        content="Remember that I prefer temporal memory.",
    )
)

worker = GrilloV2Worker(
    runtime=runtime,
    config=GrilloV2WorkerConfig(scope_key="app:demo:persona:v2", interval_seconds=60),
)
```

The host supplies `completion(request) -> dict`. The request includes:

- `schema`
- `mode`
- `scope_key`
- `persona_id`
- `available_tools`
- `context_packet`
- `episodes`

Return `tool_calls` using the memory tools above.

## Parity Decision

Current decision: GRILLO v2 does not replace the V1/WebWaifu GRILLO implementation one-to-one.

Why:

- V2 uses a new temporal graph schema: episodes, evidence, temporal facts, opinion edges, memory documents, and reflection cursors.
- V1/WebWaifu-style GRILLO behavior is organized around turns, candidates, diary entries, slots, relationship cadence beats, and worker traces.
- V2 can backfill some V1 data and expose Ladybug-compatible diagnostics, but that is compatibility, not parity.
- The Discord V2 bot now uses V1 Brain streaming/tool/memory response plumbing while keeping V2 GRILLO context packets. That makes the bot more capable, but it does not make the memory worker one-to-one.

Required direction:

- Keep V2 diagnostics honest: label them as GRILLO v2 / Ladybug-compatible graph views.
- Restore true WebWaifu/V1 parity only by porting the original prompt contract, JSON schema, cadence beats, diary/slot/candidate lifecycle, and worker trace semantics directly.
- Do not call V2 "real one-to-one GRILLO" until those behaviors are implemented and verified against the source WebWaifu/V1 flow.
