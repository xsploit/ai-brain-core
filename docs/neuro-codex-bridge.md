# Neuro Codex Bridge Grounding

## Purpose

Neuro can act as a Discord-side control plane for bounded requests into this Codex thread. The bridge must keep Codex attached to the current thread, keep local work grounded in the Brain repo, and route optional subagent work through the local Agency Harness.

## Current Thread Contract

- Current Codex thread id: `019e53da-7adc-7251-a203-e9da141553f7`
- Use the Codex app heartbeat with `destination=thread` so wakeups resume this current Codex thread.
- Do not create a new Codex thread for bridge work unless Subby explicitly asks.
- The heartbeat should read this document first, then inspect the queue.
- If no queued request exists, do not invent work.
- If a queued request exists, process one request, report in this thread, write a result file, and stop.

## Local Paths

- Brain repo: `C:\Users\SUBSECT\Documents\GitHub\Brain`
- Harness repo: `C:\Users\SUBSECT\Documents\Harness`
- Bridge root: `C:\Users\SUBSECT\Documents\GitHub\Brain\codex_bridge`
- Inbox: `C:\Users\SUBSECT\Documents\GitHub\Brain\codex_bridge\inbox`
- Outbox: `C:\Users\SUBSECT\Documents\GitHub\Brain\codex_bridge\outbox`
- Archive: `C:\Users\SUBSECT\Documents\GitHub\Brain\codex_bridge\archive`
- State file: `C:\Users\SUBSECT\Documents\GitHub\Brain\codex_bridge\state.json`

## Request Shape

Each inbox file should be JSON:

```json
{
  "id": "uuid-or-snowflake",
  "created_at": "2026-06-21T00:00:00-07:00",
  "source": "discord",
  "requester_id": "120418341775998976",
  "requester_name": "SUBSECT",
  "guild_id": "optional",
  "channel_id": "optional",
  "message_id": "optional",
  "intent": "ask_codex|review|research|harness|implement",
  "priority": "normal",
  "prompt": "What Neuro wants Codex to handle.",
  "context": {
    "recent_messages": [],
    "attachments": [],
    "model_suggestion": "optional"
  },
  "harness": {
    "agent": "claude",
    "permission_profile": "inspect"
  }
}
```

## Routing Rules

- Default to Codex handling the request directly in this thread.
- Use Harness only for bounded review, second opinion, research, or verification.
- Default Harness permission profile is `inspect`.
- Use `edit` only for an explicit implementation request from Subby/admin.
- Use `full-auto` only when Subby explicitly asks for full autonomous local work.
- Reject or defer requests that ask for broad arbitrary command execution.

## Harness Commands

Set `PYTHONPATH` when running from the Harness repo:

```powershell
Set-Location -LiteralPath "C:\Users\SUBSECT\Documents\Harness"
$env:PYTHONPATH = "C:\Users\SUBSECT\Documents\Harness\src"
py -3 -m agency_harness discover --json
py -3 -m agency_harness run --agent claude --permission-profile inspect --prompt "..." --cwd "C:\Users\SUBSECT\Documents\GitHub\Brain"
py -3 -m agency_harness run --agent claude --permission-profile inspect --prompt-file "C:\path\to\request.md" --cwd "C:\Users\SUBSECT\Documents\GitHub\Brain" --background
py -3 -m agency_harness jobs status <job-id>
py -3 -m agency_harness jobs tail <job-id> --lines 120
```

Verified agents on this machine as of 2026-06-21:

- `codex`
- `claude`
- `agy`
- `cursor`

## Discord Controls To Add

- `!codex ask <prompt>`: owner/admin queues a request.
- `!codex status`: shows pending, last processed, paused state.
- `!codex pause`: prevents autonomous enqueue and processing.
- `!codex resume`: resumes queueing.
- `!codex clear`: owner/admin clears pending bridge requests.

## Safety Rules

- Owner/admin only for manual bridge commands.
- Autonomous Neuro heartbeat can enqueue only if bridge is enabled, not paused, and rate limit allows it.
- Every request must include requester, channel, guild, and message metadata when available.
- Queue processing is one request per heartbeat.
- Codex must write outbox results before archiving inbox files.
- Do not expose secrets in Discord, PRs, or bridge output.
- Do not let Neuro call arbitrary local shell commands directly.

## First Implementation Slice

1. Add `src/aibrain/codex_bridge.py` with queue read/write/status helpers.
2. Add Discord `!codex` command group.
3. Add env flags:
   - `DISCORD_BRAIN_CODEX_BRIDGE_ENABLED=false`
   - `DISCORD_BRAIN_CODEX_BRIDGE_QUEUE=C:\Users\SUBSECT\Documents\GitHub\Brain\codex_bridge`
   - `DISCORD_BRAIN_CODEX_BRIDGE_HEARTBEAT_CHANCE=0.02`
   - `DISCORD_BRAIN_CODEX_BRIDGE_MIN_INTERVAL_SECONDS=1800`
4. Add a Codex thread heartbeat that reads this doc and processes one inbox file.
5. After queue is proven, add Harness routing for requests with `"intent": "harness"`.
