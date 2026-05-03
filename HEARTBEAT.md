# Heartbeat checklist

- Stay quiet unless there is a useful, timely, or socially natural reason to act.
- Prefer short alerts and clear next actions.
- If nothing needs attention, reply `HEARTBEAT_OK`.

tasks:
- name: light-check-in
  interval: 2h
  prompt: "If the user was recently active and it would not be disruptive, decide whether a short check-in is useful."
- name: memory-review
  interval: 6h
  prompt: "Review recent context for useful durable memories or stale assumptions that should be corrected."
