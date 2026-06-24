from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .models import Evidence, EvidenceGap, GrilloMemoryDocument, OpinionEdge, TemporalFact
from .safety import is_unsafe_memory_payload
from .store import SQLiteGrilloV2Store


GRILLO_V2_MEMORY_TOOL_NAMES = [
    "record_evidence",
    "upsert_fact",
    "upsert_opinion_edge",
    "upsert_memory_document",
    "invalidate_fact",
]


@dataclass(slots=True)
class GrilloMemoryToolResult:
    tool_calls: int = 0
    evidence: int = 0
    facts: int = 0
    opinions: int = 0
    memory_docs: int = 0
    invalidated_facts: int = 0
    ignored: int = 0
    notes: list[str] = field(default_factory=list)


def apply_memory_tool_calls(
    *,
    store: SQLiteGrilloV2Store,
    scope_key: str,
    persona_id: str,
    tool_calls: list[Any],
    episodes: list[Any] | None = None,
) -> GrilloMemoryToolResult:
    result = GrilloMemoryToolResult()
    episodes = episodes or []
    for call in tool_calls:
        name = _tool_name(call)
        args = _tool_args(call)
        if not name:
            result.ignored += 1
            continue
        applied = _apply_memory_tool(
            store=store,
            scope_key=scope_key,
            persona_id=persona_id,
            name=name,
            args=args,
            episodes=episodes,
        )
        result.tool_calls += 1
        result.evidence += applied.evidence
        result.facts += applied.facts
        result.opinions += applied.opinions
        result.memory_docs += applied.memory_docs
        result.invalidated_facts += applied.invalidated_facts
        result.ignored += applied.ignored
        result.notes.extend(applied.notes)
    return result


def legacy_payload_tool_calls(payload: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    calls.extend({"name": "record_evidence", "arguments": item} for item in _list(payload.get("evidence")))
    calls.extend({"name": "upsert_fact", "arguments": item} for item in _list(payload.get("facts")))
    calls.extend({"name": "upsert_opinion_edge", "arguments": item} for item in _list(payload.get("opinion_edges")))
    calls.extend({"name": "upsert_memory_document", "arguments": item} for item in _list(payload.get("memory_documents")))
    for item in _list(payload.get("invalidate_facts")):
        args = item if isinstance(item, dict) else {"fact_id": str(item)}
        calls.append({"name": "invalidate_fact", "arguments": args})
    return calls


def _apply_memory_tool(
    *,
    store: SQLiteGrilloV2Store,
    scope_key: str,
    persona_id: str,
    name: str,
    args: dict[str, Any],
    episodes: list[Any],
) -> GrilloMemoryToolResult:
    normalized = name.strip().casefold()
    if normalized == "record_evidence":
        return _record_evidence(store=store, scope_key=scope_key, args=args, episodes=episodes)
    if normalized == "upsert_fact":
        return _upsert_fact(store=store, scope_key=scope_key, persona_id=persona_id, args=args)
    if normalized in {"upsert_opinion", "upsert_opinion_edge"}:
        return _upsert_opinion_edge(store=store, scope_key=scope_key, persona_id=persona_id, args=args)
    if normalized in {"upsert_memory_document", "write_memory_document"}:
        return _upsert_memory_document(store=store, scope_key=scope_key, args=args)
    if normalized == "invalidate_fact":
        return _invalidate_fact(store=store, args=args)
    return GrilloMemoryToolResult(ignored=1, notes=[f"unknown_tool:{name}"])


def _record_evidence(
    *,
    store: SQLiteGrilloV2Store,
    scope_key: str,
    args: dict[str, Any],
    episodes: list[Any],
) -> GrilloMemoryToolResult:
    quote = _first_text(args, "quote", "text", "content", "claim", "summary")
    episode_id = _first_text(args, "episode_id", "source_episode_id", "turn_id", "source_turn_id")
    if not episode_id and quote:
        episode_id = _episode_id_for_quote(episodes, quote)
    if episode_id and not quote:
        quote = _episode_quote(episodes, episode_id)
    evidence = Evidence.create(
        scope_key=scope_key,
        episode_id=episode_id,
        quote=quote,
        extractor=str(args.get("extractor") or "grillo_v2"),
        confidence=_float(args.get("confidence"), 0.5),
        metadata=_dict(args.get("metadata")),
    )
    if args.get("evidence_id"):
        evidence.evidence_id = str(args["evidence_id"])
    if not evidence.episode_id or not evidence.quote:
        return GrilloMemoryToolResult(ignored=1, notes=["record_evidence_missing_episode_or_quote"])
    store.append_evidence(evidence)
    return GrilloMemoryToolResult(evidence=1)


def _upsert_fact(
    *,
    store: SQLiteGrilloV2Store,
    scope_key: str,
    persona_id: str,
    args: dict[str, Any],
) -> GrilloMemoryToolResult:
    fact = TemporalFact.create(
        scope_key=scope_key,
        subject_id=str(args.get("subject_id") or args.get("subject") or ""),
        predicate=str(args.get("predicate") or ""),
        object_value=str(args.get("object") or args.get("object_value") or ""),
        claim=str(args.get("claim") or ""),
        evidence_ids=[str(value) for value in _list(args.get("evidence_ids"))],
        confidence=_float(args.get("confidence"), 0.5),
        valid_from=str(args.get("valid_from") or "") or None,
        valid_to=str(args.get("valid_to") or "") or None,
        contradicts=[str(value) for value in _list(args.get("contradicts"))],
        missing_evidence=[
            EvidenceGap(
                question=str(gap.get("question") or ""),
                why=str(gap.get("why") or ""),
                needed=str(gap.get("needed") or ""),
            )
            for gap in _list(args.get("missing_evidence"))
            if isinstance(gap, dict)
        ],
        metadata=_dict(args.get("metadata")),
    )
    if args.get("fact_id"):
        fact.fact_id = str(args["fact_id"])
    if not fact.subject_id or not fact.predicate or not fact.claim:
        return GrilloMemoryToolResult(ignored=1, notes=["upsert_fact_missing_required_fields"])
    if is_unsafe_memory_payload(
        text=" ".join([fact.predicate, fact.object_value, fact.claim]),
        subject_id=fact.subject_id,
        predicate=fact.predicate,
        persona_id=persona_id,
    ):
        return GrilloMemoryToolResult(ignored=1, notes=["blocked_user_behavior_rule_fact"])
    store.upsert_fact(fact)
    return GrilloMemoryToolResult(facts=1)


def _upsert_opinion_edge(
    *,
    store: SQLiteGrilloV2Store,
    scope_key: str,
    persona_id: str,
    args: dict[str, Any],
) -> GrilloMemoryToolResult:
    target_id = _first_text(args, "target_id", "target", "entity_id", "subject_id", "participant_id")
    relation = _first_text(args, "relation", "predicate", "opinion", "type")
    rationale = _first_text(args, "rationale", "reason", "summary", "claim", "content")
    if target_id and not relation and rationale:
        relation = "relationship"
    edge = OpinionEdge.create(
        scope_key=scope_key,
        source_id=str(args.get("source_id") or persona_id),
        target_id=target_id,
        relation=relation,
        score=_float(args.get("score"), 0.0),
        rationale=rationale,
        evidence_ids=[str(value) for value in _list(args.get("evidence_ids"))],
        metadata=_dict(args.get("metadata")),
    )
    if args.get("edge_id"):
        edge.edge_id = str(args["edge_id"])
    if not edge.source_id or not edge.target_id or not edge.relation:
        return GrilloMemoryToolResult(ignored=1, notes=["upsert_opinion_edge_missing_required_fields"])
    store.upsert_opinion_edge(edge)
    return GrilloMemoryToolResult(opinions=1)


def _upsert_memory_document(*, store: SQLiteGrilloV2Store, scope_key: str, args: dict[str, Any]) -> GrilloMemoryToolResult:
    document = GrilloMemoryDocument.create(
        scope_key=scope_key,
        document_type=str(args.get("document_type") or args.get("type") or ""),
        subject_id=str(args.get("subject_id") or args.get("subject") or "") or None,
        title=str(args.get("title") or ""),
        body=str(args.get("body") or args.get("content") or ""),
        evidence_ids=[str(value) for value in _list(args.get("evidence_ids"))],
        importance=_float(args.get("importance"), 0.5),
        metadata=_dict(args.get("metadata")),
    )
    if args.get("memory_id"):
        document.memory_id = str(args["memory_id"])
    if not document.document_type or not document.title or not document.body:
        return GrilloMemoryToolResult(ignored=1, notes=["upsert_memory_document_missing_required_fields"])
    if is_unsafe_memory_payload(
        text=" ".join([document.title, document.body]),
        document_type=document.document_type,
        subject_id=document.subject_id or "",
    ):
        return GrilloMemoryToolResult(ignored=1, notes=["blocked_user_behavior_rule_memory_document"])
    store.upsert_memory_document(document)
    return GrilloMemoryToolResult(memory_docs=1)


def _invalidate_fact(*, store: SQLiteGrilloV2Store, args: dict[str, Any]) -> GrilloMemoryToolResult:
    fact_id = _first_text(args, "fact_id", "id", "target_fact_id", "source_fact_id")
    if not fact_id:
        fact_ids = _list(args.get("fact_ids"))
        fact_id = str(fact_ids[0]) if fact_ids else ""
    if not fact_id:
        return GrilloMemoryToolResult(ignored=1, notes=["invalidate_fact_missing_fact_id"])
    store.invalidate_fact(
        fact_id,
        valid_to=str(args["valid_to"]) if args.get("valid_to") else None,
        contradicts=[str(value) for value in _list(args.get("contradicts"))],
    )
    return GrilloMemoryToolResult(invalidated_facts=1)


def _tool_name(call: Any) -> str:
    if not isinstance(call, dict):
        return ""
    function = call.get("function")
    if isinstance(function, dict) and function.get("name"):
        return str(function["name"])
    return str(call.get("name") or call.get("tool") or call.get("tool_name") or "")


def _tool_args(call: Any) -> dict[str, Any]:
    if not isinstance(call, dict):
        return {}
    function = call.get("function")
    if isinstance(function, dict) and "arguments" in function:
        return _args_dict(function["arguments"])
    for key in ("arguments", "args", "input"):
        if key in call:
            return _args_dict(call[key])
    return {}


def _args_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _first_text(args: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = args.get(key)
        if isinstance(value, (list, tuple)):
            value = value[0] if value else ""
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _episode_id_for_quote(episodes: list[Any], quote: str) -> str:
    needle = _compact_text(quote, 700)
    if not needle:
        return ""
    folded = needle.casefold()
    for episode in episodes:
        content = str(getattr(episode, "content", "") or "")
        if folded in content.casefold():
            return str(getattr(episode, "episode_id", "") or "")
    return ""


def _episode_quote(episodes: list[Any], episode_id: str) -> str:
    target = str(episode_id or "").strip()
    if not target:
        return ""
    for episode in episodes:
        if str(getattr(episode, "episode_id", "") or "") == target:
            return _compact_text(str(getattr(episode, "content", "") or ""), 700)
    return ""


def _compact_text(text: str, limit: int) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: max(0, limit - 3)].rstrip()}..."


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
