from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .models import Evidence, EvidenceGap, GrilloMemoryDocument, OpinionEdge, TemporalFact
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
) -> GrilloMemoryToolResult:
    result = GrilloMemoryToolResult()
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
) -> GrilloMemoryToolResult:
    normalized = name.strip().casefold()
    if normalized == "record_evidence":
        return _record_evidence(store=store, scope_key=scope_key, args=args)
    if normalized == "upsert_fact":
        return _upsert_fact(store=store, scope_key=scope_key, args=args)
    if normalized in {"upsert_opinion", "upsert_opinion_edge"}:
        return _upsert_opinion_edge(store=store, scope_key=scope_key, persona_id=persona_id, args=args)
    if normalized in {"upsert_memory_document", "write_memory_document"}:
        return _upsert_memory_document(store=store, scope_key=scope_key, args=args)
    if normalized == "invalidate_fact":
        return _invalidate_fact(store=store, args=args)
    return GrilloMemoryToolResult(ignored=1, notes=[f"unknown_tool:{name}"])


def _record_evidence(*, store: SQLiteGrilloV2Store, scope_key: str, args: dict[str, Any]) -> GrilloMemoryToolResult:
    evidence = Evidence.create(
        scope_key=scope_key,
        episode_id=str(args.get("episode_id") or ""),
        quote=str(args.get("quote") or ""),
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


def _upsert_fact(*, store: SQLiteGrilloV2Store, scope_key: str, args: dict[str, Any]) -> GrilloMemoryToolResult:
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
    store.upsert_fact(fact)
    return GrilloMemoryToolResult(facts=1)


def _upsert_opinion_edge(
    *,
    store: SQLiteGrilloV2Store,
    scope_key: str,
    persona_id: str,
    args: dict[str, Any],
) -> GrilloMemoryToolResult:
    edge = OpinionEdge.create(
        scope_key=scope_key,
        source_id=str(args.get("source_id") or persona_id),
        target_id=str(args.get("target_id") or ""),
        relation=str(args.get("relation") or ""),
        score=_float(args.get("score"), 0.0),
        rationale=str(args.get("rationale") or ""),
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
    store.upsert_memory_document(document)
    return GrilloMemoryToolResult(memory_docs=1)


def _invalidate_fact(*, store: SQLiteGrilloV2Store, args: dict[str, Any]) -> GrilloMemoryToolResult:
    fact_id = str(args.get("fact_id") or "")
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


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
