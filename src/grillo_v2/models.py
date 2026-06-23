from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4


MemoryKind = Literal["episodic", "semantic", "opinion", "procedural"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}:{uuid4()}"


@dataclass(slots=True)
class GrilloEntity:
    entity_id: str
    entity_type: str
    name: str
    aliases: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GrilloEpisode:
    episode_id: str
    scope_key: str
    source: str
    content: str
    actor_id: str | None = None
    participant_ids: list[str] = field(default_factory=list)
    channel_id: str | None = None
    occurred_at: str = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        scope_key: str,
        source: str,
        content: str,
        actor_id: str | None = None,
        participant_ids: list[str] | None = None,
        channel_id: str | None = None,
        occurred_at: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "GrilloEpisode":
        return cls(
            episode_id=new_id("episode"),
            scope_key=scope_key,
            source=source,
            content=content,
            actor_id=actor_id,
            participant_ids=participant_ids or [],
            channel_id=channel_id,
            occurred_at=occurred_at or utc_now(),
            metadata=metadata or {},
        )


@dataclass(slots=True)
class Evidence:
    evidence_id: str
    scope_key: str
    episode_id: str
    quote: str
    extractor: str = "unknown"
    confidence: float = 0.5
    created_at: str = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        scope_key: str,
        episode_id: str,
        quote: str,
        extractor: str = "unknown",
        confidence: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> "Evidence":
        return cls(
            evidence_id=new_id("evidence"),
            scope_key=scope_key,
            episode_id=episode_id,
            quote=quote,
            extractor=extractor,
            confidence=confidence,
            metadata=metadata or {},
        )


@dataclass(slots=True)
class EvidenceGap:
    question: str
    why: str
    needed: str


@dataclass(slots=True)
class TemporalFact:
    fact_id: str
    scope_key: str
    subject_id: str
    predicate: str
    object_value: str
    claim: str
    evidence_ids: list[str] = field(default_factory=list)
    confidence: float = 0.5
    valid_from: str = field(default_factory=utc_now)
    valid_to: str | None = None
    contradicts: list[str] = field(default_factory=list)
    missing_evidence: list[EvidenceGap] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    updated_at: str = field(default_factory=utc_now)

    @classmethod
    def create(
        cls,
        *,
        scope_key: str,
        subject_id: str,
        predicate: str,
        object_value: str,
        claim: str,
        evidence_ids: list[str] | None = None,
        confidence: float = 0.5,
        valid_from: str | None = None,
        valid_to: str | None = None,
        contradicts: list[str] | None = None,
        missing_evidence: list[EvidenceGap] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "TemporalFact":
        return cls(
            fact_id=new_id("fact"),
            scope_key=scope_key,
            subject_id=subject_id,
            predicate=predicate,
            object_value=object_value,
            claim=claim,
            evidence_ids=evidence_ids or [],
            confidence=confidence,
            valid_from=valid_from or utc_now(),
            valid_to=valid_to,
            contradicts=contradicts or [],
            missing_evidence=missing_evidence or [],
            metadata=metadata or {},
        )


@dataclass(slots=True)
class OpinionEdge:
    edge_id: str
    scope_key: str
    source_id: str
    target_id: str
    relation: str
    score: float
    rationale: str
    evidence_ids: list[str] = field(default_factory=list)
    valid_from: str = field(default_factory=utc_now)
    valid_to: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    updated_at: str = field(default_factory=utc_now)

    @classmethod
    def create(
        cls,
        *,
        scope_key: str,
        source_id: str,
        target_id: str,
        relation: str,
        score: float,
        rationale: str,
        evidence_ids: list[str] | None = None,
        valid_from: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "OpinionEdge":
        return cls(
            edge_id=new_id("opinion"),
            scope_key=scope_key,
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            score=max(-1.0, min(1.0, float(score))),
            rationale=rationale,
            evidence_ids=evidence_ids or [],
            valid_from=valid_from or utc_now(),
            metadata=metadata or {},
        )


@dataclass(slots=True)
class GrilloContextPacket:
    scope_key: str
    actor_id: str | None = None
    version: str = "2.0"
    current_actor: dict[str, Any] = field(default_factory=dict)
    relationship_state: list[dict[str, Any]] = field(default_factory=list)
    active_facts: list[dict[str, Any]] = field(default_factory=list)
    evidence_gaps: list[dict[str, Any]] = field(default_factory=list)
    recent_episode_summary: list[dict[str, Any]] = field(default_factory=list)
    retrieval_notes: list[str] = field(default_factory=list)
    instructions: list[str] = field(default_factory=list)

    def as_prompt_text(self) -> str:
        sections = [f'<grillo_context version="{self.version}" scope="{self.scope_key}">']
        if self.current_actor:
            sections.extend(_json_section("current_actor", [self.current_actor]))
        sections.extend(_json_section("relationship_state", self.relationship_state))
        sections.extend(_json_section("active_facts", self.active_facts))
        sections.extend(_json_section("evidence_gaps", self.evidence_gaps))
        sections.extend(_json_section("recent_episode_summary", self.recent_episode_summary))
        if self.retrieval_notes:
            sections.append("<retrieval_notes>")
            sections.extend(f"- {note}" for note in self.retrieval_notes if note.strip())
            sections.append("</retrieval_notes>")
        if self.instructions:
            sections.append("<instructions>")
            sections.extend(f"- {instruction}" for instruction in self.instructions if instruction.strip())
            sections.append("</instructions>")
        sections.append("</grillo_context>")
        return "\n".join(sections)


def to_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def from_json_dict(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def from_json_list(value: str | None) -> list[Any]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def dataclass_dict(value: Any) -> dict[str, Any]:
    data = asdict(value)
    for key, item in list(data.items()):
        if isinstance(item, list):
            data[key] = [asdict(entry) if hasattr(entry, "__dataclass_fields__") else entry for entry in item]
    return data


def _json_section(name: str, rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    out = [f"<{name}>"]
    out.extend(to_json(row) for row in rows)
    out.append(f"</{name}>")
    return out
