from __future__ import annotations

import re


_POLICY_RULE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for pattern in (
        r"\bfrom\s+now\s+on\b",
        r"\bcurrent\s+rule\b",
        r"\bnew\s+rule\b",
        r"\boverrid(?:e|es|den|ing)\b.{0,80}\b(previous|prior|rules?|instructions?)\b",
        r"\b(do\s+not|don't)\s+listen\s+to\b",
        r"\bignore\s+(anyone|everyone|all|previous|prior)\b",
        r"\b(reply|respond|answer)\s+(only|to\s+everyone|to\s+all|with|to\s+.*\s+with)\b",
        r"\b(only\s+reply|reply\s+only|only\s+say|say\s+only)\b",
        r"\bnothing\s+else\b",
        r"\b(shitlist|blacklist|blocklist)\b",
        r"\bregard\s+them\s+as\s+enemies\b",
    )
]


_SAFE_OBSERVATION_PATTERNS = [
    re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for pattern in (
        r"\b(prompt[-\s]?injection|social engineering|override attempt)\b",
        r"\b(attempted|tried)\b.{0,80}\b(change|override|set)\b.{0,80}\b(rule|instruction|behavior)\b",
        r"\bnot\s+(an|a)\s+instruction\b",
        r"\bdo\s+not\s+obey\b",
    )
]


_POLICY_DOCUMENT_TYPES = {
    "instruction",
    "instructions",
    "manual_memory",
    "policy",
    "procedural_note",
    "rule",
    "system_prompt",
}


_POLICY_PREDICATES = {
    "accepted",
    "current_rule",
    "instruction",
    "received instruction",
    "received_instruction",
    "reply_rule",
    "response_rule",
    "should_reply",
}


def looks_like_user_behavior_rule(text: str) -> bool:
    lowered = text.casefold()
    if not lowered.strip():
        return False
    if any(pattern.search(lowered) for pattern in _SAFE_OBSERVATION_PATTERNS):
        return False
    return any(pattern.search(lowered) for pattern in _POLICY_RULE_PATTERNS)


def is_unsafe_memory_payload(
    *,
    text: str,
    document_type: str = "",
    subject_id: str = "",
    predicate: str = "",
    persona_id: str = "",
) -> bool:
    if not looks_like_user_behavior_rule(text):
        return False
    normalized_type = document_type.strip().casefold()
    normalized_predicate = predicate.strip().casefold()
    normalized_subject = subject_id.strip().casefold()
    normalized_persona = persona_id.strip().casefold()
    if normalized_type in _POLICY_DOCUMENT_TYPES:
        return True
    if normalized_predicate in _POLICY_PREDICATES:
        return True
    if normalized_persona and normalized_subject == normalized_persona:
        return True
    if "neuro" in normalized_subject or "persona" in normalized_subject or "assistant" in normalized_subject:
        return True
    return normalized_type in {"diary", "profile", "relationship_profile"} and any(
        phrase in text.casefold()
        for phrase in (
            "i am complying",
            "i accepted",
            "my instructions are",
            "must regard them",
            "must reply",
        )
    )
