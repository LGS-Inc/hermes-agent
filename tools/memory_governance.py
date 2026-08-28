"""Deterministic governance classifier for durable learning candidates.

This module is deliberately small and model-independent.  It does not decide
whether a policy is good or current; it only detects text that would change or
encode control-plane authority and therefore must not become autonomous
background memory.

The classifier returns bounded reason codes and never logs or persists the
candidate text.  Callers decide whether to stage, block, or content-free
suppress a governance-affecting candidate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Tuple


NORMAL = "normal"
GOVERNANCE = "governance"
UNKNOWN = "unknown"

MAX_CLASSIFIER_CHARS = 32_768


@dataclass(frozen=True)
class GovernanceMemoryDecision:
    classification: str
    reason_codes: Tuple[str, ...] = ()

    @property
    def requires_review(self) -> bool:
        return self.classification != NORMAL


_NAMED_PROTOCOL_RE = re.compile(
    r"\b(?:protocol\s+alpha|protocol\s+omega|fable(?:\s+gate)?|"
    r"independent\s+review|exact[-\s]?hash\s+review)\b",
    re.I,
)

_CHAIRMAN_AUTHORITY_RE = re.compile(
    r"\bchairman\b.{0,100}\b(?:authori[sz](?:e|ation|ed)?|authority|approval|"
    r"permission|gate|instruction|may|must|shall|requires?|no\s+longer|"
    r"can(?:not|'t)?|override|supersed(?:e|es|ed))\b|"
    r"\b(?:authori[sz](?:e|ation|ed)?|authority|approval|permission|override|"
    r"supersed(?:e|es|ed))\b.{0,100}\bchairman\b",
    re.I | re.S,
)

_APPROVAL_GATE_RE = re.compile(
    r"\b(?:approval|authorization|safety|lifecycle|chairman|tool)\s*[- ]?gate\b|"
    r"\b(?:approval|authorization)\b.{0,100}\b(?:bypass|disable|skip|remove|"
    r"automatic|auto[- ]?approve|not\s+required|no\s+longer|required|default)\b|"
    r"\b(?:bypass|disable|skip|remove|auto[- ]?approve|never\s+require|"
    r"no\s+longer\s+require)\b.{0,100}\b(?:approval|authorization|gate)\b",
    re.I | re.S,
)

_MUTATION_AUTHORITY_RE = re.compile(
    r"\b(?:write|edit|mutat(?:e|ion)|commit|deploy|publish|send|delete|remove|"
    r"authenticate|restore|rollback|production)\b.{0,100}\b(?:authority|"
    r"authori[sz](?:e|ation|ed)?|permission|permit(?:ted)?|allow(?:ed)?|approval|"
    r"without\s+approval|may|must|shall|can)\b|"
    r"\b(?:authority|authori[sz](?:e|ation|ed)?|permission|permit(?:ted)?|"
    r"allow(?:ed)?|approval|without\s+approval)\b.{0,100}\b(?:write|edit|"
    r"mutat(?:e|ion)|commit|deploy|publish|send|delete|remove|authenticate|"
    r"restore|rollback|production)\b",
    re.I | re.S,
)

_CREDENTIAL_POLICY_RE = re.compile(
    r"\b(?:credential|password|secret|api\s*key|access\s*token|authentication)"
    r"\b.{0,100}\b(?:policy|authority|approval|permission|gate|always|never|"
    r"must|shall|may|automatic|bypass|without\s+approval|no\s+longer)\b|"
    r"\b(?:policy|authority|approval|permission|gate|always|never|must|shall|"
    r"automatic|bypass|without\s+approval)\b.{0,100}\b(?:credential|password|"
    r"secret|api\s*key|access\s*token|authentication)\b",
    re.I | re.S,
)

_REVIEW_AUTHORITY_RE = re.compile(
    r"\b(?:review|verdict|proceed|exact[-\s]?hash)\b.{0,100}\b(?:authority|"
    r"authori[sz](?:e|ation|ed)?|permission|deploy|commit|mutat(?:e|ion)|"
    r"automatic|mandatory|required|approval)\b|"
    r"\b(?:authority|authori[sz](?:e|ation|ed)?|permission|automatic|mandatory|"
    r"required)\b.{0,100}\b(?:review|verdict|proceed|exact[-\s]?hash)\b",
    re.I | re.S,
)

_MODEL_ROUTING_RE = re.compile(
    r"\b(?:model\s+routing|routing\s+(?:policy|authority|default)|default\s+model|"
    r"model\s+selection\s+(?:policy|authority))\b|"
    r"(?:^|[.!?]\s+)(?:dumbledore\s+)?(?:always\s+)?(?:use|route|select|choose|"
    r"default\s+to|prefer)\s+(?:qwen[\w.:-]*|gpt[\w.:-]*|claude[\w.:-]*|"
    r"gemini[\w.:-]*|[\w.-]+\s+model)\b|"
    r"\bdumbledore\b.{0,60}\b(?:must|shall|should|always)\s+(?:use|route|"
    r"select|choose)\b.{0,80}\bmodel\b",
    re.I | re.S,
)

_ABSOLUTE_DIRECTIVE_RE = re.compile(
    r"(?:^|[.!?]\s+)(?:dumbledore\s+)?(?:always|never|must|shall)\s+"
    r"(?:run|use|activate|invoke|start|continue|skip|bypass|allow|permit|require|"
    r"approve|authori[sz]e|write|edit|commit|deploy|publish|send|delete|remove|"
    r"authenticate|restore|rollback|route|select|choose)\b|"
    r"\b(?:automatically|auto[- ]?)(?:run|activate|invoke|start|approve|deploy|"
    r"commit|route|select|review)\b|"
    r"\b(?:do\s+not|don't|never)\s+require\s+(?:approval|authorization)\b|"
    r"\bno\s+longer\s+(?:needs?|requires?)\s+(?:approval|authorization|a\s+gate)\b",
    re.I | re.S,
)


_RULES = (
    ("named_protocol_activation", _NAMED_PROTOCOL_RE),
    ("chairman_authority", _CHAIRMAN_AUTHORITY_RE),
    ("approval_or_safety_gate", _APPROVAL_GATE_RE),
    ("mutation_or_deployment_authority", _MUTATION_AUTHORITY_RE),
    ("credential_policy", _CREDENTIAL_POLICY_RE),
    ("review_authority", _REVIEW_AUTHORITY_RE),
    ("model_routing_authority", _MODEL_ROUTING_RE),
    ("absolute_behavior_directive", _ABSOLUTE_DIRECTIVE_RE),
)


def classify_governance_texts(values: Iterable[str]) -> GovernanceMemoryDecision:
    """Classify one bounded collection of candidate-memory strings.

    ``UNKNOWN`` is intentionally review-requiring.  It is returned when the
    caller supplies malformed or oversized input rather than guessing that the
    content is safe.
    """

    texts = []
    total = 0
    try:
        iterator = iter(values)
    except TypeError:
        return GovernanceMemoryDecision(UNKNOWN, ("invalid_classifier_input",))

    for value in iterator:
        if not isinstance(value, str):
            return GovernanceMemoryDecision(UNKNOWN, ("invalid_classifier_input",))
        total += len(value)
        if total > MAX_CLASSIFIER_CHARS:
            return GovernanceMemoryDecision(UNKNOWN, ("classifier_input_too_large",))
        if value:
            texts.append(value)

    if not texts:
        return GovernanceMemoryDecision(NORMAL)

    combined = "\n".join(texts)
    reasons = tuple(code for code, pattern in _RULES if pattern.search(combined))
    if reasons:
        return GovernanceMemoryDecision(GOVERNANCE, reasons)
    return GovernanceMemoryDecision(NORMAL)


__all__ = [
    "GOVERNANCE",
    "MAX_CLASSIFIER_CHARS",
    "NORMAL",
    "UNKNOWN",
    "GovernanceMemoryDecision",
    "classify_governance_texts",
]
