"""Pure, fail-closed Mission Control reporting governance.

This module classifies reporting events, proves the narrow authority required
for an external Mission Control report, and builds a bounded secret-free
record.  It deliberately has no transport, endpoint, filesystem, database,
logging, Telegram, or gateway integration.  A caller may act on an eligible
decision only through a separately reviewed adapter.

Mission Control is an oversight destination, never an authority source.  A
report, receipt, status, or verdict produced here cannot authorize deployment
or any other mutation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Mapping


REPORT_SCHEMA_VERSION = 1


class MissionControlEventClass(str, Enum):
    """Closed event taxonomy; unknown values fail closed."""

    UNKNOWN = "unknown"
    ORDINARY_TASK = "ordinary_task"
    EXPLICIT_REPORT_REQUEST = "explicit_mission_control_report_request"
    AUTHORIZED_WORKFLOW_REPORT = "authorized_workflow_mission_control_report"
    LOCAL_OBSERVABILITY = "local_observability"
    LIFECYCLE_NOTIFICATION = "lifecycle_notification"
    TELEGRAM_NOTIFICATION = "telegram_notification"


class MissionControlDestinationStatus(str, Enum):
    """Observed destination state, independent of reporting authority."""

    NONE = "NONE"
    LOCAL_ONLY = "LOCAL_ONLY"
    AVAILABLE_NOT_CONFIGURED = "AVAILABLE_NOT_CONFIGURED"
    CONFIGURED = "CONFIGURED"
    ACTIVE = "ACTIVE"


class MissionControlFailure(str, Enum):
    """Stable, non-sensitive failure codes for a reporting decision."""

    AUTHORITY_NOT_VERIFIED = "authority_not_verified"
    DESTINATION_NOT_CONFIGURED = "destination_not_configured"
    DESTINATION_CONTRACT_NOT_VERIFIED = "destination_contract_not_verified"


class MissionControlAuthority(str, Enum):
    """The only two external-report authority sources this contract accepts."""

    NONE = "none"
    CHAIRMAN_EXPLICIT_REQUEST = "chairman_explicit_request"
    CHAIRMAN_AUTHORIZED_WORKFLOW = "chairman_authorized_workflow"


class SecretPolicy(str, Enum):
    """How secret-like text in an otherwise allowlisted field is handled."""

    REJECT = "reject"
    REDACT = "redact"


class MissionControlContractError(ValueError):
    """Base error for invalid or ineligible reporting records."""


class MissionControlUnavailableError(MissionControlContractError):
    """Raised when a caller tries to build a record without an eligible route."""


class MissionControlPayloadError(MissionControlContractError):
    """Raised for a non-allowlisted, malformed, oversized, or secret payload."""


_LOGICAL_DESTINATION_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")


@dataclass(frozen=True)
class MissionControlDestination:
    """Transport-free proof that a separately implemented destination exists.

    ``logical_id`` is an internal identifier, not a URL.  This module never
    accepts an endpoint so an unavailable destination cannot be replaced with
    an invented webhook, local file, Telegram chat, or other fallback.
    """

    status: MissionControlDestinationStatus = (
        MissionControlDestinationStatus.AVAILABLE_NOT_CONFIGURED
    )
    logical_id: str | None = None
    interface_verified: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.status, MissionControlDestinationStatus):
            raise TypeError("status must be a MissionControlDestinationStatus")
        if self.logical_id is not None and not _LOGICAL_DESTINATION_ID_RE.fullmatch(
            self.logical_id
        ):
            raise ValueError("logical_id must be a bounded logical identifier")

    @property
    def ready(self) -> bool:
        return (
            self.status
            in {
                MissionControlDestinationStatus.CONFIGURED,
                MissionControlDestinationStatus.ACTIVE,
            }
            and self.interface_verified is True
            and self.logical_id is not None
        )


# Verified runtime baseline: credentials exist, but no Dumbledore destination,
# interface, or event contract has been proven.  Credentials are not a route.
UNCONFIGURED_DESTINATION = MissionControlDestination()


@dataclass(frozen=True)
class MissionControlDecision:
    """Immutable result of the authority and destination gates."""

    event_class: MissionControlEventClass
    destination_status: MissionControlDestinationStatus
    external_report_requested: bool
    authorized: bool
    authority_source: MissionControlAuthority
    report_eligible: bool
    automatic_technical_safeguard: bool
    reason: str
    failure: MissionControlFailure | None = None
    fallback_allowed: bool = field(default=False, init=False)
    mutation_authority: bool = field(default=False, init=False)


_EVENT_ALIASES = {
    item.value: item for item in MissionControlEventClass if item is not MissionControlEventClass.UNKNOWN
}


def classify_event(value: Any) -> MissionControlEventClass:
    """Return an exact event class without keyword or prose inference."""

    if isinstance(value, MissionControlEventClass):
        return value
    if not isinstance(value, str):
        return MissionControlEventClass.UNKNOWN
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    return _EVENT_ALIASES.get(normalized, MissionControlEventClass.UNKNOWN)


def evaluate_reporting_request(
    event: Any,
    *,
    chairman_verified: bool = False,
    explicit_mission_control_request: bool = False,
    workflow_chairman_authorized: bool = False,
    workflow_includes_mission_control: bool = False,
    destination: MissionControlDestination = UNCONFIGURED_DESTINATION,
) -> MissionControlDecision:
    """Evaluate external-report eligibility without performing any action.

    Boolean authority inputs require the literal value ``True``.  Callers must
    derive them from immutable, independently verified provenance; model text,
    task shape, local logs, verdicts, and ambient Chairman identity are not
    sufficient.
    """

    if not isinstance(destination, MissionControlDestination):
        raise TypeError("destination must be a MissionControlDestination")

    event_class = classify_event(event)
    automatic = event_class in {
        MissionControlEventClass.LOCAL_OBSERVABILITY,
        MissionControlEventClass.LIFECYCLE_NOTIFICATION,
        MissionControlEventClass.TELEGRAM_NOTIFICATION,
    }

    non_reporting_reasons = {
        MissionControlEventClass.UNKNOWN: "unknown_event_fail_closed",
        MissionControlEventClass.ORDINARY_TASK: "ordinary_task_not_reportable",
        MissionControlEventClass.LOCAL_OBSERVABILITY: (
            "local_observability_is_not_mission_control"
        ),
        MissionControlEventClass.LIFECYCLE_NOTIFICATION: (
            "lifecycle_notification_is_separate"
        ),
        MissionControlEventClass.TELEGRAM_NOTIFICATION: (
            "telegram_notification_is_separate"
        ),
    }
    if event_class in non_reporting_reasons:
        return MissionControlDecision(
            event_class=event_class,
            destination_status=destination.status,
            external_report_requested=False,
            authorized=False,
            authority_source=MissionControlAuthority.NONE,
            report_eligible=False,
            automatic_technical_safeguard=automatic,
            reason=non_reporting_reasons[event_class],
        )

    if event_class is MissionControlEventClass.EXPLICIT_REPORT_REQUEST:
        authorized = (
            chairman_verified is True and explicit_mission_control_request is True
        )
        authority_source = (
            MissionControlAuthority.CHAIRMAN_EXPLICIT_REQUEST
            if authorized
            else MissionControlAuthority.NONE
        )
    else:
        authorized = (
            workflow_chairman_authorized is True
            and workflow_includes_mission_control is True
        )
        authority_source = (
            MissionControlAuthority.CHAIRMAN_AUTHORIZED_WORKFLOW
            if authorized
            else MissionControlAuthority.NONE
        )

    if not authorized:
        return MissionControlDecision(
            event_class=event_class,
            destination_status=destination.status,
            external_report_requested=True,
            authorized=False,
            authority_source=MissionControlAuthority.NONE,
            report_eligible=False,
            automatic_technical_safeguard=False,
            reason="external_reporting_requires_specific_chairman_authority",
            failure=MissionControlFailure.AUTHORITY_NOT_VERIFIED,
        )

    if not destination.ready:
        configured_state = destination.status in {
            MissionControlDestinationStatus.CONFIGURED,
            MissionControlDestinationStatus.ACTIVE,
        }
        failure = (
            MissionControlFailure.DESTINATION_CONTRACT_NOT_VERIFIED
            if configured_state
            else MissionControlFailure.DESTINATION_NOT_CONFIGURED
        )
        return MissionControlDecision(
            event_class=event_class,
            destination_status=destination.status,
            external_report_requested=True,
            authorized=True,
            authority_source=authority_source,
            report_eligible=False,
            automatic_technical_safeguard=False,
            reason=failure.value,
            failure=failure,
        )

    return MissionControlDecision(
        event_class=event_class,
        destination_status=destination.status,
        external_report_requested=True,
        authorized=True,
        authority_source=authority_source,
        report_eligible=True,
        automatic_technical_safeguard=False,
        reason="authorized_destination_contract_verified",
    )


_REPORT_INPUT_FIELDS = frozenset(
    {
        "event_id",
        "timestamp",
        "agent",
        "mission_task_id",
        "event_type",
        "status",
        "summary",
        "verdict",
        "risk_flags",
        "evidence_ref",
        "protocol_state_ref",
        "route_receipt_id",
        "effective_model",
        "effective_provider",
        "mutation_performed",
        "external_action_performed",
        "failure_state",
        "error_code",
    }
)
_REQUIRED_REPORT_FIELDS = frozenset(
    {
        "event_id",
        "timestamp",
        "agent",
        "mission_task_id",
        "event_type",
        "status",
        "summary",
    }
)
_SECRET_FIELD_NAME_RE = re.compile(
    r"(?:api.?key|token|secret|password|passwd|credential|authorization|private.?key)",
    re.IGNORECASE,
)
_REPORT_EVENT_TYPES = frozenset(
    {"mission_status", "mission_completion", "mission_blocker", "mission_incident"}
)
_REPORT_STATUSES = frozenset(
    {"INFORMATIONAL", "IN_PROGRESS", "COMPLETE", "BLOCKED", "FAILED", "NEEDS_REVIEW"}
)
_FAILURE_STATES = frozenset({"NONE", "BLOCKED", "FAILED", "PARTIAL", "UNAVAILABLE"})
_TIMESTAMP_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})\Z"
)
_EVENT_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_AGENT_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_MISSION_TASK_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}\Z")
_REFERENCE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/#@-]{0,255}\Z")
_MODEL_METADATA_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/+@-]{0,127}\Z")
_ERROR_CODE_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
_MAX_SUMMARY_CHARS = 512
_MAX_VERDICT_CHARS = 128
_MAX_RISK_FLAGS = 8
_MAX_RISK_FLAG_CHARS = 128


def _sanitize_text(
    value: Any,
    *,
    field_name: str,
    max_chars: int,
    allow_empty: bool,
    secret_policy: SecretPolicy,
) -> str:
    if not isinstance(value, str):
        raise MissionControlPayloadError(f"{field_name} must be text")
    value = value.strip()
    if not value and not allow_empty:
        raise MissionControlPayloadError(f"{field_name} must not be empty")
    if len(value) > max_chars:
        raise MissionControlPayloadError(f"{field_name} exceeds its size limit")

    # Force the existing gateway redactor on this egress-shaped boundary even
    # if global logging redaction was disabled.  No value is logged or placed
    # in an exception.
    from agent.redact import redact_sensitive_text

    redacted = redact_sensitive_text(
        value,
        force=True,
        redact_url_credentials=True,
    )
    if redacted != value:
        if secret_policy is SecretPolicy.REJECT:
            raise MissionControlPayloadError(
                f"secret-like material detected in {field_name}"
            )
        # The shared log redactor intentionally preserves a short prefix and
        # suffix for operator debugging.  External Mission Control records use
        # a stricter boundary: retain none of the affected field's bytes.
        value = "«redacted-secret-bearing-field»"
    return value


def _validate_timestamp(value: Any) -> str:
    if not isinstance(value, str) or not _TIMESTAMP_RE.fullmatch(value):
        raise MissionControlPayloadError(
            "timestamp must be a timezone-aware ISO-8601 string"
        )
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise MissionControlPayloadError("timestamp is not a valid date-time") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise MissionControlPayloadError("timestamp must include a timezone")
    return value


def _validate_identifier(
    value: Any,
    *,
    field_name: str,
    pattern: re.Pattern[str],
    secret_policy: SecretPolicy,
) -> str:
    max_chars = pattern.pattern.count("255") and 256 or 128
    value = _sanitize_text(
        value,
        field_name=field_name,
        max_chars=max_chars,
        allow_empty=False,
        secret_policy=secret_policy,
    )
    if "://" in value or not pattern.fullmatch(value):
        raise MissionControlPayloadError(
            f"{field_name} must be a bounded opaque identifier"
        )
    return value


def _optional_identifier(
    fields: Mapping[str, Any],
    field_name: str,
    pattern: re.Pattern[str],
    secret_policy: SecretPolicy,
) -> str | None:
    value = fields.get(field_name)
    if value is None:
        return None
    return _validate_identifier(
        value,
        field_name=field_name,
        pattern=pattern,
        secret_policy=secret_policy,
    )


def _strict_bool(fields: Mapping[str, Any], field_name: str) -> bool:
    value = fields.get(field_name, False)
    if type(value) is not bool:
        raise MissionControlPayloadError(f"{field_name} must be a boolean")
    return value


@dataclass(frozen=True)
class MissionControlReportRecord:
    """Minimal trace record; transcripts and authority grants are excluded."""

    event_id: str
    timestamp: str
    agent: str
    mission_task_id: str
    event_type: str
    status: str
    summary: str
    verdict: str
    risk_flags: tuple[str, ...]
    evidence_ref: str | None
    protocol_state_ref: str | None
    route_receipt_id: str | None
    effective_model: str | None
    effective_provider: str | None
    mutation_performed: bool
    external_action_performed: bool
    failure_state: str
    error_code: str | None
    authorization_status: str
    schema_version: int = field(default=REPORT_SCHEMA_VERSION, init=False)
    authority_effect: str = field(default="none", init=False)
    mutation_authority: bool = field(default=False, init=False)

    def to_payload(self) -> dict[str, Any]:
        """Return only the stable allowlist; never include routing or secrets."""

        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "agent": self.agent,
            "mission_task_id": self.mission_task_id,
            "event_type": self.event_type,
            "status": self.status,
            "summary": self.summary,
            "verdict": self.verdict,
            "risk_flags": list(self.risk_flags),
            "evidence_ref": self.evidence_ref,
            "protocol_state_ref": self.protocol_state_ref,
            "route_receipt_id": self.route_receipt_id,
            "effective_model": self.effective_model,
            "effective_provider": self.effective_provider,
            "mutation_performed": self.mutation_performed,
            "external_action_performed": self.external_action_performed,
            "failure_state": self.failure_state,
            "error_code": self.error_code,
            "authorization_status": self.authorization_status,
            "authority_effect": self.authority_effect,
            "mutation_authority": self.mutation_authority,
        }


def build_report_record(
    decision: MissionControlDecision,
    fields: Mapping[str, Any],
    *,
    secret_policy: SecretPolicy = SecretPolicy.REJECT,
) -> MissionControlReportRecord:
    """Build a bounded record only after both independent gates succeeded."""

    if not isinstance(decision, MissionControlDecision):
        raise TypeError("decision must be a MissionControlDecision")
    if not decision.report_eligible:
        failure = decision.failure.value if decision.failure else "report_not_eligible"
        raise MissionControlUnavailableError(failure)
    if not isinstance(fields, Mapping):
        raise MissionControlPayloadError("report fields must be a mapping")
    if not isinstance(secret_policy, SecretPolicy):
        raise TypeError("secret_policy must be a SecretPolicy")

    supplied = set(fields)
    if any(_SECRET_FIELD_NAME_RE.search(str(name)) for name in supplied):
        raise MissionControlPayloadError("secret-bearing report fields are forbidden")
    if not supplied.issubset(_REPORT_INPUT_FIELDS):
        raise MissionControlPayloadError("report contains fields outside the allowlist")
    missing = _REQUIRED_REPORT_FIELDS - supplied
    if missing:
        raise MissionControlPayloadError("report is missing required allowlisted fields")

    event_id = _validate_identifier(
        fields["event_id"],
        field_name="event_id",
        pattern=_EVENT_ID_RE,
        secret_policy=secret_policy,
    )
    timestamp = _validate_timestamp(fields["timestamp"])
    agent = _validate_identifier(
        fields["agent"],
        field_name="agent",
        pattern=_AGENT_ID_RE,
        secret_policy=secret_policy,
    )
    mission_task_id = _validate_identifier(
        fields["mission_task_id"],
        field_name="mission_task_id",
        pattern=_MISSION_TASK_ID_RE,
        secret_policy=secret_policy,
    )

    event_type = _sanitize_text(
        fields["event_type"],
        field_name="event_type",
        max_chars=64,
        allow_empty=False,
        secret_policy=secret_policy,
    ).lower()
    if event_type not in _REPORT_EVENT_TYPES:
        raise MissionControlPayloadError("event_type is not allowlisted")

    status = _sanitize_text(
        fields["status"],
        field_name="status",
        max_chars=32,
        allow_empty=False,
        secret_policy=secret_policy,
    ).upper()
    if status not in _REPORT_STATUSES:
        raise MissionControlPayloadError("status is not allowlisted")

    summary = _sanitize_text(
        fields["summary"],
        field_name="summary",
        max_chars=_MAX_SUMMARY_CHARS,
        allow_empty=False,
        secret_policy=secret_policy,
    )
    verdict = _sanitize_text(
        fields.get("verdict", ""),
        field_name="verdict",
        max_chars=_MAX_VERDICT_CHARS,
        allow_empty=True,
        secret_policy=secret_policy,
    )

    raw_risk_flags = fields.get("risk_flags", ())
    if isinstance(raw_risk_flags, (str, bytes)) or not isinstance(
        raw_risk_flags, (list, tuple)
    ):
        raise MissionControlPayloadError("risk_flags must be a list or tuple")
    if len(raw_risk_flags) > _MAX_RISK_FLAGS:
        raise MissionControlPayloadError("risk_flags exceeds its item limit")
    risk_flags = tuple(
        _sanitize_text(
            item,
            field_name="risk_flags",
            max_chars=_MAX_RISK_FLAG_CHARS,
            allow_empty=False,
            secret_policy=secret_policy,
        )
        for item in raw_risk_flags
    )

    evidence_ref = _optional_identifier(
        fields, "evidence_ref", _REFERENCE_RE, secret_policy
    )
    protocol_state_ref = _optional_identifier(
        fields, "protocol_state_ref", _REFERENCE_RE, secret_policy
    )
    route_receipt_id = _optional_identifier(
        fields, "route_receipt_id", _REFERENCE_RE, secret_policy
    )
    effective_model = _optional_identifier(
        fields, "effective_model", _MODEL_METADATA_RE, secret_policy
    )
    effective_provider = _optional_identifier(
        fields, "effective_provider", _MODEL_METADATA_RE, secret_policy
    )
    mutation_performed = _strict_bool(fields, "mutation_performed")
    external_action_performed = _strict_bool(fields, "external_action_performed")

    failure_state = _sanitize_text(
        fields.get("failure_state", "NONE"),
        field_name="failure_state",
        max_chars=16,
        allow_empty=False,
        secret_policy=secret_policy,
    ).upper()
    if failure_state not in _FAILURE_STATES:
        raise MissionControlPayloadError("failure_state is not allowlisted")
    error_code = _optional_identifier(
        fields, "error_code", _ERROR_CODE_RE, secret_policy
    )
    if (failure_state == "NONE") != (error_code is None):
        raise MissionControlPayloadError(
            "failure_state and error_code must be supplied together"
        )

    return MissionControlReportRecord(
        event_id=event_id,
        timestamp=timestamp,
        agent=agent,
        mission_task_id=mission_task_id,
        event_type=event_type,
        status=status,
        summary=summary,
        verdict=verdict,
        risk_flags=risk_flags,
        evidence_ref=evidence_ref,
        protocol_state_ref=protocol_state_ref,
        route_receipt_id=route_receipt_id,
        effective_model=effective_model,
        effective_provider=effective_provider,
        mutation_performed=mutation_performed,
        external_action_performed=external_action_performed,
        failure_state=failure_state,
        error_code=error_code,
        authorization_status=decision.authority_source.value,
    )


def report_can_authorize_mutation(_record: object) -> bool:
    """Mission Control evidence never grants deployment or mutation authority."""

    return False


__all__ = [
    "MissionControlAuthority",
    "MissionControlContractError",
    "MissionControlDecision",
    "MissionControlDestination",
    "MissionControlDestinationStatus",
    "MissionControlEventClass",
    "MissionControlFailure",
    "MissionControlPayloadError",
    "MissionControlReportRecord",
    "MissionControlUnavailableError",
    "REPORT_SCHEMA_VERSION",
    "SecretPolicy",
    "UNCONFIGURED_DESTINATION",
    "build_report_record",
    "classify_event",
    "evaluate_reporting_request",
    "report_can_authorize_mutation",
]
