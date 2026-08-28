"""MC1-MC8 governance contract for unavailable Mission Control reporting."""

from __future__ import annotations

import pytest

import gateway.mission_control_reporting as mc


def _ready_explicit_decision() -> mc.MissionControlDecision:
    destination = mc.MissionControlDestination(
        status=mc.MissionControlDestinationStatus.CONFIGURED,
        logical_id="test-mission-control",
        interface_verified=True,
    )
    return mc.evaluate_reporting_request(
        mc.MissionControlEventClass.EXPLICIT_REPORT_REQUEST,
        chairman_verified=True,
        explicit_mission_control_request=True,
        destination=destination,
    )


def _safe_fields(**overrides):
    fields = {
        "event_id": "evt-p2-mc-001",
        "timestamp": "2026-08-28T16:30:00Z",
        "agent": "Dumbledore",
        "mission_task_id": "DUMBLEDORE-P2-MC",
        "event_type": "mission_status",
        "status": "INFORMATIONAL",
        "summary": "The bounded governance check completed.",
        "verdict": "OBSERVED",
        "risk_flags": [],
    }
    fields.update(overrides)
    return fields


def test_mc1_no_destination_is_unavailable_and_cannot_report():
    decision = mc.evaluate_reporting_request(
        mc.MissionControlEventClass.EXPLICIT_REPORT_REQUEST,
        chairman_verified=True,
        explicit_mission_control_request=True,
    )

    assert decision.destination_status is mc.MissionControlDestinationStatus.AVAILABLE_NOT_CONFIGURED
    assert decision.authorized is True
    assert decision.report_eligible is False
    assert decision.failure is mc.MissionControlFailure.DESTINATION_NOT_CONFIGURED
    assert decision.fallback_allowed is False


def test_mc2_ordinary_task_never_becomes_an_external_report():
    decision = mc.evaluate_reporting_request(
        mc.MissionControlEventClass.ORDINARY_TASK,
        chairman_verified=True,
        explicit_mission_control_request=True,
        workflow_chairman_authorized=True,
        workflow_includes_mission_control=True,
        destination=mc.MissionControlDestination(
            status=mc.MissionControlDestinationStatus.ACTIVE,
            logical_id="otherwise-ready",
            interface_verified=True,
        ),
    )

    assert decision.external_report_requested is False
    assert decision.authorized is False
    assert decision.report_eligible is False
    assert decision.reason == "ordinary_task_not_reportable"


def test_mc3_local_health_logging_is_observability_not_mission_control():
    decision = mc.evaluate_reporting_request(
        mc.MissionControlEventClass.LOCAL_OBSERVABILITY,
        chairman_verified=True,
        explicit_mission_control_request=True,
    )

    assert decision.event_class is mc.MissionControlEventClass.LOCAL_OBSERVABILITY
    assert decision.automatic_technical_safeguard is True
    assert decision.external_report_requested is False
    assert decision.report_eligible is False
    assert decision.failure is None


@pytest.mark.parametrize(
    ("event", "kwargs", "authority"),
    [
        (
            mc.MissionControlEventClass.EXPLICIT_REPORT_REQUEST,
            {
                "chairman_verified": True,
                "explicit_mission_control_request": True,
            },
            mc.MissionControlAuthority.CHAIRMAN_EXPLICIT_REQUEST,
        ),
        (
            mc.MissionControlEventClass.AUTHORIZED_WORKFLOW_REPORT,
            {
                "workflow_chairman_authorized": True,
                "workflow_includes_mission_control": True,
            },
            mc.MissionControlAuthority.CHAIRMAN_AUTHORIZED_WORKFLOW,
        ),
    ],
)
def test_mc4_specific_chairman_authority_is_recognized_but_isolated(
    event, kwargs, authority
):
    decision = mc.evaluate_reporting_request(event, **kwargs)

    assert decision.external_report_requested is True
    assert decision.authorized is True
    assert decision.authority_source is authority
    assert decision.report_eligible is False
    assert decision.failure is mc.MissionControlFailure.DESTINATION_NOT_CONFIGURED


def test_mc4_generic_workflow_authority_without_mc_scope_is_insufficient():
    decision = mc.evaluate_reporting_request(
        mc.MissionControlEventClass.AUTHORIZED_WORKFLOW_REPORT,
        workflow_chairman_authorized=True,
        workflow_includes_mission_control=False,
    )

    assert decision.authorized is False
    assert decision.failure is mc.MissionControlFailure.AUTHORITY_NOT_VERIFIED


def test_mc5_unavailable_destination_fails_cleanly_without_fallback_or_send_path():
    decision = mc.evaluate_reporting_request(
        mc.MissionControlEventClass.EXPLICIT_REPORT_REQUEST,
        chairman_verified=True,
        explicit_mission_control_request=True,
    )

    with pytest.raises(
        mc.MissionControlUnavailableError,
        match="^destination_not_configured$",
    ):
        mc.build_report_record(decision, _safe_fields())

    assert decision.fallback_allowed is False
    assert not hasattr(mc, "send_report")
    assert not hasattr(mc, "deliver_report")
    assert not hasattr(mc, "post_report")


def test_mc6_payload_is_allowlisted_and_rejects_or_redacts_secrets():
    decision = _ready_explicit_decision()
    raw_secret = "sk-abcdefghijklmnopqrstuvwxyz123456"

    with pytest.raises(mc.MissionControlPayloadError) as unknown_error:
        mc.build_report_record(
            decision,
            {**_safe_fields(), "api_key": raw_secret},
        )
    assert raw_secret not in str(unknown_error.value)

    with pytest.raises(mc.MissionControlPayloadError) as embedded_error:
        mc.build_report_record(
            decision,
            _safe_fields(summary=f"Credential observed: {raw_secret}"),
        )
    assert raw_secret not in str(embedded_error.value)

    record = mc.build_report_record(
        decision,
        _safe_fields(summary=f"Credential observed: {raw_secret}"),
        secret_policy=mc.SecretPolicy.REDACT,
    )
    payload = record.to_payload()
    assert raw_secret not in payload["summary"]
    assert "redacted" in payload["summary"].lower()
    assert set(payload) == {
        "schema_version",
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
        "authorization_status",
        "authority_effect",
        "mutation_authority",
    }


def test_mc7_report_status_or_verdict_never_authorizes_mutation():
    record = mc.build_report_record(
        _ready_explicit_decision(),
        _safe_fields(
            status="COMPLETE",
            verdict="PROCEED",
            mutation_performed=True,
            external_action_performed=True,
        ),
    )

    assert record.status == "COMPLETE"
    assert record.verdict == "PROCEED"
    assert record.mutation_performed is True
    assert record.external_action_performed is True
    assert record.authority_effect == "none"
    assert record.mutation_authority is False
    assert mc.report_can_authorize_mutation(record) is False

    with pytest.raises(mc.MissionControlPayloadError):
        mc.build_report_record(
            _ready_explicit_decision(),
            {**_safe_fields(), "authorize_deployment": True},
        )


def test_trace_record_carries_bounded_evidence_route_and_failure_metadata():
    record = mc.build_report_record(
        _ready_explicit_decision(),
        _safe_fields(
            event_id="evt-20260828-001",
            timestamp="2026-08-28T16:30:00.123456-04:00",
            agent="Dumbledore",
            mission_task_id="P2/MC-TRACE-001",
            evidence_ref="evidence/p2/mc-trace.json#sha256",
            protocol_state_ref="protocol-state/no-active-protocol",
            route_receipt_id="route-receipt-001",
            effective_model="openai-codex/gpt-5.6-sol",
            effective_provider="openai-codex",
            failure_state="UNAVAILABLE",
            error_code="destination_not_configured",
        ),
    )

    payload = record.to_payload()
    assert payload["event_id"] == "evt-20260828-001"
    assert payload["timestamp"] == "2026-08-28T16:30:00.123456-04:00"
    assert payload["agent"] == "Dumbledore"
    assert payload["mission_task_id"] == "P2/MC-TRACE-001"
    assert payload["evidence_ref"] == "evidence/p2/mc-trace.json#sha256"
    assert payload["protocol_state_ref"] == "protocol-state/no-active-protocol"
    assert payload["route_receipt_id"] == "route-receipt-001"
    assert payload["effective_model"] == "openai-codex/gpt-5.6-sol"
    assert payload["effective_provider"] == "openai-codex"
    assert payload["failure_state"] == "UNAVAILABLE"
    assert payload["error_code"] == "destination_not_configured"


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("event_id", "contains spaces"),
        ("timestamp", "2026-08-28T16:30:00"),
        ("timestamp", 1787950000),
        ("agent", ["Dumbledore"]),
        ("mission_task_id", ""),
        ("evidence_ref", "https://unconfigured.example/report"),
        ("mutation_performed", "false"),
        ("external_action_performed", 0),
        ("failure_state", "UNKNOWN"),
    ],
)
def test_trace_record_rejects_malformed_or_ambiguous_metadata(field_name, bad_value):
    with pytest.raises(mc.MissionControlPayloadError):
        mc.build_report_record(
            _ready_explicit_decision(),
            _safe_fields(**{field_name: bad_value}),
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"failure_state": "FAILED"},
        {"error_code": "failure_without_state"},
        {"full_text": "raw request and response transcript"},
        {"raw_error": "unbounded exception text"},
    ],
)
def test_trace_record_rejects_incomplete_failure_or_full_text_fields(overrides):
    with pytest.raises(mc.MissionControlPayloadError):
        mc.build_report_record(
            _ready_explicit_decision(),
            _safe_fields(**overrides),
        )


@pytest.mark.parametrize(
    "event",
    [
        mc.MissionControlEventClass.LIFECYCLE_NOTIFICATION,
        mc.MissionControlEventClass.TELEGRAM_NOTIFICATION,
    ],
)
def test_mc8_lifecycle_and_telegram_are_separate_automatic_safeguards(event):
    decision = mc.evaluate_reporting_request(
        event,
        chairman_verified=True,
        explicit_mission_control_request=True,
        workflow_chairman_authorized=True,
        workflow_includes_mission_control=True,
        destination=mc.MissionControlDestination(
            status=mc.MissionControlDestinationStatus.ACTIVE,
            logical_id="otherwise-ready",
            interface_verified=True,
        ),
    )

    assert decision.event_class is event
    assert decision.automatic_technical_safeguard is True
    assert decision.external_report_requested is False
    assert decision.authorized is False
    assert decision.report_eligible is False
    assert decision.failure is None
