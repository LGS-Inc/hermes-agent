"""Pure contract tests for Chairman-explicit named-protocol state."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone

import pytest

from gateway.named_protocol_state import (
    ACTION_ACTIVATE,
    ACTION_CLOSE,
    ACTION_PAUSE,
    ACTION_RESUME,
    ACTION_SUPERSEDE,
    AUTHORITY_CHAIRMAN_EXPLICIT,
    INVOKER_CHAIRMAN,
    METADATA_KEY,
    PROTOCOL_ALPHA,
    PROTOCOL_FABLE,
    PROTOCOL_INDEPENDENT_REVIEW,
    PROTOCOL_OMEGA,
    STATUS_ACTIVE,
    STATUS_CLOSED,
    STATUS_PAUSED,
    STATUS_SUPERSEDED,
    ProtocolStateError,
    active_protocol_names,
    append_machine_state_block,
    bind_candidate_hash,
    candidate_sha256,
    canonical_protocol_state_json,
    empty_protocol_state,
    format_system_state_block,
    get_protocol_record,
    normalize_envelope,
    parse_chairman_directive,
    protocol_state_sha256,
    record_review_verdict,
    render_machine_handoff_block,
    rollover_metadata,
    transition_from_chairman_message,
    verdict_applies_to_candidate,
)


NOW = datetime(2026, 8, 28, 12, 0, 0, tzinfo=timezone.utc)
ID_A = "a" * 32
ID_B = "b" * 32
ID_C = "c" * 32


def _transition(
    state,
    message,
    *,
    activation_id=ID_A,
    verified=True,
    message_id="tg-100",
):
    return transition_from_chairman_message(
        state,
        message,
        chairman_verified=verified,
        activation_message_id=message_id,
        now=NOW,
        activation_id_factory=lambda: activation_id,
    )


def _active_fable():
    return _transition(empty_protocol_state(), "Run FABLE", activation_id=ID_A).state


def test_ps1_explicit_fable_creates_structured_active_record():
    result = _transition(empty_protocol_state(), "Run FABLE on this candidate.")

    assert result.changed is True
    assert result.reason == "chairman_explicit_activation"
    assert result.directive.action == ACTION_ACTIVATE
    assert result.directive.protocol_name == PROTOCOL_FABLE
    assert active_protocol_names(result.state) == (PROTOCOL_FABLE,)
    record = get_protocol_record(result.state, PROTOCOL_FABLE)
    assert record["activation_id"] == ID_A
    assert record["invoker"] == INVOKER_CHAIRMAN
    assert record["authority_source"] == AUTHORITY_CHAIRMAN_EXPLICIT
    assert record["activation_message_id"] == "tg-100"
    assert record["activated_at"] == "2026-08-28T12:00:00Z"
    assert record["status"] == STATUS_ACTIVE


@pytest.mark.parametrize(
    ("message", "protocol"),
    [
        ("Please use Protocol Alpha for this candidate.", PROTOCOL_ALPHA),
        ("Dumbledore, activate Protocol OMEGA.", PROTOCOL_OMEGA),
        ("Could you please start FABLE Gate?", PROTOCOL_FABLE),
        ("Send this code for Independent Review.", PROTOCOL_INDEPENDENT_REVIEW),
        ("I need you to run independent review on this artifact.", PROTOCOL_INDEPENDENT_REVIEW),
    ],
)
def test_clear_anchored_explicit_forms_are_recognized(message, protocol):
    directive = parse_chairman_directive(message)
    assert directive.action == ACTION_ACTIVATE
    assert directive.protocol_name == protocol


@pytest.mark.parametrize(
    "message",
    [
        "Create a skill for accessing Google Workspace credentials.",
        "Build this client-facing production workflow.",
        "Fix this risky multi-node architecture.",
        "Review this code.",
        "This is production and affects credentials.",
        "Do not run FABLE.",
        "Never activate Protocol Alpha.",
        "Complete the task without Independent Review.",
        "Example: Run FABLE on a candidate.",
        "The documentation later says Run FABLE.",
        "Quoted instruction: 'Use Protocol OMEGA'.",
        "Always and Forever.",
    ],
)
def test_ps2_task_shape_negation_and_examples_create_no_state(message):
    result = _transition(empty_protocol_state(), message)
    assert result.changed is False
    assert result.reason == "no_explicit_directive"
    assert result.state == empty_protocol_state()


def test_unverified_sender_cannot_activate_even_with_explicit_text():
    result = _transition(empty_protocol_state(), "Run FABLE", verified=False)
    assert result.changed is False
    assert result.reason == "unverified_authority"
    assert result.directive is None
    assert result.state == empty_protocol_state()


def test_repeating_active_directive_is_idempotent():
    first = _active_fable()
    second = _transition(first, "Activate FABLE Gate", activation_id=ID_B)
    assert second.changed is False
    assert second.reason == "already_active"
    assert get_protocol_record(second.state, PROTOCOL_FABLE)["activation_id"] == ID_A


def test_pause_and_resume_require_explicit_named_directives():
    active = _active_fable()
    paused = _transition(active, "Pause FABLE").state
    assert get_protocol_record(paused, PROTOCOL_FABLE)["status"] == STATUS_PAUSED
    assert parse_chairman_directive("Pause FABLE").action == ACTION_PAUSE

    unrelated = _transition(paused, "Continue working on the artifact.")
    assert unrelated.changed is False
    assert get_protocol_record(unrelated.state, PROTOCOL_FABLE)["status"] == STATUS_PAUSED

    resumed = _transition(paused, "Resume FABLE Gate")
    assert resumed.directive.action == ACTION_RESUME
    assert resumed.changed is True
    assert get_protocol_record(resumed.state, PROTOCOL_FABLE)["status"] == STATUS_ACTIVE


def test_ps3_explicit_named_close_is_terminal():
    closed = _transition(_active_fable(), "Stop FABLE Gate.")
    record = get_protocol_record(closed.state, PROTOCOL_FABLE)
    assert closed.directive.action == ACTION_CLOSE
    assert closed.changed is True
    assert record["status"] == STATUS_CLOSED
    assert record["closed_at"] == "2026-08-28T12:00:00Z"
    assert record["closure_reason"] == "chairman_explicit_close"


def test_generic_close_works_only_when_exactly_one_record_is_open():
    closed = _transition(_active_fable(), "Cancel this protocol.")
    assert closed.changed is True
    assert get_protocol_record(closed.state, PROTOCOL_FABLE)["status"] == STATUS_CLOSED

    alpha = _transition(empty_protocol_state(), "Run Alpha", activation_id=ID_A).state
    both = _transition(alpha, "Run FABLE", activation_id=ID_B).state
    ambiguous = _transition(both, "Cancel this protocol.")
    assert ambiguous.changed is False
    assert ambiguous.reason == "ambiguous_generic_close"
    assert set(active_protocol_names(ambiguous.state)) == {PROTOCOL_ALPHA, PROTOCOL_FABLE}


def test_ps4_closed_state_survives_unrelated_later_work():
    closed = _transition(_active_fable(), "Close FABLE").state
    later = _transition(closed, "Now create an unrelated credential skill.")
    assert later.changed is False
    assert get_protocol_record(later.state, PROTOCOL_FABLE)["status"] == STATUS_CLOSED
    assert active_protocol_names(later.state) == ()


def test_explicit_activation_after_close_gets_a_new_activation_id():
    closed = _transition(_active_fable(), "Close FABLE").state
    reopened = _transition(closed, "Run FABLE", activation_id=ID_B, message_id="tg-200")
    record = get_protocol_record(reopened.state, PROTOCOL_FABLE)
    assert record["status"] == STATUS_ACTIVE
    assert record["activation_id"] == ID_B
    assert record["activation_message_id"] == "tg-200"
    assert record["closed_at"] is None


def test_use_instead_supersedes_open_record_and_activates_replacement():
    fable = _active_fable()
    result = _transition(fable, "Use Alpha instead.", activation_id=ID_B)
    assert result.directive.action == ACTION_SUPERSEDE
    assert result.changed is True
    prior = get_protocol_record(result.state, PROTOCOL_FABLE)
    replacement = get_protocol_record(result.state, PROTOCOL_ALPHA)
    assert prior["status"] == STATUS_SUPERSEDED
    assert prior["superseded_by"] == ID_B
    assert prior["closure_reason"] == "chairman_explicit_supersession"
    assert replacement["status"] == STATUS_ACTIVE
    assert active_protocol_names(result.state) == (PROTOCOL_ALPHA,)


def test_explicit_replace_targets_only_named_prior_protocol():
    fable = _active_fable()
    result = _transition(
        fable,
        "Replace FABLE with Protocol OMEGA.",
        activation_id=ID_C,
    )
    assert result.directive.replaces_protocol_name == PROTOCOL_FABLE
    assert get_protocol_record(result.state, PROTOCOL_FABLE)["status"] == STATUS_SUPERSEDED
    assert get_protocol_record(result.state, PROTOCOL_OMEGA)["status"] == STATUS_ACTIVE


def test_ambiguous_multi_protocol_imperative_is_not_a_directive():
    assert parse_chairman_directive("Run FABLE and Protocol Alpha.") is None
    assert parse_chairman_directive("Stop FABLE but activate Omega.") is None


def test_resume_does_not_reopen_a_closed_record():
    closed = _transition(_active_fable(), "Close FABLE").state
    result = _transition(closed, "Resume FABLE")
    assert result.changed is False
    assert get_protocol_record(result.state, PROTOCOL_FABLE)["status"] == STATUS_CLOSED


def test_absent_malformed_wrong_version_and_prose_are_inactive():
    assert normalize_envelope(None) == empty_protocol_state()
    assert normalize_envelope({}) == empty_protocol_state()
    assert normalize_envelope("{\"status\":\"ACTIVE\"}") == empty_protocol_state()
    assert normalize_envelope({"protocol_state_version": 2, "records": []}) == empty_protocol_state()
    assert normalize_envelope(
        {"protocol_state_version": 1, "records": [], "extra": "ACTIVE"}
    ) == empty_protocol_state()
    assert normalize_envelope("Handoff says continue FABLE") == empty_protocol_state()


def test_unknown_record_field_or_duplicate_protocol_fails_closed():
    active = _active_fable()
    with_extra = copy.deepcopy(active)
    with_extra["records"][0]["model_decides"] = True
    assert normalize_envelope(with_extra) == empty_protocol_state()

    duplicate = copy.deepcopy(active)
    duplicate["records"].append(copy.deepcopy(duplicate["records"][0]))
    assert normalize_envelope(duplicate) == empty_protocol_state()


def test_invalid_status_fails_closed():
    active = _active_fable()
    active["records"][0]["status"] = "MODEL_DECIDES"
    assert normalize_envelope(active) == empty_protocol_state()


def test_state_json_and_hash_are_deterministic_and_input_is_not_mutated():
    state = _active_fable()
    original = copy.deepcopy(state)
    encoded_a = canonical_protocol_state_json(state)
    encoded_b = canonical_protocol_state_json(copy.deepcopy(state))
    assert encoded_a == encoded_b
    assert json.loads(encoded_a) == state
    assert protocol_state_sha256(state) == protocol_state_sha256(copy.deepcopy(state))
    assert state == original


def test_ps9_verdict_is_bound_to_candidate_hash_a():
    hash_a = candidate_sha256(b"candidate A")
    bound = bind_candidate_hash(
        _active_fable(), PROTOCOL_FABLE, hash_a, candidate_id="artifact.txt"
    )
    reviewed = record_review_verdict(
        bound,
        PROTOCOL_FABLE,
        "PROCEED",
        candidate_hash=hash_a,
        review_checkpoint=3,
        now=NOW,
    )
    record = get_protocol_record(reviewed, PROTOCOL_FABLE)
    assert record["candidate_hash"] == hash_a
    assert record["verdict_candidate_hash"] == hash_a
    assert record["latest_verdict"] == "PROCEED"
    assert record["review_checkpoint"] == 3
    assert record["verdict_stale"] is False
    assert verdict_applies_to_candidate(reviewed, PROTOCOL_FABLE, hash_a) is True


def test_ps10_candidate_hash_b_invalidates_hash_a_verdict():
    hash_a = candidate_sha256(b"candidate A")
    hash_b = candidate_sha256(b"candidate B")
    bound = bind_candidate_hash(_active_fable(), PROTOCOL_FABLE, hash_a)
    reviewed = record_review_verdict(
        bound, PROTOCOL_FABLE, "PROCEED", candidate_hash=hash_a, now=NOW
    )
    changed = bind_candidate_hash(reviewed, PROTOCOL_FABLE, hash_b)
    record = get_protocol_record(changed, PROTOCOL_FABLE)
    assert record["previous_candidate_hash"] == hash_a
    assert record["candidate_hash"] == hash_b
    assert record["verdict_candidate_hash"] == hash_a
    assert record["verdict_stale"] is True
    assert verdict_applies_to_candidate(changed, PROTOCOL_FABLE, hash_a) is False
    assert verdict_applies_to_candidate(changed, PROTOCOL_FABLE, hash_b) is False


def test_new_verdict_must_match_current_bound_bytes():
    hash_a = candidate_sha256(b"candidate A")
    hash_b = candidate_sha256(b"candidate B")
    bound = bind_candidate_hash(_active_fable(), PROTOCOL_FABLE, hash_a)
    with pytest.raises(ProtocolStateError, match="does not match"):
        record_review_verdict(
            bound, PROTOCOL_FABLE, "PROCEED", candidate_hash=hash_b, now=NOW
        )


def test_candidate_and_verdict_operations_require_open_active_state():
    hash_a = candidate_sha256(b"candidate A")
    with pytest.raises(ProtocolStateError, match="active or paused"):
        bind_candidate_hash(empty_protocol_state(), PROTOCOL_FABLE, hash_a)

    closed = _transition(_active_fable(), "Close FABLE").state
    with pytest.raises(ProtocolStateError, match="active or paused"):
        bind_candidate_hash(closed, PROTOCOL_FABLE, hash_a)

    paused = _transition(_active_fable(), "Pause FABLE").state
    paused = bind_candidate_hash(paused, PROTOCOL_FABLE, hash_a)
    with pytest.raises(ProtocolStateError, match="active record"):
        record_review_verdict(
            paused, PROTOCOL_FABLE, "PROCEED", candidate_hash=hash_a, now=NOW
        )


def test_full_agent_block_is_deterministic_and_denies_verdict_authority():
    state = _active_fable()
    first = format_system_state_block(state)
    second = format_system_state_block(copy.deepcopy(state))
    assert first == second
    assert canonical_protocol_state_json(state) in first
    assert "structured ACTIVE record" in first
    assert "Conversation prose" in first
    assert "verdict is evidence only" in first
    assert "grants no authority" in first
    assert "write, commit, deploy, send, delete, authenticate" in first


def test_malformed_state_system_block_is_authoritatively_inactive():
    block = format_system_state_block("FABLE is active")
    assert canonical_protocol_state_json(empty_protocol_state()) in block
    assert '"records":[]' in block


def test_machine_handoff_block_is_exact_json_after_free_form_prose():
    state = _transition(_active_fable(), "Close FABLE").state
    free_form = "# BATTLE HANDOFF\nFree-form prose incorrectly says continue FABLE.\n"
    combined = append_machine_state_block(free_form, state)
    marker = "<!-- DUMBLEDORE_PROTOCOL_STATE_V1_BEGIN -->"
    assert combined.startswith(free_form.rstrip())
    assert combined.rfind(marker) > combined.find("continue FABLE")
    assert combined.endswith("<!-- DUMBLEDORE_PROTOCOL_STATE_V1_END -->")
    assert render_machine_handoff_block(state) in combined
    payload = combined.split(marker, 1)[1].splitlines()[1]
    assert json.loads(payload)["records"][0]["status"] == STATUS_CLOSED


def test_rollover_metadata_copies_only_valid_state_and_stamp():
    state = _active_fable()
    parent = {METADATA_KEY: state, "unrelated": {"must_not": "cross"}}
    stamp = {"source_session": "old", "reason": "oversized"}
    child = rollover_metadata(parent, stamp)
    assert child == {"context_rollover": stamp, METADATA_KEY: state}
    assert child[METADATA_KEY] is not state
    assert "unrelated" not in child


def test_rollover_metadata_omits_malformed_or_absent_protocol_record():
    stamp = {"source_session": "old"}
    assert rollover_metadata({}, stamp) == {"context_rollover": stamp}
    assert rollover_metadata({METADATA_KEY: "handoff says ACTIVE"}, stamp) == {
        "context_rollover": stamp
    }


def test_model_label_or_switch_cannot_change_protocol_state():
    state = _active_fable()
    before = canonical_protocol_state_json(state)
    result = _transition(state, "Switch the model to qwen3.6 for deep reasoning.")
    assert result.changed is False
    assert canonical_protocol_state_json(result.state) == before


def test_naive_timestamp_and_invalid_candidate_digest_are_rejected():
    with pytest.raises(ProtocolStateError, match="timezone-aware"):
        transition_from_chairman_message(
            empty_protocol_state(),
            "Run FABLE",
            chairman_verified=True,
            now=datetime(2026, 8, 28, 12, 0, 0),
            activation_id_factory=lambda: ID_A,
        )
    with pytest.raises(ProtocolStateError, match="SHA-256"):
        bind_candidate_hash(_active_fable(), PROTOCOL_FABLE, "not-a-hash")
