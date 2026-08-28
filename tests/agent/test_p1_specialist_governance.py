"""Pure P1 specialist-governance contract tests (no model or gateway I/O)."""

import pytest

from agent import dumbledore_capability_router as cap


def test_specialist_governance_block_has_complete_authority_boundary():
    block = cap.SPECIALIST_GOVERNANCE_BLOCK
    lowered = block.lower()

    assert "advisory and task-scoped" in lowered
    assert "model selection grants no authority" in lowered
    assert "do not expand task scope" in lowered
    for named_workflow in (
        "Protocol Alpha",
        "Protocol OMEGA",
        "FABLE Gate",
        "Independent Review",
    ):
        assert named_workflow in block
    for action in ("write", "deploy", "send", "delete", "authenticate"):
        assert action in lowered
    assert "evidence and proposed output only" in lowered
    assert "parent dumbledore" in lowered
    assert "runtime authorization gates remain authoritative" in lowered


@pytest.mark.parametrize(
    ("model", "route"),
    [
        (cap.CODE_FAST_MODEL, cap.CODE_FAST),
        (cap.CODE_HEAVY_MODEL, cap.CODE_HEAVY),
        (cap.DEEP_LOCAL_MODEL, cap.DEEP_LOCAL),
    ],
)
def test_direct_coding_and_deep_prompts_include_one_common_block(model, route):
    prompt = cap.specialist_system_prompt(model, route)

    assert model in prompt
    assert prompt.endswith(cap.SPECIALIST_GOVERNANCE_BLOCK)
    assert prompt.count(cap.SPECIALIST_GOVERNANCE_BLOCK) == 1


def test_explicit_local_specialists_receive_same_governance_block():
    for model in cap.LOCAL_SPECIALIST_MODELS:
        prompt = cap.specialist_system_prompt(model, cap.EXPLICIT_PIN)
        assert prompt.endswith(cap.SPECIALIST_GOVERNANCE_BLOCK)
        assert prompt.count(cap.SPECIALIST_GOVERNANCE_BLOCK) == 1


def test_result_boundary_is_advisory_and_idempotent():
    raw = "Suggested patch and verification evidence."
    marked = cap.mark_specialist_result(raw)

    assert marked.startswith(cap.SPECIALIST_RESULT_BOUNDARY + "\n\n")
    assert marked.endswith(raw)
    assert marked.count(cap.SPECIALIST_RESULT_BOUNDARY) == 1
    assert cap.mark_specialist_result(marked) == marked


def test_result_boundary_normalizes_repeated_leading_markers():
    marker = cap.SPECIALIST_RESULT_BOUNDARY
    repeated = marker + "\n\n" + marker + "\n\nproposal"

    normalized = cap.mark_specialist_result(repeated)
    assert normalized == marker + "\n\nproposal"
    assert normalized.count(marker) == 1


def test_empty_result_still_carries_the_authority_boundary():
    assert cap.mark_specialist_result(None) == cap.SPECIALIST_RESULT_BOUNDARY
    assert cap.mark_specialist_result("") == cap.SPECIALIST_RESULT_BOUNDARY
