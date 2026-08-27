"""Post-commit skill mutation lifecycle and approval-replay contracts."""

import hashlib
import json

import pytest

from tools import skill_manager_tool


SKILL_V1 = """\
---
name: lifecycle-skill
description: A lifecycle test skill.
---

# Lifecycle Skill

Step 1: Do the thing.
"""


SKILL_V2 = """\
---
name: lifecycle-skill
description: An updated lifecycle test skill.
---

# Lifecycle Skill

Step 1: Do the updated thing.
"""


@pytest.fixture
def skill_env(tmp_path, monkeypatch):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    monkeypatch.setattr(skill_manager_tool, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr(
        "agent.skill_utils.get_all_skills_dirs",
        lambda: [skills_dir],
    )
    monkeypatch.setattr(
        skill_manager_tool,
        "_maybe_debounced_sync_push",
        lambda name: None,
    )
    return skills_dir


@pytest.fixture
def lifecycle_events(monkeypatch):
    events = []
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: True)
    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook",
        lambda hook_name, **payload: events.append((hook_name, payload)),
    )
    return events


def _manage(**kwargs):
    return json.loads(skill_manager_tool.skill_manage(**kwargs))


def test_all_successful_mutations_emit_one_exact_post_commit_event(
    skill_env, lifecycle_events
):
    common = {"name": "lifecycle-skill", "task_id": "task-7", "session_id": "session-9"}

    results = [
        _manage(action="create", content=SKILL_V1, execution_id="exec-create", **common),
        _manage(action="edit", content=SKILL_V2, execution_id="exec-edit", **common),
        _manage(
            action="patch",
            old_string="Do the updated thing.",
            new_string="Do the updated thing safely.",
            execution_id="exec-patch",
            **common,
        ),
        _manage(
            action="write_file",
            file_path="references/evidence.md",
            file_content="# Evidence\n",
            execution_id="exec-write-file",
            **common,
        ),
        _manage(
            action="remove_file",
            file_path="references/evidence.md",
            execution_id="exec-remove-file",
            **common,
        ),
        _manage(
            action="delete",
            absorbed_into="",
            execution_id="exec-delete",
            **common,
        ),
    ]

    assert all(result["success"] is True for result in results)
    mutation_events = [
        payload
        for hook_name, payload in lifecycle_events
        if hook_name == "on_skill_lifecycle" and payload.get("mutation_event") is True
    ]
    assert [event["action"] for event in mutation_events] == [
        "create",
        "edit",
        "patch",
        "write_file",
        "remove_file",
        "delete",
    ]
    assert [event["changed_file"] for event in mutation_events] == [
        "SKILL.md",
        "SKILL.md",
        "SKILL.md",
        "references/evidence.md",
        "references/evidence.md",
        ".",
    ]
    assert all(event["skill_name"] == "lifecycle-skill" for event in mutation_events)
    assert all(event["task_id"] == "task-7" for event in mutation_events)
    assert all(event["session_id"] == "session-9" for event in mutation_events)
    assert [event["execution_id"] for event in mutation_events] == [
        "exec-create",
        "exec-edit",
        "exec-patch",
        "exec-write-file",
        "exec-remove-file",
        "exec-delete",
    ]
    assert all(event["provenance"] != "unknown" for event in mutation_events)
    assert all(event["success"] is True for event in mutation_events)
    assert all(event["result"]["success"] is True for event in mutation_events)
    assert all(event["result"]["message"] for event in mutation_events)
    expected_hashes = [
        hashlib.sha256(SKILL_V1.encode()).hexdigest(),
        hashlib.sha256(SKILL_V2.encode()).hexdigest(),
        hashlib.sha256(
            SKILL_V2.replace(
                "Do the updated thing.",
                "Do the updated thing safely.",
            ).encode()
        ).hexdigest(),
        hashlib.sha256(b"# Evidence\n").hexdigest(),
        None,
        None,
    ]
    assert [event["resulting_sha256"] for event in mutation_events] == expected_hashes


def test_approved_replay_preserves_correlation_in_event_and_ledger(
    skill_env, lifecycle_events, monkeypatch
):
    from tools import skill_ledger, write_approval

    monkeypatch.setattr(
        write_approval,
        "evaluate_gate",
        lambda subsystem: write_approval.GateDecision(
            stage=True,
            message="staged for test approval",
        ),
    )

    staged = _manage(
        action="create",
        name="lifecycle-skill",
        content=SKILL_V1,
        task_id="task-staged",
        session_id="session-staged",
        execution_id="tool-call-staged",
    )

    assert staged["success"] is True
    assert staged["staged"] is True
    assert not (skill_env / "lifecycle-skill" / "SKILL.md").exists()
    assert not any(
        payload.get("mutation_event") is True
        for _, payload in lifecycle_events
    )

    pending = write_approval.get_pending(write_approval.SKILLS, staged["pending_id"])
    assert pending is not None
    assert pending["payload"]["task_id"] == "task-staged"
    assert pending["payload"]["session_id"] == "session-staged"
    assert pending["payload"]["execution_id"] == "tool-call-staged"

    applied = json.loads(skill_manager_tool.apply_skill_pending(pending["payload"]))
    assert applied["success"] is True

    mutation_events = [
        payload
        for hook_name, payload in lifecycle_events
        if hook_name == "on_skill_lifecycle" and payload.get("mutation_event") is True
    ]
    assert len(mutation_events) == 1
    assert mutation_events[0]["action"] == "create"
    assert mutation_events[0]["changed_file"] == "SKILL.md"
    assert mutation_events[0]["task_id"] == "task-staged"
    assert mutation_events[0]["session_id"] == "session-staged"
    assert mutation_events[0]["execution_id"] == "tool-call-staged"
    assert mutation_events[0]["success"] is True
    assert mutation_events[0]["result"]["success"] is True
    assert mutation_events[0]["resulting_sha256"] == hashlib.sha256(
        SKILL_V1.encode()
    ).hexdigest()

    entries = [
        entry
        for entry in skill_ledger.list_entries(skill="lifecycle-skill")
        if entry["action"] == "create"
    ]
    assert len(entries) == 1
    assert entries[0]["evidence"]["task_id"] == "task-staged"
    assert entries[0]["evidence"]["session_id"] == "session-staged"
    assert entries[0]["evidence"]["execution_id"] == "tool-call-staged"


def test_registry_handler_uses_original_tool_call_id_as_execution_id(
    skill_env, lifecycle_events
):
    from tools.registry import registry

    entry = registry.get_entry("skill_manage")
    assert entry is not None
    result = json.loads(entry.handler(
        {
            "action": "create",
            "name": "lifecycle-skill",
            "content": SKILL_V1,
        },
        task_id="task-registry",
        session_id="session-registry",
        tool_call_id="call-registry-123",
    ))

    assert result["success"] is True
    mutation_events = [
        payload
        for _, payload in lifecycle_events
        if payload.get("mutation_event") is True
    ]
    assert len(mutation_events) == 1
    assert mutation_events[0]["execution_id"] == "call-registry-123"


@pytest.mark.parametrize(
    "file_path",
    [
        "SKILL.md",
        "./SKILL.md",
        "lifecycle-skill/SKILL.md",
        "lifecycle-skill\\SKILL.md",
    ],
)
def test_skill_md_file_path_spellings_emit_one_canonical_changed_file(
    file_path,
    skill_env,
    lifecycle_events,
):
    created = _manage(
        action="create",
        name="lifecycle-skill",
        content=SKILL_V1,
        execution_id="create-canonical",
    )
    assert created["success"] is True
    lifecycle_events.clear()

    written = _manage(
        action="write_file",
        name="lifecycle-skill",
        file_path=file_path,
        file_content=SKILL_V2,
        execution_id="write-canonical",
    )
    assert written["success"] is True
    assert (skill_env / "lifecycle-skill" / "SKILL.md").read_text() == SKILL_V2
    assert not (
        skill_env / "lifecycle-skill" / "lifecycle-skill" / "SKILL.md"
    ).exists()

    removed = _manage(
        action="remove_file",
        name="lifecycle-skill",
        file_path=file_path,
        execution_id="remove-canonical",
    )
    assert removed["success"] is True
    assert not (skill_env / "lifecycle-skill" / "SKILL.md").exists()

    mutation_events = [
        payload
        for _, payload in lifecycle_events
        if payload.get("mutation_event") is True
    ]
    assert [event["action"] for event in mutation_events] == [
        "write_file",
        "remove_file",
    ]
    assert [event["changed_file"] for event in mutation_events] == [
        "SKILL.md",
        "SKILL.md",
    ]


def test_mismatched_skill_prefix_is_rejected_before_mutation_event(
    skill_env,
    lifecycle_events,
):
    assert _manage(
        action="create",
        name="lifecycle-skill",
        content=SKILL_V1,
        execution_id="create-prefix",
    )["success"] is True
    lifecycle_events.clear()

    result = _manage(
        action="write_file",
        name="lifecycle-skill",
        file_path="different-skill/SKILL.md",
        file_content=SKILL_V2,
        execution_id="write-prefix",
    )
    assert result["success"] is False
    assert lifecycle_events == []
