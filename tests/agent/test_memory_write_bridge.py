"""Behavior tests for the built-in memory → external provider bridge.

The bridge lives behind the MemoryManager interface
(``MemoryManager.notify_memory_tool_write``): the agent loop hands over the raw
built-in memory tool result + args, and the manager decides whether/what to
mirror to external providers. These tests drive that method with a fake
external provider and assert which ``on_memory_write`` calls land.
"""

import json

import pytest

from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider


class _RecordingProvider(MemoryProvider):
    """Minimal external provider that records on_memory_write calls."""

    def __init__(self) -> None:
        self.calls = []

    @property
    def name(self) -> str:
        return "recording"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        pass

    def get_tool_schemas(self):
        return []

    def shutdown(self) -> None:
        pass

    def on_memory_write(self, action, target, content, metadata=None):
        self.calls.append({
            "action": action,
            "target": target,
            "content": content,
            "metadata": dict(metadata or {}),
        })


def _manager_with_provider():
    mgr = MemoryManager()
    provider = _RecordingProvider()
    mgr.add_provider(provider)
    return mgr, provider


def _single_commit(*, noop=False):
    return {
        "success": True,
        "noop": noop,
        "applied_operation_indexes": [] if noop else [0],
        "noop_operation_indexes": [0] if noop else [],
    }


def test_notifies_remove_with_old_text_after_success():
    mgr, provider = _manager_with_provider()
    mgr.notify_memory_tool_write(
        json.dumps(_single_commit()),
        {"action": "remove", "target": "memory", "old_text": "stale preference entry"},
    )
    assert provider.calls == [
        {
            "action": "remove",
            "target": "memory",
            "content": "",
            "metadata": {"old_text": "stale preference entry"},
        }
    ]


def test_single_new_text_alias_is_mirrored():
    mgr, provider = _manager_with_provider()
    mgr.notify_memory_tool_write(
        _single_commit(),
        {"action": "add", "target": "memory", "new_text": "aliased fact"},
    )

    assert provider.calls == [
        {
            "action": "add",
            "target": "memory",
            "content": "aliased fact",
            "metadata": {},
        }
    ]


def test_partial_batch_mirrors_only_committed_operation_indexes():
    mgr, provider = _manager_with_provider()
    mgr.notify_memory_tool_write(
        {
            "success": True,
            "noop": False,
            "applied_operation_indexes": [1],
            "noop_operation_indexes": [0],
        },
        {
            "target": "memory",
            "operations": [
                {"action": "add", "content": "duplicate"},
                {"action": "add", "new_text": "committed"},
            ],
        },
    )

    assert provider.calls == [
        {
            "action": "add",
            "target": "memory",
            "content": "committed",
            "metadata": {},
        }
    ]


def test_native_noop_result_is_not_mirrored():
    mgr, provider = _manager_with_provider()
    mgr.notify_memory_tool_write(
        {
            "success": True,
            "noop": True,
            "applied_operation_indexes": [],
            "noop_operation_indexes": [0],
        },
        {
            "target": "memory",
            "operations": [{"action": "add", "content": "duplicate"}],
        },
    )

    assert provider.calls == []


@pytest.mark.parametrize(
    "result",
    [
        {"success": True, "noop": False},
        {
            "success": True,
            "noop": False,
            "applied_operation_indexes": [0],
        },
        {
            "success": True,
            "noop": False,
            "applied_operation_indexes": [0, 0],
            "noop_operation_indexes": [1],
        },
        {
            "success": True,
            "noop": False,
            "applied_operation_indexes": [0],
            "noop_operation_indexes": [1, 1],
        },
        {
            "success": True,
            "noop": False,
            "applied_operation_indexes": [0],
            "noop_operation_indexes": [0, 1],
        },
        {
            "success": True,
            "noop": False,
            "applied_operation_indexes": [99],
            "noop_operation_indexes": [0],
        },
        {
            "success": True,
            "noop": False,
            "applied_operation_indexes": [True],
            "noop_operation_indexes": [1],
        },
        {
            "success": True,
            "noop": False,
            "applied_operation_indexes": [0],
            "noop_operation_indexes": [],
        },
        {
            "success": True,
            "noop": True,
            "applied_operation_indexes": [0],
            "noop_operation_indexes": [1],
        },
        {
            "success": True,
            "noop": False,
            "applied_operation_indexes": [],
            "noop_operation_indexes": [0, 1],
        },
        {
            "success": True,
            "noop": "false",
            "applied_operation_indexes": [0],
            "noop_operation_indexes": [1],
        },
    ],
)
def test_malformed_native_commit_metadata_fails_closed(result):
    mgr, provider = _manager_with_provider()
    mgr.notify_memory_tool_write(
        result,
        {
            "target": "memory",
            "operations": [
                {"action": "add", "content": "fact"},
                {"action": "add", "content": "second"},
            ],
        },
    )

    assert provider.calls == []






@pytest.mark.parametrize("tool_result", [None, [], object(), "not-json"])
def test_skips_unrecognized_tool_result_shape(tool_result):
    mgr, provider = _manager_with_provider()
    mgr.notify_memory_tool_write(
        tool_result,
        {"action": "add", "target": "memory", "content": "new fact"},
    )
    assert provider.calls == []






def test_build_metadata_callback_is_merged_per_op():
    mgr, provider = _manager_with_provider()
    mgr.notify_memory_tool_write(
        json.dumps(_single_commit()),
        {"action": "add", "target": "memory", "content": "fact"},
        build_metadata=lambda: {"session_id": "s1", "tool_name": "memory"},
    )
    assert provider.calls == [
        {
            "action": "add",
            "target": "memory",
            "content": "fact",
            "metadata": {"session_id": "s1", "tool_name": "memory"},
        }
    ]
