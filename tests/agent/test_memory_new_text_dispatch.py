"""Behavior coverage for both built-in memory dispatch paths."""

import json
from types import SimpleNamespace

import pytest

from tools.memory_tool import MemoryStore


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
    memory_store = MemoryStore()
    memory_store.load_from_disk()
    return memory_store


class _FakeAgent:
    session_id = "session-new-text"
    _current_turn_id = "turn-new-text"
    _current_api_request_id = "request-new-text"
    _memory_manager = None

    def __init__(self, store):
        self._memory_store = store


def test_runtime_helper_forwards_single_new_text(store):
    from agent.agent_runtime_helpers import invoke_tool

    result = invoke_tool(
        _FakeAgent(store),
        "memory",
        {"action": "add", "target": "memory", "new_text": "runtime alias"},
        "task-new-text",
        tool_call_id="call-runtime",
        pre_tool_block_checked=True,
        skip_tool_request_middleware=True,
        skip_tool_execution_middleware=True,
    )

    assert json.loads(result)["success"] is True
    assert store.memory_entries == ["runtime alias"]


def test_sequential_executor_forwards_single_new_text(store, monkeypatch):
    from agent import tool_executor

    class _ExecutionCaptured(Exception):
        pass

    def _capture_execution(agent, **kwargs):
        kwargs["execute"](kwargs["function_args"])
        raise _ExecutionCaptured

    monkeypatch.setattr(tool_executor, "_budget_for_agent", lambda agent: object())
    monkeypatch.setattr(
        tool_executor,
        "_run_sequential_tool_execution_middleware",
        _capture_execution,
    )

    agent = _FakeAgent(store)
    agent._incremental_persistence_failed = False
    agent._interrupt_requested = False
    assistant_message = SimpleNamespace(tool_calls=[SimpleNamespace(
        id="call-sequential",
        function=SimpleNamespace(
            name="memory",
            arguments=json.dumps({
                "action": "add",
                "target": "memory",
                "new_text": "sequential alias",
            }),
        ),
    )])

    with pytest.raises(_ExecutionCaptured):
        tool_executor.execute_tool_calls_sequential(
            agent,
            assistant_message,
            [],
            "task-new-text",
        )

    assert store.memory_entries == ["sequential alias"]
