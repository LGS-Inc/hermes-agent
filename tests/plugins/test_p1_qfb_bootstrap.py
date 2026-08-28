"""Behavior contracts for the bounded private QFB bootstrap index."""

from __future__ import annotations

import importlib.util
from pathlib import Path


PLUGIN_PATH = Path(
    "/home/qws/.hermes/plugins/quantum-fleet-brain-bootstrap/__init__.py"
)


def _load_plugin():
    spec = importlib.util.spec_from_file_location(
        "p1_qfb_bootstrap_fixture",
        PLUGIN_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_runtime_index_fits_existing_cap_and_exposes_explicit_only_policy():
    plugin = _load_plugin()

    text, status = plugin._read_runtime_index()

    assert status == "ok"
    assert text is not None
    assert len(text) <= plugin.MAX_RUNTIME_INDEX_CHARS == 3_400
    assert len(plugin._compose_bootstrap_section(text)) <= (
        plugin.SYSTEM_PROMPT_SECTION_MAX_CHARS
    )
    assert "Protocol Alpha, Protocol OMEGA, FABLE Gate, and Independent Review" in text
    assert "Chairman-explicit only" in text
    assert "Automatic technical safeguards remain independent" in text
    assert "Retrieve the smallest relevant note" in text


def test_fresh_private_render_is_complete_and_unauthorized_render_is_empty(monkeypatch):
    plugin = _load_plugin()
    plugin._BOOTSTRAPPED_SESSION_IDS.clear()
    monkeypatch.setattr(
        plugin,
        "_private_main_gate",
        lambda info: (True, "eligible", "fresh-fixture"),
    )
    monkeypatch.setattr(plugin, "_bootstrap_reason", lambda sid: "fresh_session")

    section = plugin.render_fleet_brain_section({"session_id": "fresh-fixture"})

    assert section.count("# Dumbledore Fleet Brain Runtime Index") == 1
    assert section.count("PRIVATE QFB BOOTSTRAP") == 1
    assert plugin._session_bootstrapped_in_process("fresh-fixture") is True

    monkeypatch.setattr(
        plugin,
        "_private_main_gate",
        lambda info: (False, "not_unthreaded_dm", "unauthorized-fixture"),
    )
    assert plugin.render_fleet_brain_section({"session_id": "unauthorized-fixture"}) == ""


def test_rollover_bridge_injects_at_most_once_when_frozen_prompt_lacks_index(monkeypatch):
    plugin = _load_plugin()
    plugin._BOOTSTRAPPED_SESSION_IDS.clear()
    monkeypatch.setattr(plugin, "_is_rollover_created_session", lambda sid: True)
    monkeypatch.setattr(plugin, "_stored_system_prompt", lambda sid: None)

    first = plugin._rollover_bootstrap_context("rollover-child")
    second = plugin._rollover_bootstrap_context("rollover-child")

    assert first is not None
    assert first.count("# Dumbledore Fleet Brain Runtime Index") == 1
    assert second is None


def test_missing_malformed_and_oversized_indexes_fail_closed(tmp_path):
    plugin = _load_plugin()
    plugin.RUNTIME_INDEX_PATH = tmp_path / "missing.md"
    assert plugin._read_runtime_index() == (None, "index_missing")

    plugin.RUNTIME_INDEX_PATH.write_text("not the runtime index", encoding="utf-8")
    assert plugin._read_runtime_index() == (None, "index_malformed")

    plugin.RUNTIME_INDEX_PATH.write_text(
        "# Dumbledore Fleet Brain Runtime Index\n" + "x" * 3_500,
        encoding="utf-8",
    )
    assert plugin._read_runtime_index() == (None, "index_oversized")


def test_targeted_retrieval_and_registration_contract_remain_intact():
    plugin = _load_plugin()
    route = plugin._route_for_query("What does Protocol Alpha require?")
    assert "Shared/FLEET PROTOCOL REGISTRY.md" in route
    assert "/home/qws/.hermes/fleet-brain" in route

    class FakeContext:
        def __init__(self):
            self.section = None
            self.hooks = []

        def get_config(self, name, default=""):
            return "a" * 64

        def register_system_prompt_section(self, *args, **kwargs):
            self.section = (args, kwargs)

        def register_hook(self, name, callback):
            self.hooks.append((name, callback))

    context = FakeContext()
    plugin.register(context)

    args, kwargs = context.section
    assert args[0] == "dumbledore.quantum_fleet_brain"
    assert args[1] is plugin.render_fleet_brain_section
    assert kwargs == {"position": "after_memory", "max_chars": 3_800}
    assert [name for name, _ in context.hooks] == [
        "pre_tool_call",
        "pre_llm_call",
        "post_tool_call",
    ]
