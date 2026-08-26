"""Gateway regressions for Dumbledore attachment-lane toolset scoping."""

import asyncio
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import agent
import gateway.run as gateway_run
from gateway.config import Platform
from gateway.session import SessionSource


class _CapturingAgent:
    init_calls = []

    def __init__(self, *args, **kwargs):
        type(self).init_calls.append(dict(kwargs))
        self.tools = []

    def run_conversation(self, user_message, conversation_history=None, task_id=None):
        return {"final_response": "ok", "messages": [], "api_calls": 1}


def _runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner.config = MagicMock(multiplex_profiles=False)
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._session_reasoning_overrides = {}
    runner._service_tier = None
    runner._show_reasoning = False
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner._session_model_overrides = {}
    runner._session_overrides_loaded = set()
    runner._last_resolved_model = {}
    runner._session_db = None
    runner._agent_cache = None
    runner._agent_cache_lock = None
    runner._dmbl_pending_image = False
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.hooks.loaded_hooks = []
    runner._get_or_create_gateway_honcho = lambda session_key: (None, None)
    return runner


def _configure(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "platform_toolsets:\n  cli: [web, memory]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr(gateway_run, "_env_path", hermes_home / ".env")
    monkeypatch.setattr(gateway_run, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setenv("DUMBLEDORE_ROUTER", "1")
    monkeypatch.setenv("DUMBLEDORE_IMAGE_LANE", "1")
    fake_dmbl = types.ModuleType("agent.dumbledore_router")
    fake_dmbl.load_mode = lambda: {"mode": "home"}
    fake_dmbl.log_decision = lambda **kwargs: None
    fake_dmbl.assert_not_abliterated = lambda model: None
    fake_dmbl.classify_home = lambda prompt, has_image=False: SimpleNamespace(
        model="gemma4:12b" if has_image else "qwen3.5:9b-131k-fleet",
        provider="ollama",
        notice="",
        rule="image" if has_image else "default",
        est_prompt_tokens=1,
    )
    monkeypatch.setitem(sys.modules, "agent.dumbledore_router", fake_dmbl)
    monkeypatch.setattr(agent, "dumbledore_router", fake_dmbl, raising=False)
    runtime = {
        "provider": "ollama",
        "api_mode": "chat_completions",
        "base_url": "http://127.0.0.1:11434/v1",
        "api_key": "test-key",
    }
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: dict(runtime))
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs_for_provider",
        lambda provider: dict(runtime),
    )
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _CapturingAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    _CapturingAgent.init_calls = []
    return fake_dmbl


def _run_turn(runner, message="Hello", *, has_image=False, session_id="session-1"):
    runner._dmbl_pending_image = has_image
    source = SessionSource(
        platform=Platform.LOCAL,
        chat_id="cli",
        chat_name="CLI",
        chat_type="dm",
        user_id="user-1",
    )
    return asyncio.run(
        runner._run_agent(
            message=message,
            context_prompt="",
            history=[],
            source=source,
            session_id=session_id,
            session_key="agent:main:local:dm",
        )
    )


def test_normal_text_constructs_with_image_lane_flags_enabled(tmp_path, monkeypatch):
    _configure(monkeypatch, tmp_path)
    result = _run_turn(_runner(), "Hello")
    assert result["final_response"] == "ok"
    assert {"memory", "web"} <= set(_CapturingAgent.init_calls[-1]["enabled_toolsets"])
    assert _CapturingAgent.init_calls[-1]["dumbledore_answer_only_attachment"] is False
    assert "image_gen" in _CapturingAgent.init_calls[-1]["disabled_toolsets"]


def test_normal_text_constructs_immediately_after_direct_flux_generation(tmp_path, monkeypatch):
    fake_dmbl = _configure(monkeypatch, tmp_path)
    generated = tmp_path / "fresh-flux.png"
    fake_dmbl.run_image_generation = lambda subject: {
        "path": str(generated), "cold": False
    }
    assert fake_dmbl.run_image_generation("test image")["path"] == str(generated)
    runner = _runner()
    runner._dmbl_comfy_last_used = 1.0
    result = _run_turn(runner, "Hello after FLUX", session_id="session-after-flux")
    assert result["final_response"] == "ok"
    assert {"memory", "web"} <= set(_CapturingAgent.init_calls[-1]["enabled_toolsets"])


def test_attachment_turn_gets_zero_tools_and_32k_exemption_marker(tmp_path, monkeypatch):
    _configure(monkeypatch, tmp_path)
    result = _run_turn(_runner(), "Describe this image", has_image=True)
    assert result["final_response"] == "ok"
    call = _CapturingAgent.init_calls[-1]
    assert call["model"] == "gemma4:12b"
    assert call["enabled_toolsets"] == []
    assert call["disabled_toolsets"] is None
    assert call["dumbledore_answer_only_attachment"] is True


def test_normal_turn_after_attachment_restores_configured_tools(tmp_path, monkeypatch):
    _configure(monkeypatch, tmp_path)
    runner = _runner()
    _run_turn(runner, "Describe this image", has_image=True, session_id="attachment")
    _run_turn(runner, "Hello again", session_id="normal")
    attachment, normal = _CapturingAgent.init_calls[-2:]
    assert attachment["enabled_toolsets"] == []
    assert {"memory", "web"} <= set(normal["enabled_toolsets"])
    assert normal["dumbledore_answer_only_attachment"] is False


def test_reset_state_followed_by_normal_text_constructs(tmp_path, monkeypatch):
    _configure(monkeypatch, tmp_path)
    runner = _runner()
    # /reset creates a fresh session and evicts the old cached agent.  Model
    # that resulting gateway state without invoking persistence or adapters.
    runner._agent_cache = {}
    runner._agent_cache_lock = None
    runner._dmbl_pending_image = False
    result = _run_turn(runner, "Hello", session_id="fresh-after-reset")
    assert result["final_response"] == "ok"
    assert {"memory", "web"} <= set(_CapturingAgent.init_calls[-1]["enabled_toolsets"])
