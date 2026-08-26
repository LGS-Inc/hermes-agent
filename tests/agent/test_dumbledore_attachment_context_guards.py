import inspect
from types import SimpleNamespace

import pytest

from agent.conversation_loop import (
    DUMBLEDORE_IMAGE_TOOL_REDIRECT,
    _dumbledore_agent_loop_image_redirect,
    _ollama_context_limit_error,
    run_conversation,
)
from run_agent import AIAgent


def _build_gemma(monkeypatch, *, marked: bool, router: str = "1", image_lane: str = "1"):
    monkeypatch.setenv("DUMBLEDORE_ROUTER", router)
    monkeypatch.setenv("DUMBLEDORE_IMAGE_LANE", image_lane)
    return AIAgent(
        model="gemma4:12b",
        base_url="https://example.invalid/v1",
        api_key="test-key",
        provider="custom",
        api_mode="chat_completions",
        quiet_mode=True,
        enabled_toolsets=["safe"],
        session_id="dumbledore-attachment-guard-test",
        platform="telegram",
        dumbledore_answer_only_attachment=marked,
    )


def test_attachment_lane_constructs_at_32k_with_no_tools(monkeypatch):
    monkeypatch.setattr(
        "agent.context_compressor.get_model_context_length", lambda *a, **k: 32768
    )
    agent = _build_gemma(monkeypatch, marked=True)
    assert agent._dumbledore_answer_only_attachment is True
    assert agent.context_compressor.context_length == 32768
    assert agent.enabled_toolsets == []
    assert agent.tools == []


@pytest.mark.parametrize(
    ("marked", "router", "image_lane"),
    [(False, "1", "1"), (True, "0", "1"), (True, "1", "0")],
)
def test_declared_context_guard_remains_active_outside_lane(
    monkeypatch, marked, router, image_lane
):
    monkeypatch.setattr(
        "agent.context_compressor.get_model_context_length", lambda *a, **k: 32768
    )
    with pytest.raises(ValueError, match="below the minimum 64,000"):
        _build_gemma(
            monkeypatch, marked=marked, router=router, image_lane=image_lane
        )


def test_runtime_guard_exempts_only_marked_attachment_lane():
    base = dict(
        tools=[{"type": "function", "function": {"name": "probe"}}],
        _ollama_num_ctx=32768,
        model="gemma4:12b",
        base_url="http://127.0.0.1:11434/v1",
        provider="custom",
        session_id="guard-test",
    )
    exempt = SimpleNamespace(
        **base, _dumbledore_answer_only_attachment=True
    )
    ordinary = SimpleNamespace(
        **base, _dumbledore_answer_only_attachment=False
    )
    assert _ollama_context_limit_error(exempt, 1000) is None
    error = _ollama_context_limit_error(ordinary, 1000)
    assert error is not None
    assert "only 32,768 tokens" in error
    assert "at least 64,000 tokens" in error


def _tool_message(
    name="computer_use", arguments='{"action":"open image editor"}',
    content="I will create that image now.",
):
    call = SimpleNamespace(
        function=SimpleNamespace(name=name, arguments=arguments)
    )
    return SimpleNamespace(tool_calls=[call], content=content)


def test_dumbledore_agent_loop_blocks_missed_image_generation_before_tool(monkeypatch):
    monkeypatch.setenv("DUMBLEDORE_ROUTER", "1")
    monkeypatch.setenv("DUMBLEDORE_IMAGE_LANE", "1")
    agent = SimpleNamespace(
        model="qwen3.5:9b-131k-fleet",
        _dumbledore_answer_only_attachment=False,
    )
    # Explicit-command contract: unprefixed prose — even keyword-laden
    # follow-ups — passes through to normal tool dispatch.
    prose = [{
        "role": "user",
        "content": "same thing with engraved bronze serif lettering",
    }]
    assert _dumbledore_agent_loop_image_redirect(
        agent, _tool_message(), prose
    ) is None
    # A leading image control prefix still triggers the redirect…
    for prefix in ("/quality", "/literal", "/brand"):
        prefixed = [{
            "role": "user",
            "content": f"{prefix} same thing with engraved bronze serif lettering",
        }]
        assert _dumbledore_agent_loop_image_redirect(
            agent, _tool_message(), prefixed
        ) == DUMBLEDORE_IMAGE_TOOL_REDIRECT
    # …as does the gateway-armed control state, without any prefix.
    armed = SimpleNamespace(
        model="qwen3.5:9b-131k-fleet",
        _dumbledore_answer_only_attachment=False,
        _dumbledore_image_control_armed=True,
    )
    assert _dumbledore_agent_loop_image_redirect(
        armed, _tool_message(), prose
    ) == DUMBLEDORE_IMAGE_TOOL_REDIRECT
    # Terminal/exec/file tools are never blocked, even on a prefixed order.
    assert _dumbledore_agent_loop_image_redirect(
        agent,
        _tool_message("terminal", '{"command":"magick in.png -crop 100x100+0+0 out.png"}'),
        [{"role": "user", "content": "/quality crop this image"}],
    ) is None


def test_dumbledore_agent_loop_guard_leaves_image_reading_untouched(monkeypatch):
    monkeypatch.setenv("DUMBLEDORE_ROUTER", "1")
    monkeypatch.setenv("DUMBLEDORE_IMAGE_LANE", "1")
    gemma = SimpleNamespace(
        model="gemma4:12b", _dumbledore_answer_only_attachment=True
    )
    messages = [{"role": "user", "content": "read the attached image"}]
    assert _dumbledore_agent_loop_image_redirect(
        gemma, _tool_message("vision_analyze", "{}"), messages
    ) is None


def test_dumbledore_agent_loop_guard_does_not_block_ordinary_computer_use(monkeypatch):
    monkeypatch.setenv("DUMBLEDORE_ROUTER", "1")
    monkeypatch.setenv("DUMBLEDORE_IMAGE_LANE", "1")
    agent = SimpleNamespace(
        model="qwen3.5:9b-131k-fleet",
        _dumbledore_answer_only_attachment=False,
    )
    messages = [{"role": "user", "content": "open the settings window"}]
    assert _dumbledore_agent_loop_image_redirect(
        agent,
        _tool_message(
            "computer_use", '{"action":"open settings"}', "Opening settings."
        ),
        messages,
    ) is None


def test_dumbledore_image_redirect_terminates_before_tool_executor():
    source = inspect.getsource(run_conversation)
    guard_at = source.index("_dumbledore_agent_loop_image_redirect(")
    executor_at = source.index("agent._execute_tool_calls(")
    between = source[guard_at:executor_at]
    assert guard_at < executor_at
    assert '"final_response": _image_tool_redirect' in between
    assert "return {" in between
