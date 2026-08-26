from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key
from hermes_cli.commands import resolve_command
from gateway.run import (
    _dumbledore_nonlocal_image_provenance,
    _redact_historical_image_generate_evidence,
)


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="chairman",
        chat_id="telegram-dm",
        user_name="Chairman",
        chat_type="dm",
    )


def _event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_source(), message_id="m1")


def _runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    adapter.send_image_file = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(
        emit=AsyncMock(), emit_collect=AsyncMock(return_value=[]), loaded_hooks=False
    )
    entry = SessionEntry(
        session_key=build_session_key(_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._should_send_voice_reply = lambda *_a, **_kw: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *_a, **_kw: None
    runner._emit_gateway_run_progress = AsyncMock()

    async def execute(fn, *args, **kwargs):
        return fn()

    runner._run_in_executor_with_context = execute
    return runner, adapter


def _stub_image_lane(monkeypatch, tmp_path):
    from agent import dumbledore_router as router

    calls = {"renders": [], "enrichments": [], "paid_renders": []}
    image = tmp_path / "image.png"
    image.write_bytes(b"png")
    monkeypatch.setenv("DUMBLEDORE_ROUTER", "1")
    monkeypatch.setenv("DUMBLEDORE_IMAGE_LANE", "1")
    monkeypatch.setattr(router, "comfy_is_up", lambda: False)
    monkeypatch.setattr(router, "load_mode", lambda: {"mode": "home"})
    monkeypatch.setattr(router, "log_decision", lambda **kwargs: None)

    def enrich(prompt):
        calls["enrichments"].append(prompt)
        return {
            "prompt": prompt + ", enriched detail",
            "seconds": 0.01,
            "enriched": True,
            "reason": "ok",
        }

    def render(prompt, *, steps, width=1024, height=1024):
        calls["renders"].append((prompt, steps, width, height))
        return {
            "path": str(image),
            "seconds": 0.01,
            "startup_seconds": 0.0,
            "cold": False,
            "provider": "comfyui",
            "model": router.IMAGE_GEN_MODEL,
            "width": 1024,
            "height": 1024,
        }

    monkeypatch.setattr(router, "enrich_image_prompt", enrich)
    monkeypatch.setattr(router, "run_image_generation", render)
    def paid_render(prompt, *, width=1024, height=1024):
        calls["paid_renders"].append((prompt, width, height))
        return {
            "path": str(image), "provider": "black-forest-labs",
            "model": "flux-2-flex", "width": width, "height": height,
            "cost_credits": 5.0, "cost_usd": 0.05,
            "request_id": "paid-1", "seconds": 1.0,
        }
    monkeypatch.setattr(router, "run_bfl_brand_generation", paid_render)
    monkeypatch.setattr(
        router, "check_paid_brand_spelling",
        lambda _path, approved: {
            "status": "match", "observed_text": approved, "seconds": 0.01,
        },
    )
    return calls


def test_commands_are_registered_and_unknown_stays_unknown():
    assert resolve_command("quality").gateway_only
    assert resolve_command("literal").gateway_only
    assert resolve_command("brand").gateway_only
    assert resolve_command("definitely-unrelated-command") is None


@pytest.mark.asyncio
async def test_quality_prefix_generates_immediately(monkeypatch, tmp_path):
    runner, _ = _runner()
    calls = _stub_image_lane(monkeypatch, tmp_path)
    await runner._handle_message(
        _event("/quality create me an image of girls playing volleyball on the beach")
    )
    assert calls["renders"] == [
        ("girls playing volleyball on the beach, enriched detail", 16, 1280, 1280)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/quality", "/literal"])
async def test_standalone_command_arms_with_ack(monkeypatch, tmp_path, command):
    runner, adapter = _runner()
    _stub_image_lane(monkeypatch, tmp_path)
    await runner._handle_message(_event(command))
    adapter.send.assert_awaited_once()
    assert "armed for the next image" in adapter.send.await_args.kwargs["content"]


@pytest.mark.asyncio
async def test_brand_prefix_is_explicit_paid_one_shot(monkeypatch, tmp_path):
    runner, adapter = _runner()
    calls = _stub_image_lane(monkeypatch, tmp_path)
    await runner._handle_message(
        _event("/brand create an image for Quantum Web Studios Inc")
    )
    assert len(calls["paid_renders"]) == 1
    assert '"Quantum Web Studios Inc"' in calls["paid_renders"][0][0]
    assert calls["paid_renders"][0][1:] == (1024, 1024)
    assert calls["renders"] == []
    caption = adapter.send_image_file.await_args.kwargs["caption"]
    assert "PAID RENDER" in caption
    assert "black-forest-labs/flux-2-flex" in caption
    assert "actual charged cost $0.05 (5 credits)" in caption


def test_paid_brand_proof_prompt_has_company_once_and_preserves_remainder():
    from agent import dumbledore_router as router

    original = "Quantum Web Studios Inc in a premium technology boardroom"
    prompt = router.build_paid_brand_prompt(original)
    assert prompt.count("Quantum Web Studios Inc") == 1
    assert '"Quantum Web Studios Inc"' in prompt
    assert " in a premium technology boardroom" in prompt
    assert "soft directional studio daylight" in prompt
    assert "50mm lens" in prompt
    assert "Do not invent company names" in prompt


def test_paid_brand_keeps_supplied_lighting_lens_and_hex_binding():
    from agent import dumbledore_router as router

    original = (
        "Quantum Web Studios Inc on a navy #112244 wall, dramatic neon lighting, "
        "photographed with an 85mm lens"
    )
    prompt = router.build_paid_brand_prompt(original)
    assert " on a navy #112244 wall, dramatic neon lighting, photographed with an 85mm lens" in prompt
    assert "Preserve every supplied hex color on its named object exactly" in prompt
    assert "soft directional studio daylight" not in prompt
    assert "50mm lens" not in prompt


@pytest.mark.asyncio
async def test_paid_brand_accepts_one_arbitrary_exact_quoted_string(monkeypatch, tmp_path):
    runner, adapter = _runner()
    calls = _stub_image_lane(monkeypatch, tmp_path)
    await runner._handle_message(
        _event('/brand "Heroes United Foundation" on a banner above a veterans event')
    )
    assert len(calls["paid_renders"]) == 1
    prompt, width, height = calls["paid_renders"][0]
    assert prompt.count("Heroes United Foundation") == 1
    assert '"Heroes United Foundation"' in prompt
    assert " on a banner above a veterans event" in prompt
    assert (width, height) == (1024, 1024)
    assert adapter.send_image_file.await_args.kwargs["caption"].startswith("PAID RENDER")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "error_fragment"),
    [
        ('/brand "First" beside "Second"', "multiple quoted strings"),
        ('/brand a banner above a veterans event', "quoted string or a detectable company name"),
        ('/brand "unterminated text', "complete double-quoted string"),
    ],
)
async def test_paid_brand_validation_fails_before_generating_ack(
    monkeypatch, tmp_path, text, error_fragment
):
    runner, adapter = _runner()
    calls = _stub_image_lane(monkeypatch, tmp_path)
    await runner._handle_message(_event(text))
    assert calls["paid_renders"] == []
    assert adapter.send_image_file.await_count == 0
    sent = [call.kwargs["content"] for call in adapter.send.await_args_list]
    assert len(sent) == 1
    assert sent[0].startswith("⚠️ /brand not started:")
    assert error_fragment in sent[0]
    assert all("Generating:" not in message for message in sent)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("preset", "dimensions"),
    [("x", (1200, 675)), ("fb", (1200, 630)), ("ig", (1080, 1080)),
     ("ig-portrait", (1080, 1350)), ("landscape", (1536, 1024))],
)
async def test_paid_brand_optional_dimension_presets(
    monkeypatch, tmp_path, preset, dimensions
):
    runner, _ = _runner()
    calls = _stub_image_lane(monkeypatch, tmp_path)
    await runner._handle_message(
        _event(f"/brand {preset} Quantum Web Studios Inc in a boardroom")
    )
    assert calls["paid_renders"][0][1:] == dimensions


@pytest.mark.asyncio
async def test_paid_brand_spelling_mismatch_is_labeled_without_retry(monkeypatch, tmp_path):
    runner, adapter = _runner()
    calls = _stub_image_lane(monkeypatch, tmp_path)
    from agent import dumbledore_router as router
    monkeypatch.setattr(
        router, "check_paid_brand_spelling",
        lambda *_a: {"status": "mismatch", "observed_text": "Quantum Web Studio Inc"},
    )
    await runner._handle_message(_event("/brand Quantum Web Studios Inc in a boardroom"))
    assert len(calls["paid_renders"]) == 1
    assert "SPELLING CHECK" in adapter.send_image_file.await_args.kwargs["caption"]
    assert "No automatic retry" in adapter.send_image_file.await_args.kwargs["caption"]


@pytest.mark.asyncio
async def test_bare_brand_arms_next_image_then_reverts_local(monkeypatch, tmp_path):
    runner, adapter = _runner()
    calls = _stub_image_lane(monkeypatch, tmp_path)
    runner._handle_message_with_agent = AsyncMock(return_value=None)
    await runner._handle_message(_event("/brand"))
    assert "PAID BRAND" in adapter.send.await_args_list[-1].kwargs["content"]
    await runner._handle_message(_event("Quantum Web Studios Inc in a boardroom"))
    # Brand armed state is consumed: the next EXPLICIT command renders local.
    await runner._handle_message(_event("/quality generate image of a moonlit castle"))
    # Keyword prose without a prefix no longer routes to the image lane.
    await runner._handle_message(_event("generate image of a moonlit castle"))
    assert len(calls["paid_renders"]) == 1
    assert len(calls["renders"]) == 1
    runner._handle_message_with_agent.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prefix", ["/brand /quality", "/quality /brand", "/brand /literal", "/literal /brand"]
)
async def test_brand_composes_with_existing_controls_without_local_fallback(
    monkeypatch, tmp_path, prefix
):
    runner, _ = _runner()
    calls = _stub_image_lane(monkeypatch, tmp_path)
    await runner._handle_message(
        _event(f"{prefix} Quantum Web Studios Inc in a boardroom")
    )
    assert len(calls["paid_renders"]) == 1
    assert calls["renders"] == []


@pytest.mark.asyncio
async def test_paid_brand_failure_never_falls_back_local(monkeypatch, tmp_path):
    runner, adapter = _runner()
    calls = _stub_image_lane(monkeypatch, tmp_path)
    from agent import dumbledore_router as router
    monkeypatch.setattr(
        router, "run_bfl_brand_generation",
        lambda _prompt: (_ for _ in ()).throw(TimeoutError("paid timeout")),
    )
    await runner._handle_message(_event("/brand Quantum Web Studios Inc"))
    assert calls["renders"] == []
    failure = adapter.send.await_args_list[-1].kwargs["content"]
    assert "no local fallback was attempted" in failure


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prefix",
    ["/quality /literal", "/literal /quality"],
)
async def test_quality_and_literal_compose_in_both_orders(
    monkeypatch, tmp_path, prefix
):
    runner, _ = _runner()
    calls = _stub_image_lane(monkeypatch, tmp_path)
    prompt = "create me an image of girls playing volleyball on the beach"
    await runner._handle_message(_event(f"{prefix} {prompt}"))
    assert calls["enrichments"] == []
    assert calls["renders"] == [("girls playing volleyball on the beach", 16, 1280, 1280)]


@pytest.mark.asyncio
async def test_quality_auto_reverts_after_one_image(monkeypatch, tmp_path):
    runner, _ = _runner()
    calls = _stub_image_lane(monkeypatch, tmp_path)
    runner._handle_message_with_agent = AsyncMock(return_value=None)
    await runner._handle_message(_event("/quality"))
    # Armed by the bare control: the next message renders at 16 steps.
    await runner._handle_message(_event("generate image of a moonlit castle"))
    # Quality reverted: the next EXPLICIT command renders at the 4-step default.
    await runner._handle_message(_event("/literal generate image of a sunlit garden"))
    assert [item[1] for item in calls["renders"]] == [16, 4]
    # Prefix-only contract: unprefixed prose now passes to the agent loop.
    await runner._handle_message(_event("generate image of a rainy street"))
    assert len(calls["renders"]) == 2
    runner._handle_message_with_agent.assert_awaited_once()


@pytest.mark.asyncio
async def test_cinematic_portrait_regression_stays_enriched(monkeypatch, tmp_path):
    runner, _ = _runner()
    calls = _stub_image_lane(monkeypatch, tmp_path)
    runner._handle_message_with_agent = AsyncMock(return_value=None)
    # Keyword prose no longer routes to the image lane…
    await runner._handle_message(_event("Generate a cinematic portrait of a wizard"))
    assert calls["renders"] == []
    runner._handle_message_with_agent.assert_awaited_once()
    # …but the explicit command still strips the wrapper and enriches.
    await runner._handle_message(
        _event("/quality Generate a cinematic portrait of a wizard")
    )
    assert calls["enrichments"] == ["a cinematic portrait of a wizard"]
    assert calls["renders"][0][1:] == (16, 1280, 1280)


@pytest.mark.asyncio
async def test_verbless_cinematic_portrait_hard_dispatches_locally(monkeypatch, tmp_path):
    runner, _ = _runner()
    calls = _stub_image_lane(monkeypatch, tmp_path)
    runner._handle_message_with_agent = AsyncMock(return_value=None)
    # Verbless prose alone passes through to the agent loop.
    await runner._handle_message(_event("a cinematic portrait of a wizard"))
    assert calls["renders"] == []
    runner._handle_message_with_agent.assert_awaited_once()
    # With an explicit control prefix the verbless remainder hard-dispatches.
    await runner._handle_message(_event("/quality a cinematic portrait of a wizard"))
    assert calls["enrichments"] == ["a cinematic portrait of a wizard"]
    assert calls["renders"][0][1] == 16


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", ["/quality", "/literal"])
async def test_control_flag_makes_verbless_remainder_image_intent(
    monkeypatch, tmp_path, flag
):
    runner, _ = _runner()
    calls = _stub_image_lane(monkeypatch, tmp_path)
    await runner._handle_message(_event(f"{flag} girls playing volleyball on the beach"))
    assert len(calls["renders"]) == 1


def test_verbless_classifier_keeps_question_and_empty_vetoes():
    from agent import dumbledore_router as router

    assert router.is_image_generation("a cinematic portrait of a wizard")
    assert not router.is_image_generation("what is a cinematic portrait?")
    assert not router.is_image_generation("can you make images?")
    assert not router.is_image_generation("generate")
    assert not router.is_image_generation("hello, how are you")


def test_nonlocal_image_provenance_extracts_paid_tool_result():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "call-1",
                "function": {"name": "image_generate", "arguments": "{}"},
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": (
                '{"success":true,"provider":"openai-codex",'
                '"model":"gpt-image-2-medium","size":"1024x1536",'
                '"image":"/tmp/paid.png"}'
            ),
        },
    ]
    assert _dumbledore_nonlocal_image_provenance(messages) == {
        "provider": "openai-codex",
        "model": "gpt-image-2-medium",
        "width": 1024,
        "height": 1536,
        "path": "/tmp/paid.png",
    }


def test_text_only_turn_after_nonlocal_image_gets_no_stale_label():
    historical = [
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "old-call",
                "function": {"name": "image_generate", "arguments": "{}"},
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "old-call",
            "content": (
                '{"success":true,"provider":"openai-codex",'
                '"model":"gpt-image-2-medium","size":"1024x1536"}'
            ),
        },
        {"role": "assistant", "content": "MEDIA:/tmp/old.png"},
    ]
    current = [
        {"role": "user", "content": "Are you using Flux 2?"},
        {"role": "assistant", "content": "Text-only answer"},
    ]
    assert _dumbledore_nonlocal_image_provenance(
        historical + current, history_offset=len(historical)
    ) is None


def test_text_only_and_local_image_have_no_nonlocal_label():
    text_only = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hello"},
    ]
    local = [
        {"role": "assistant", "content": "MEDIA:/tmp/local-flux.png"},
    ]
    assert _dumbledore_nonlocal_image_provenance(text_only) is None
    assert _dumbledore_nonlocal_image_provenance(local) is None


def test_historical_nonlocal_tool_evidence_is_redacted_from_next_turn():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "old-call",
                "function": {"name": "image_generate", "arguments": "{}"},
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "old-call",
            "content": '{"provider":"openai-codex","model":"gpt-image-2-medium"}',
        },
        {
            "role": "assistant",
            "content": "That image used openai-codex/gpt-image-2-medium.",
        },
        {"role": "user", "content": "What about the current request?"},
    ]
    redacted = _redact_historical_image_generate_evidence(messages)
    visible = "\n".join(str(m.get("content") or "") for m in redacted)
    assert "openai-codex" not in visible
    assert "gpt-image-2-medium" not in visible
    assert "not provenance for the current turn" in visible


@pytest.mark.asyncio
async def test_provenance_question_after_local_image_receives_local_evidence(
    monkeypatch, tmp_path
):
    runner, _ = _runner()
    _stub_image_lane(monkeypatch, tmp_path)
    await runner._handle_message(_event("/quality generate image of a moonlit castle"))
    runner._run_agent = AsyncMock(return_value={
        "final_response": "The most recent image was local.",
        "messages": [],
        "api_calls": 1,
        "history_offset": 0,
    })
    await runner._handle_message(_event("Was that image generated locally?"))
    sent_context = runner._run_agent.await_args.kwargs["context_prompt"]
    assert "most recent image actually delivered" in sent_context
    assert "LOCAL via comfyui/" in sent_context
    assert "openai-codex" not in sent_context


@pytest.mark.asyncio
async def test_unrelated_unknown_command_keeps_normal_message(monkeypatch):
    runner, _ = _runner()
    monkeypatch.setenv("DUMBLEDORE_ROUTER", "1")
    monkeypatch.setenv("DUMBLEDORE_IMAGE_LANE", "1")
    runner._run_agent = AsyncMock(
        side_effect=AssertionError("unknown command leaked to agent")
    )
    result = await runner._handle_message(_event("/definitely-unrelated-command"))
    assert "Unknown command" in result
    assert "/commands" in result
    runner._run_agent.assert_not_called()
