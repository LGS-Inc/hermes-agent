"""Gateway integration tests for the Dumbledore capability router.

Drives the REAL ``GatewayRunner._handle_message`` / ``_resolve_session_agent_runtime``
with every accelerator side effect (Ollama load/unload, ComfyUI, specialist
inference) stubbed at the module boundary of ``agent.dumbledore_capability_router``.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from agent import dumbledore_router as dr
from agent import dumbledore_capability_router as cap
from hermes_cli.commands import resolve_command
from tests.gateway.test_dumbledore_image_commands import _event, _runner, _source


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("DUMBLEDORE_ROUTER", "1")
    monkeypatch.setenv("DUMBLEDORE_IMAGE_LANE", "1")
    monkeypatch.delenv(cap.ROUTE_SIGNATURE_ENV, raising=False)
    monkeypatch.setattr(dr, "comfy_is_up", lambda timeout=2.0: False)
    monkeypatch.setattr(dr, "load_mode", lambda: {"mode": "home"})
    monkeypatch.setattr(dr, "log_decision", lambda **kw: None)
    mode_writes = []
    monkeypatch.setattr(dr, "save_mode", lambda mode, model=None: mode_writes.append((mode, model)))
    monkeypatch.setattr(cap, "TELEMETRY_PATH", str(tmp_path / "router.jsonl"))
    monkeypatch.setattr(cap, "LOCK_PATH", str(tmp_path / "accel.lock"))
    monkeypatch.setattr(cap, "RENDER_DIR", str(tmp_path / "renders"))
    img = tmp_path / "flux.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    calls = {"prepare": [], "specialist": [], "flux": [], "uncut": [], "mode_writes": mode_writes}

    def prepare(model, *, route, keep_alive, **kw):
        calls["prepare"].append((model, route, keep_alive))
        return {"previous_loaded": ["prev"], "unloaded": ["prev"], "load_seconds": 0.0, "comfy_stopped": False}

    def specialist(model, pack, *, route, keep_alive=None, **kw):
        calls["specialist"].append((model, route, keep_alive, pack))
        return {"content": f"{model} says hi", "seconds": 0.1, "load_seconds": 0.0}

    def flux(prompt, *, steps, width, height, **kw):
        calls["flux"].append((prompt, steps, width, height))
        return {"path": str(img), "seconds": 0.1, "startup_seconds": 0.0, "cold": True,
                "provider": "comfyui", "model": dr.IMAGE_GEN_MODEL, "width": width,
                "height": height, "start_mechanism": "systemd", "bytes": 8}

    monkeypatch.setattr(cap, "prepare_local_target", prepare)
    monkeypatch.setattr(cap, "run_specialist", specialist)
    monkeypatch.setattr(cap, "run_flux_generation", flux)
    monkeypatch.setattr(cap, "prepare_for_flux", lambda: {"previous_loaded": [], "unloaded": []})
    monkeypatch.setattr(cap, "ollama_loaded_names", lambda timeout=5.0: [])
    monkeypatch.setattr(cap, "ollama_ps", lambda timeout=5.0: [])
    monkeypatch.setattr(cap, "comfy_stop", lambda wait=30.0: True)
    monkeypatch.setattr(cap, "comfy_unit_state", lambda: "inactive")
    monkeypatch.setattr(dr, "enrich_image_prompt",
                        lambda subject: {"prompt": subject, "seconds": 0.0, "enriched": False, "reason": "t"})
    monkeypatch.setattr(dr, "run_uncut", lambda prompt, alt=False: calls["uncut"].append(prompt) or "[UNCUT] ok")
    return calls


def _gw():
    runner, adapter = _runner()
    runner._handle_message_with_agent = AsyncMock(return_value=None)
    runner._session_model_overrides = {}
    runner._session_overrides_loaded = set()
    runner._last_resolved_model = {}
    return runner, adapter


def _sent(adapter):
    return [c.kwargs["content"] for c in adapter.send.await_args_list]


def _decision(runner):
    return (getattr(runner, "_dmbl_turn", None) or {}).get("decision")


def _telemetry(tmp_path):
    p = tmp_path / "router.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _receipts(tmp_path):
    return [
        event
        for event in _telemetry(tmp_path)
        if event.get("v") == cap.ROUTE_RECEIPT_VERSION
        and event.get("kind") == cap.ROUTE_RECEIPT_KIND
    ]


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def test_route_command_registered_without_conflicts():
    c = resolve_command("route")
    assert c is not None and c.gateway_only and c.busy_policy == "reject"
    for existing in ("model", "quality", "literal", "brand", "new", "stop"):
        assert resolve_command(existing) is not None


@pytest.mark.asyncio
async def test_route_status_is_read_only(env, tmp_path):
    runner, adapter = _gw()
    await runner._handle_message(_event("/route status"))
    txt = _sent(adapter)[0]
    assert "Route status" in txt and "HOME_FAST=" in txt and "accelerator lock: free" in txt
    runner._handle_message_with_agent.assert_not_awaited()
    assert env["mode_writes"] == [] and env["prepare"] == []


@pytest.mark.asyncio
async def test_R12_bare_route_deep_arms_then_applies_one_turn(env, tmp_path):
    runner, adapter = _gw()
    await runner._handle_message(_event("/route deep"))
    assert "DEEP_LOCAL armed" in _sent(adapter)[0]
    await runner._handle_message(_event("what time is it?"))
    d = _decision(runner)
    assert d.route == cap.DEEP_LOCAL and d.model == cap.DEEP_LOCAL_MODEL and d.dispatch == "agent"
    assert env["prepare"][-1] == (cap.DEEP_LOCAL_MODEL, cap.DEEP_LOCAL, "5m")
    runner._handle_message_with_agent.assert_awaited_once()
    # one-turn: the next ordinary message is HOME_FAST again
    await runner._handle_message(_event("what time is it?"))
    assert _decision(runner).route == cap.HOME_FAST
    assert env["prepare"][-1] == (cap.HOME_FAST_MODEL, cap.HOME_FAST, "15m")


@pytest.mark.asyncio
async def test_route_auto_clears_armed_override(env):
    runner, adapter = _gw()
    await runner._handle_message(_event("/route cloud"))
    await runner._handle_message(_event("/route auto"))
    assert "cleared" in _sent(adapter)[-1]
    await runner._handle_message(_event("hello"))
    assert _decision(runner).route == cap.HOME_FAST


@pytest.mark.asyncio
async def test_R13_route_code_heavy_dispatches_specialist_with_pack(env, tmp_path):
    runner, adapter = _gw()
    await runner._handle_message(_event("/route code-heavy write hello world"))
    assert env["prepare"][-1][0] == cap.CODE_HEAVY_MODEL
    model, route, keep_alive, pack = env["specialist"][-1]
    assert model == cap.CODE_HEAVY_MODEL and route == cap.CODE_HEAVY
    assert keep_alive == cap.KEEP_ALIVE["CODE_HEAVY_SESSION"]        # explicit heavy session
    assert pack["messages"][-1]["content"] == "CURRENT TASK:\nwrite hello world"
    assert any("qwen3-coder-next:latest says hi" in m for m in _sent(adapter))
    runner._handle_message_with_agent.assert_not_awaited()
    ev = [e for e in _telemetry(tmp_path) if e.get("outcome") == "ok"][-1]
    assert ev["route"] == cap.CODE_HEAVY and ev["lock"] == "acquired" and "pack_tokens" in ev
    assert "hello world" not in json.dumps(ev)


@pytest.mark.asyncio
async def test_auto_code_heavy_is_one_shot_keep_alive_zero(env):
    runner, adapter = _gw()
    await runner._handle_message(_event("Refactor the whole codebase to replace requests with httpx across every module"))
    model, route, keep_alive, _ = env["specialist"][-1]
    assert model == cap.CODE_HEAVY_MODEL and keep_alive == 0
    assert any("Routing to qwen3-coder-next:latest" in m for m in _sent(adapter))


@pytest.mark.asyncio
async def test_R6_code_fast_dispatch(env):
    runner, adapter = _gw()
    await runner._handle_message(_event("Write a function that reverses a string in python"))
    model, route, keep_alive, _ = env["specialist"][-1]
    assert model == cap.CODE_FAST_MODEL and route == cap.CODE_FAST and keep_alive == "5m"
    assert _sent(adapter)[-1].endswith("qwen2.5-coder:14b says hi")
    assert cap.SPECIALIST_RESULT_BOUNDARY in _sent(adapter)[-1]


@pytest.mark.asyncio
async def test_route_model_explicit_local_tag(env):
    runner, adapter = _gw()
    await runner._handle_message(_event("/route model qwen2.5-coder:14b say hi"))
    model, route, _, pack = env["specialist"][-1]
    assert model == cap.CODE_FAST_MODEL and route == cap.EXPLICIT_PIN
    assert pack["messages"][-1]["content"].endswith("say hi")


@pytest.mark.asyncio
async def test_route_model_rejects_abliterated(env):
    runner, adapter = _gw()
    await runner._handle_message(_event("/route model qwen3-abliterated-hermes:8b hi"))
    assert "uncut" in _sent(adapter)[0]
    assert env["specialist"] == []


# ---------------------------------------------------------------------------
# Pins (R14 / R15 / R16)
# ---------------------------------------------------------------------------

def _pin_35b(runner, monkeypatch):
    monkeypatch.setattr(dr, "load_mode", lambda: {"mode": "pinned"})
    key = runner._session_key_for_source(_source())
    runner._session_model_overrides[key] = {
        "model": cap.DEEP_LOCAL_MODEL, "provider": cap.DEEP_LOCAL_PROVIDER,
        "api_key": "ollama", "base_url": "http://127.0.0.1:11434/v1",
    }
    return key


@pytest.mark.asyncio
async def test_R14_R15_R16_pin_survives_heavy_specialist(env, monkeypatch):
    runner, adapter = _gw()
    key = _pin_35b(runner, monkeypatch)
    before = dict(runner._session_model_overrides[key])
    # R15: pinned 35B + repository-wide task -> coder runs, conflict unloaded
    await runner._handle_message(_event("Refactor the whole codebase to replace requests with httpx across every module"))
    d = _decision(runner)
    assert d.route == cap.CODE_HEAVY and d.reason.overrides_pin
    assert env["prepare"][-1][0] == cap.CODE_HEAVY_MODEL
    assert env["specialist"][-1][0] == cap.CODE_HEAVY_MODEL
    # R14: the persistent pin was never rewritten
    assert runner._session_model_overrides[key] == before
    assert env["mode_writes"] == []
    # R16: next ordinary turn is the pin again, prepared with the pin keep-alive
    await runner._handle_message(_event("what time is it?"))
    d = _decision(runner)
    assert d.route == cap.EXPLICIT_PIN and d.dispatch == "pin" and d.model == cap.DEEP_LOCAL_MODEL
    assert env["prepare"][-1] == (cap.DEEP_LOCAL_MODEL, cap.EXPLICIT_PIN, cap.KEEP_ALIVE["DEEP_LOCAL_PIN"])
    assert runner._session_model_overrides[key] == before
    runner._handle_message_with_agent.assert_awaited_once()


# ---------------------------------------------------------------------------
# Failures (R17 / R18 / R29)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_R17_local_load_failure_routes_cloud_safe_for_this_turn_only(env, monkeypatch, tmp_path):
    runner, adapter = _gw()

    def boom(model, *, route, keep_alive, **kw):
        raise cap.LocalLoadError(f"load of {model} exceeded 180s")

    monkeypatch.setattr(cap, "prepare_local_target", boom)
    await runner._handle_message(_event("Think deeply about whether we should sell the studio"))
    d = _decision(runner)
    assert d.route == cap.CLOUD_SAFE and d.model == "gpt-5.6-sol" and d.provider == "openai-codex"
    assert d.reason.reason_code == "fallback:deep_local:LocalLoadError"
    runner._handle_message_with_agent.assert_awaited_once()
    ev = [e for e in _telemetry(tmp_path) if e.get("outcome") == "fallback_cloud"][-1]
    assert ev["fallback"] == cap.DEEP_LOCAL and ev["error"] == "LocalLoadError"
    assert env["mode_writes"] == []


@pytest.mark.asyncio
async def test_specialist_chain_14b_then_heavy_then_cloud(env, monkeypatch):
    runner, adapter = _gw()

    def failing(model, pack, *, route, keep_alive=None, **kw):
        raise cap.LocalLoadError(f"ollama 500 for {model}")

    monkeypatch.setattr(cap, "run_specialist", failing)
    await runner._handle_message(_event("Write a function that reverses a string in python"))
    # 14B attempted, then heavy, then cloud through the agent loop
    assert [p[0] for p in env["prepare"]] == [cap.CODE_FAST_MODEL, cap.CODE_HEAVY_MODEL]
    d = _decision(runner)
    assert d.route == cap.CLOUD_SAFE
    assert runner._dmbl_coder_failed and "cloud-safe gpt-5.6-sol" in runner._dmbl_coder_failed_text
    runner._handle_message_with_agent.assert_awaited_once()


@pytest.mark.asyncio
async def test_R18_local_only_never_uses_cloud(env, monkeypatch, tmp_path):
    runner, adapter = _gw()

    def failing(model, pack, *, route, keep_alive=None, **kw):
        raise cap.LocalLoadError("ollama overloaded")

    monkeypatch.setattr(cap, "run_specialist", failing)
    await runner._handle_message(_event("Local only: write a function that reverses a string in python"))
    msg = _sent(adapter)[-1]
    assert "no cloud fallback was used" in msg and "LocalLoadError" in msg
    runner._handle_message_with_agent.assert_not_awaited()
    assert all(e.get("model") != "gpt-5.6-sol" for e in _telemetry(tmp_path))
    # agent-loop variant: preflight failure on a local-only deep request
    def boom(model, *, route, keep_alive, **kw):
        raise cap.LocalLoadError("nope")
    monkeypatch.setattr(cap, "prepare_local_target", boom)
    await runner._handle_message(_event("Keep it local. Think deeply about the second-order effects of selling the studio"))
    assert _decision(runner) is None
    assert "no cloud fallback was used" in runner._dmbl_coder_failed_text


@pytest.mark.asyncio
async def test_R29_gateway_survives_router_explosions(env, monkeypatch):
    runner, adapter = _gw()
    real_decide = cap.decide_route
    monkeypatch.setattr(cap, "decide_route", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("classifier exploded")))
    await runner._handle_message(_event("hello there"))
    runner._handle_message_with_agent.assert_awaited_once()      # message not lost
    monkeypatch.setattr(cap, "decide_route", real_decide)
    monkeypatch.setattr(cap, "prepare_local_target", lambda *a, **k: (_ for _ in ()).throw(OSError("disk")))
    runner2, _ = _gw()
    await runner2._handle_message(_event("hello again"))
    runner2._handle_message_with_agent.assert_awaited_once()
    assert _decision(runner2).route == cap.HOME_FAST                # unexpected prep error: proceed normally


# ---------------------------------------------------------------------------
# Image lanes (R10 / R11 / R22-R24 at the gateway)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_R11_prose_image_order_routes_flux_under_lock(env, tmp_path):
    runner, adapter = _gw()
    await runner._handle_message(_event("Create an image of a red fox in snow"))
    assert env["flux"] == [("a red fox in snow", dr.IMAGE_GEN_STEPS, 1024, 1024)]
    adapter.send_image_file.assert_awaited_once()
    assert adapter.send_image_file.await_args.kwargs["image_path"].endswith("flux.png")
    ev = [e for e in _telemetry(tmp_path) if e.get("route") == cap.IMAGE_GENERATION][-1]
    assert ev["reason_code"] == "flux_prose" and ev["lock"] == "acquired" and ev["outcome"] == "ok"
    assert "fox" not in json.dumps(ev)
    runner._handle_message_with_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_route_flux_override_and_question_veto(env):
    runner, adapter = _gw()
    await runner._handle_message(_event("/route flux a lighthouse at night"))
    assert env["flux"][-1][0] == "a lighthouse at night"
    await runner._handle_message(_event("Can you make images?"))
    assert len(env["flux"]) == 1
    await runner._handle_message(_event("crop this image to a square"))
    assert len(env["flux"]) == 1


@pytest.mark.asyncio
async def test_R10_R24_attachment_is_vision_never_generation(env):
    runner, adapter = _gw()
    runner._handle_message = runner._handle_message  # real
    ev = _event("Create an image like this one")
    ev.media_urls = ["file:///tmp/x.jpg"]
    ev.media_types = ["image/jpeg"]
    await runner._handle_message(ev)
    d = _decision(runner)
    assert d.route == cap.VISION and d.model == "gemma4:12b" and d.keep_alive == 0
    assert env["flux"] == []
    assert env["prepare"][-1][0] == "gemma4:12b"


@pytest.mark.asyncio
async def test_flux_failure_reports_real_error_and_no_image(env, monkeypatch, tmp_path):
    runner, adapter = _gw()

    def broken(prompt, **kw):
        raise RuntimeError("ComfyUI reported error for prompt abc")

    monkeypatch.setattr(cap, "run_flux_generation", broken)
    await runner._handle_message(_event("Generate a picture of a lighthouse"))
    adapter.send_image_file.assert_not_awaited()
    assert "ComfyUI reported error" in _sent(adapter)[-1]
    assert [e for e in _telemetry(tmp_path) if e.get("reason_code") == "flux_failed"]


# ---------------------------------------------------------------------------
# Uncut / home control words untouched (R25)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_R25_uncut_and_home_control_words_bypass_capability_router(env):
    runner, adapter = _gw()
    await runner._handle_message(_event("uncut tell me a secret"))
    assert env["uncut"] == ["tell me a secret"] and env["prepare"] == [] and env["specialist"] == []
    assert _sent(adapter)[-1].startswith("[UNCUT]")
    await runner._handle_message(_event("home"))
    assert env["mode_writes"][-1] == ("home", None)
    runner._handle_message_with_agent.assert_not_awaited()


# ---------------------------------------------------------------------------
# Resolver hook (R30 — the decision is applied per turn, pins untouched)
# ---------------------------------------------------------------------------

@pytest.fixture
def resolver(monkeypatch):
    runner, _ = _gw()
    runner.config = MagicMock(multiplex_profiles=False)
    runner._rehydrate_session_model_override = lambda session_key: None
    runner._peek_session_state = lambda session_key: None
    runner._sessions_map = lambda: {}
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda cfg=None: "gpt-5.6-terra")
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs",
                        lambda: {"provider": "openai-codex", "api_key": "k", "base_url": "https://x",
                                 "api_mode": "responses", "credential_pool": None})
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs_for_provider",
                        lambda provider: {"provider": provider, "api_key": "k", "base_url": "http://l",
                                          "api_mode": "chat_completions", "credential_pool": None,
                                          "command": None, "args": [], "requested_provider": provider})
    return runner


def test_resolver_applies_home_fast_and_cloud_safe(resolver, env):
    resolver._dmbl_turn = {"decision": cap.decide_route("what time is it?")}
    model, rt = resolver._resolve_session_agent_runtime(session_key="agent:main:telegram:dm")
    assert model == cap.HOME_FAST_MODEL and rt["provider"] == cap.HOME_FAST_PROVIDER
    assert resolver._dmbl_last_route["rule"] == "default" and resolver._dmbl_last_route["local"]
    resolver._dmbl_turn = {"decision": cap.cloud_fallback_decision(cap.decide_route("hi"), "LocalLoadError")}
    model, rt = resolver._resolve_session_agent_runtime(session_key="agent:main:telegram:dm")
    assert model == "gpt-5.6-sol" and rt["provider"] == "openai-codex"
    assert resolver._dmbl_last_route["local"] is False and "cloud-safe" in resolver._dmbl_last_route["notice"]


def test_resolver_vision_keeps_answer_only_marker(resolver, env):
    resolver._dmbl_turn = {"decision": cap.decide_route("x" * 200_000, has_image=True)}
    model, _ = resolver._resolve_session_agent_runtime(session_key="k")
    assert model == "gemma4:12b" and resolver._dmbl_last_route["rule"] == "image_overflow_warn"
    resolver._dmbl_turn = {"decision": cap.decide_route("describe", has_image=True)}
    resolver._resolve_session_agent_runtime(session_key="k")
    assert resolver._dmbl_last_route["rule"] == "image"


def test_resolver_pin_decision_leaves_hermes_resolution_alone(resolver, env, monkeypatch):
    monkeypatch.setattr(dr, "load_mode", lambda: {"mode": "pinned"})
    d = cap.decide_route("hi", mode="pinned", pinned_model="gpt-5.6-terra", pinned_provider="openai-codex")
    resolver._dmbl_turn = {"decision": d}
    model, rt = resolver._resolve_session_agent_runtime(session_key="k")
    assert model == "gpt-5.6-terra" and rt["provider"] == "openai-codex"
    assert resolver._dmbl_last_route["rule"] == "pin"


def test_resolver_without_decision_keeps_legacy_home_behaviour(resolver, env):
    resolver._dmbl_turn = {"prompt": "hello", "has_image": False}
    model, rt = resolver._resolve_session_agent_runtime(session_key="k")
    assert model == cap.HOME_FAST_MODEL


def test_resolver_survives_provider_resolution_failure(resolver, env, monkeypatch):
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs_for_provider",
                        lambda provider: (_ for _ in ()).throw(RuntimeError("no creds")))
    resolver._dmbl_turn = {"decision": cap.decide_route("Think deeply about the studio's future")}
    model, rt = resolver._resolve_session_agent_runtime(session_key="k")
    assert model == "gpt-5.6-terra"      # already-resolved pair survives (R29)


# ---------------------------------------------------------------------------
# R30: no permanent configuration drift
# ---------------------------------------------------------------------------

def test_R30_router_constants_match_verified_live_models():
    assert cap.HOME_FAST_MODEL == "qwen3.5:9b-131k-fleet"
    assert cap.DEEP_LOCAL_MODEL == "qwen3.6:35b-a3b-64k"
    assert cap.CODE_FAST_MODEL == "qwen2.5-coder:14b"
    assert cap.CODE_HEAVY_MODEL == "qwen3-coder-next:latest"
    assert cap.VISION_MODEL == "gemma4:12b"
    assert cap.CLOUD_SAFE_MODEL == "gpt-5.6-sol" and cap.CLOUD_SAFE_PROVIDER == "openai-codex"
    assert dr.FLUX_CKPT == "flux-2-klein-4b-fp8.safetensors" and dr.FLUX_VAE == "flux2-vae.safetensors"
    assert dr.COMFY_URL.startswith("http://127.0.0.1:")
    # the legacy classifier still exists and is unchanged for the fallback path
    assert dr.classify_home("hi", has_image=False).model == cap.HOME_FAST_MODEL


# ---------------------------------------------------------------------------
# Review findings 2026-08-27 (independent reviewer) — regression guards
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_slash_commands_never_run_preflight(env):
    runner, adapter = _gw()
    runner._handle_reset_command = AsyncMock(return_value="reset")
    for cmd in ("/help", "/status", "/model qwen36", "/new"):
        try:
            await runner._handle_message(_event(cmd))
        except Exception:
            pass    # the minimal harness cannot run every command; the preflight gate is what matters
    assert env["prepare"] == [] and env["specialist"] == []


@pytest.mark.asyncio
async def test_home_fast_under_held_lock_stays_local(env, monkeypatch, tmp_path):
    runner, adapter = _gw()
    holder = cap.AcceleratorLock(str(tmp_path / "accel.lock")).acquire(owner="other", route=cap.CODE_HEAVY, timeout=1)
    monkeypatch.setattr(cap, "LOCK_TIMEOUT_LIGHT", 0.2)
    try:
        await runner._handle_message(_event("what time is it?"))
    finally:
        holder.release()
    d = _decision(runner)
    assert d.route == cap.HOME_FAST and d.model == cap.HOME_FAST_MODEL     # never the cloud
    assert env["prepare"] == []                                            # proceeded without preflight
    ev = [e for e in _telemetry(tmp_path) if e.get("outcome") == "proceed_local_without_preflight"]
    assert ev and ev[-1]["lock"] == "timeout"
    runner._handle_message_with_agent.assert_awaited_once()


@pytest.mark.asyncio
async def test_heavy_route_under_held_lock_falls_to_cloud(env, monkeypatch, tmp_path):
    runner, adapter = _gw()
    holder = cap.AcceleratorLock(str(tmp_path / "accel.lock")).acquire(owner="other", route=cap.CODE_HEAVY, timeout=1)
    monkeypatch.setattr(cap, "LOCK_TIMEOUT_DEFAULT", 0.2)
    try:
        await runner._handle_message(_event("/route deep what should the fleet do next year?"))
    finally:
        holder.release()
    assert _decision(runner).route == cap.CLOUD_SAFE


@pytest.mark.asyncio
async def test_cloud_question_is_not_cloud_selection(env):
    runner, adapter = _gw()
    await runner._handle_message(_event("which cloud model do we use?"))
    assert _decision(runner).route == cap.HOME_FAST


@pytest.mark.asyncio
async def test_decisions_are_session_keyed(env):
    runner, adapter = _gw()
    await runner._handle_message(_event("what time is it?"))
    key = runner._session_key_for_source(_source())
    assert runner._dmbl_turns[key]["decision"].route == cap.HOME_FAST
    # a stale instance-global stash from another chat must not be used when the
    # session-keyed one exists
    runner._dmbl_turn = {"decision": cap.decide_route("hi", override=cap.CLOUD_SAFE)}
    runner._rehydrate_session_model_override = lambda sk: None
    runner._peek_session_state = lambda sk: None
    runner._sessions_map = lambda: {}
    runner.config = MagicMock(multiplex_profiles=False)
    import gateway.run as _gr
    orig = (_gr._resolve_gateway_model, _gr._resolve_runtime_agent_kwargs, _gr._resolve_runtime_agent_kwargs_for_provider)
    _gr._resolve_gateway_model = lambda cfg=None: "gpt-5.6-terra"
    _gr._resolve_runtime_agent_kwargs = lambda: {"provider": "openai-codex", "api_key": "k", "base_url": "u", "api_mode": "r", "credential_pool": None}
    _gr._resolve_runtime_agent_kwargs_for_provider = lambda provider: {"provider": provider, "api_key": "k", "base_url": "l", "api_mode": "c", "credential_pool": None, "command": None, "args": [], "requested_provider": provider}
    try:
        model, _ = runner._resolve_session_agent_runtime(session_key=key)
    finally:
        _gr._resolve_gateway_model, _gr._resolve_runtime_agent_kwargs, _gr._resolve_runtime_agent_kwargs_for_provider = orig
    assert model == cap.HOME_FAST_MODEL


@pytest.mark.asyncio
async def test_specialist_exchange_is_persisted_to_transcript(env):
    runner, adapter = _gw()
    store = MagicMock()
    entry = MagicMock(session_id="sess-9")
    store.lookup_by_session_key = AsyncMock(return_value=entry)
    store.load_transcript = AsyncMock(return_value=[])
    store.append_to_transcript = AsyncMock()
    import gateway.run as _gr
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_gr.GatewayRunner, "async_session_store", property(lambda self: store))
        await runner._handle_message(_event("Write a function that reverses a string in python"))
    roles = [c.args[1]["role"] for c in store.append_to_transcript.await_args_list]
    assert roles == ["user", "assistant"]
    assert store.append_to_transcript.await_args_list[0].args[0] == "sess-9"
    assert "says hi" in store.append_to_transcript.await_args_list[1].args[1]["content"]


@pytest.mark.asyncio
async def test_stale_decision_never_survives_to_a_bypassing_turn(env, monkeypatch):
    runner, adapter = _gw()

    def boom(model, *, route, keep_alive, **kw):
        raise cap.LocalLoadError("nope")

    monkeypatch.setattr(cap, "prepare_local_target", boom)
    await runner._handle_message(_event("Think deeply about whether we should sell the studio"))
    key = runner._session_key_for_source(_source())
    assert runner._dmbl_turns[key]["decision"].route == cap.CLOUD_SAFE
    # a slash command bypasses dispatch: the stash must be fresh, not the old cloud decision
    try:
        await runner._handle_message(_event("/help"))
    except Exception:
        pass
    assert runner._dmbl_turns[key].get("decision") is None


def test_image_prose_trailing_question_mark_vetoes():
    assert not cap.is_image_generation_prose("draw a picture of a cat?")
    assert cap.is_image_generation_prose("draw a picture of a cat")


def test_turn_runner_agent_path_applies_session_keyed_decision(tmp_path, monkeypatch):
    """The real _run_agent → TurnRunner → _resolve_session_agent_runtime path
    must construct the agent with the capability router's model."""
    from tests.gateway import test_dumbledore_attachment_toolsets as tk

    runner = tk._runner()
    tk._configure(monkeypatch, tmp_path)
    monkeypatch.setattr(cap, "TELEMETRY_PATH", str(tmp_path / "router.jsonl"))
    monkeypatch.setattr(cap, "apply_keep_alive", lambda model, keep_alive, timeout=30.0: True)
    decision = cap.decide_route("what time is it?", override=cap.DEEP_LOCAL)
    runner._dmbl_turns = {"agent:main:local:dm": {"prompt": "what time is it?", "has_image": False,
                                                  "decision": decision}}
    tk._run_turn(runner, "what time is it?")
    assert tk._CapturingAgent.init_calls[-1]["model"] == cap.DEEP_LOCAL_MODEL


def test_RP_agent_result_exports_actual_provider_turn_and_attempts(tmp_path, monkeypatch):
    from tests.gateway import test_dumbledore_attachment_toolsets as tk

    runner = tk._runner()
    tk._configure(monkeypatch, tmp_path)

    def _run(self, user_message, conversation_history=None, task_id=None):
        self.model = cap.CLOUD_SAFE_MODEL
        self.provider = cap.CLOUD_SAFE_PROVIDER
        self._current_turn_id = "raw-core-turn"
        self._route_provenance_attempts = [
            {
                "ordinal": 1,
                "api_request_id": "raw-core-api",
                "retry_index": 0,
                "provider": cap.CLOUD_SAFE_PROVIDER,
                "model": cap.CLOUD_SAFE_MODEL,
                "transport": "agent",
                "local_invocation": False,
                "outcome": "returned",
            }
        ]
        self._route_provenance_fallbacks = []
        self._route_provenance_attempts_truncated = 0
        return {"final_response": "ok", "messages": [], "api_calls": 1}

    monkeypatch.setattr(tk._CapturingAgent, "run_conversation", _run)
    result = tk._run_turn(runner, "hello", session_id="session-provenance")
    assert result["model"] == cap.CLOUD_SAFE_MODEL
    assert result["provider"] == cap.CLOUD_SAFE_PROVIDER
    assert result["route_provenance_turn_id"] == "raw-core-turn"
    assert result["route_provenance_attempts"][0]["api_request_id"] == "raw-core-api"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("prompt", "expected_route", "local"),
    [
        ("what time is it?", cap.HOME_FAST, True),
        (
            "Design a fault-tolerant architecture and compare the trade-offs",
            cap.DEEP_LOCAL,
            True,
        ),
        ("switch to the cloud and summarize the document", cap.CLOUD_SAFE, False),
    ],
)
async def test_RP1_RP2_RP5_agent_success_receipts(
    env, tmp_path, prompt, expected_route, local
):
    runner, _adapter = _gw()
    event = _event(prompt)
    event.message_id = f"rp-agent-{expected_route}"
    session_key = runner._session_key_for_source(event.source)
    decision = cap.decide_route(prompt)
    assert decision.route == expected_route
    receipt = await runner._dmbl_build_receipt_context(
        event=event,
        source=event.source,
        session_key=session_key,
        decision=decision,
        mode="home",
        pin_model=None,
        pin_provider=None,
        route_armed=None,
    )
    receipt["input_session_id"] = "session-agent"
    turn = {
        "requested_decision": decision,
        "decision": decision,
        "route_receipt": receipt,
    }
    provider = decision.provider or "ollama"
    result = {
        "provider": provider,
        "model": decision.model,
        "session_id": "session-agent",
        "final_response": "ok",
        "completed": True,
        "route_provenance_turn_id": f"turn-{expected_route}",
        "route_provenance_attempts": [
            {
                "ordinal": 1,
                "api_request_id": f"api-{expected_route}",
                "retry_index": 0,
                "provider": provider,
                "model": decision.model,
                "response_model": decision.model,
                "transport": "agent",
                "local_invocation": local,
                "outcome": "returned",
            }
        ],
    }
    await runner._dmbl_finalize_route_receipt(turn, agent_result=result)

    rec = _receipts(tmp_path)[-1]
    assert rec["requested_route"] == expected_route
    assert rec["requested_model"] == decision.model
    assert rec["actual_model"] == decision.model
    assert rec["final_effective_model"] == decision.model
    assert rec["local_invocation"] is local
    assert rec["cloud_fallback"] is False
    assert rec["completion_outcome"] == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("prompt", "expected_route", "expected_model"),
    [
        ("Write a function that reverses a string in python", cap.CODE_FAST, cap.CODE_FAST_MODEL),
        (
            "Refactor the whole codebase to replace requests with httpx across every module",
            cap.CODE_HEAVY,
            cap.CODE_HEAVY_MODEL,
        ),
    ],
)
async def test_RP3_RP4_direct_specialist_receipts(
    env, tmp_path, prompt, expected_route, expected_model
):
    runner, _adapter = _gw()
    event = _event(prompt)
    event.message_id = f"rp-specialist-{expected_route}"
    await runner._handle_message(event)

    receipts = _receipts(tmp_path)
    assert len(receipts) == 1
    rec = receipts[0]
    assert rec["requested_route"] == expected_route
    assert rec["requested_model"] == expected_model
    assert rec["actual_provider"] == "ollama"
    assert rec["actual_model"] == expected_model
    assert rec["local_invocation"] is True
    assert rec["cloud_fallback"] is False
    assert rec["attempts"][-1]["transport"] == "direct_ollama"
    assert rec["attempts"][-1]["outcome"] == "returned"


@pytest.mark.asyncio
async def test_RP6_local_failure_receipt_preserves_request_and_cloud_actual(env, tmp_path):
    runner, _adapter = _gw()
    event = _event("what time is it?")
    event.message_id = "rp-local-fallback"
    session_key = runner._session_key_for_source(event.source)
    requested = cap.decide_route(event.text)
    cloud = cap.cloud_fallback_decision(requested, "LocalLoadError")
    receipt = await runner._dmbl_build_receipt_context(
        event=event,
        source=event.source,
        session_key=session_key,
        decision=requested,
        mode="home",
        pin_model=None,
        pin_provider=None,
        route_armed=None,
    )
    receipt["input_session_id"] = "session-fallback"
    turn = {
        "requested_decision": requested,
        "decision": cloud,
        "route_receipt": receipt,
        "route_fallback_reason": cloud.reason.reason_code,
    }
    runner._dmbl_add_receipt_attempt(
        turn,
        provider=requested.provider,
        model=requested.model,
        transport="local_preflight",
        outcome="failed",
        local_invocation=False,
        error_class="LocalLoadError",
    )
    await runner._dmbl_finalize_route_receipt(
        turn,
        agent_result={
            "provider": cap.CLOUD_SAFE_PROVIDER,
            "model": cap.CLOUD_SAFE_MODEL,
            "session_id": "session-fallback",
            "final_response": "cloud answer",
            "completed": True,
            "route_provenance_attempts": [
                {
                    "ordinal": 1,
                    "api_request_id": "cloud-api-1",
                    "provider": cap.CLOUD_SAFE_PROVIDER,
                    "model": cap.CLOUD_SAFE_MODEL,
                    "response_model": cap.CLOUD_SAFE_MODEL,
                    "transport": "agent",
                    "local_invocation": False,
                    "outcome": "returned",
                    "fallback_reason_code": cloud.reason.reason_code,
                    "fallback_from_model": requested.model,
                    "fallback_from_provider": requested.provider,
                }
            ],
        },
    )
    rec = _receipts(tmp_path)[-1]
    assert rec["requested_route"] == cap.HOME_FAST
    assert rec["requested_model"] == cap.HOME_FAST_MODEL
    assert rec["actual_provider"] == cap.CLOUD_SAFE_PROVIDER
    assert rec["actual_model"] == cap.CLOUD_SAFE_MODEL
    assert rec["fallback_reason_code"].startswith("fallback:home_fast:")
    assert rec["local_invocation"] is False
    assert rec["cloud_fallback"] is True


@pytest.mark.asyncio
async def test_RP6_local_only_failure_never_claims_cloud_fallback(env, tmp_path):
    runner, _adapter = _gw()
    event = _event("what time is it?")
    event.message_id = "rp-local-only-failure"
    session_key = runner._session_key_for_source(event.source)
    requested = cap.decide_route(event.text)
    receipt = await runner._dmbl_build_receipt_context(
        event=event,
        source=event.source,
        session_key=session_key,
        decision=requested,
        mode="home",
        pin_model=None,
        pin_provider=None,
        route_armed=None,
    )
    receipt["input_session_id"] = "session-local-only"
    turn = {
        "requested_decision": requested,
        "decision": None,
        "route_receipt": receipt,
        "route_fallback_reason": "local_only:LocalLoadError",
    }
    runner._dmbl_add_receipt_attempt(
        turn,
        provider=requested.provider,
        model=requested.model,
        transport="local_preflight",
        outcome="failed",
        local_invocation=False,
        error_class="LocalLoadError",
    )
    await runner._dmbl_finalize_route_receipt(
        turn,
        completion_outcome="local_only_failed",
        effective_session_id="session-local-only",
    )

    rec = _receipts(tmp_path)[-1]
    assert rec["local_invocation"] is False
    assert rec["cloud_fallback"] is False
    assert rec["completion_outcome"] == "local_only_failed"


@pytest.mark.asyncio
async def test_RP7_RP8_pin_receipt_then_normal_router_receipt(env, tmp_path):
    runner, _adapter = _gw()
    source = _source()
    session_key = runner._session_key_for_source(source)
    pin_model = "gpt-5.6-terra"
    pin_provider = "openai-codex"

    pinned_event = _event("hello")
    pinned_event.message_id = "rp-pin-active"
    pinned = cap.decide_route(
        pinned_event.text,
        mode="pinned",
        pinned_model=pin_model,
        pinned_provider=pin_provider,
    )
    pinned_receipt = await runner._dmbl_build_receipt_context(
        event=pinned_event,
        source=source,
        session_key=session_key,
        decision=pinned,
        mode="pinned",
        pin_model=pin_model,
        pin_provider=pin_provider,
        route_armed=None,
    )
    pinned_receipt["input_session_id"] = "session-pin"
    await runner._dmbl_finalize_route_receipt(
        {
            "requested_decision": pinned,
            "decision": pinned,
            "route_receipt": pinned_receipt,
        },
        agent_result={
            "provider": pin_provider,
            "model": pin_model,
            "session_id": "session-pin",
            "final_response": "pinned",
            "completed": True,
        },
    )

    normal_event = _event("hello")
    normal_event.message_id = "rp-pin-removed"
    normal = cap.decide_route(normal_event.text, mode="home")
    normal_receipt = await runner._dmbl_build_receipt_context(
        event=normal_event,
        source=source,
        session_key=session_key,
        decision=normal,
        mode="home",
        pin_model=None,
        pin_provider=None,
        route_armed=None,
    )
    normal_receipt["input_session_id"] = "session-pin"
    await runner._dmbl_finalize_route_receipt(
        {
            "requested_decision": normal,
            "decision": normal,
            "route_receipt": normal_receipt,
        },
        agent_result={
            "provider": normal.provider,
            "model": normal.model,
            "session_id": "session-pin",
            "final_response": "normal",
            "completed": True,
        },
    )

    first, second = _receipts(tmp_path)
    assert first["requested_route"] == cap.EXPLICIT_PIN
    assert first["override_type"] == "persistent_pin"
    assert first["override_model"] == pin_model
    assert first["pin_preserved"] is True
    assert second["requested_route"] == cap.HOME_FAST
    assert second["override_type"] == "none"
    assert second["pin_preserved"] is False


@pytest.mark.asyncio
async def test_RP9_RP10_RP11_RP12_retry_rollover_telegram_and_single_receipt(
    env, tmp_path, monkeypatch
):
    runner, _adapter = _gw()
    entry = MagicMock(
        session_id="child-session",
        metadata={
            "context_rollover": {
                "source_session": "parent-session",
                "reason": "oversized",
            }
        },
    )
    store = MagicMock()
    store.lookup_by_session_key = AsyncMock(return_value=entry)
    monkeypatch.setattr(
        gateway_run.GatewayRunner,
        "async_session_store",
        property(lambda self: store),
    )

    event = _event("what time is it?")
    event.message_id = "telegram-private-message"
    event.platform_update_id = 424242
    session_key = runner._session_key_for_source(event.source)
    requested = cap.decide_route(event.text)
    receipt = await runner._dmbl_build_receipt_context(
        event=event,
        source=event.source,
        session_key=session_key,
        decision=requested,
        mode="home",
        pin_model=None,
        pin_provider=None,
        route_armed=None,
    )
    receipt["input_session_id"] = "parent-session"
    turn = {
        "requested_decision": requested,
        "decision": requested,
        "route_receipt": receipt,
    }
    result = {
        "provider": cap.CLOUD_SAFE_PROVIDER,
        "model": cap.CLOUD_SAFE_MODEL,
        "session_id": "child-session",
        "final_response": "recovered",
        "completed": True,
        "route_provenance_turn_id": "parent-session:task:turn-1",
        "route_provenance_run_generation": 12,
        "route_provenance_attempts": [
            {
                "ordinal": 1,
                "api_request_id": "logical-api-1",
                "retry_index": 0,
                "provider": requested.provider,
                "model": requested.model,
                "transport": "agent",
                "local_invocation": True,
                "outcome": "error",
                "error_class": "TimeoutError",
            },
            {
                "ordinal": 2,
                "api_request_id": "logical-api-1",
                "retry_index": 1,
                "provider": cap.CLOUD_SAFE_PROVIDER,
                "model": cap.CLOUD_SAFE_MODEL,
                "response_model": cap.CLOUD_SAFE_MODEL,
                "transport": "agent",
                "local_invocation": False,
                "outcome": "returned",
                "fallback_reason_code": "timeout",
                "fallback_from_provider": requested.provider,
                "fallback_from_model": requested.model,
            },
        ],
        "route_provenance_fallbacks": [
            {
                "from_provider": requested.provider,
                "from_model": requested.model,
                "to_provider": cap.CLOUD_SAFE_PROVIDER,
                "to_model": cap.CLOUD_SAFE_MODEL,
                "reason_code": "timeout",
            }
        ],
    }
    await runner._dmbl_finalize_route_receipt(turn, agent_result=result)
    await runner._dmbl_finalize_route_receipt(turn, agent_result=result)

    receipts = _receipts(tmp_path)
    assert len(receipts) == 1
    rec = receipts[0]
    assert [a["ordinal"] for a in rec["attempts"]] == [1, 2]
    assert rec["attempts"][0]["api_request_sha256"] == rec["attempts"][1]["api_request_sha256"]
    assert rec["fallback_reason_code"] == "timeout"
    assert rec["actual_provider"] == cap.CLOUD_SAFE_PROVIDER
    assert rec["final_effective_model"] == cap.CLOUD_SAFE_MODEL
    assert rec["platform"] == "telegram"
    assert rec["platform_message_sha256"] == cap.hash_route_identifier(event.message_id)
    assert rec["platform_update_sha256"] == cap.hash_route_identifier(event.platform_update_id)
    assert rec["input_session_sha256"] == cap.hash_route_identifier("parent-session")
    assert rec["effective_session_sha256"] == cap.hash_route_identifier("child-session")
    assert rec["agent_turn_sha256"] == cap.hash_route_identifier(
        "parent-session:task:turn-1"
    )
    assert rec["rollover_occurred"] is True
    assert rec["rollover_parent_sha256"] == cap.hash_route_identifier("parent-session")
    assert rec["rollover_child_sha256"] == cap.hash_route_identifier("child-session")
    assert rec["rollover_reason"] == "oversized"
    serialized = json.dumps(rec)
    assert event.message_id not in serialized
    assert event.source.chat_id not in serialized


@pytest.mark.asyncio
async def test_RP10_persisted_rollover_stamp_does_not_relabel_later_child_turn(
    env, tmp_path, monkeypatch
):
    runner, _adapter = _gw()
    entry = MagicMock(
        session_id="child-session",
        metadata={
            "context_rollover": {
                "source_session": "parent-session",
                "reason": "oversized",
            }
        },
    )
    store = MagicMock()
    store.lookup_by_session_key = AsyncMock(return_value=entry)
    monkeypatch.setattr(
        gateway_run.GatewayRunner,
        "async_session_store",
        property(lambda self: store),
    )

    event = _event("hello after rollover")
    event.message_id = "post-rollover-child-turn"
    session_key = runner._session_key_for_source(event.source)
    requested = cap.decide_route(event.text)
    receipt = await runner._dmbl_build_receipt_context(
        event=event,
        source=event.source,
        session_key=session_key,
        decision=requested,
        mode="home",
        pin_model=None,
        pin_provider=None,
        route_armed=None,
    )
    assert receipt["input_session_id"] == "child-session"
    await runner._dmbl_finalize_route_receipt(
        {
            "requested_decision": requested,
            "decision": requested,
            "route_receipt": receipt,
        },
        agent_result={
            "provider": requested.provider,
            "model": requested.model,
            "session_id": "child-session",
            "final_response": "ok",
            "completed": True,
        },
    )

    rec = _receipts(tmp_path)[-1]
    assert rec["rollover_occurred"] is False
    assert "rollover_parent_sha256" not in rec
    assert "rollover_child_sha256" not in rec
    assert "rollover_reason" not in rec


@pytest.mark.asyncio
async def test_RP10_session_change_without_rollover_evidence_is_not_rollover(
    env, tmp_path, monkeypatch
):
    runner, _adapter = _gw()
    entry = MagicMock(session_id="reset-session", metadata={})
    store = MagicMock()
    store.lookup_by_session_key = AsyncMock(return_value=entry)
    monkeypatch.setattr(
        gateway_run.GatewayRunner,
        "async_session_store",
        property(lambda self: store),
    )

    event = _event("first message after an idle reset")
    event.message_id = "post-reset-message"
    session_key = runner._session_key_for_source(event.source)
    requested = cap.decide_route(event.text)
    receipt = await runner._dmbl_build_receipt_context(
        event=event,
        source=event.source,
        session_key=session_key,
        decision=requested,
        mode="home",
        pin_model=None,
        pin_provider=None,
        route_armed=None,
    )
    receipt["input_session_id"] = "pre-reset-session"
    await runner._dmbl_finalize_route_receipt(
        {
            "requested_decision": requested,
            "decision": requested,
            "route_receipt": receipt,
        },
        agent_result={
            "provider": requested.provider,
            "model": requested.model,
            "session_id": "reset-session",
            "final_response": "ok",
            "completed": True,
            "route_provenance_session_was_split": False,
        },
    )

    rec = _receipts(tmp_path)[-1]
    assert rec["input_session_sha256"] == cap.hash_route_identifier(
        "pre-reset-session"
    )
    assert rec["effective_session_sha256"] == cap.hash_route_identifier(
        "reset-session"
    )
    assert rec["rollover_occurred"] is False
    assert "rollover_parent_sha256" not in rec
    assert "rollover_child_sha256" not in rec
    assert "rollover_reason" not in rec


@pytest.mark.asyncio
async def test_RP10_agent_compression_signal_records_rollover_without_store_stamp(
    env, tmp_path, monkeypatch
):
    runner, _adapter = _gw()
    entry = MagicMock(session_id="compressed-child", metadata={})
    store = MagicMock()
    store.lookup_by_session_key = AsyncMock(return_value=entry)
    monkeypatch.setattr(
        gateway_run.GatewayRunner,
        "async_session_store",
        property(lambda self: store),
    )

    event = _event("continue after agent compression")
    event.message_id = "agent-compression-message"
    session_key = runner._session_key_for_source(event.source)
    requested = cap.decide_route(event.text)
    receipt = await runner._dmbl_build_receipt_context(
        event=event,
        source=event.source,
        session_key=session_key,
        decision=requested,
        mode="home",
        pin_model=None,
        pin_provider=None,
        route_armed=None,
    )
    receipt["input_session_id"] = "compression-parent"
    await runner._dmbl_finalize_route_receipt(
        {
            "requested_decision": requested,
            "decision": requested,
            "route_receipt": receipt,
        },
        agent_result={
            "provider": requested.provider,
            "model": requested.model,
            "session_id": "compressed-child",
            "final_response": "ok",
            "completed": True,
            "route_provenance_session_was_split": True,
        },
    )

    rec = _receipts(tmp_path)[-1]
    assert rec["rollover_occurred"] is True
    assert rec["rollover_parent_sha256"] == cap.hash_route_identifier(
        "compression-parent"
    )
    assert rec["rollover_child_sha256"] == cap.hash_route_identifier(
        "compressed-child"
    )
    assert rec["rollover_reason"] == "compression_split"


@pytest.mark.asyncio
async def test_RP6_remote_custom_fallback_is_cloud_not_local(env, tmp_path):
    runner, _adapter = _gw()
    event = _event("what time is it?")
    event.message_id = "remote-custom-fallback"
    session_key = runner._session_key_for_source(event.source)
    requested = cap.decide_route(event.text)
    receipt = await runner._dmbl_build_receipt_context(
        event=event,
        source=event.source,
        session_key=session_key,
        decision=requested,
        mode="home",
        pin_model=None,
        pin_provider=None,
        route_armed=None,
    )
    receipt["input_session_id"] = "remote-custom-session"
    turn = {
        "requested_decision": requested,
        "decision": requested,
        "route_receipt": receipt,
        "route_fallback_reason": "timeout",
    }
    await runner._dmbl_finalize_route_receipt(
        turn,
        agent_result={
            "provider": "custom:remote-https",
            "model": cap.HOME_FAST_MODEL,
            "session_id": "remote-custom-session",
            "final_response": "remote answer",
            "completed": True,
            "route_provenance_attempts": [
                {
                    "ordinal": 1,
                    "provider": "custom:remote-https",
                    "model": cap.HOME_FAST_MODEL,
                    "response_model": cap.HOME_FAST_MODEL,
                    "transport": "agent",
                    "local_invocation": False,
                    "outcome": "returned",
                    "fallback_reason_code": "timeout",
                }
            ],
        },
    )

    rec = _receipts(tmp_path)[-1]
    assert rec["actual_provider"] == "custom:remote-https"
    assert rec["local_invocation"] is False
    assert rec["cloud_fallback"] is True


@pytest.mark.asyncio
async def test_RP12_queued_turns_get_separate_correlated_receipts(env, tmp_path):
    runner, _adapter = _gw()
    first_event = _event("first queued-chain message")
    first_event.message_id = "queue-message-one"
    second_event = _event("second queued-chain message")
    second_event.message_id = "queue-message-two"
    session_key = runner._session_key_for_source(first_event.source)
    decision = cap.decide_route(first_event.text)
    first_receipt = await runner._dmbl_build_receipt_context(
        event=first_event,
        source=first_event.source,
        session_key=session_key,
        decision=decision,
        mode="home",
        pin_model=None,
        pin_provider=None,
        route_armed=None,
    )
    first_receipt["input_session_id"] = "queue-session"
    first_turn = {
        "requested_decision": decision,
        "decision": decision,
        "session_key": session_key,
        "route_receipt": first_receipt,
    }
    runner._dmbl_turns = {session_key: first_turn}
    first_result = {
        "provider": decision.provider,
        "model": decision.model,
        "session_id": "queue-session",
        "final_response": "first answer",
        "completed": True,
        "route_provenance_turn_id": "queue-turn-one",
        "route_provenance_attempts": [
            {
                "ordinal": 1,
                "provider": decision.provider,
                "model": decision.model,
                "response_model": decision.model,
                "transport": "agent",
                "local_invocation": True,
                "outcome": "returned",
            }
        ],
    }

    queued_turn = await runner._dmbl_transition_queued_route_receipt(
        current_session_key=session_key,
        current_agent_result=first_result,
        queued_event=second_event,
        queued_message=second_event.text,
        queued_message_id=second_event.message_id,
        queued_source=second_event.source,
        queued_session_key=session_key,
        run_generation=7,
    )
    assert queued_turn is runner._dmbl_turns[session_key]
    assert queued_turn["decision"] is decision

    second_result = {
        "provider": decision.provider,
        "model": decision.model,
        "session_id": "queue-session",
        "final_response": "second answer",
        "completed": True,
        "route_provenance_turn_id": "queue-turn-two",
        "route_provenance_attempts": [
            {
                "ordinal": 1,
                "provider": decision.provider,
                "model": decision.model,
                "response_model": decision.model,
                "transport": "agent",
                "local_invocation": True,
                "outcome": "returned",
            }
        ],
    }
    await runner._dmbl_finalize_route_receipt(
        queued_turn,
        agent_result=second_result,
        run_generation=7,
    )

    receipts = _receipts(tmp_path)
    assert len(receipts) == 2
    by_message = {rec["platform_message_sha256"]: rec for rec in receipts}
    first = by_message[cap.hash_route_identifier(first_event.message_id)]
    second = by_message[cap.hash_route_identifier(second_event.message_id)]
    assert first["agent_turn_sha256"] == cap.hash_route_identifier("queue-turn-one")
    assert second["agent_turn_sha256"] == cap.hash_route_identifier("queue-turn-two")
    assert first["receipt_id"] != second["receipt_id"]

    synthetic_first = await runner._dmbl_transition_queued_route_receipt(
        current_session_key=session_key,
        current_agent_result=second_result,
        queued_event=None,
        queued_message="repeated identifier-less queued text",
        queued_message_id=None,
        queued_source=second_event.source,
        queued_session_key=session_key,
        run_generation=7,
    )
    synthetic_first_result = {
        "provider": decision.provider,
        "model": decision.model,
        "session_id": "queue-session",
        "final_response": "third answer",
        "completed": True,
        "route_provenance_turn_id": "queue-turn-three",
        "route_provenance_attempts": second_result["route_provenance_attempts"],
    }
    synthetic_second = await runner._dmbl_transition_queued_route_receipt(
        current_session_key=session_key,
        current_agent_result=synthetic_first_result,
        queued_event=None,
        queued_message="repeated identifier-less queued text",
        queued_message_id=None,
        queued_source=second_event.source,
        queued_session_key=session_key,
        run_generation=7,
    )
    assert synthetic_first is not None
    assert synthetic_second is not None
    assert (
        synthetic_first["route_receipt"]["receipt_id"]
        != synthetic_second["route_receipt"]["receipt_id"]
    )


@pytest.mark.asyncio
async def test_RP12_cancelled_direct_specialist_finalizes_receipt_then_reraises(
    env, tmp_path, monkeypatch
):
    runner, _adapter = _gw()
    event = _event("Write a function that reverses a string in python")
    event.message_id = "cancelled-specialist-message"
    monkeypatch.setattr(
        runner,
        "_run_in_executor_with_context",
        AsyncMock(side_effect=asyncio.CancelledError()),
    )

    with pytest.raises(asyncio.CancelledError):
        await runner._handle_message(event)

    receipts = _receipts(tmp_path)
    assert len(receipts) == 1
    rec = receipts[0]
    assert rec["platform_message_sha256"] == cap.hash_route_identifier(
        event.message_id
    )
    assert rec["completion_outcome"] == "cancelled"
    assert rec["attempts"][-1]["outcome"] == "cancelled"
    assert rec["attempts"][-1]["error_class"] == "CancelledError"
