"""Gateway integration tests for the Dumbledore capability router.

Drives the REAL ``GatewayRunner._handle_message`` / ``_resolve_session_agent_runtime``
with every accelerator side effect (Ollama load/unload, ComfyUI, specialist
inference) stubbed at the module boundary of ``agent.dumbledore_capability_router``.
"""
from __future__ import annotations

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
    assert _sent(adapter)[-1] == "qwen2.5-coder:14b says hi"


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
