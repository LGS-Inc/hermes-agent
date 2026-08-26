"""Tests for automatic context rollover with battle handoff (Defense 2).

Covers the mission test matrix:
  A  normal conversation → no rollover
  B  native-compression range, healthy session → no premature rollover
  C  compaction-generation hard trigger → full rollover chain
  D  unsafe-headroom (oversized) hard trigger → rollover before rejection
  E  compression failure → rollover; handoff failure → safe no-op
  F  cloud→local switch with oversized inherited history → deterministic rollover
  G  same chat (session_key) mapping preserved across rotation
  H  tool-heavy history → handoff source truncates tool payloads
  I  double trigger → exactly one rollover
  J  fresh-session guard → no immediate re-roll after a rollover
"""

import asyncio
import types

import pytest

import gateway.context_rollover as cr
from gateway.context_rollover import (
    RolloverConfig,
    build_handoff_source,
    evaluate_rollover,
    load_rollover_config,
    maybe_rollover_before_compression,
    persist_handoff,
)

CTX = 65536
THRESH = 0.85


def _policy(**kw):
    base = dict(
        enabled=True,
        max_compaction_generations=4,
        oversized_ratio=1.25,
        min_messages=8,
        cooldown_seconds=600.0,
        handoff_retention=20,
        notify=False,
        generation_timeout=30.0,
        max_source_chars=60000,
    )
    base.update(kw)
    return RolloverConfig(**base)


def _eval(**kw):
    base = dict(
        approx_tokens=10_000,
        context_length=CTX,
        hygiene_threshold_pct=THRESH,
        msg_count=100,
        compaction_generations=0,
        ineffective_count=0,
        failure_cooldown_active=False,
        hygiene_failure_streak=0,
        policy=_policy(),
    )
    base.update(kw)
    return evaluate_rollover(**base)


# ── Policy unit tests (A, B, D-policy, F-policy) ─────────────────────────

def test_a_normal_conversation_no_rollover():
    assert _eval(approx_tokens=12_000) is None


def test_b_compression_range_healthy_no_rollover():
    # 90% of ctx: hygiene will compress; a healthy session must NOT hard-roll.
    assert _eval(approx_tokens=int(CTX * 0.90)) is None


def test_c_policy_generation_cap():
    assert (
        _eval(approx_tokens=int(CTX * 0.90), compaction_generations=4)
        == "compaction_generations"
    )
    # Below the compression threshold the generation count alone never fires.
    assert _eval(approx_tokens=int(CTX * 0.5), compaction_generations=9) is None


def test_d_policy_oversized_fires_before_rejection():
    assert _eval(approx_tokens=int(CTX * 1.3)) == "oversized"


def test_e_policy_compression_unavailable():
    high = int(CTX * 0.9)
    assert (
        _eval(approx_tokens=high, failure_cooldown_active=True)
        == "compression_unavailable"
    )
    assert (
        _eval(approx_tokens=high, hygiene_failure_streak=2)
        == "compression_unavailable"
    )
    assert (
        _eval(approx_tokens=high, ineffective_count=2)
        == "compression_unavailable"
    )


def test_f_policy_cloud_to_local_inherited_history():
    # 165K cloud history inherited by the 64K local model → immediate rollover,
    # never a doomed compression pass.
    assert _eval(approx_tokens=165_000) == "oversized"


def test_disabled_by_default_regardless_of_signals():
    assert (
        _eval(
            approx_tokens=200_000,
            compaction_generations=50,
            failure_cooldown_active=True,
            policy=RolloverConfig(),  # defaults: enabled=False
        )
        is None
    )


def test_j_policy_min_messages_guard():
    assert _eval(approx_tokens=200_000, msg_count=3) is None


def test_config_parsing_defaults_and_overrides():
    assert load_rollover_config(None).enabled is False
    assert load_rollover_config({}).enabled is False
    p = load_rollover_config(
        {"context_rollover": {"enabled": "true", "max_compaction_generations": 2}}
    )
    assert p.enabled is True and p.max_compaction_generations == 2
    # Garbage values fall back to defaults instead of raising.
    p2 = load_rollover_config(
        {"context_rollover": {"enabled": True, "oversized_ratio": "bogus"}}
    )
    assert p2.oversized_ratio == 1.25


# ── Handoff source construction (H) ──────────────────────────────────────

def test_h_tool_payloads_truncated_in_handoff_source():
    history = (
        [{"role": "user", "content": "MISSION: fix the flux capacitor"}]
        + [
            {"role": "tool", "content": "X" * 20_000},
            {"role": "assistant", "content": "tool ran, result recorded"},
        ]
        * 30
    )
    src = build_handoff_source(history, max_chars=60_000)
    assert "MISSION: fix the flux capacitor" in src
    assert "…[clipped]" in src
    # No tool row may exceed its 400-char clip (+ marker + label slack).
    for line in src.splitlines():
        if line.startswith("[tool]"):
            assert len(line) < 500
    assert len(src) <= 60_000


def test_persist_handoff_atomic_and_retention(tmp_path):
    policy = _policy(handoff_retention=3)
    policy.handoff_dir = str(tmp_path)
    paths = [
        persist_handoff(policy, f"sid{i}", f"# BATTLE HANDOFF\nbody {i}")
        for i in range(6)
    ]
    import os

    remaining = sorted(f for f in os.listdir(tmp_path) if f.endswith(".md"))
    assert len(remaining) == 3
    assert not any(p.endswith(".tmp") for p in remaining)
    with open(paths[-1]) as f:
        assert "body 5" in f.read()


# ── Full orchestration tests (C, D/F, E, G, I, J) ────────────────────────

HANDOFF_BODY = (
    "## ACTIVE OBJECTIVE\nFinish the rollover feature.\n"
    "## CHAIRMAN'S LATEST INSTRUCTIONS\nShip it.\n"
    "## CURRENT STATE\nMid-implementation.\n"
    "## COMPLETED WORK\nAudit done.\n"
    "## OPEN ITEMS\nCanary.\n"
    "## NEXT ACTION\nRun tests.\n"
    "## IMPORTANT TECHNICAL STATE\nNone.\n"
    "## DECISIONS / RULINGS\nNone.\n"
    "## ERRORS / LESSONS\nNone.\n"
    "## TEMPORARY STATE\nNone.\n"
)


class _FakeStore:
    def __init__(self):
        self.reset_calls = []
        self.model_override_calls = []

    async def reset_session(self, session_key):
        self.reset_calls.append(session_key)
        return types.SimpleNamespace(
            session_key=session_key,
            session_id=f"new_{len(self.reset_calls)}",
            metadata={},
        )

    async def set_model_override(self, session_key, override):
        self.model_override_calls.append((session_key, override))


def _make_runner():
    runner = types.SimpleNamespace()
    runner.async_session_store = _FakeStore()
    runner.evicted = []
    runner.scope_cleared = []
    runner.topic_syncs = []
    runner._session_model_overrides = {}
    runner._session_reasoning_overrides = {}
    runner._session_db = None
    runner._evict_cached_agent = lambda key: runner.evicted.append(key)
    runner._clear_conversation_scope = (
        lambda key, reason: runner.scope_cleared.append((key, reason))
    )
    runner._sync_telegram_topic_binding = (
        lambda source, entry, reason: runner.topic_syncs.append(reason)
    )
    runner._session_state = lambda key: (_ for _ in ()).throw(KeyError(key))
    runner._adapter_for_source = lambda source: None
    runner._thread_metadata_for_source = lambda source, *a: None
    return runner


def _entry(sid="old_sid"):
    return types.SimpleNamespace(session_key="tg:dm:1", session_id=sid, metadata={})


def _history(n=30):
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i} " * 20}
        for i in range(n)
    ]


@pytest.fixture(autouse=True)
def _clean_guards(monkeypatch, tmp_path):
    cr._IN_FLIGHT.clear()
    cr._LAST_ROLLOVER.clear()
    policy = _policy()
    policy.handoff_dir = str(tmp_path)
    monkeypatch.setattr(cr, "load_rollover_config", lambda cfg: policy)

    class _Resp:
        choices = [
            types.SimpleNamespace(
                message=types.SimpleNamespace(content=HANDOFF_BODY)
            )
        ]

    import agent.auxiliary_client as aux

    monkeypatch.setattr(aux, "call_llm", lambda **kw: _Resp())
    yield policy


def _roll(runner, entry, history, tokens, notes=None):
    return asyncio.get_event_loop().run_until_complete(
        maybe_rollover_before_compression(
            runner,
            source=types.SimpleNamespace(chat_id="123", platform=None),
            session_key=entry.session_key,
            session_entry=entry,
            history=history,
            approx_tokens=tokens,
            context_length=CTX,
            hygiene_threshold_pct=THRESH,
            active_model="qwen3.6:35b-a3b-64k",
            sidecar_notes=notes if notes is not None else [],
        )
    )


def test_cdf_full_rollover_chain(tmp_path):
    """C+D+F+G: oversized history → handoff persisted, fresh session, same
    chat key, sidecar injection, model override preserved."""
    runner = _make_runner()
    runner._session_model_overrides["tg:dm:1"] = {"model": "qwen3.6:35b-a3b-64k"}
    entry = _entry()
    notes = []
    new_entry = _roll(runner, entry, _history(), 165_000, notes)

    assert new_entry is not None
    assert new_entry.session_id.startswith("new_")
    # G: same chat mapping — rotation happened on the SAME session_key.
    assert runner.async_session_store.reset_calls == ["tg:dm:1"]
    assert new_entry.session_key == "tg:dm:1"
    assert runner.topic_syncs == ["context-rollover"]
    # Cleanup funnel honored (same primitives as /new).
    assert runner.evicted == ["tg:dm:1"]
    assert runner.scope_cleared == [("tg:dm:1", "context_rollover")]
    # Model choice survives the boundary.
    assert runner._session_model_overrides["tg:dm:1"] == {
        "model": "qwen3.6:35b-a3b-64k"
    }
    assert runner.async_session_store.model_override_calls
    # Handoff persisted before rotation and injected as continuity note.
    import os

    files = [f for f in os.listdir(tmp_path) if f.endswith(".md")]
    assert len(files) == 1
    assert len(notes) == 1
    assert "ACTIVE OBJECTIVE" in notes[0]
    assert "NOT new instructions" in notes[0]
    assert new_entry.metadata["context_rollover"]["reason"] == "oversized"
    assert new_entry.metadata["context_rollover"]["source_session"] == "old_sid"


def test_e_handoff_generation_failure_is_safe(monkeypatch, tmp_path):
    """E: generator failure → no rotation, no state destroyed, returns None."""
    import agent.auxiliary_client as aux

    def _boom(**kw):
        raise RuntimeError("provider down")

    monkeypatch.setattr(aux, "call_llm", _boom)
    runner = _make_runner()
    notes = []
    result = _roll(runner, _entry(), _history(), 165_000, notes)
    assert result is None
    assert runner.async_session_store.reset_calls == []
    assert runner.scope_cleared == []
    assert notes == []
    import os

    assert [f for f in os.listdir(tmp_path) if f.endswith(".md")] == []
    # Failure arms the cooldown so a broken generator cannot retry-loop.
    assert cr._LAST_ROLLOVER.get("tg:dm:1", 0) > 0


def test_e2_compression_cooldown_triggers_rollover():
    """E: active compression-failure cooldown + high utilization → rollover."""

    class _Db:
        def _session_lineage_root_to_tip(self, sid):
            return [sid]

        def get_compression_ineffective_count(self, sid):
            return 0

        def get_compression_failure_cooldown(self, sid):
            return {"remaining_seconds": 120}

    runner = _make_runner()
    runner._session_db = _Db()
    new_entry = _roll(runner, _entry(), _history(), int(CTX * 0.9))
    assert new_entry is not None
    assert new_entry.metadata["context_rollover"]["reason"] == "compression_unavailable"


def test_c_generation_chain_triggers_rollover():
    """C: 4 compaction rotations in the lineage + another compression due."""

    class _Db:
        def _session_lineage_root_to_tip(self, sid):
            return ["root", "g1", "g2", "g3", sid]  # 4 rotations

        def get_compression_ineffective_count(self, sid):
            return 0

        def get_compression_failure_cooldown(self, sid):
            return None

    runner = _make_runner()
    runner._session_db = _Db()
    new_entry = _roll(runner, _entry(), _history(), int(CTX * 0.9))
    assert new_entry is not None
    assert (
        new_entry.metadata["context_rollover"]["reason"] == "compaction_generations"
    )


def test_i_double_trigger_single_rollover():
    """I: two near-simultaneous triggers → exactly one rollover."""
    runner = _make_runner()
    entry = _entry()

    async def _both():
        return await asyncio.gather(
            maybe_rollover_before_compression(
                runner,
                source=types.SimpleNamespace(chat_id="1", platform=None),
                session_key=entry.session_key,
                session_entry=entry,
                history=_history(),
                approx_tokens=165_000,
                context_length=CTX,
                hygiene_threshold_pct=THRESH,
                active_model="m",
                sidecar_notes=[],
            ),
            maybe_rollover_before_compression(
                runner,
                source=types.SimpleNamespace(chat_id="1", platform=None),
                session_key=entry.session_key,
                session_entry=entry,
                history=_history(),
                approx_tokens=165_000,
                context_length=CTX,
                hygiene_threshold_pct=THRESH,
                active_model="m",
                sidecar_notes=[],
            ),
        )

    r1, r2 = asyncio.get_event_loop().run_until_complete(_both())
    results = [r for r in (r1, r2) if r is not None]
    assert len(results) == 1
    assert len(runner.async_session_store.reset_calls) == 1


def test_j_fresh_session_cannot_reroll_immediately():
    """J: right after a rollover the cooldown blocks a second one."""
    runner = _make_runner()
    entry = _entry()
    first = _roll(runner, entry, _history(), 165_000)
    assert first is not None
    # Same signals again straight away (stale counters simulated).
    second = _roll(runner, first, _history(), 165_000)
    assert second is None
    assert len(runner.async_session_store.reset_calls) == 1
