"""Automatic context rollover with battle handoff (Dumbledore Defense 2).

Native Hermes compression (agent compressor + gateway session hygiene) is
Defense 1 and remains untouched. This module adds a config-gated hard
defense: when compression can no longer be trusted to keep a conversation
healthy — too many compaction generations, ineffective/failed compression,
or a transcript grossly oversized for the active model (cloud→local model
switches) — the gateway:

  1. generates a structured BATTLE HANDOFF from the old transcript,
  2. persists it to disk BEFORE any state is touched,
  3. rotates to a genuinely fresh session using the same primitives as
     ``/new`` (reset_session + conversation-scope clear + agent eviction),
  4. re-applies the session's model/reasoning override so the user keeps
     their selected model,
  5. injects the handoff into the fresh session's first turn as a sidecar
     note, and
  6. continues in the SAME chat — the user never types /new.

Config (config.yaml, all optional — the feature is OFF unless enabled):

    context_rollover:
      enabled: true
      max_compaction_generations: 4   # rotations before hard rollover
      oversized_ratio: 1.25           # tokens >= ratio*ctx → immediate rollover
      min_messages: 8                 # fresh-session guard
      cooldown_seconds: 600           # min seconds between rollovers per chat
      handoff_retention: 20           # files kept in handoff_dir
      handoff_dir: ~/.hermes/handoffs/dumbledore
      notify: true                    # brief user-facing notice
      generation_timeout: 120         # seconds for the handoff LLM call
      max_source_chars: 60000         # transcript slice cap fed to the LLM

The handoff LLM call runs as auxiliary task ``battle_handoff`` so
``auxiliary.battle_handoff.provider/model`` chooses the summarizer
(recommended: a large-context cloud model; the oversized-transcript case is
exactly when a small local model cannot do this job).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Module-level rollover guards. Keyed by session_key (the stable chat
# identity). _IN_FLIGHT prevents two concurrent rollovers for one chat;
# _LAST_ROLLOVER enforces the cooldown even if entry metadata is lost.
_IN_FLIGHT: set = set()
_LAST_ROLLOVER: Dict[str, float] = {}

_HANDOFF_SECTIONS = (
    "ACTIVE OBJECTIVE",
    "CHAIRMAN'S LATEST INSTRUCTIONS",
    "CURRENT STATE",
    "COMPLETED WORK",
    "OPEN ITEMS",
    "NEXT ACTION",
    "IMPORTANT TECHNICAL STATE",
    "DECISIONS / RULINGS",
    "ERRORS / LESSONS",
    "TEMPORARY STATE",
)


@dataclass
class RolloverConfig:
    enabled: bool = False
    max_compaction_generations: int = 4
    oversized_ratio: float = 1.25
    min_messages: int = 8
    cooldown_seconds: float = 600.0
    handoff_retention: int = 20
    handoff_dir: str = "~/.hermes/handoffs/dumbledore"
    notify: bool = True
    generation_timeout: float = 120.0
    max_source_chars: int = 60000


def load_rollover_config(cfg: Optional[dict]) -> RolloverConfig:
    """Parse the ``context_rollover`` config section. Defaults preserve
    current fleet behavior (disabled)."""
    out = RolloverConfig()
    section = (cfg or {}).get("context_rollover")
    if not isinstance(section, dict):
        return out
    def _num(key, cur, cast, lo):
        try:
            v = cast(section.get(key, cur))
            return v if v >= lo else cur
        except (TypeError, ValueError):
            return cur
    out.enabled = str(section.get("enabled", False)).lower() in {"true", "1", "yes"}
    out.max_compaction_generations = _num("max_compaction_generations", out.max_compaction_generations, int, 1)
    out.oversized_ratio = _num("oversized_ratio", out.oversized_ratio, float, 1.0)
    out.min_messages = _num("min_messages", out.min_messages, int, 1)
    out.cooldown_seconds = _num("cooldown_seconds", out.cooldown_seconds, float, 0)
    out.handoff_retention = _num("handoff_retention", out.handoff_retention, int, 1)
    out.generation_timeout = _num("generation_timeout", out.generation_timeout, float, 5)
    out.max_source_chars = _num("max_source_chars", out.max_source_chars, int, 4000)
    if isinstance(section.get("handoff_dir"), str) and section["handoff_dir"].strip():
        out.handoff_dir = section["handoff_dir"].strip()
    if "notify" in section:
        out.notify = str(section.get("notify", True)).lower() in {"true", "1", "yes"}
    return out


def evaluate_rollover(
    *,
    approx_tokens: int,
    context_length: int,
    hygiene_threshold_pct: float,
    msg_count: int,
    compaction_generations: int,
    ineffective_count: int,
    failure_cooldown_active: bool,
    hygiene_failure_streak: int,
    policy: RolloverConfig,
) -> Optional[str]:
    """Pure trigger policy. Returns a reason string, or None (no rollover).

    Triggers (checked in order):
      oversized               — transcript >= oversized_ratio * ctx. The
                                cloud→local switch case: compressing a
                                grossly oversized history through the small
                                model is slow and lossy; roll instead.
      compaction_generations  — the conversation has already been rotated by
                                compression >= N times AND it needs another
                                pass. This is the 40+-compressions spiral cap.
      compression_unavailable — compression is needed but cannot be trusted:
                                active failure cooldown, repeated hygiene
                                failures, or the anti-thrash ineffective
                                counter shows compaction stopped reclaiming.
    """
    if not policy.enabled or context_length <= 0:
        return None
    if msg_count < policy.min_messages:
        return None  # fresh-session guard: never roll a young session

    utilization = approx_tokens / float(context_length)
    if utilization >= policy.oversized_ratio:
        return "oversized"

    needs_compress = approx_tokens >= int(context_length * hygiene_threshold_pct)
    if not needs_compress:
        return None
    if compaction_generations >= policy.max_compaction_generations:
        return "compaction_generations"
    if failure_cooldown_active or hygiene_failure_streak >= 2 or ineffective_count >= 2:
        return "compression_unavailable"
    return None


def count_compaction_generations(session_db: Any, session_id: str) -> int:
    """Rotations this conversation has been through (parent-chain length - 1).

    Context compression rotates ``session_id`` into a child linked via
    ``parent_session_id``; the lineage length is therefore the number of
    major compactions survived. 0 when unavailable."""
    try:
        db = getattr(session_db, "_db", session_db)
        chain = db._session_lineage_root_to_tip(session_id)
        return max(0, len(chain) - 1)
    except Exception:
        return 0


# ── Handoff generation ────────────────────────────────────────────────────

def _clip(text: str, limit: int) -> str:
    text = text if isinstance(text, str) else str(text)
    return text if len(text) <= limit else text[:limit] + " …[clipped]"


def _msg_text(msg: dict) -> str:
    content = msg.get("content")
    if isinstance(content, list):  # multimodal parts
        content = " ".join(
            str(p.get("text", "")) for p in content if isinstance(p, dict)
        )
    return content if isinstance(content, str) else ("" if content is None else str(content))


def build_handoff_source(history: List[dict], max_chars: int) -> str:
    """Bounded transcript slice: mission head + recent tail, tool payloads
    truncated hard. Never the whole transcript."""
    rows: List[str] = []
    usable = [
        m for m in history
        if isinstance(m, dict) and m.get("role") in ("user", "assistant", "tool")
    ]
    head = usable[:4]
    tail = usable[-40:]
    seen_ids = set()
    for m in head + tail:
        mid = id(m)
        if mid in seen_ids:
            continue
        seen_ids.add(mid)
        role = m.get("role")
        limit = 400 if role == "tool" else 2500
        text = _clip(_msg_text(m).strip(), limit)
        if not text and role != "tool":
            continue
        rows.append(f"[{role}] {text}")
    src = "\n".join(rows)
    return src[-max_chars:] if len(src) > max_chars else src


def build_handoff_messages(history: List[dict], *, max_chars: int) -> List[dict]:
    source = build_handoff_source(history, max_chars)
    sections = "\n".join(f"## {s}" for s in _HANDOFF_SECTIONS)
    instruction = (
        "You are generating a BATTLE HANDOFF: a concise continuity brief for "
        "an AI assistant whose conversation context is about to be reset. A "
        "fresh session will receive ONLY this handoff (plus its normal system "
        "instructions). Extract continuity-critical state from the transcript "
        "excerpt below and write the handoff using EXACTLY these markdown "
        "sections, in this order:\n\n"
        f"{sections}\n\n"
        "Rules: be concise and operational; include concrete paths, hosts, "
        "branches, SHAs, model names, IDs, ports, and service names ONLY "
        "where relevant; record decisions already made so they are not "
        "reopened; record failures already hit so they are not repeated; do "
        "NOT copy tool logs or transcripts wholesale; do NOT include "
        "credentials, API keys, or tokens; if a section has nothing, write "
        "'None.'. Output only the handoff markdown."
    )
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": f"TRANSCRIPT EXCERPT (oldest first):\n\n{source}"},
    ]


def generate_handoff_text(
    history: List[dict],
    *,
    policy: RolloverConfig,
    reason: str,
    old_session_id: str,
    active_model: str,
    approx_tokens: int,
    context_length: int,
) -> str:
    """Blocking. Raises on failure — caller treats any exception as
    'rollover unavailable this turn' and leaves the session untouched."""
    from agent.auxiliary_client import call_llm
    from agent.redact import redact_sensitive_text

    response = call_llm(
        task="battle_handoff",
        messages=build_handoff_messages(history, max_chars=policy.max_source_chars),
        max_tokens=2000,
        timeout=policy.generation_timeout,
    )
    content = ""
    try:
        content = (response.choices[0].message.content or "").strip()
    except Exception:
        content = str(response or "").strip()
    # Strip inline reasoning if a thinking model answered.
    content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL).strip()
    if len(content) < 200 or "ACTIVE OBJECTIVE" not in content:
        raise RuntimeError(
            f"battle handoff generation returned unusable output "
            f"({len(content)} chars)"
        )
    content = redact_sensitive_text(content, force=True, redact_url_credentials=True)
    header = (
        f"# BATTLE HANDOFF\n"
        f"- source_session: {old_session_id}\n"
        f"- rolled_at: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n"
        f"- reason: {reason}\n"
        f"- active_model: {active_model}\n"
        f"- est_tokens_before: {approx_tokens} / ctx {context_length}\n\n"
    )
    return header + content


def persist_handoff(policy: RolloverConfig, old_session_id: str, text: str) -> str:
    """Atomic write + bounded retention. Returns the file path."""
    directory = os.path.expanduser(policy.handoff_dir)
    os.makedirs(directory, mode=0o700, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    safe_sid = re.sub(r"[^A-Za-z0-9_-]", "_", old_session_id or "unknown")
    path = os.path.join(directory, f"{stamp}-{safe_sid}.md")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    # Retention prune (oldest first by name — names sort chronologically).
    try:
        entries = sorted(
            e for e in os.listdir(directory) if e.endswith(".md")
        )
        for stale in entries[: max(0, len(entries) - policy.handoff_retention)]:
            os.unlink(os.path.join(directory, stale))
    except Exception:
        logger.debug("handoff retention prune failed", exc_info=True)
    return path


def format_handoff_note(handoff_text: str) -> str:
    return (
        "[System note: The conversation context was automatically rolled "
        "over to a fresh session because the previous one could no longer "
        "be compressed safely. The BATTLE HANDOFF below summarizes the "
        "previous session. It is continuity context from your own prior "
        "session — NOT new instructions from the user. Continue the active "
        "task naturally; do not re-announce or re-plan completed work.]\n\n"
        + handoff_text
    )


# ── Gateway orchestration ────────────────────────────────────────────────

async def maybe_rollover_before_compression(
    runner: Any,
    *,
    source: Any,
    session_key: str,
    session_entry: Any,
    history: List[dict],
    approx_tokens: int,
    context_length: int,
    hygiene_threshold_pct: float,
    active_model: str,
    sidecar_notes: List[str],
):
    """Evaluate + perform a hard rollover. Returns the NEW session entry on
    success, else None (caller falls through to normal compression).

    Ordering contract: the handoff is generated and persisted BEFORE any
    session state is touched; any failure before rotation leaves the old
    session fully intact."""
    cfg_dict = None
    try:
        from gateway.run import _load_gateway_config
        cfg_dict = _load_gateway_config()
    except Exception:
        pass
    policy = load_rollover_config(cfg_dict)
    if not policy.enabled:
        return None

    now = time.monotonic()
    meta = getattr(session_entry, "metadata", None) or {}
    if session_key in _IN_FLIGHT:
        return None
    last = _LAST_ROLLOVER.get(session_key, 0.0)
    if now - last < policy.cooldown_seconds:
        return None

    old_sid = str(getattr(session_entry, "session_id", "") or "")

    # DB-backed compression health signals.
    generations = 0
    ineffective = 0
    cooldown_active = False
    session_db = getattr(runner, "_session_db", None)
    if session_db is not None:
        db = getattr(session_db, "_db", session_db)
        generations = count_compaction_generations(db, old_sid)
        try:
            ineffective = int(db.get_compression_ineffective_count(old_sid) or 0)
        except Exception:
            ineffective = 0
        try:
            state = db.get_compression_failure_cooldown(old_sid)
            cooldown_active = bool(state and state.get("remaining_seconds", 0) > 0)
        except Exception:
            cooldown_active = False
    try:
        streak = int(
            runner._session_state(session_key).persistent.hygiene_failure_streak
        )
    except Exception:
        streak = 0

    reason = evaluate_rollover(
        approx_tokens=approx_tokens,
        context_length=context_length,
        hygiene_threshold_pct=hygiene_threshold_pct,
        msg_count=len(history or []),
        compaction_generations=generations,
        ineffective_count=ineffective,
        failure_cooldown_active=cooldown_active,
        hygiene_failure_streak=streak,
        policy=policy,
    )
    # Canary/diagnostic instrumentation: a one-shot force trigger, so a live
    # canary can prove the complete rollover chain WITHOUT lowering any
    # production threshold. Armed by writing the target session_key (or a
    # substring of it) to <handoff_dir>/../ROLLOVER_FORCE, or by setting
    # HERMES_ROLLOVER_FORCE_SESSION. The flag file is consumed (deleted) the
    # moment it matches, making it strictly one-shot; both are absent in
    # normal production operation.
    if reason is None and len(history or []) >= 2:
        _force = os.environ.get("HERMES_ROLLOVER_FORCE_SESSION", "").strip()
        _flag_path = os.path.join(
            os.path.dirname(os.path.expanduser(policy.handoff_dir)),
            "ROLLOVER_FORCE",
        )
        if not _force:
            try:
                with open(_flag_path, "r", encoding="utf-8") as f:
                    _force = f.read().strip()
            except OSError:
                _force = ""
            if _force and _force in session_key:
                try:
                    os.unlink(_flag_path)  # consume: strictly one-shot
                except OSError:
                    pass
        if _force and _force in session_key:
            reason = "canary_forced"
    if reason is None:
        return None

    _IN_FLIGHT.add(session_key)
    started = time.monotonic()
    try:
        logger.info(
            "Context rollover triggered: key=%s old_sid=%s reason=%s model=%s "
            "tokens=~%s ctx=%s generations=%s ineffective=%s cooldown=%s streak=%s",
            session_key, old_sid, reason, active_model,
            f"{approx_tokens:,}", f"{context_length:,}",
            generations, ineffective, cooldown_active, streak,
        )

        # 1. Generate + persist the handoff FIRST. Failure → no rollover.
        # The LLM-authored prose is followed by a deterministic machine block;
        # rollover never attempts to infer protocol state from the prose.
        handoff = await asyncio.to_thread(
            generate_handoff_text,
            history,
            policy=policy,
            reason=reason,
            old_session_id=old_sid,
            active_model=active_model,
            approx_tokens=approx_tokens,
            context_length=context_length,
        )
        from gateway.named_protocol_state import (
            METADATA_KEY as _PROTOCOL_STATE_KEY,
            append_machine_state_block,
            normalize_envelope,
            rollover_metadata,
        )

        protocol_state = normalize_envelope(meta.get(_PROTOCOL_STATE_KEY))
        handoff = append_machine_state_block(handoff, protocol_state)
        handoff_path = await asyncio.to_thread(
            persist_handoff, policy, old_sid, handoff
        )

        # 2. Snapshot per-conversation overrides worth carrying across the
        #    boundary (the user's /model choice must survive — losing it was
        #    a documented pain point of /new).
        model_override = dict(
            (getattr(runner, "_session_model_overrides", {}) or {}).get(session_key) or {}
        )
        reasoning_override = (
            getattr(runner, "_session_reasoning_overrides", {}) or {}
        ).get(session_key)
        if isinstance(reasoning_override, dict):
            reasoning_override = dict(reasoning_override)

        # 3. Rotate — same primitive chain as /new and the
        #    compression-exhausted reset (reset → evict → scope clear). Only
        #    validated named-protocol state and the rollover stamp cross this
        #    boundary; unrelated metadata does not.
        rollover_stamp = {
            "at": time.time(),
            "reason": reason,
            "source_session": old_sid,
            "handoff_path": handoff_path,
        }
        child_metadata = rollover_metadata(meta, rollover_stamp)
        new_entry = await runner.async_session_store.reset_session(
            session_key,
            initial_metadata=child_metadata,
        )
        if new_entry is None:
            raise RuntimeError("reset_session returned no new entry")
        runner._evict_cached_agent(session_key)
        runner._clear_conversation_scope(session_key, reason="context_rollover")

        # 4. Restore the model/reasoning selection onto the fresh conversation.
        if model_override:
            runner._session_model_overrides[session_key] = model_override
            try:
                await runner.async_session_store.set_model_override(
                    session_key, model_override
                )
            except Exception:
                logger.debug("rollover: model override persist failed", exc_info=True)
        if reasoning_override:
            try:
                runner._session_reasoning_overrides[session_key] = reasoning_override
            except Exception:
                pass

        # 5. Telegram topic-binding heal (same as compression-exhausted reset;
        #    no-op on non-topic lanes).
        try:
            await asyncio.to_thread(
                runner._sync_telegram_topic_binding,
                source, new_entry, reason="context-rollover",
            )
        except Exception:
            logger.debug("rollover: topic binding sync failed", exc_info=True)

        # 6. Clear hygiene failure streak for the chat — fresh conversation.
        try:
            from gateway.run import _reset_hygiene_failure_streak
            await asyncio.to_thread(_reset_hygiene_failure_streak, runner, session_key)
        except Exception:
            pass

        # 7. Stamp the process-local cooldown guard. The durable rollover stamp
        #    and protocol envelope were written atomically with reset_session.
        _LAST_ROLLOVER[session_key] = time.monotonic()

        # 8. Inject the handoff into this very turn. The full-agent system
        #    state remains authoritative over the free-form handoff prose.
        sidecar_notes.append(format_handoff_note(handoff))

        # 9. Brief user notice (optional).
        if policy.notify:
            try:
                adapter = runner._adapter_for_source(source)
                if adapter and getattr(source, "chat_id", None):
                    await adapter.send(
                        source.chat_id,
                        "♻️ Context refreshed. Continuing from the saved handoff.",
                        metadata=runner._thread_metadata_for_source(source),
                    )
            except Exception:
                logger.debug("rollover: user notice failed", exc_info=True)

        logger.info(
            "Context rollover complete: key=%s old_sid=%s new_sid=%s reason=%s "
            "handoff=%s duration=%.1fs",
            session_key, old_sid,
            getattr(new_entry, "session_id", "?"), reason,
            handoff_path, time.monotonic() - started,
        )
        return new_entry
    except Exception as exc:
        # Fail visibly, destroy nothing: old session untouched unless
        # rotation already happened (steps after reset_session are all
        # individually guarded and non-fatal).
        logger.warning(
            "Context rollover failed for %s (reason=%s): %s — session left "
            "intact; normal compression path continues.",
            session_key, reason, exc,
        )
        _LAST_ROLLOVER[session_key] = time.monotonic()  # back off retries
        return None
    finally:
        _IN_FLIGHT.discard(session_key)
