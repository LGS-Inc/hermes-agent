"""Session persistence contracts for P1 named-protocol continuity."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.named_protocol_state import (
    METADATA_KEY,
    PROTOCOL_FABLE,
    STATUS_CLOSED,
    get_protocol_record,
    rollover_metadata,
    transition_from_chairman_message,
)
from gateway.session import SessionSource, SessionStore


def _closed_fable_state():
    active = transition_from_chairman_message(
        None,
        "Run FABLE.",
        chairman_verified=True,
        activation_message_id="fixture-activate",
        now=datetime(2026, 8, 28, tzinfo=timezone.utc),
        activation_id_factory=lambda: "a" * 32,
    ).state
    return transition_from_chairman_message(
        active,
        "Stop FABLE.",
        chairman_verified=True,
        activation_message_id="fixture-close",
        now=datetime(2026, 8, 28, 0, 1, tzinfo=timezone.utc),
    ).state


def _source():
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        user_id="12345",
    )


def _store(path):
    store = SessionStore(sessions_dir=path, config=GatewayConfig())
    store._db = None
    return store


def test_protocol_metadata_survives_restart_and_explicit_rollover_reset(tmp_path):
    state = _closed_fable_state()
    store = _store(tmp_path)
    entry = store.get_or_create_session(_source())
    assert store.set_session_metadata(entry.session_key, METADATA_KEY, state)

    restarted = _store(tmp_path)
    durable = restarted.get_session_metadata(entry.session_key, METADATA_KEY)
    assert get_protocol_record(durable, PROTOCOL_FABLE)["status"] == STATUS_CLOSED

    child_metadata = rollover_metadata(
        {METADATA_KEY: durable, "unrelated": "drop-me"},
        {"source_session": entry.session_id, "reason": "fixture"},
    )
    child = restarted.reset_session(
        entry.session_key,
        initial_metadata=child_metadata,
    )
    assert "unrelated" not in child.metadata
    assert get_protocol_record(
        child.metadata[METADATA_KEY], PROTOCOL_FABLE
    )["status"] == STATUS_CLOSED

    restarted_again = _store(tmp_path)
    final = restarted_again.get_session_metadata(entry.session_key, METADATA_KEY)
    assert get_protocol_record(final, PROTOCOL_FABLE)["status"] == STATUS_CLOSED


def test_ordinary_reset_remains_a_clean_protocol_boundary(tmp_path):
    store = _store(tmp_path)
    entry = store.get_or_create_session(_source())
    assert store.set_session_metadata(
        entry.session_key,
        METADATA_KEY,
        _closed_fable_state(),
    )

    reset = store.reset_session(entry.session_key)

    assert METADATA_KEY not in reset.metadata


def test_metadata_write_is_failure_atomic(tmp_path, monkeypatch):
    store = _store(tmp_path)
    entry = store.get_or_create_session(_source())
    original = dict(entry.metadata)

    def fail_save(*args, **kwargs):
        raise OSError("simulated durable-store failure")

    monkeypatch.setattr(store, "_save_entry", fail_save)
    with pytest.raises(OSError, match="durable-store failure"):
        store.set_session_metadata(
            entry.session_key,
            METADATA_KEY,
            _closed_fable_state(),
        )

    assert entry.metadata == original


def test_metadata_limits_reject_non_json_and_oversized_values(tmp_path):
    store = _store(tmp_path)
    entry = store.get_or_create_session(_source())

    with pytest.raises(ValueError, match="JSON-serializable"):
        store.set_session_metadata(entry.session_key, "bad", {"x": object()})
    with pytest.raises(ValueError, match="16 KiB"):
        store.set_session_metadata(entry.session_key, "large", "x" * 20_000)
