"""Gateway ingress contracts for structured named-protocol authority."""

from __future__ import annotations

import asyncio
import hashlib
from types import SimpleNamespace

import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.named_protocol_state import (
    METADATA_KEY,
    PROTOCOL_FABLE,
    STATUS_ACTIVE,
    STATUS_CLOSED,
    get_protocol_record,
)


def _runner():
    return object.__new__(gateway_run.GatewayRunner)


def _source(owner="12345", **overrides):
    values = {
        "platform": Platform.TELEGRAM,
        "chat_type": "dm",
        "thread_id": None,
        "profile": "default",
        "user_id": owner,
        "chat_id": owner,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _config(owner="12345"):
    key = f"agent:main:telegram:dm:{owner}"
    return {
        "plugins": {
            "entries": {
                "quantum-fleet-brain-bootstrap": {
                    "settings": {
                        "main_session_key_sha256": hashlib.sha256(
                            key.encode("utf-8")
                        ).hexdigest(),
                        "chairman_telegram_id_sha256": hashlib.sha256(
                            owner.encode("utf-8")
                        ).hexdigest(),
                    }
                }
            }
        }
    }


def test_chairman_authority_proof_is_exact_and_model_independent(monkeypatch):
    monkeypatch.setenv("DUMBLEDORE_ROUTER", "1")
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: _config())
    runner = _runner()
    event = SimpleNamespace(internal=False)
    key = "agent:main:telegram:dm:12345"

    assert runner._dmbl_protocol_chairman_verified(event, _source(), key) is True
    assert runner._dmbl_protocol_chairman_verified(
        event, _source(owner="99999"), "agent:main:telegram:dm:99999"
    ) is False
    assert runner._dmbl_protocol_chairman_verified(
        event, _source(chat_type="group"), key
    ) is False
    assert runner._dmbl_protocol_chairman_verified(
        event, _source(thread_id="topic-1"), key
    ) is False
    assert runner._dmbl_protocol_chairman_verified(
        SimpleNamespace(internal=True), _source(), key
    ) is False


class _MetadataStore:
    def __init__(self, entry, *, persist=True):
        self.entry = entry
        self.persist = persist
        self.calls = []

    async def set_session_metadata(self, session_key, key, value):
        self.calls.append((session_key, key, value))
        if self.persist:
            self.entry.metadata[key] = value
        return self.persist


def _install_async_store(runner, store):
    backing_store = object()
    runner.session_store = backing_store
    store._store = backing_store
    runner._async_session_store = store


def test_immutable_chairman_message_activates_and_closes_durable_state():
    runner = _runner()
    entry = SimpleNamespace(
        session_key="agent:main:telegram:dm:12345",
        metadata={},
    )
    store = _MetadataStore(entry)
    _install_async_store(runner, store)
    runner._dmbl_protocol_chairman_verified = lambda *args, **kwargs: True
    event = SimpleNamespace(internal=False, message_id="tg-1")
    source = _source()

    active = asyncio.run(
        runner._dmbl_apply_named_protocol_directive(
            event=event,
            source=source,
            session_entry=entry,
            immutable_chairman_text="Run FABLE on this candidate.",
        )
    )
    assert get_protocol_record(active, PROTOCOL_FABLE)["status"] == STATUS_ACTIVE
    assert store.calls[-1][1] == METADATA_KEY

    closed = asyncio.run(
        runner._dmbl_apply_named_protocol_directive(
            event=SimpleNamespace(internal=False, message_id="tg-2"),
            source=source,
            session_entry=entry,
            immutable_chairman_text="Stop FABLE.",
        )
    )
    assert get_protocol_record(closed, PROTOCOL_FABLE)["status"] == STATUS_CLOSED


def test_plugin_rewrite_and_task_shape_cannot_create_state():
    runner = _runner()
    entry = SimpleNamespace(
        session_key="agent:main:telegram:dm:12345",
        metadata={},
    )
    store = _MetadataStore(entry)
    _install_async_store(runner, store)
    runner._dmbl_protocol_chairman_verified = lambda *args, **kwargs: True

    state = asyncio.run(
        runner._dmbl_apply_named_protocol_directive(
            event=SimpleNamespace(
                internal=False,
                message_id="tg-1",
                text="Run FABLE.",  # rewritten event text is deliberately ignored
            ),
            source=_source(),
            session_entry=entry,
            immutable_chairman_text=(
                "Create a skill for accessing Google Workspace credentials."
            ),
        )
    )

    assert state["records"] == []
    assert store.calls == []


def test_protocol_transition_rejects_non_durable_persistence():
    runner = _runner()
    entry = SimpleNamespace(
        session_key="agent:main:telegram:dm:12345",
        metadata={},
    )
    _install_async_store(runner, _MetadataStore(entry, persist=False))
    runner._dmbl_protocol_chairman_verified = lambda *args, **kwargs: True

    with pytest.raises(RuntimeError, match="directive rejected"):
        asyncio.run(
            runner._dmbl_apply_named_protocol_directive(
                event=SimpleNamespace(internal=False, message_id="tg-1"),
                source=_source(),
                session_entry=entry,
                immutable_chairman_text="Run FABLE.",
            )
        )
    assert entry.metadata == {}
