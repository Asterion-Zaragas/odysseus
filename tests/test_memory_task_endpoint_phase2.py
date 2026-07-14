"""Phase 2 of the memory upgrade: model-role plumbing.

Covers `resolve_memory_candidates` / `memory_llm_call_async` (src/task_endpoint.py)
and the new settings surface (src/settings.py, routes/auth_routes.py int clamps).
"""

import pytest

from src import task_endpoint
from src.settings import DEFAULT_SETTINGS, _PER_USER_KEYS


# ── resolve_memory_candidates ──

def test_unknown_role_raises():
    with pytest.raises(ValueError):
        task_endpoint.resolve_memory_candidates("medium")


def test_prefers_configured_role_endpoint_over_task_chain(monkeypatch):
    def fake_resolve_endpoint(prefix, *args, **kwargs):
        assert prefix == "memory_fast"
        return ("http://fast-llm", "fast-model", {"x": "1"})

    def fake_task_candidates(**kwargs):
        return [("http://task-llm", "task-model", {})]

    monkeypatch.setattr(task_endpoint, "resolve_endpoint", fake_resolve_endpoint)
    monkeypatch.setattr(task_endpoint, "resolve_task_candidates", fake_task_candidates)

    candidates = task_endpoint.resolve_memory_candidates("fast")
    assert candidates[0] == ("http://fast-llm", "fast-model", {"x": "1"})
    assert candidates[1] == ("http://task-llm", "task-model", {})


def test_falls_back_to_task_chain_when_role_unconfigured(monkeypatch):
    monkeypatch.setattr(task_endpoint, "resolve_endpoint", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(
        task_endpoint, "resolve_task_candidates",
        lambda **k: [("http://task-llm", "task-model", {})],
    )

    candidates = task_endpoint.resolve_memory_candidates("smart")
    assert candidates == [("http://task-llm", "task-model", {})]


def test_dedupes_role_endpoint_against_task_chain(monkeypatch):
    same = ("http://llm", "model")
    monkeypatch.setattr(task_endpoint, "resolve_endpoint", lambda *a, **k: (*same, {}))
    monkeypatch.setattr(
        task_endpoint, "resolve_task_candidates",
        lambda **k: [(*same, {}), ("http://other", "other-model", {})],
    )

    candidates = task_endpoint.resolve_memory_candidates("fast")
    assert candidates == [
        ("http://llm", "model", {}),
        ("http://other", "other-model", {}),
    ]


def test_owner_forwarded_to_both_resolvers(monkeypatch):
    seen = {}

    def fake_resolve_endpoint(prefix, *args, **kwargs):
        seen["endpoint_owner"] = kwargs.get("owner")
        return (None, None, None)

    def fake_task_candidates(**kwargs):
        seen["task_owner"] = kwargs.get("owner")
        return []

    monkeypatch.setattr(task_endpoint, "resolve_endpoint", fake_resolve_endpoint)
    monkeypatch.setattr(task_endpoint, "resolve_task_candidates", fake_task_candidates)

    task_endpoint.resolve_memory_candidates("smart", owner="alice")
    assert seen == {"endpoint_owner": "alice", "task_owner": "alice"}


# ── memory_llm_call_async ──

@pytest.mark.asyncio
async def test_raises_when_no_candidates(monkeypatch):
    monkeypatch.setattr(task_endpoint, "resolve_memory_candidates", lambda *a, **k: [])
    with pytest.raises(RuntimeError):
        await task_endpoint.memory_llm_call_async("fast", [{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_waits_for_interactive_quiet_by_default(monkeypatch):
    waited = []

    async def fake_wait(label):
        waited.append(label)
        return False

    async def fake_llm_call(candidates, messages, **kwargs):
        return "ok"

    monkeypatch.setattr(
        task_endpoint, "resolve_memory_candidates",
        lambda *a, **k: [("http://llm", "model", {})],
    )
    monkeypatch.setattr(task_endpoint, "wait_for_interactive_quiet", fake_wait)
    monkeypatch.setattr(task_endpoint, "llm_call_async_with_fallback", fake_llm_call)

    result = await task_endpoint.memory_llm_call_async("smart", [{"role": "user", "content": "hi"}])
    assert result == "ok"
    assert len(waited) == 1


@pytest.mark.asyncio
async def test_interactive_flag_skips_the_gate(monkeypatch):
    waited = []

    async def fake_wait(label):
        waited.append(label)
        return False

    async def fake_llm_call(candidates, messages, **kwargs):
        return "ok"

    monkeypatch.setattr(
        task_endpoint, "resolve_memory_candidates",
        lambda *a, **k: [("http://llm", "model", {})],
    )
    monkeypatch.setattr(task_endpoint, "wait_for_interactive_quiet", fake_wait)
    monkeypatch.setattr(task_endpoint, "llm_call_async_with_fallback", fake_llm_call)

    await task_endpoint.memory_llm_call_async(
        "fast", [{"role": "user", "content": "hi"}], interactive=True
    )
    assert waited == []


@pytest.mark.asyncio
async def test_workload_defaults_to_background(monkeypatch):
    captured = {}

    async def fake_wait(label):
        return False

    async def fake_llm_call(candidates, messages, **kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(
        task_endpoint, "resolve_memory_candidates",
        lambda *a, **k: [("http://llm", "model", {})],
    )
    monkeypatch.setattr(task_endpoint, "wait_for_interactive_quiet", fake_wait)
    monkeypatch.setattr(task_endpoint, "llm_call_async_with_fallback", fake_llm_call)

    await task_endpoint.memory_llm_call_async("fast", [{"role": "user", "content": "hi"}])
    assert captured.get("workload") == "background"


@pytest.mark.asyncio
async def test_interactive_workload_is_foreground(monkeypatch):
    # interactive=True must clear BOTH foreground gates: skip
    # wait_for_interactive_quiet AND go out as workload="foreground", or
    # llm_core's _local_model_slot parks the call in its background
    # wait-for-quiet loop — which self-deadlocks against the chat request
    # the call runs inline in (the live facet-timeout bug, 2026-07-14).
    captured = {}

    async def fake_wait(label):
        return False

    async def fake_llm_call(candidates, messages, **kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(
        task_endpoint, "resolve_memory_candidates",
        lambda *a, **k: [("http://llm", "model", {})],
    )
    monkeypatch.setattr(task_endpoint, "wait_for_interactive_quiet", fake_wait)
    monkeypatch.setattr(task_endpoint, "llm_call_async_with_fallback", fake_llm_call)

    await task_endpoint.memory_llm_call_async(
        "fast", [{"role": "user", "content": "hi"}], interactive=True
    )
    assert captured.get("workload") == "foreground"


# ── settings surface ──

def test_default_settings_has_memory_role_and_scalar_keys():
    for key, expected in {
        "memory_fast_endpoint_id": "",
        "memory_fast_model": "",
        "memory_smart_endpoint_id": "",
        "memory_smart_model": "",
        "memory_tag_registry_cap": 50,
        "memory_curator_nightly": True,
        "memory_curator_hour": 3,
        "memory_curator_batch": 25,
        "memory_retrieval_effort": "medium",
        "memory_context_doc_injection": False,
        "memory_archive_expiry_days": 90,
        "memory_protected_tags": "contact,identity",
    }.items():
        assert key in DEFAULT_SETTINGS, key
        assert DEFAULT_SETTINGS[key] == expected, key


def test_memory_role_endpoints_are_per_user_overridable():
    for key in (
        "memory_fast_endpoint_id", "memory_fast_model",
        "memory_smart_endpoint_id", "memory_smart_model",
    ):
        assert key in _PER_USER_KEYS, key


def test_memory_scalars_are_not_per_user():
    # Registry/curator/retrieval-default knobs are admin-global, unlike the
    # role endpoint pickers above.
    for key in (
        "memory_tag_registry_cap", "memory_curator_nightly", "memory_curator_hour",
        "memory_curator_batch", "memory_retrieval_effort",
        "memory_context_doc_injection", "memory_archive_expiry_days",
        "memory_protected_tags",
    ):
        assert key not in _PER_USER_KEYS, key
