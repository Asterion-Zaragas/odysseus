"""_local_model_slot (src/llm_core.py): the local-model foreground/background
arbitration gate.

Covers the 2026-07-14 fixes: foreground no longer cancels an in-flight
background call (it queues behind it — pause-and-resume, not kill), the
waiting-foreground counter is decremented exactly once per caller (the old
double decrement let background work jump the queue past a still-waiting
foreground caller), and foreground never waits on `has_foreground_activity()`
(which is what made interactive memory calls sent as workload="background"
self-deadlock).
"""

import asyncio

import pytest

import src.llm_core as llm_core


URL = "http://localhost:8001/v1"


@pytest.fixture(autouse=True)
def gate_state(monkeypatch):
    """Fresh gate globals per test; treat every endpoint as local."""
    monkeypatch.setattr(llm_core, "_LOCAL_MODEL_LOCK", asyncio.Lock())
    monkeypatch.setattr(llm_core, "_LOCAL_MODEL_WAITING_FOREGROUND", 0)
    monkeypatch.setattr(llm_core, "_LOCAL_MODEL_CURRENT", {})
    monkeypatch.setattr(llm_core, "is_local_endpoint", lambda url: True)
    monkeypatch.delenv("ODYSSEUS_LOCAL_MODEL_GATE", raising=False)
    yield


async def test_foreground_does_not_cancel_inflight_background(monkeypatch):
    monkeypatch.setattr("src.interactive_gate.has_foreground_activity", lambda now=None: False)

    bg_inside = asyncio.Event()
    bg_release = asyncio.Event()
    fg_inside = asyncio.Event()
    bg_cancelled = False

    async def bg():
        nonlocal bg_cancelled
        try:
            async with llm_core._local_model_slot(URL, "bg-model", "background"):
                bg_inside.set()
                await bg_release.wait()
        except asyncio.CancelledError:
            bg_cancelled = True
            raise

    async def fg():
        async with llm_core._local_model_slot(URL, "fg-model", "foreground"):
            fg_inside.set()

    bg_task = asyncio.create_task(bg())
    await asyncio.wait_for(bg_inside.wait(), 2)

    fg_task = asyncio.create_task(fg())
    # Give the foreground caller ample time to (wrongly) cancel/overtake.
    await asyncio.sleep(0.1)
    assert not bg_task.cancelled() and not bg_cancelled
    assert not fg_inside.is_set()  # queued behind the in-flight background call

    bg_release.set()
    await asyncio.wait_for(fg_task, 2)
    await asyncio.wait_for(bg_task, 2)
    assert fg_inside.is_set()
    assert not bg_cancelled


async def test_foreground_never_waits_on_foreground_activity(monkeypatch):
    # workload="foreground" must enter immediately even while the app reports
    # foreground activity — this is the path interactive memory calls take.
    monkeypatch.setattr("src.interactive_gate.has_foreground_activity", lambda now=None: True)

    async def fg():
        async with llm_core._local_model_slot(URL, "fg-model", "foreground"):
            return "entered"

    assert await asyncio.wait_for(fg(), 1) == "entered"


async def test_background_defers_while_foreground_activity(monkeypatch):
    monkeypatch.setattr("src.interactive_gate.has_foreground_activity", lambda now=None: True)

    async def bg():
        async with llm_core._local_model_slot(URL, "bg-model", "background"):
            return "entered"

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(bg(), 0.4)


async def test_waiting_counter_not_double_decremented(monkeypatch):
    # A completes its slot while B and C still wait on the lock. The old code
    # decremented the counter again on A's exit, eating C's waiting mark
    # (2 -> ... -> 0 by the time B held the slot) and letting background work
    # jump the queue past the still-waiting C. Now: exactly one decrement per
    # caller, so with B inside and C waiting the counter must still be 1.
    monkeypatch.setattr("src.interactive_gate.has_foreground_activity", lambda now=None: False)

    a_inside = asyncio.Event(); a_release = asyncio.Event()
    b_inside = asyncio.Event(); b_release = asyncio.Event()
    c_inside = asyncio.Event()

    async def slot(model, inside, release=None):
        async with llm_core._local_model_slot(URL, model, "foreground"):
            inside.set()
            if release is not None:
                await release.wait()

    a_task = asyncio.create_task(slot("a", a_inside, a_release))
    await asyncio.wait_for(a_inside.wait(), 2)
    b_task = asyncio.create_task(slot("b", b_inside, b_release))
    c_task = asyncio.create_task(slot("c", c_inside))
    await asyncio.sleep(0.05)  # let B and C reach the lock wait
    assert llm_core._LOCAL_MODEL_WAITING_FOREGROUND == 2

    a_release.set()
    await asyncio.wait_for(b_inside.wait(), 2)  # B now holds the slot
    assert not c_inside.is_set()
    assert llm_core._LOCAL_MODEL_WAITING_FOREGROUND == 1  # C's mark survives

    b_release.set()
    await asyncio.wait_for(c_task, 2)
    await asyncio.wait_for(a_task, 2)
    await asyncio.wait_for(b_task, 2)
    assert llm_core._LOCAL_MODEL_WAITING_FOREGROUND == 0


async def test_foreground_cancelled_while_waiting_undoes_its_mark(monkeypatch):
    monkeypatch.setattr("src.interactive_gate.has_foreground_activity", lambda now=None: False)

    a_inside = asyncio.Event(); a_release = asyncio.Event()

    async def holder():
        async with llm_core._local_model_slot(URL, "a", "foreground"):
            a_inside.set()
            await a_release.wait()

    async def waiter():
        async with llm_core._local_model_slot(URL, "b", "foreground"):
            pass

    a_task = asyncio.create_task(holder())
    await asyncio.wait_for(a_inside.wait(), 2)
    b_task = asyncio.create_task(waiter())
    await asyncio.sleep(0.05)
    assert llm_core._LOCAL_MODEL_WAITING_FOREGROUND == 1

    b_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await b_task
    assert llm_core._LOCAL_MODEL_WAITING_FOREGROUND == 0

    a_release.set()
    await asyncio.wait_for(a_task, 2)


async def test_gate_skips_non_local_endpoints(monkeypatch):
    # Cloud endpoints bypass the gate entirely, even under foreground activity
    # and with the lock held.
    monkeypatch.setattr(llm_core, "is_local_endpoint", lambda url: False)
    monkeypatch.setattr("src.interactive_gate.has_foreground_activity", lambda now=None: True)
    await llm_core._LOCAL_MODEL_LOCK.acquire()
    try:
        async def bg():
            async with llm_core._local_model_slot("https://api.example.com", "m", "background"):
                return "entered"

        assert await asyncio.wait_for(bg(), 1) == "entered"
    finally:
        llm_core._LOCAL_MODEL_LOCK.release()
