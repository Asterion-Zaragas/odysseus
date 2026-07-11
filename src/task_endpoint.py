"""Shared resolver for background-task AI endpoints."""

from src.endpoint_resolver import (
    resolve_chat_fallback_candidates,
    resolve_endpoint,
    resolve_utility_fallback_candidates,
)
from src.llm_core import llm_call_async_with_fallback
from src.interactive_gate import wait_for_interactive_quiet


def resolve_task_endpoint(fallback_url=None, fallback_model=None, fallback_headers=None, owner=None):
    """Return (endpoint_url, model, headers) for background tasks.

    Reads task_endpoint_id / task_model from admin settings.
    Falls back to the provided values when the setting is empty or the
    endpoint cannot be resolved.
    """
    return resolve_endpoint("task", fallback_url, fallback_model, fallback_headers, owner=owner)


def resolve_task_candidates(
    fallback_url=None,
    fallback_model=None,
    fallback_headers=None,
    owner=None,
):
    """Return ordered background-task LLM candidates.

    Order:
    1. configured Background Tasks endpoint/model, or caller fallback
    2. Utility endpoint/model
    3. Default endpoint/model
    4. Utility fallback chain
    5. Default fallback chain
    """
    candidates = []

    def _append(url, model, headers):
        if not url or not model:
            return
        key = (url, model)
        if any((u, m) == key for u, m, _ in candidates):
            return
        candidates.append((url, model, headers or {}))

    _append(*resolve_task_endpoint(fallback_url, fallback_model, fallback_headers, owner=owner))
    _append(*resolve_endpoint("utility", owner=owner))
    _append(*resolve_endpoint("default", owner=owner))
    for url, model, headers in resolve_utility_fallback_candidates(owner=owner):
        _append(url, model, headers)
    for url, model, headers in resolve_chat_fallback_candidates(owner=owner):
        _append(url, model, headers)

    return candidates


async def task_llm_call_async(
    messages,
    *,
    fallback_url=None,
    fallback_model=None,
    fallback_headers=None,
    owner=None,
    **kwargs,
):
    """Call the shared background-task LLM candidate chain."""
    candidates = resolve_task_candidates(
        fallback_url=fallback_url,
        fallback_model=fallback_model,
        fallback_headers=fallback_headers,
        owner=owner,
    )
    if not candidates:
        raise RuntimeError("No LLM endpoint available for background task")
    await wait_for_interactive_quiet("background task LLM")
    kwargs.setdefault("workload", "background")
    return await llm_call_async_with_fallback(candidates, messages=messages, **kwargs)


# Memory-system model roles (memory upgrade plan, Part "Model classes per
# agent"): "fast" serves the tagger/facet-extractor/verifier, "smart" serves
# the distiller/curator. Both resolve through the generic `resolve_endpoint`
# prefix support, falling back to the same task->utility->default chain as
# everything else in this module when unconfigured.
MEMORY_ROLES = ("fast", "smart")


def resolve_memory_candidates(role, owner=None):
    """Return ordered LLM candidates for a memory-system agent role.

    Order:
    1. configured `memory_{role}_endpoint_id` / `memory_{role}_model`
    2. the background-task candidate chain (resolve_task_candidates)
    """
    if role not in MEMORY_ROLES:
        raise ValueError(f"unknown memory role: {role!r}")

    candidates = []

    def _append(url, model, headers):
        if not url or not model:
            return
        key = (url, model)
        if any((u, m) == key for u, m, _ in candidates):
            return
        candidates.append((url, model, headers or {}))

    _append(*resolve_endpoint(f"memory_{role}", owner=owner))
    for url, model, headers in resolve_task_candidates(owner=owner):
        _append(url, model, headers)

    return candidates


async def memory_llm_call_async(
    role,
    messages,
    *,
    interactive=False,
    owner=None,
    **kwargs,
):
    """Call the shared candidate chain for a memory-system agent role.

    `interactive=True` skips the background-task foreground gate. Only the
    retrieval hot path (Phase 6) should pass it: that call happens inline in
    a chat turn, so it must not queue behind the same "UI is busy" window it
    is itself part of. Every other memory role (tagger, curator, distiller)
    runs off the interactive path and keeps the gate.
    """
    candidates = resolve_memory_candidates(role, owner=owner)
    if not candidates:
        raise RuntimeError(f"No LLM endpoint available for memory {role} task")
    if not interactive:
        await wait_for_interactive_quiet(f"memory {role} task")
    kwargs.setdefault("workload", "background")
    return await llm_call_async_with_fallback(candidates, messages=messages, **kwargs)
