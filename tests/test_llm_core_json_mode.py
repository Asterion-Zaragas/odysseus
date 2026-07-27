"""JSON-mode (`response_format`) plumbing for the async LLM path (curator rec #1).

Covers `llm_call_async`: the OpenAI-compatible payload carries `response_format`,
and a backend that rejects it (HTTP 400/422) is retried once without it rather
than failing the call.
"""
import httpx

from src import llm_core


class _NullSlot:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *a):
        return False


def _patch_async_post(monkeypatch, responder):
    """Route llm_call_async's HTTP through `responder(payload) -> httpx.Response`,
    bypassing the real client/slot/host-liveness machinery."""
    monkeypatch.setattr(llm_core, "_local_model_slot", lambda *a, **k: _NullSlot())
    monkeypatch.setattr(llm_core, "_is_host_dead", lambda url: False)
    monkeypatch.setattr(llm_core, "_get_http_client", lambda: object())
    llm_core._response_cache.clear()

    async def fake_post(client, url, headers, json=None, timeout=None):
        return responder(json)

    monkeypatch.setattr(llm_core, "httpx_post_kimi_aware_async", fake_post)


def _ok(_payload):
    req = httpx.Request("POST", "http://x/v1/chat/completions")
    return httpx.Response(200, request=req, json={"choices": [{"message": {"content": "OK"}}]})


async def test_response_format_in_openai_payload(monkeypatch):
    seen = []
    _patch_async_post(monkeypatch, lambda p: (seen.append(p), _ok(p))[1])

    out = await llm_core.llm_call_async(
        "http://localhost:9/v1", "m", [{"role": "user", "content": "hi"}],
        response_format={"type": "json_object"},
    )
    assert out == "OK"
    assert seen[0]["response_format"] == {"type": "json_object"}


async def test_400_on_response_format_retries_without_it(monkeypatch):
    seen = []

    def responder(payload):
        seen.append(dict(payload))
        if "response_format" in payload:
            req = httpx.Request("POST", "http://x/v1/chat/completions")
            return httpx.Response(400, request=req, text="unknown field response_format")
        return _ok(payload)

    _patch_async_post(monkeypatch, responder)

    out = await llm_core.llm_call_async(
        "http://localhost:9/v1", "m", [{"role": "user", "content": "hi"}],
        response_format={"type": "json_object"},
    )
    assert out == "OK"
    # First attempt carried it and 400'd; retry dropped it and succeeded.
    assert len(seen) == 2
    assert "response_format" in seen[0]
    assert "response_format" not in seen[1]


async def test_no_response_format_leaves_payload_clean(monkeypatch):
    seen = []
    _patch_async_post(monkeypatch, lambda p: (seen.append(p), _ok(p))[1])

    await llm_core.llm_call_async(
        "http://localhost:9/v1", "m", [{"role": "user", "content": "hi"}],
    )
    assert "response_format" not in seen[0]
