from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COOKBOOK_RUNNING = ROOT / "static" / "js" / "cookbookRunning.js"


def _source() -> str:
    return COOKBOOK_RUNNING.read_text(encoding="utf-8")


def test_cookbook_marks_local_endpoint_registration_as_container_local():
    src = _source()
    assert "function _appendCookbookEndpointScope" in src
    assert "fd.append('container_local', 'true')" in src
    assert src.count("_appendCookbookEndpointScope(fd,") >= 3


def test_cookbook_does_not_use_local_as_endpoint_hostname():
    src = _source()
    assert "function _connectHostFromRemote" in src
    assert "if (!host || host === 'local') return fallback;" in src
    assert "const rawHost = task.remoteHost || 'localhost';" not in src


def test_cookbook_advertised_bind_urls_keep_connectable_host():
    src = _source()
    assert "function _endpointFromAdvertisedUrl" in src
    assert "_isAnyBindHost(u.hostname) ? currentHost" in src
    assert "host = u.hostname || host;" not in src


def test_cookbook_pin_and_label_use_serve_intent_model_id():
    # bugs/2026-07-15-cookbook-model-id-mismatch.md: the friendly-name key
    # and the pinned id must be the launch-INTENT id (repo_id for llama.cpp,
    # whose wire id is only known after /v1/models answers), not
    # _serveExpectedModel's model_path fallback (a directory path that never
    # appears in the endpoint's model list — orphaned label).
    src = _source()
    assert "function _serveIntentModelId" in src
    assert "backend !== 'llamacpp' ? fields.model_path : ''" in src

    def _fn_body(name: str) -> str:
        start = src.index(f"function {name}")
        return src[start:src.index("\n}", start)]

    assert "_serveIntentModelId(task)" in _fn_body("_friendlyLabelFor")
    assert "_serveExpectedModel" not in _fn_body("_friendlyLabelFor")
    assert "_serveIntentModelId(task)" in _fn_body("_appendPinnedServeModel")
    assert "_serveExpectedModel" not in _fn_body("_appendPinnedServeModel")


def test_cookbook_serve_request_carries_friendly_name():
    # The serve POST body must include friendly_name so the backend's
    # _auto_register_llm_endpoint can label the endpoint at serve time,
    # before the frontend's own PATCH path runs.
    src = _source()
    assert "friendly_name: String(fields?.friendly_name || '').trim() || undefined" in src
