"""Cookbook model-id reconciliation helpers (routes/model_routes.py).

A Cookbook serve can name the same model three ways: the launch-intent
repo/dir name ("Gemma_4_31B", pinned at registration), the serve panel's
model path ("/data/Gemma_4_31B", friendly-label key), and the wire id the
server reports from /v1/models ("/data/Gemma_4_31B/Gemma-4-31B.gguf").
_reconcile_cookbook_model_ids migrates the first two onto the third whenever
fresh probe results are persisted, so the Added Models tab shows one row and
friendly names don't orphan under phantom keys.
"""
import json
import sys
from types import SimpleNamespace

from tests.helpers.import_state import clear_fake_endpoint_resolver_modules, preserve_import_state

with preserve_import_state("core.database", "src.database", "routes.model_routes"):
    clear_fake_endpoint_resolver_modules()
    from routes.model_routes import (
        _is_cookbook_managed_endpoint,
        _model_ids_equivalent,
        _normalize_model_id_key,
        _reconcile_cookbook_model_ids,
    )


GGUF = "/data/models/Gemma_4_31B/Gemma-4-31B.gguf"


def _ep(**kw):
    base = dict(
        id="local-abc12345",
        endpoint_kind="local",
        pinned_models=None,
        model_labels=None,
        cached_models=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


# ── normalization / equivalence ─────────────────────────────────────────


def test_normalize_strips_path_gguf_split_suffix_and_separators():
    assert _normalize_model_id_key(GGUF) == "gemma431b"
    assert _normalize_model_id_key("Gemma_4_31B") == "gemma431b"
    assert _normalize_model_id_key("/data/models/Gemma_4_31B") == "gemma431b"
    assert _normalize_model_id_key("Gemma-4-31B-00001-of-00002.gguf") == "gemma431b"
    assert _normalize_model_id_key("C\\models\\Gemma-4-31B.GGUF") == "gemma431b"
    assert _normalize_model_id_key("") == ""


def test_equivalent_dir_name_vs_gguf_path():
    assert _model_ids_equivalent("Gemma_4_31B", GGUF)
    assert _model_ids_equivalent("/data/models/Gemma_4_31B", GGUF)
    assert _model_ids_equivalent(GGUF, "Gemma_4_31B")


def test_equivalent_quant_stem_contains_dir_name():
    assert _model_ids_equivalent("Gemma_4_31B", "Gemma-4-31B-Q4_K_M-00001-of-00002.gguf")


def test_not_equivalent_different_models_or_tags():
    assert not _model_ids_equivalent("gemma:2b", "gemma:7b")
    assert not _model_ids_equivalent("llama-3", "gemma-3")
    assert not _model_ids_equivalent("", GGUF)
    assert not _model_ids_equivalent("Gemma_4_31B", "")


# ── endpoint gating ──────────────────────────────────────────────────────


def test_cookbook_managed_by_raw_kind_and_id_prefix():
    # Raw column check matters: "ollama" is not in _ENDPOINT_KINDS, so the
    # normalizing _endpoint_kind() helper would report "auto" for it.
    assert _is_cookbook_managed_endpoint(_ep(endpoint_kind="local", id="ep1"))
    assert _is_cookbook_managed_endpoint(_ep(endpoint_kind="ollama", id="ep1"))
    assert _is_cookbook_managed_endpoint(_ep(endpoint_kind="api", id="local-deadbeef"))
    assert not _is_cookbook_managed_endpoint(_ep(endpoint_kind="api", id="ep1"))


def test_reconcile_skips_non_cookbook_endpoints():
    ep = _ep(endpoint_kind="api", id="azure-ep", pinned_models=json.dumps(["my-deploy-alias"]))
    assert _reconcile_cookbook_model_ids(ep, ["my-deploy-alias-v2"]) is False
    assert json.loads(ep.pinned_models) == ["my-deploy-alias"]


# ── pinned migration ─────────────────────────────────────────────────────


def test_pinned_dir_name_replaced_by_probed_gguf():
    ep = _ep(pinned_models=json.dumps(["Gemma_4_31B"]))
    assert _reconcile_cookbook_model_ids(ep, [GGUF]) is True
    assert json.loads(ep.pinned_models) == [GGUF]


def test_pinned_variants_dedupe_to_single_probed_id():
    ep = _ep(pinned_models=json.dumps(["Gemma_4_31B", "/data/models/Gemma_4_31B"]))
    assert _reconcile_cookbook_model_ids(ep, [GGUF]) is True
    assert json.loads(ep.pinned_models) == [GGUF]


def test_pinned_unrelated_id_kept():
    ep = _ep(pinned_models=json.dumps(["Gemma_4_31B", "some-other-model"]))
    assert _reconcile_cookbook_model_ids(ep, [GGUF]) is True
    assert json.loads(ep.pinned_models) == [GGUF, "some-other-model"]


def test_pinned_exact_probed_id_is_noop():
    ep = _ep(pinned_models=json.dumps([GGUF]))
    assert _reconcile_cookbook_model_ids(ep, [GGUF]) is False
    assert json.loads(ep.pinned_models) == [GGUF]


# ── label migration ──────────────────────────────────────────────────────


def test_label_under_dir_path_migrates_to_probed_id():
    ep = _ep(model_labels=json.dumps({"/data/models/Gemma_4_31B": "Gemma 4"}))
    assert _reconcile_cookbook_model_ids(ep, [GGUF]) is True
    assert json.loads(ep.model_labels) == {GGUF: "Gemma 4"}


def test_label_never_overwrites_existing_probed_key():
    ep = _ep(model_labels=json.dumps({GGUF: "Kept", "Gemma_4_31B": "Provisional"}))
    assert _reconcile_cookbook_model_ids(ep, [GGUF]) is True
    assert json.loads(ep.model_labels) == {GGUF: "Kept"}


def test_label_without_equivalent_probed_id_untouched():
    ep = _ep(model_labels=json.dumps({"unrelated-model": "Other"}))
    assert _reconcile_cookbook_model_ids(ep, [GGUF]) is False
    assert json.loads(ep.model_labels) == {"unrelated-model": "Other"}


def test_reconcile_noop_without_probed_ids():
    ep = _ep(pinned_models=json.dumps(["Gemma_4_31B"]))
    assert _reconcile_cookbook_model_ids(ep, []) is False
    assert _reconcile_cookbook_model_ids(ep, None) is False
    assert json.loads(ep.pinned_models) == ["Gemma_4_31B"]
