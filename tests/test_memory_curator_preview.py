"""Curator log overhaul: per-action detail + the full preview changelog.

See .AGENT_CONTEXT/plans/2026-08-05-curator-log-detail-and-preview.md.

Two halves, one plan:

**Part B (Python)** — a dry run returns the actions it *would* have taken
inline in `curate()`'s response, so the Curator tab's preview can show a
proposed changelog instead of only a count delta. The tempting shortcut —
relaxing `read_curation_log(include_dry_run=False)` — is deliberately NOT
taken: that filter is what stops the panel offering to undo a change that
never happened (the Phase 7 fix these tests also re-assert from the other
side). Live runs must therefore keep returning nothing, so the history feed
and the preview feed never share a source.

**Part A (JavaScript, under Node)** — `_describeCuratorAction` in
static/js/memory.js used to return one string and fell through to
`after.text || before.text` for everything it didn't special-case, so every
action whose interesting change isn't the text rendered as the entry text and
looked identical before and after. It now returns `{summary, detail}`. These
run the real function rather than asserting on source strings, since the
whole point is what it produces for each action's actual logged payload.
"""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from services.memory import memory_context as mc
from services.memory import memory_curator as cur
from src.memory import MemoryManager


_REPO = Path(__file__).resolve().parents[1]
_MEMORY_JS = (_REPO / "static" / "js" / "memory.js").read_text(encoding="utf-8")
_HAS_NODE = shutil.which("node") is not None


@pytest.fixture()
def _isolated_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(cur, "CURATION_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(cur, "CURATION_LOG_DIR", str(tmp_path / "log"))
    monkeypatch.setattr(cur, "CURATION_FAILURES_DIR", str(tmp_path / "failures"))
    monkeypatch.setattr(cur, "TAG_BACKFILL_STATE_DIR", str(tmp_path / "tag_backfill_state"))
    monkeypatch.setattr(cur, "REWORD_STATE_DIR", str(tmp_path / "reword_state"))
    monkeypatch.setattr(mc, "CONTEXT_DIR", str(tmp_path / "context"))
    return tmp_path


class _FakeVectorStore:
    healthy = True

    def __init__(self, similarity=0.97):
        self._similarity = similarity

    def text_similarity(self, a, b):
        return self._similarity

    def rebuild(self, memories):
        pass


def _settings(monkeypatch, **overrides):
    monkeypatch.setattr(
        "src.settings.get_setting", lambda key, default=None: overrides.get(key, default)
    )


def _store(d, texts, owner="alice", generality=2):
    """Seed a store. `generality=2` marks entries already triaged, so only the
    passes a test cares about have anything to do; `generality=None` leaves
    them pending so the triage pass fires."""
    mgr = MemoryManager(d)
    entries = []
    for text in texts:
        e = mgr.add_entry(text, owner=owner)
        e["generality"] = generality
        entries.append(e)
    mgr.save(entries)
    return mgr, entries


def _quiet_llm(**by_prompt):
    """A memory-smart double that answers every pass with a no-op unless the
    test supplies a reply keyed by a distinctive phrase from that pass's
    system prompt."""

    async def fake_llm(role, messages, owner=None, **kwargs):
        system = messages[0]["content"]
        for needle, reply in by_prompt.items():
            if needle in system:
                return reply
        if "MERGE" in system:
            return json.dumps({"merges": [], "remove": []})
        return "[]"

    return fake_llm


# ── curate(dry_run=True) returns its proposed actions ──

async def test_dry_run_returns_proposed_actions(_isolated_dirs, monkeypatch):
    _settings(monkeypatch)
    with tempfile.TemporaryDirectory() as d:
        mgr, entries = _store(d, ["The user's cat is called Mimi"], generality=None)
        eid = entries[0]["id"]
        monkeypatch.setattr(
            "src.task_endpoint.memory_llm_call_async",
            _quiet_llm(**{
                "triage personal memory entries": json.dumps(
                    [{"id": eid, "generality": 3, "promote_tags": []}]
                ),
            }),
        )

        result = await cur.curate(mgr, _FakeVectorStore(), owner="alice", dry_run=True)

    assert result["status"] == "dry_run"
    actions = result["proposed_actions"]
    assert actions, "a dry run that proposed changes must return them"
    # The triage pass's retag, alongside whatever else the run proposed — the
    # point is that the response carries the actions at all, not only counts.
    assert "retag" in {a["action"] for a in actions}
    assert result["proposed_total"] == len(actions)
    assert result["proposed_truncated"] is False


async def test_live_run_does_not_return_proposed_actions(_isolated_dirs, monkeypatch):
    """The history list (GET /curation-log) already covers a live run. Keeping
    the two feeds on separate sources is the whole reason a proposed action
    can never reach the undo path."""
    _settings(monkeypatch)
    with tempfile.TemporaryDirectory() as d:
        mgr, entries = _store(d, ["The user's cat is called Mimi"])
        eid = entries[0]["id"]
        monkeypatch.setattr(
            "src.task_endpoint.memory_llm_call_async",
            _quiet_llm(**{
                "triage personal memory entries": json.dumps(
                    [{"id": eid, "generality": 3, "promote_tags": []}]
                ),
            }),
        )

        result = await cur.curate(mgr, _FakeVectorStore(), owner="alice", dry_run=False)

    assert result["status"] == "done"
    assert "proposed_actions" not in result
    assert "proposed_total" not in result
    assert "proposed_truncated" not in result


async def test_proposed_actions_all_carry_the_dry_run_marker(_isolated_dirs, monkeypatch):
    """The frontend's undo guard reads `dry_run` off the row itself rather
    than trusting the call site, so every returned action must carry it."""
    _settings(monkeypatch)
    with tempfile.TemporaryDirectory() as d:
        mgr, entries = _store(d, ["The user's cat is called Mimi"])
        eid = entries[0]["id"]
        monkeypatch.setattr(
            "src.task_endpoint.memory_llm_call_async",
            _quiet_llm(**{
                "triage personal memory entries": json.dumps(
                    [{"id": eid, "generality": 1, "promote_tags": []}]
                ),
            }),
        )

        result = await cur.curate(mgr, _FakeVectorStore(), owner="alice", dry_run=True)

    assert result["proposed_actions"]
    assert all(a["dry_run"] is True for a in result["proposed_actions"])


async def test_proposed_actions_are_capped_and_flag_truncation(_isolated_dirs, monkeypatch):
    """This response is assembled in memory inside a request; a first run over
    a large store across several passes can propose a lot."""
    _settings(monkeypatch, memory_curator_preview_cap=2)
    with tempfile.TemporaryDirectory() as d:
        mgr, entries = _store(d, [f"Fact number {i}" for i in range(5)])
        monkeypatch.setattr(
            "src.task_endpoint.memory_llm_call_async",
            _quiet_llm(**{
                "triage personal memory entries": json.dumps(
                    [{"id": e["id"], "generality": 3, "promote_tags": []} for e in entries]
                ),
            }),
        )

        result = await cur.curate(mgr, _FakeVectorStore(), owner="alice", dry_run=True)

    assert len(result["proposed_actions"]) == 2
    assert result["proposed_total"] == 5
    assert result["proposed_truncated"] is True


async def test_reword_refused_appears_in_the_proposed_list(_isolated_dirs, monkeypatch):
    """"What the guards rejected" is exactly what's needed to tune
    `_REWORD_MIN_SIMILARITY` — a refusal that only ever reached the log file
    is invisible where the tuning decision gets made."""
    _settings(
        monkeypatch,
        memory_curator_reword_enabled=True,
        memory_curator_reword_judge_enabled=False,
    )
    with tempfile.TemporaryDirectory() as d:
        mgr, entries = _store(d, ["The user's rent is $1450/mo, due on the 1st."])
        eid = entries[0]["id"]
        monkeypatch.setattr(
            "src.task_endpoint.memory_llm_call_async",
            _quiet_llm(**{
                "lightly improve the wording": json.dumps(
                    [{"id": eid, "text": "Rent is about fifteen hundred a month."}]
                ),
            }),
        )

        # Similarity below the floor -> the guard refuses, entry unchanged.
        result = await cur.curate(
            mgr, _FakeVectorStore(similarity=0.20), owner="alice", dry_run=True
        )

    refusals = [a for a in result["proposed_actions"] if a["action"] == "reword_refused"]
    assert len(refusals) == 1
    assert "similarity" in refusals[0]["before"]["reason"]
    assert refusals[0]["before"]["text"] == "The user's rent is $1450/mo, due on the 1st."


async def test_preview_trims_pathological_text_but_the_log_keeps_it(_isolated_dirs, monkeypatch):
    _settings(monkeypatch)
    long_text = "x" * (cur._PREVIEW_TEXT_LIMIT + 500)
    with tempfile.TemporaryDirectory() as d:
        mgr, entries = _store(d, [long_text], generality=None)
        eid = entries[0]["id"]
        monkeypatch.setattr(
            "src.task_endpoint.memory_llm_call_async",
            _quiet_llm(**{
                "triage personal memory entries": json.dumps(
                    [{"id": eid, "generality": 3, "promote_tags": []}]
                ),
            }),
        )

        result = await cur.curate(mgr, _FakeVectorStore(), owner="alice", dry_run=True)

    shown = result["proposed_actions"][0]["before"]["text"]
    assert len(shown) == cur._PREVIEW_TEXT_LIMIT + 1  # + the ellipsis
    assert result["proposed_actions"][0]["before"]["text_truncated"] is True
    # The on-disk changelog is not a display surface and stays complete.
    logged = cur.read_curation_log("alice")
    assert logged[0]["before"]["text"] == long_text


async def test_dry_run_still_writes_log_lines_and_they_stay_filtered(_isolated_dirs, monkeypatch):
    """Unchanged behavior, restated here because Part B depends on it: dry-run
    lines remain on disk for debugging AND remain hidden from the history
    feed. Returning them in the response is an addition, not a relaxation."""
    _settings(monkeypatch)
    with tempfile.TemporaryDirectory() as d:
        mgr, entries = _store(d, ["The user's cat is called Mimi"])
        eid = entries[0]["id"]
        monkeypatch.setattr(
            "src.task_endpoint.memory_llm_call_async",
            _quiet_llm(**{
                "triage personal memory entries": json.dumps(
                    [{"id": eid, "generality": 3, "promote_tags": []}]
                ),
            }),
        )

        await cur.curate(mgr, _FakeVectorStore(), owner="alice", dry_run=True)

    assert cur.read_curation_log("alice"), "dry-run lines are still logged"
    assert cur.read_curation_log("alice", include_dry_run=False) == []


def test_collector_is_inert_outside_a_dry_run(_isolated_dirs):
    """`_log_action` is the collection funnel, so it must stay a plain logger
    when no run installed a sink — including for the undo paths, which log
    from outside `curate()` entirely."""
    cur._log_action("alice", "undo_reword", before=None, after={"id": "1", "text": "hi"})
    assert cur._preview_sink.get() is None
    assert len(cur.read_curation_log("alice")) == 1


def test_collector_sink_is_removed_after_the_block(_isolated_dirs):
    with cur._collect_preview(10) as sink:
        cur._log_action("alice", "expire", before={"id": "1"}, after=None, dry_run=True)
        assert len(sink["actions"]) == 1
    assert cur._preview_sink.get() is None
    cur._log_action("alice", "expire", before={"id": "2"}, after=None, dry_run=True)
    assert len(sink["actions"]) == 1, "actions logged after the block must not leak in"


# ── POST /api/memory/audit passes the list through ──

def _route(router, path, method):
    for r in router.routes:
        if r.path == path and method in getattr(r, "methods", set()):
            return r.endpoint
    raise AssertionError(path)


def _request_for(user):
    from types import SimpleNamespace
    return SimpleNamespace(
        state=SimpleNamespace(current_user=user),
        app=SimpleNamespace(state=SimpleNamespace(auth_manager=None)),
    )


async def test_audit_route_passes_proposed_actions_through(monkeypatch):
    import routes.memory_routes as mr
    from unittest.mock import MagicMock

    monkeypatch.setattr(mr, "get_current_user", lambda request: "alice", raising=False)
    proposed = [{"ts": 1.0, "action": "retag", "before": {}, "after": {}, "dry_run": True}]

    async def fake_curate(*args, **kwargs):
        return {
            "status": "dry_run", "before": 3, "after": 3, "already_tidy": False,
            "had_failures": False, "failure_count": 0,
            "proposed_actions": proposed, "proposed_total": 7, "proposed_truncated": True,
        }

    monkeypatch.setattr(mr, "curate", fake_curate)
    router = mr.setup_memory_routes(MagicMock(), MagicMock())
    handler = _route(router, "/api/memory/audit", "POST")

    out = await handler(request=_request_for("alice"), dry_run=True, force=False)
    assert out["proposed_actions"] == proposed
    assert out["proposed_total"] == 7
    assert out["proposed_truncated"] is True


async def test_audit_route_omits_proposed_actions_for_a_live_run(monkeypatch):
    import routes.memory_routes as mr
    from unittest.mock import MagicMock

    monkeypatch.setattr(mr, "get_current_user", lambda request: "alice", raising=False)

    async def fake_curate(*args, **kwargs):
        return {
            "status": "done", "before": 3, "after": 2, "already_tidy": False,
            "had_failures": False, "failure_count": 0,
        }

    monkeypatch.setattr(mr, "curate", fake_curate)
    router = mr.setup_memory_routes(MagicMock(), MagicMock())
    handler = _route(router, "/api/memory/audit", "POST")

    out = await handler(request=_request_for("alice"), dry_run=False, force=False)
    assert "proposed_actions" not in out
    assert out["removed"] == 1


# ── Part A: the JS renderings, run under Node ──


def _extract_source(source: str, start: str, end: str) -> str:
    """Slice module source between two anchors, failing loudly if one moved.

    The extracted region ships to Node verbatim, so the anchors must stay
    unique strings in the module. A refactor that renames or duplicates an
    anchor fails here with the anchor named, not with an opaque split error.
    """
    assert source.count(start) == 1, f"start anchor not unique in source: {start!r}"
    tail = source.split(start, 1)[1]
    assert end in tail, f"end anchor not found after start anchor: {end!r}"
    return start + tail.split(end, 1)[0]


# memory.js is a browser ES module with a heavy import chain, so the pure
# description helpers are lifted out and run standalone with the two module
# constants they close over stubbed in. Anything DOM-touching stays out of
# this slice by construction — which is also the property Part A's plan asks
# for ("keep the function pure and total").
_DESCRIBE_SRC = _extract_source(
    _MEMORY_JS,
    "function _curatorTagList(obj, field) {",
    "function _buildCuratorDetailBody(blocks) {",
)

_JS_PRELUDE = """
const MEMORY_TIER_NAMES = { 0: 'core', 1: 'durable', 2: 'situational', 3: 'archive' };
function relativeTime(ts) { return 'a while ago'; }
"""


def _describe(entries):
    """Run `_describeCuratorAction` (and `_curatorActionCanUndo`) under Node
    over a list of changelog lines; returns one result object per entry."""
    source = _JS_PRELUDE + _DESCRIBE_SRC + f"""
const entries = {json.dumps(entries)};
const out = entries.map(e => {{
  const d = _describeCuratorAction(e);
  return {{ summary: d.summary, detail: d.detail, canUndo: _curatorActionCanUndo(e) }};
}});
console.log(JSON.stringify(out));
"""
    proc = subprocess.run(
        ["node", "--input-type=module"],
        input=source, capture_output=True, text=True, cwd=_REPO, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


pytestmark_node = pytest.mark.skipif(not _HAS_NODE, reason="node binary not on PATH")


@pytestmark_node
def test_js_reword_shows_both_old_and_new_text():
    """The headline feature: the reword pass can't be judged — and its
    similarity floor can't be tuned — without the old text on screen."""
    (out,) = _describe([{
        "action": "reword", "dry_run": False,
        "before": {"id": "m1", "text": "The user he likes the coffee black"},
        "after": {"id": "m1", "text": "The user likes black coffee"},
    }])
    labels = {b["label"]: b["value"] for b in out["detail"]}
    assert labels["before"] == "The user he likes the coffee black"
    assert labels["after"] == "The user likes black coffee"
    assert out["summary"] == "The user likes black coffee"


@pytestmark_node
def test_js_retag_shows_the_tag_delta_not_the_entry_text():
    """The bug this plan opens with: a retag row rendered the entry's text,
    which is identical before and after, instead of what changed."""
    (out,) = _describe([{
        "action": "retag", "dry_run": False,
        "before": {"id": "m1", "text": "Lives in Berlin", "tags": ["identity", "job"],
                   "provisional_tags": ["place-berlin"]},
        "after": {"id": "m1", "text": "Lives in Berlin", "tags": ["identity", "place-berlin"],
                  "provisional_tags": []},
    }])
    assert out["summary"] == "+place-berlin −job −place-berlin"
    blocks = {b["label"]: b for b in out["detail"]}
    tags = {t["name"]: t["state"] for t in blocks["tags"]["tags"]}
    assert tags == {"place-berlin": "added", "job": "removed", "identity": "kept"}
    # Provisional tags render as their own row so promotion is legible.
    assert blocks["provisional"]["provisional"] is True
    assert blocks["provisional"]["tags"] == [{"name": "place-berlin", "state": "removed"}]
    assert blocks["entry"]["value"] == "Lives in Berlin"


@pytestmark_node
def test_js_tag_backfill_marks_new_provisional_tags():
    (out,) = _describe([{
        "action": "tag_backfill", "dry_run": False,
        "before": {"id": "m1", "text": "Uses VeraCrypt", "tags": [], "provisional_tags": []},
        "after": {"id": "m1", "text": "Uses VeraCrypt", "tags": [],
                  "provisional_tags": ["software-veracrypt"]},
    }])
    assert out["summary"] == "+software-veracrypt"
    blocks = {b["label"]: b for b in out["detail"]}
    assert blocks["provisional"]["tags"] == [{"name": "software-veracrypt", "state": "added"}]


@pytestmark_node
def test_js_tag_rename_is_not_read_as_an_entry_retag():
    """tag_normalize logs `retag` with {tag: from}/{tag: into} and no entry —
    same action name, different payload shape from triage's retag."""
    (out,) = _describe([{
        "action": "retag", "dry_run": False,
        "before": {"tag": "job"}, "after": {"tag": "work"},
    }])
    assert out["summary"] == 'tag "job" → "work" — renamed across every entry'
    assert out["detail"] is None


@pytestmark_node
def test_js_merge_names_kept_dropped_and_the_tag_union():
    """dedupe logs merges with LISTS: before = every dropped entry,
    after = [the kept entry] with the union already applied."""
    (out,) = _describe([{
        "action": "merge", "dry_run": False,
        "before": [{"id": "m2", "text": "The user is called Sam", "tags": ["identity"]}],
        "after": [{"id": "m1", "text": "User's name is Sam", "tags": ["identity", "person-sam"]}],
    }])
    assert out["summary"] == 'kept "User\'s name is Sam" — dropped 1 duplicate'
    blocks = [(b["label"], b.get("value")) for b in out["detail"]]
    assert ("kept", "User's name is Sam") in blocks
    assert ("dropped", "The user is called Sam") in blocks
    union = [b for b in out["detail"] if b["label"] == "tags after merge"][0]
    assert [t["name"] for t in union["tags"]] == ["identity", "person-sam"]


@pytestmark_node
def test_js_expire_explains_why():
    (out,) = _describe([{
        "action": "expire", "dry_run": False,
        "before": {"id": "m1", "text": "Was reading a blog post", "tier": 3,
                   "tags": ["fact"], "uses": 0, "last_used_at": 1700000000},
        "after": None,
    }])
    why = [b["value"] for b in out["detail"] if b["label"] == "why"][0]
    assert "tier 3 (archive)" in why
    assert "0 uses" in why
    assert out["summary"] == "Was reading a blog post"


@pytestmark_node
def test_js_tier_changes_show_old_and_new_tier():
    out = _describe([
        {"action": "promote", "dry_run": False,
         "before": {"id": "m1", "tier": 2}, "after": {"id": "m1", "tier": 1}},
        {"action": "demote", "dry_run": False,
         "before": {"id": "m1", "tier": 2}, "after": {"id": "m1", "tier": 3}},
    ])
    assert out[0]["summary"] == "tier 2 (situational) → 1 (durable)"
    assert out[1]["summary"] == "tier 2 (situational) → 3 (archive)"


@pytestmark_node
def test_js_reword_refused_keeps_its_reason_and_gains_the_original():
    (out,) = _describe([{
        "action": "reword_refused", "dry_run": False,
        "before": {"id": "m1", "text": "Rent is $1450/mo",
                   "reason": "similarity 0.412 below floor 0.9"},
        "after": None,
    }])
    assert out["summary"] == "Reword declined: similarity 0.412 below floor 0.9"
    assert out["detail"] == [{"label": "original", "kind": "text", "value": "Rent is $1450/mo"}]


@pytestmark_node
def test_js_generic_fallback_survives_for_unknown_actions():
    """The fallback is why `tag_backfill` needed no frontend change when it
    landed. An action added later must still render something useful."""
    (out,) = _describe([{
        "action": "some_future_pass", "dry_run": False,
        "before": {"id": "m1", "text": "a fact"}, "after": {"id": "m1", "text": "a fact"},
    }])
    assert out["summary"] == "a fact"
    assert out["detail"] is None


@pytestmark_node
def test_js_malformed_legacy_lines_never_throw():
    """The changelog is append-only JSONL going back to Phase 4; older lines
    may lack any field a newer rendering wants."""
    out = _describe([
        {"action": "retag"},
        {"action": "merge", "before": None, "after": None},
        {"action": "expire", "before": "not-a-dict", "after": None},
        {"action": "reword", "before": {}, "after": {}},
        {"action": "promote", "before": {"tier": None}, "after": {}},
        {},
    ])
    assert len(out) == 6
    assert all(isinstance(o["summary"], str) and o["summary"] for o in out)


# ── the phantom-undo invariant ──

@pytestmark_node
def test_js_proposed_rows_never_offer_undo():
    """The single most important invariant: an undo button on a proposed
    action would reverse a change that never happened. Guarded on the row's
    own `dry_run` marker so it holds regardless of which feed rendered it."""
    out = _describe([
        {"action": "expire", "dry_run": True, "before": {"id": "m1", "text": "x"}, "after": None},
        {"action": "reword", "dry_run": True,
         "before": {"id": "m1", "text": "x"}, "after": {"id": "m1", "text": "y"}},
    ])
    assert [o["canUndo"] for o in out] == [False, False]


@pytestmark_node
def test_js_applied_expire_and_reword_still_offer_undo():
    out = _describe([
        {"action": "expire", "dry_run": False, "before": {"id": "m1", "text": "x"}, "after": None},
        {"action": "reword", "dry_run": False,
         "before": {"id": "m1", "text": "x"}, "after": {"id": "m1", "text": "y"}},
        # merge's `before` is a list, so there is no single id to undo — this
        # matched nothing before the rewrite either.
        {"action": "merge", "dry_run": False, "before": [{"id": "m2"}], "after": [{"id": "m1"}]},
    ])
    assert [o["canUndo"] for o in out] == [True, True, False]


def test_render_path_refuses_undo_buttons_in_proposed_mode():
    """Belt and braces on top of the per-row `dry_run` check above: the render
    function itself must not build an undo button when rendering the preview
    feed. Asserted in source — the branch is DOM-coupled, so the Node slice
    above can't reach it."""
    assert "if (!proposed && _curatorActionCanUndo(entry)) {" in _MEMORY_JS


def test_preview_renders_through_the_shared_row_component():
    assert "renderCuratorLogList(actions, listEl, { proposed: true });" in _MEMORY_JS
