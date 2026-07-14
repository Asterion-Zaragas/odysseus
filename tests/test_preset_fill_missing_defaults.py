"""An older / partial presets.json must still serve every built-in preset,
WITHOUT clobbering user edits.

Built-ins are no longer persisted per store — `PresetManager.get_all()` merges
`DEFAULT_PRESETS` in at read time (defaults first, stored values win), so a
missing built-in can never be absent from the picker served by
GET /api/presets. A missing built-in is never an intentional user action —
there is no delete path for the built-in keys (only `user_templates` entries
can be deleted), and presets are hidden via an `enabled: False` flag, not
removal — so serving them back is safe.
"""
import json
import os
import tempfile

from src.preset_manager import PresetManager


def _write_presets(data: dict) -> str:
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "presets.json"), "w", encoding="utf-8") as f:
        json.dump(data, f)
    return d


def test_missing_builtin_presets_are_served():
    # Partial file: has code_analyze + brainstorm, missing reason + custom.
    data_dir = _write_presets({
        "code_analyze": {"name": "Code Analyze", "temperature": 0.2,
                         "max_tokens": 8000, "system_prompt": "analyze"},
        "brainstorm": {"name": "Brainstorm", "temperature": 0.9,
                       "max_tokens": 4096, "system_prompt": "ideate"},
    })
    pm = PresetManager(data_dir)
    merged = pm.get_all()
    for key in PresetManager.DEFAULT_PRESETS:
        assert key in merged, f"built-in preset {key!r} should be present"
    # Stored (possibly edited) values win over the defaults.
    assert merged["code_analyze"]["system_prompt"] == "analyze"


def test_merge_does_not_clobber_user_edits():
    # An edited `custom` (enabled, bespoke prompt) plus a missing `reason`.
    edited_custom = {
        "name": "My Persona",
        "character_name": "My Persona",
        "temperature": 0.55,
        "max_tokens": 1234,
        "system_prompt": "You are my bespoke assistant.",
        "inject_prefix": "PRE",
        "inject_suffix": "SUF",
        "enabled": True,
    }
    data_dir = _write_presets({
        "code_analyze": {"name": "Code Analyze", "temperature": 0.2,
                         "max_tokens": 8000, "system_prompt": "analyze"},
        "brainstorm": {"name": "Brainstorm", "temperature": 0.9,
                       "max_tokens": 4096, "system_prompt": "ideate"},
        "custom": edited_custom,
        "user_templates": [{"id": "t1", "name": "Tmpl"}],
        # missing: reason
    })
    pm = PresetManager(data_dir)
    merged = pm.get_all()
    # reason is served...
    assert "reason" in merged
    # ...but the user's edited custom + templates are untouched.
    assert merged["custom"] == edited_custom
    assert merged["user_templates"] == [{"id": "t1", "name": "Tmpl"}]
    assert pm.get_user_templates() == [{"id": "t1", "name": "Tmpl"}]


def test_partial_file_is_not_rewritten():
    # The read-time merge must not rewrite the file (writes only happen on
    # explicit user saves, which then land in that user's slot).
    data = {"code_analyze": {"name": "Code Analyze", "temperature": 0.2,
                             "max_tokens": 8000, "system_prompt": "analyze"}}
    data_dir = _write_presets(data)
    pm = PresetManager(data_dir)
    pm.get_all()
    with open(os.path.join(data_dir, "presets.json"), encoding="utf-8") as f:
        on_disk = json.load(f)
    assert on_disk == data
