"""Per-user preset isolation (plans/2026-07-13-per-user-personas-and-model-defaults.md).

Presets live in one data/presets.json partitioned by owner
(``{"_users": {"alice": {...}, "bob": {...}}}``). Built-ins are merged in at
read time; a user's slot only holds their overrides/custom content/templates/
groups. A legacy flat file keeps serving everyone (shared, pre-multi-user
behavior) until the first authenticated write migrates it into that writer's
slot. owner=None (auth disabled) mirrors routes/prefs_routes.py: flat store
stays flat, a multi-user store reads/writes the first slot.
"""
import json

from src.preset_manager import PresetManager


def _read(tmp_path):
    return json.loads((tmp_path / "presets.json").read_text(encoding="utf-8"))


def test_two_owners_are_isolated(tmp_path):
    pm = PresetManager(str(tmp_path))

    assert pm.update_custom(0.7, 1000, "I am Alice's persona", name="Aria",
                            enabled=True, owner="alice") is True
    assert pm.update_custom(0.2, 500, "I am Bob's persona", name="Bort",
                            enabled=True, owner="bob") is True

    a = pm.get("custom", "alice")
    b = pm.get("custom", "bob")
    assert a["system_prompt"] == "I am Alice's persona"
    assert b["system_prompt"] == "I am Bob's persona"
    assert a["character_name"] == "Aria" and b["character_name"] == "Bort"

    # Built-ins are present for both.
    for owner in ("alice", "bob"):
        merged = pm.get_all(owner)
        for key in PresetManager.DEFAULT_PRESETS:
            assert key in merged

    # A user with no slot at all still gets the defaults.
    assert pm.get_all("carol") == PresetManager.DEFAULT_PRESETS


def test_templates_and_groups_are_isolated(tmp_path):
    pm = PresetManager(str(tmp_path))

    assert pm.save_user_template({"id": "t1", "name": "A-tmpl"}, owner="alice")
    assert pm.save_group_presets([{"name": "A-group"}], owner="alice")

    assert pm.get_user_templates("alice") == [{"id": "t1", "name": "A-tmpl"}]
    assert pm.get_user_templates("bob") == []
    assert pm.get_group_presets("alice") == [{"name": "A-group"}]
    assert pm.get_group_presets("bob") == []

    # Bob's writes don't touch Alice's slot.
    assert pm.save_user_template({"id": "t2", "name": "B-tmpl"}, owner="bob")
    assert pm.get_user_templates("alice") == [{"id": "t1", "name": "A-tmpl"}]

    # Template update-by-id and delete stay scoped.
    assert pm.save_user_template({"id": "t1", "name": "A-tmpl-2"}, owner="alice")
    assert pm.get_user_templates("alice") == [{"id": "t1", "name": "A-tmpl-2"}]
    assert pm.delete_user_template("t1", owner="alice")
    assert pm.get_user_templates("alice") == []
    assert pm.get_user_templates("bob") == [{"id": "t2", "name": "B-tmpl"}]


def test_legacy_flat_file_migrates_to_first_authenticated_writer(tmp_path):
    # A shared pre-multi-user file with a configured persona + template.
    legacy = {
        "custom": {"name": "Shared", "character_name": "Shared",
                   "temperature": 0.5, "max_tokens": 100,
                   "system_prompt": "shared persona", "enabled": True},
        "user_templates": [{"id": "t1", "name": "Shared tmpl"}],
    }
    (tmp_path / "presets.json").write_text(json.dumps(legacy), encoding="utf-8")
    pm = PresetManager(str(tmp_path))

    # Before any write, everyone still sees the shared store (status quo).
    assert pm.get("custom", "alice")["system_prompt"] == "shared persona"
    assert pm.get("custom", "bob")["system_prompt"] == "shared persona"

    # First authenticated write migrates the flat content into that owner's
    # slot — the admin who configured the shared personas keeps them.
    assert pm.save_user_template({"id": "t2", "name": "New"}, owner="admin")
    raw = _read(tmp_path)
    assert set(raw) == {"_users"}
    assert raw["_users"]["admin"]["custom"]["system_prompt"] == "shared persona"

    assert pm.get("custom", "admin")["system_prompt"] == "shared persona"
    assert pm.get_user_templates("admin") == [
        {"id": "t1", "name": "Shared tmpl"}, {"id": "t2", "name": "New"}]
    # Other users start from defaults.
    assert pm.get("custom", "bob")["enabled"] is False
    assert pm.get_user_templates("bob") == []


def test_owner_none_keeps_legacy_flat_behavior(tmp_path):
    pm = PresetManager(str(tmp_path))

    assert pm.update_custom(0.9, 200, "single-user persona", name="Solo",
                            enabled=True, owner=None) is True
    # No-auth mode never grows a _users wrapper.
    assert "_users" not in _read(tmp_path)
    assert pm.get("custom", None)["system_prompt"] == "single-user persona"

    reloaded = PresetManager(str(tmp_path))
    assert reloaded.get("custom")["system_prompt"] == "single-user persona"


def test_owner_none_on_multiuser_store_uses_first_slot(tmp_path):
    # Auth turned off on a previously multi-user deployment: owner=None reads
    # and writes the first slot, preserving every other user's presets.
    pm = PresetManager(str(tmp_path))
    assert pm.update_custom(0.7, 100, "alice persona", name="A", owner="alice")
    assert pm.update_custom(0.2, 100, "bob persona", name="B", owner="bob")

    assert pm.get("custom", None)["system_prompt"] == "alice persona"
    assert pm.save_user_template({"id": "t1", "name": "T"}, owner=None)
    raw = _read(tmp_path)
    assert raw["_users"]["alice"]["user_templates"] == [{"id": "t1", "name": "T"}]
    assert raw["_users"]["bob"]["custom"]["system_prompt"] == "bob persona"


def test_export_import_round_trips_users_map(tmp_path):
    pm = PresetManager(str(tmp_path))
    assert pm.update_custom(0.7, 100, "alice persona", name="A", owner="alice")
    assert pm.save_user_template({"id": "t1", "name": "T"}, owner="bob")

    exported = pm.export_all()
    assert set(exported["_users"]) == {"alice", "bob"}

    # Restore into a fresh store — both users' presets survive.
    restored = PresetManager(str(tmp_path / "restore"))
    (tmp_path / "restore").mkdir()
    assert restored.import_all(exported, owner="admin") is True
    assert restored.get("custom", "alice")["system_prompt"] == "alice persona"
    assert restored.get_user_templates("bob") == [{"id": "t1", "name": "T"}]


def test_import_legacy_flat_backup_lands_in_importer_slot(tmp_path):
    # Old exports were one merged flat view — they import into the importing
    # user's slot instead of overwriting everyone.
    pm = PresetManager(str(tmp_path))
    assert pm.update_custom(0.2, 100, "bob persona", name="B", owner="bob")

    flat_backup = {
        "custom": {"name": "Restored", "temperature": 0.5, "max_tokens": 10,
                   "system_prompt": "restored persona", "enabled": True},
        "user_templates": [{"id": "t9", "name": "Old tmpl"}],
        "not_a_preset": "ignored-scalar",
    }
    assert pm.import_all(flat_backup, owner="admin") is True
    assert pm.get("custom", "admin")["system_prompt"] == "restored persona"
    assert pm.get_user_templates("admin") == [{"id": "t9", "name": "Old tmpl"}]
    # Bob untouched.
    assert pm.get("custom", "bob")["system_prompt"] == "bob persona"
