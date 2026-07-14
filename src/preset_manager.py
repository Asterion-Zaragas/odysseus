import os
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# System prompt shipped by old builds as the default "custom" preset. Used to
# detect (and neutralize) a legacy file where "custom" was never user-edited.
_LEGACY_CUSTOM_PROMPT = (
    "You are a helpful, balanced assistant. Match your response style to the user's needs."
)


class PresetManager:
    """Owner-scoped preset store.

    Storage lives in one ``data/presets.json`` file, partitioned per user:

        {"_users": {"alice": {"custom": {...}, "user_templates": [...], ...},
                    "bob":   {...}}}

    A user's slot only holds their overrides + custom content + templates +
    group presets; ``DEFAULT_PRESETS`` are merged in at read time (built-ins
    always present, user edits win). Two legacy layouts are still understood:

    - **flat file** (top-level preset keys, no ``_users``): served as a shared
      store to every reader (pre-multi-user behavior). The first authenticated
      write migrates the flat content into that writer's slot, so the admin who
      configured the shared personas keeps them; other users start from
      defaults.
    - **owner=None** (auth disabled): reads/writes the first slot when the file
      is already multi-user, otherwise stays flat — mirroring
      ``routes/prefs_routes.py`` so no-auth deployments behave exactly as before.
    """

    DEFAULT_PRESETS = {
        "code_analyze": {
            "name": "Code Analyze",
            "temperature": 0.2,
            "max_tokens": 8000,
            "system_prompt": """You are a code analyzer.
ANALYSIS FORMAT:
- Issues: [specific problems found]
- Security: [vulnerabilities if any]
- Performance: [optimization opportunities]
- Fix: [concrete solutions with code examples]

Start directly with findings. No preamble. If input isn't code, state: "Input is not code. Please provide code to analyze."
"""
        },
        "brainstorm": {
            "name": "Brainstorm",
            "temperature": 0.9,
            "max_tokens": 4096,
            "system_prompt": """You are a creative ideation assistant focused on divergent thinking.

Generate diverse, unexpected ideas that span from practical to experimental.
- Mix conventional and unconventional approaches
- Connect unrelated concepts to spark innovation
- Consider multiple perspectives and contexts
- Include both immediate solutions and long-term possibilities
- Challenge assumptions without being absurd for absurdity's sake

Structure ideas clearly but allow creative freedom in presentation. Aim for quantity and variety over filtering.
"""
        },
        "reason": {
            "name": "Reason",
            "temperature": 0.3,
            "max_tokens": 6000,
            "system_prompt": """You are a systematic reasoning assistant.

Structure all responses using clear logical progression:
1. Identify key components of the question
2. State relevant principles or facts
3. Build argument step by step
4. Address potential counterarguments
5. Conclude with justified answer

Use precise language. Show causal relationships explicitly. Quantify uncertainty where applicable.
"""
        },
        "custom": {
            "name": "Custom",
            "temperature": 1.0,
            "max_tokens": 0,
            "system_prompt": "",
            "inject_prefix": "",
            "inject_suffix": "",
            "enabled": False,
        }
    }

    def __init__(self, data_dir: str):
        self.presets_file = os.path.join(data_dir, "presets.json")

    # ------------------------------------------------------------------
    # Raw storage
    # ------------------------------------------------------------------

    def _load_raw(self) -> Dict[str, Any]:
        """Load the raw store — either ``{"_users": {...}}`` or a legacy flat
        preset dict. Missing/corrupt file → empty dict (defaults still get
        merged in at read time)."""
        if not os.path.exists(self.presets_file):
            return {}
        try:
            with open(self.presets_file, 'r', encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                logger.error("Error loading presets: expected an object")
                return {}
            return data
        except Exception as e:
            logger.error(f"Error loading presets: {e}")
            return {}

    def _save_raw(self, data: Dict[str, Any]) -> bool:
        try:
            # Atomic write (tmp file + os.replace) so a crash or serialization
            # error mid-write can't truncate presets.json and lose every saved
            # preset. Lazy import keeps this module free of the heavy core
            # package import graph at load time.
            from core.atomic_io import atomic_write_json
            atomic_write_json(self.presets_file, data, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving presets: {e}")
            return False

    # ------------------------------------------------------------------
    # Per-owner slots (same pattern as routes/prefs_routes.py)
    # ------------------------------------------------------------------

    def _load_for_user(self, owner: Optional[str]) -> Dict[str, Any]:
        raw = self._load_raw()
        users = raw.get("_users")
        if isinstance(users, dict):
            if owner is None:
                # Auth disabled — first slot, for backward compat.
                return dict(next(iter(users.values()), {}) or {})
            slot = users.get(owner)
            return dict(slot) if isinstance(slot, dict) else {}
        # Legacy flat file — shared store served to everyone until the first
        # authenticated write migrates it into that writer's slot.
        return dict(raw)

    def _save_for_user(self, owner: Optional[str], slot: Dict[str, Any]) -> bool:
        raw = self._load_raw()
        users = raw.get("_users")
        if owner is None:
            # Auth disabled. If the store is already multi-user, write back
            # into the same (first) slot _load_for_user(None) reads from so the
            # other users' presets are preserved; otherwise stay flat.
            if isinstance(users, dict):
                first_key = next(iter(users), None)
                if first_key is not None:
                    users[first_key] = slot
                    return self._save_raw(raw)
            return self._save_raw(slot)
        if not isinstance(users, dict):
            # First authenticated write: `slot` was loaded from the legacy flat
            # content and mutated, so this migrates the shared personas into
            # this owner's slot. Other users start from defaults.
            raw = {"_users": {}}
        raw["_users"][owner] = slot
        return self._save_raw(raw)

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    @staticmethod
    def _heal_custom(merged: Dict[str, Any]) -> Dict[str, Any]:
        """Neutralize the never-edited legacy default `custom` preset (old
        builds shipped it enabled with a stock prompt and no `enabled` flag)."""
        custom = merged.get("custom")
        if isinstance(custom, dict) and "enabled" not in custom:
            if (
                custom.get("name") == "Custom"
                and not custom.get("character_name")
                and custom.get("system_prompt") == _LEGACY_CUSTOM_PROMPT
            ):
                custom = dict(custom)
                custom["enabled"] = False
                custom["system_prompt"] = ""
                custom["temperature"] = 1.0
                custom["max_tokens"] = 0
                custom.setdefault("inject_prefix", "")
                custom.setdefault("inject_suffix", "")
                merged["custom"] = custom
        return merged

    def get_all(self, owner: Optional[str] = None) -> Dict[str, Any]:
        """All presets visible to `owner`: built-ins merged with their slot
        (user edits win), plus `user_templates` / `group_presets` if present."""
        slot = self._load_for_user(owner)
        merged = {k: dict(v) for k, v in self.DEFAULT_PRESETS.items()}
        merged.update(slot)
        return self._heal_custom(merged)

    def get(self, preset_id: str, owner: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get a specific preset from `owner`'s merged view."""
        return self.get_all(owner).get(preset_id)

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def update_custom(
        self,
        temperature: float,
        max_tokens: int,
        system_prompt: str,
        name: str = "",
        enabled: bool = True,
        inject_prefix: str = "",
        inject_suffix: str = "",
        owner: Optional[str] = None,
    ) -> bool:
        """Update `owner`'s custom preset."""
        slot = self._load_for_user(owner)
        slot["custom"] = {
            "name": name or "Custom",
            "character_name": name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "system_prompt": system_prompt,
            "inject_prefix": inject_prefix,
            "inject_suffix": inject_suffix,
            "enabled": enabled,
        }
        return self._save_for_user(owner, slot)

    def get_user_templates(self, owner: Optional[str] = None) -> List[dict]:
        """Get `owner`'s saved character templates."""
        templates = self._load_for_user(owner).get("user_templates", [])
        return templates if isinstance(templates, list) else []

    def save_user_template(self, template: dict, owner: Optional[str] = None) -> bool:
        """Save a new template for `owner` or update an existing one by id."""
        slot = self._load_for_user(owner)
        templates = slot.get("user_templates", [])
        if not isinstance(templates, list):
            templates = []
        existing = next((i for i, t in enumerate(templates) if t.get("id") == template.get("id")), None)
        if existing is not None:
            templates[existing] = template
        else:
            templates.append(template)
        slot["user_templates"] = templates
        return self._save_for_user(owner, slot)

    def delete_user_template(self, template_id: str, owner: Optional[str] = None) -> bool:
        """Delete one of `owner`'s templates by id."""
        slot = self._load_for_user(owner)
        templates = slot.get("user_templates", [])
        if not isinstance(templates, list):
            templates = []
        slot["user_templates"] = [t for t in templates if t.get("id") != template_id]
        return self._save_for_user(owner, slot)

    def get_group_presets(self, owner: Optional[str] = None) -> list:
        """Get `owner`'s saved group chat presets."""
        groups = self._load_for_user(owner).get("group_presets", [])
        return groups if isinstance(groups, list) else []

    def save_group_presets(self, groups: list, owner: Optional[str] = None) -> bool:
        """Save `owner`'s group chat presets."""
        slot = self._load_for_user(owner)
        slot["group_presets"] = groups
        return self._save_for_user(owner, slot)

    # ------------------------------------------------------------------
    # Backup / restore
    # ------------------------------------------------------------------

    def export_all(self) -> Dict[str, Any]:
        """Full raw store (including the `_users` map) for backup."""
        return self._load_raw()

    def import_all(self, data: Dict[str, Any], owner: Optional[str] = None) -> bool:
        """Merge a backup into the store.

        Multi-user backups (``{"_users": {...}}``) merge slot-by-slot so every
        user's presets round-trip. Legacy flat backups (the old export was one
        merged view) merge into `owner`'s slot.
        """
        if not isinstance(data, dict):
            return False
        incoming_users = data.get("_users")
        if isinstance(incoming_users, dict):
            raw = self._load_raw()
            users = raw.get("_users")
            if not isinstance(users, dict):
                # Current store is legacy flat: park its content under the
                # importing owner's slot so it isn't lost by the merge.
                users = {owner: raw} if (owner and raw) else {}
            for name, slot in incoming_users.items():
                if not isinstance(slot, dict):
                    continue
                cur = users.get(name)
                users[name] = {**cur, **slot} if isinstance(cur, dict) else dict(slot)
            return self._save_raw({"_users": users})
        slot = self._load_for_user(owner)
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                slot[key] = value
        return self._save_for_user(owner, slot)
