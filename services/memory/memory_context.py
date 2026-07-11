"""
memory_context.py

Per-owner memory context document: the shared source of truth for the
tagger (this phase), and the curator/distiller/chat injection that land in
later phases. Pure data + rendering — no LLM calls, no network.

Storage: data/memory_context/<owner-slug>.json

Cold start: a missing file is just an empty registry. The tagger (below)
runs registry-free in that state and every tag it proposes comes back
provisional — there is nothing to promote them into until the first curator
run (Phase 4) bootstraps the registry from whatever got tagged overnight.
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional, Set

from src.constants import DATA_DIR

logger = logging.getLogger(__name__)

CONTEXT_DIR = os.path.join(DATA_DIR, "memory_context")

_SCHEMA_VERSION = 1


def _owner_slug(owner: Optional[str]) -> str:
    """Filesystem-safe filename stem for an owner string.

    Mirrors the slugging used for other per-owner state files (e.g.
    src/builtin_actions.py's note-ping / email-urgency state) so path
    handling stays consistent across the codebase.
    """
    return "".join(c if (c.isalnum() or c in "-_.@") else "_" for c in (owner or "default"))


def _context_path(owner: Optional[str]) -> str:
    return os.path.join(CONTEXT_DIR, f"{_owner_slug(owner)}.json")


def _empty_context() -> Dict:
    return {
        "version": _SCHEMA_VERSION,
        # Curator-maintained (Phase 4) summary sentences for tier-0/pinned facts.
        "core_facts": [],
        # list[{name, description, aliases, count, protected, provisional}]
        "tag_registry": [],
        # Curator-maintained tier/category counters (Phase 4).
        "stats": {},
        "updated_at": 0,
    }


class MemoryContext:
    """Loads/saves one owner's context document."""

    def __init__(self, owner: Optional[str] = None):
        self.owner = owner
        self.path = _context_path(owner)

    def load(self) -> Dict:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                merged = _empty_context()
                merged.update(data)
                return merged
        except FileNotFoundError:
            pass
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Could not read memory context for %r: %s", self.owner, e)
        return _empty_context()

    def save(self, context: Dict) -> None:
        os.makedirs(CONTEXT_DIR, exist_ok=True)
        context = dict(context)
        context["updated_at"] = int(time.time())
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(context, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    # ---- lookups ----

    def registry_names(self) -> Set[str]:
        """Tag names currently in the registry (cold-start: empty set)."""
        ctx = self.load()
        return {t.get("name") for t in (ctx.get("tag_registry") or []) if t.get("name")}

    # ---- compact excerpts for prompts ----

    def registry_excerpt(self, max_tags: int = 50) -> str:
        """Highest-usage tags first, capped at `max_tags` — keeps the tagger's
        prompt bounded regardless of how large the registry grows."""
        ctx = self.load()
        registry = ctx.get("tag_registry") or []
        if not registry:
            return "(no tags registered yet — propose any that fit; they start provisional)"
        ordered = sorted(registry, key=lambda t: -(t.get("count") or 0))[:max_tags]
        lines = []
        for t in ordered:
            name = t.get("name")
            if not name:
                continue
            bits = [name]
            if t.get("description"):
                bits.append(f"— {t['description']}")
            if t.get("aliases"):
                bits.append(f"(aka {', '.join(t['aliases'])})")
            lines.append("- " + " ".join(bits))
        return "\n".join(lines) if lines else "(no tags registered yet — propose any that fit; they start provisional)"

    def core_facts_excerpt(self) -> str:
        ctx = self.load()
        facts = ctx.get("core_facts") or []
        if not facts:
            return "(no core facts recorded yet)"
        return "\n".join(f"- {f}" for f in facts)

    def render_markdown(self) -> str:
        """Full document render — used for the optional chat context-doc
        injection (Phase 6) and as a human-readable debug view."""
        ctx = self.load()
        facts = ctx.get("core_facts") or []
        registry = ctx.get("tag_registry") or []

        lines: List[str] = ["# Memory context", "", "## Core facts"]
        if facts:
            lines.extend(f"- {f}" for f in facts)
        else:
            lines.append("(none yet)")
        lines += ["", "## Tags"]
        if registry:
            for t in sorted(registry, key=lambda t: -(t.get("count") or 0)):
                name = t.get("name")
                if not name:
                    continue
                desc = f" — {t['description']}" if t.get("description") else ""
                lines.append(f"- **{name}**{desc} ({t.get('count', 0)})")
        else:
            lines.append("(none yet)")
        return "\n".join(lines)
