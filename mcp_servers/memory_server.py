"""
memory_server.py

MCP server exposing memory management (list, add, edit, delete, search).
Imports MemoryManager and MemoryVectorStore from the Odysseus codebase.
"""

import asyncio
import os
import sys
import time
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.memory import MemoryStoreUnreadable

server = Server("memory")

# Late-initialized managers (set during first tool call)
_memory_manager = None
_memory_vector = None
_initialized = False

_OWNER_ENV_KEYS = ("ODYSSEUS_MCP_MEMORY_OWNER", "ODYSSEUS_MEMORY_OWNER")
_OWNER_SCOPE_ERROR = (
    "Error: Memory MCP owner is not configured for an owner-scoped memory store. "
    "Set ODYSSEUS_MCP_MEMORY_OWNER for this server or use the owner-aware native memory tool."
)
_UNREADABLE_STORE_ERROR = (
    "Error: Memory store is temporarily unreadable — nothing was saved. "
    "Repair or restore memory.json, then retry."
)


def _configured_owner() -> str | None:
    for key in _OWNER_ENV_KEYS:
        owner = os.environ.get(key, "").strip()
        if owner:
            return owner
    return None


def _entry_owner(entry: dict) -> str | None:
    owner = entry.get("owner")
    if owner is None:
        return None
    owner_text = str(owner).strip()
    return owner_text or None


def _owner_scoped_store(entries: list[dict]) -> bool:
    return any(_entry_owner(entry) for entry in entries if isinstance(entry, dict))


def _scope_entries(for_update: bool = False) -> tuple[str | None, list[dict], list[dict], str | None]:
    """Return configured owner, all entries, visible entries, and optional error.

    ``for_update=True`` is for read-modify-write callers. They save the ``all
    entries`` list back, so an unreadable store must be reported as an error
    instead of degrading to ``[]`` — otherwise the save writes their one new
    entry over the whole store (issue #5673).
    """
    if for_update:
        try:
            entries = _memory_manager.load_all_for_update()
        except MemoryStoreUnreadable as e:
            return None, [], [], f"{_UNREADABLE_STORE_ERROR} ({e})"
    else:
        entries = _memory_manager.load_all()
    owner = _configured_owner()
    if owner is None and _owner_scoped_store(entries):
        return None, entries, [], _OWNER_SCOPE_ERROR
    if owner is None:
        visible = [
            entry for entry in entries
            if isinstance(entry, dict) and _entry_owner(entry) is None
        ]
    else:
        visible = [
            entry for entry in entries
            if isinstance(entry, dict) and _entry_owner(entry) == owner
        ]
    return owner, entries, visible, None


def _text_result(text: str) -> list[TextContent]:
    return [TextContent(type="text", text=text)]


def _ensure_init():
    """Lazy-init memory managers on first use."""
    global _memory_manager, _memory_vector, _initialized
    if _initialized:
        return
    _initialized = True

    from src.constants import DATA_DIR
    from src.memory import MemoryManager
    _memory_manager = MemoryManager(DATA_DIR)

    try:
        from src.memory_vector import MemoryVectorStore
        _memory_vector = MemoryVectorStore(DATA_DIR)
        if not _memory_vector.healthy:
            _memory_vector = None
    except Exception:
        _memory_vector = None


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="manage_memory",
            description="Manage the user's memory system: list, add, edit, delete, or search memories.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "add", "edit", "delete", "search"],
                        "description": "The action to perform",
                    },
                    "text": {"type": "string", "description": "Memory text (add/edit) or search query (search)"},
                    "memory_id": {"type": "string", "description": "Memory ID (edit/delete)"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Facet tags for add, or a single-tag filter for list. Optional — omit to let the memory tagger pick tags automatically.",
                    },
                    "category": {
                        "type": "string",
                        "description": "Deprecated alias for a single tag (add/list filter); prefer 'tags'.",
                    },
                },
                "required": ["action"],
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name != "manage_memory":
        return _text_result(f"Unknown tool: {name}")

    _ensure_init()
    if not _memory_manager:
        return _text_result("Error: Memory manager not available")

    from src.memory import normalize_tag

    def _tag_label(m) -> str:
        tags = m.get("tags") or []
        return ",".join(tags) if tags else "fact"

    action = arguments.get("action", "")

    if action == "list":
        tags_arg = arguments.get("tags")
        filter_raw = tags_arg[0] if isinstance(tags_arg, list) and tags_arg else arguments.get("category", "")
        tag_filter = normalize_tag(filter_raw)
        _owner, _all_memories, memories, scope_error = _scope_entries()
        if scope_error:
            return _text_result(scope_error)
        if tag_filter:
            memories = [m for m in memories if tag_filter in (m.get("tags") or [])]
        if not memories:
            msg = "No memories found"
            if tag_filter:
                msg += f" with tag '{tag_filter}'"
            return _text_result(msg + ".")

        lines = [f"Found {len(memories)} memory entries:\n"]
        for m in memories:
            mid = m.get("id", "?")[:8]
            text = m.get("text", "")
            if len(text) > 150:
                text = text[:150] + "..."
            lines.append(f"- [{_tag_label(m)}] `{mid}` — {text}")
        return _text_result("\n".join(lines))

    elif action == "add":
        text = arguments.get("text", "")
        tags_arg = arguments.get("tags")
        explicit_tags = [str(t) for t in tags_arg] if isinstance(tags_arg, list) and tags_arg else None
        if explicit_tags is None and arguments.get("category"):
            explicit_tags = [str(arguments["category"])]
        if not text:
            return _text_result("Error: Memory text cannot be empty")
        owner, memories, _visible, scope_error = _scope_entries(for_update=True)
        if scope_error:
            return _text_result(scope_error)
        entry = _memory_manager.add_entry(text, source="ai_agent", owner=owner)
        # interactive=True: this MCP call is the synchronous tail of a chat/
        # agent tool call blocking on our reply — even though this subprocess
        # has no interactive_gate state of its own to deadlock against, treat
        # it the same as the other inline write paths for consistency.
        from services.memory.memory_tagger import tag_memory, apply_tags
        tag_result = await tag_memory(text, owner=owner, interactive=True)
        apply_tags(entry, tag_result, user_tags=explicit_tags)
        with _memory_manager.lock:
            memories.append(entry)
            _memory_manager.save(memories)
        if _memory_vector and _memory_vector.healthy:
            try:
                _memory_vector.add(entry["id"], text)
            except Exception:
                pass
        return _text_result(f"Memory added: [{_tag_label(entry)}] {text} (id: {entry['id'][:8]})")

    elif action == "edit":
        memory_id = arguments.get("memory_id", "")
        new_text = arguments.get("text", "")
        if not memory_id or not new_text:
            return _text_result("Error: edit needs memory_id and text")
        _owner, memories, visible, scope_error = _scope_entries()
        if scope_error:
            return _text_result(scope_error)
        full_id = None
        for m in visible:
            if m.get("id", "").startswith(memory_id):
                full_id = m["id"]
                break
        if not full_id:
            return _text_result(f"Error: Memory '{memory_id}' not found")
        with _memory_manager.lock:
            for m in memories:
                if m.get("id") == full_id:
                    m["text"] = new_text
                    m["timestamp"] = int(time.time())
                    break
            _memory_manager.save(memories)

        # Re-tag from the new text (mirrors the memory-page PUT /{id} route
        # and the manage_memory builtin tool's edit action): an edit changes
        # what the memory is about, so tags/generality/tier must be
        # recomputed too, or they silently drift from the text.
        # interactive=True: this call runs inline inside the MCP request
        # handling — waiting on interactive_gate's foreground-quiet check
        # here would deadlock the request against itself.
        from services.memory.memory_tagger import tag_memory, apply_tags
        tag_result = await tag_memory(new_text, owner=_owner, interactive=True)
        with _memory_manager.lock:
            try:
                fresh = _memory_manager.load_all_for_update()
            except MemoryStoreUnreadable as e:
                logger.info("Skipping tag re-apply, memory store unreadable: %s", e)
                fresh = None
            target = None if fresh is None else next(
                (m for m in fresh if m.get("id") == full_id), None
            )
            if target:
                apply_tags(target, tag_result)
                _memory_manager.save(fresh)

        if _memory_vector and _memory_vector.healthy and full_id:
            try:
                _memory_vector.remove(full_id)
                _memory_vector.add(full_id, new_text)
            except Exception:
                pass
        return _text_result(f"Memory updated: {new_text}")

    elif action == "delete":
        memory_id = arguments.get("memory_id", "")
        if not memory_id:
            return _text_result("Error: delete needs memory_id")
        _owner, memories, visible, scope_error = _scope_entries()
        if scope_error:
            return _text_result(scope_error)
        full_id = None
        deleted_text = ""
        deleted_label = ""
        for m in visible:
            if m.get("id", "").startswith(memory_id):
                full_id = m["id"]
                deleted_text = m.get("text", "")
                deleted_label = _tag_label(m)
                break
        if not full_id:
            return _text_result(f"Error: Memory '{memory_id}' not found")
        memories = [m for m in memories if m.get("id") != full_id]
        _memory_manager.save(memories)
        if _memory_vector and _memory_vector.healthy and full_id:
            try:
                _memory_vector.remove(full_id)
            except Exception:
                pass
        cat = f"[{deleted_label}] " if deleted_label else ""
        snippet = deleted_text if len(deleted_text) <= 120 else deleted_text[:117] + "..."
        return _text_result(f"Memory deleted: {cat}{snippet} (id: {memory_id})")

    elif action == "search":
        query = arguments.get("text", "")
        if not query:
            return _text_result("Error: search needs text (query)")
        _owner, _all_memories, memories, scope_error = _scope_entries()
        if scope_error:
            return _text_result(scope_error)
        if hasattr(_memory_manager, 'get_relevant_memories'):
            results = _memory_manager.get_relevant_memories(query, memories, threshold=0.05, max_items=20)
        else:
            query_lower = query.lower()
            results = [m for m in memories if query_lower in m.get("text", "").lower()][:20]
        if not results:
            return _text_result(f"No memories found matching '{query}'.")
        lines = [f"Found {len(results)} matching memories:\n"]
        for m in results:
            mid = m.get("id", "?")[:8]
            text = m.get("text", "")
            lines.append(f"- [{_tag_label(m)}] `{mid}` — {text}")
        return _text_result("\n".join(lines))

    else:
        return _text_result(f"Error: Unknown action '{action}'. Use: list, add, edit, delete, search")


async def run():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(run())
