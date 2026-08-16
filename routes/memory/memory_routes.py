# routes/memory_routes.py
from fastapi import APIRouter, Form, HTTPException, Request, UploadFile, File
from typing import Dict, Any, Optional, List
import json
import os
import re
import tempfile
import time
from datetime import datetime
import logging

# Leading list-marker like "1.", "12)", or "3:" plus surrounding whitespace.
# Strips one prefix per call so import-from-LLM-output doesn't leave the
# numbering inside the saved memory text. Bullet markers (-, *, •) are
# also peeled here for the same reason.
_LIST_PREFIX_RE = re.compile(r"^\s*(?:\d{1,3}[.):]\s+|[-*•]\s+)")


def _strip_list_prefix(text: str) -> str:
    if not text:
        return text
    return _LIST_PREFIX_RE.sub("", text, count=1).strip()

from services.memory import MemoryManager, MemoryStoreUnreadable
from core.session_manager import SessionManager
from src.memory import compat_category, normalize_tags
from src.request_models import MemoryAddRequest
from core.database import SessionLocal
from src.llm_core import llm_call_async
from services.memory.memory_curator import curate, read_curation_log, undo_expire, undo_reword, clear_quarantine, list_quarantined
from services.memory.memory_context import MemoryContext
from src.auth_helpers import get_current_user, require_user
from src.endpoint_resolver import resolve_endpoint
from src.task_endpoint import resolve_task_endpoint
from src.upload_limits import read_upload_limited, MEMORY_IMPORT_MAX_BYTES

logger = logging.getLogger(__name__)


def _load_for_update(memory_manager) -> List[Dict[str, Any]]:
    """Load the whole store for a read-modify-write cycle.

    A transient read failure must not look like an empty store: the caller
    would append to ``[]`` and save that back, atomically destroying every
    existing memory (issue #5673). Surface it as a 503 and change nothing.
    """
    try:
        return memory_manager.load_all_for_update()
    except MemoryStoreUnreadable as e:
        logger.error("Refusing to rewrite the memory store: %s", e)
        raise HTTPException(
            503, "Memory store is temporarily unreadable — no changes were made."
        )


def setup_memory_routes(memory_manager: MemoryManager, session_manager: SessionManager, memory_vector=None):
    """Set up memory-related routes."""
    router = APIRouter(prefix="/api/memory", tags=["memory"])

    def _owner(request: Request) -> Optional[str]:
        return get_current_user(request)

    def _assert_session_owner(session_obj, user):
        """SECURITY: 404 if the caller does not own this session.

        SessionManager.get_session is NOT owner-scoped — it returns any
        session by id. These routes accept a caller-supplied session id, so
        without this gate a user could target another tenant's session and
        leak their chat history, their session-scoped LLM credentials, or the
        session title. Mirrors session_routes / webhook_routes ownership.
        """
        if user is not None and getattr(session_obj, "owner", None) != user:
            raise HTTPException(404, "Session not found")

    def _verify_memory_owner(memory: dict, user: Optional[str]):
        """Raise 404 if user doesn't own this memory.

        SECURITY: strict ownership — previously `mem_owner and mem_owner != user`
        allowed any user to read/edit/delete memories with an empty/null owner
        field, which leaked legacy data across the multi-user deploy.
        """
        if user is None:
            return  # Auth disabled
        if memory.get("owner") != user:
            raise HTTPException(404, "Memory not found")

    def _with_compat(memory: dict) -> dict:
        """Response copy carrying a computed legacy `category` field.

        Storage is tags-only; the memory page reads `tags`/`tier` directly
        since Phase 7. `category` (= first tag) is kept as a one-release
        back-compat alias for API consumers that never migrated (MCP
        clients, old exports) — see the implementation plan's Phase 8
        cleanup notes.
        """
        return {**memory, "category": compat_category(memory)}

    @router.post("/debug")
    def debug_memory_relevance(request: Request, query: str = Form(...)):
        """Debug which memories would be triggered for a query"""
        user = _owner(request)
        memories = memory_manager.load(owner=user)
        relevant = memory_manager.get_relevant_memories(query, memories, threshold=0.05)

        return {
            "query": query,
            "total_memories": len(memories),
            "relevant_count": len(relevant),
            "relevant_memories": [{"text": m["text"], "tags": m.get("tags") or [],
                                   "category": compat_category(m)}
                                 for m in relevant]
        }

    @router.post("/add", response_model=Dict[str, Any])
    async def api_add_memory(
        request: Request,
        memory_data: Optional[MemoryAddRequest] = None
    ):
        """Add a new memory entry with optional tags, source, and session reference."""
        from src.auth_helpers import require_privilege
        require_privilege(request, "can_manage_memory")
        if memory_data is None:
            form = await request.form()
            raw_tags = form.get("tags")
            tags = None
            if raw_tags:
                try:
                    parsed = json.loads(raw_tags)
                    tags = parsed if isinstance(parsed, list) else None
                except json.JSONDecodeError:
                    tags = [t for t in raw_tags.split(",") if t.strip()]
            memory_data = MemoryAddRequest(
                text=form.get("text"),
                tags=tags,
                category=form.get("category"),
                source=form.get("source", "user"),
                session_id=form.get("session_id")
            )

        user = _owner(request)
        text = (memory_data.text or "").strip()
        if not text:
            raise HTTPException(400, "empty memory")
        user_mem = memory_manager.load(owner=user)
        if memory_manager.find_duplicates(text, user_mem):
            return {"ok": True, "count": len(user_mem), "message": "Memory already exists"}

        if memory_data.session_id:
            try:
                session_obj = session_manager.get_session(memory_data.session_id)
            except KeyError:
                raise HTTPException(404, "Session not found")
            _assert_session_owner(session_obj, user)

        new_entry = memory_manager.add_entry(
            text, memory_data.source, tags=memory_data.tags,
            category=memory_data.category, owner=user,
        )
        if memory_data.session_id:
            new_entry["session_id"] = memory_data.session_id
        # interactive=True: this route runs inline inside its own tracked HTTP
        # request, so waiting on interactive_gate's foreground-quiet check here
        # would deadlock the request against itself.
        from services.memory.memory_tagger import tag_memory, apply_tags
        tag_result = await tag_memory(text, owner=user, interactive=True)
        apply_tags(new_entry, tag_result, user_tags=memory_data.tags)
        with memory_manager.lock:
            all_mem = _load_for_update(memory_manager)
            all_mem.append(new_entry)
            memory_manager.save(all_mem)
        # Sync vector index
        if memory_vector and memory_vector.healthy:
            memory_vector.add(new_entry["id"], text)
        try:
            from src.event_bus import fire_event
            fire_event("memory_added", user)
        except Exception:
            logger.debug("memory_added event dispatch failed", exc_info=True)
        return {"ok": True, "count": len([m for m in all_mem if m.get("owner") == user])}

    @router.get("")
    def api_get_memory(request: Request):
        """Return all memory entries with their metadata."""
        user = _owner(request)
        return {"memory": [_with_compat(m) for m in memory_manager.load(owner=user)]}

    @router.post("/search")
    def search_memories(request: Request, query: str = Form(...), session_id: str = Form(None),
                        tag: str = Form(None), category: str = Form(None)):
        """Search across all memories with optional filters.

        `tag` filters on the tags list; `category` is a legacy alias for it.
        """
        user = _owner(request)
        memories = memory_manager.load(owner=user)

        if session_id:
            memories = [m for m in memories if m.get("session_id") == session_id]

        tag_filter = tag or category
        # Direct (non-FastAPI) callers leave the Form(None) sentinel in place
        # of an omitted param — only a real string is a filter.
        if not isinstance(tag_filter, str):
            tag_filter = None
        if tag_filter:
            wanted = normalize_tags([tag_filter])
            memories = [m for m in memories if wanted and wanted[0] in (m.get("tags") or [])]

        relevant = memory_manager.get_relevant_memories(query, memories, threshold=0.05, max_items=20)

        return {"memories": [_with_compat(m) for m in relevant], "total": len(relevant), "query": query}

    @router.get("/timeline")
    def memory_timeline(request: Request):
        """Get memories in chronological order with source session information."""
        user = _owner(request)
        memories = memory_manager.load(owner=user)
        sorted_memories = sorted(memories, key=lambda x: x.get("timestamp", 0), reverse=True)

        results = []
        for memory in sorted_memories:
            memory = _with_compat(memory)
            if "timestamp" in memory:
                try:
                    dt = datetime.fromtimestamp(memory["timestamp"])
                    memory["timestamp_str"] = dt.strftime("%Y-%m-%d %H:%M:%S")
                except (ValueError, OSError, OverflowError):
                    memory["timestamp_str"] = "Unknown"
            else:
                memory["timestamp_str"] = "Unknown"

            session_id = memory.get("session_id")
            if session_id and session_id in session_manager.sessions:
                try:
                    session = session_manager.get_session(session_id)
                    if session:
                        _assert_session_owner(session, user)
                    memory["session_name"] = session.name if session else f"Session {session_id[:6]}"
                except KeyError:
                    memory["session_name"] = "Unknown"
                except HTTPException as exc:
                    if exc.status_code != 404:
                        raise
                    memory["session_name"] = "Unknown"
            else:
                memory["session_name"] = "Unknown"

            results.append(memory)

        return {"timeline": results, "total": len(results)}

    @router.get("/by-session/{session_id}")
    def get_memory_by_session(request: Request, session_id: str):
        """Get all memories associated with a specific session."""
        user = _owner(request)
        try:
            _session_obj = session_manager.get_session(session_id)
        except KeyError:
            raise HTTPException(404, f"Session {session_id} not found")
        _assert_session_owner(_session_obj, user)
        memories = memory_manager.load(owner=user)
        session_memories = [_with_compat(m) for m in memories if m.get("session_id") == session_id]

        session_memories.sort(key=lambda x: x.get("timestamp", 0), reverse=True)

        try:
            session = session_manager.get_session(session_id)
            session_name = session.name if session else f"Session {session_id[:6]}"
        except KeyError:
            session_name = f"Session {session_id[:6]}"

        for memory in session_memories:
            memory["session_name"] = session_name

        return {
            "session_id": session_id,
            "session_name": session_name,
            "memory_count": len(session_memories),
            "memories": session_memories
        }

    @router.post("/extract")
    async def extract_memory(request: Request, session: str = Form(...)) -> Dict[str, List[str]]:
        """Analyze a session's chat history and return memory suggestions."""
        require_user(request)
        try:
            sess = session_manager.get_session(session)
        except KeyError:
            raise HTTPException(404, "Session not found")
        _assert_session_owner(sess, _owner(request))

        system_msg = {
            "role": "system",
            "content": (
                "You are a helpful assistant. Analyze the entire conversation history provided and extract any "
                "useful factual statements, contacts, addresses, phone numbers, or other information that the user "
                "might want to remember for future interactions. Return each piece of information as a JSON object "
                "with a 'text' field. For example: [{'text': 'Alice lives at 123 Main St'}, {'text': 'Bob works at Acme Corp'}]. "
                "Only include information that is specific and likely to be useful later."
            ),
        }
        messages = [system_msg] + sess.get_context_messages()

        t_url, t_model, t_headers = resolve_task_endpoint(
            sess.endpoint_url, sess.model, sess.headers, owner=_owner(request)
        )

        try:
            suggestion_text = await llm_call_async(
                t_url,
                t_model,
                messages,
                temperature=0.2,
                max_tokens=500,
                headers=t_headers,
            )
            try:
                suggestions = json.loads(suggestion_text)
                if isinstance(suggestions, list):
                    suggestions = [s if isinstance(s, str) else s.get("text", "") for s in suggestions]
                else:
                    suggestions = []
            except json.JSONDecodeError:
                suggestions = [line.strip() for line in suggestion_text.splitlines() if line.strip()]

            return {"suggestions": [s for s in suggestions if s]}
        except Exception as e:
            logger.error(f"LLM memory extraction failed (session {session}): {e}")
            fallback = memory_manager.extract_memory_from_chat(sess.history, session)
            return {"suggestions": [item["text"] for item in fallback]}

    @router.post("/audit")
    async def api_audit_memories(request: Request, dry_run: bool = Form(False), force: bool = Form(False)):
        """Run the memory curator for the caller: dedupe/merge, tag
        normalization, tier rescoring, archive expiry, and a context-doc
        rebuild — batched, nightly-run's manual-trigger twin.

        Uses the "memory smart" model role (Settings -> AI Defaults ->
        Memory Models), falling back through the background-task chain like
        every other memory-system agent — no per-request model resolution
        needed here anymore. `dry_run=true` runs the full pipeline and logs
        proposed actions without writing anything back. `force=true` (the
        Curator tab's "Force run" button) bypasses the tidy-fingerprint
        short-circuit so a real pass runs even if the store already looks
        "clean" — e.g. to retry batches a prior run silently no-op'd on.
        """
        user = _owner(request)
        # interactive=True: this runs INLINE inside the audit request, which is
        # itself counted in the interactive gate's active-request tally. Passing
        # False (the nightly-loop default) would make curate()'s memory-smart
        # calls wait for the request count to hit zero — i.e. wait on this very
        # request — and deadlock (button stuck on "Running…", no LLM call made).
        result = await curate(memory_manager, memory_vector, owner=user, dry_run=dry_run, interactive=True, force=force)

        return {
            "ok": True,
            "before": result.get("before", 0),
            "after": result.get("after", 0),
            "removed": result.get("before", 0) - result.get("after", 0),
            # True when the run skipped the LLM passes because nothing
            # changed since the last curation. Frontend already says
            # "Already clean" for removed==0, so this is here for future
            # use / debugging.
            "already_tidy": bool(result.get("already_tidy")),
            "dry_run": dry_run,
            # True if any batch failed to get a usable LLM reply (or was
            # refused by the dedupe safety guard) — those batches were left
            # un-curated and will be retried automatically on the next run
            # since the tidy fingerprint isn't saved in that case.
            "had_failures": bool(result.get("had_failures")),
            "failure_count": result.get("failure_count", 0),
        }

    @router.get("/curation-log")
    def get_curation_log(request: Request, limit: int = 100):
        """Recent curator changelog entries for the caller (merge/retag/
        demote/promote/expire), most recent last. Dry-run entries are
        excluded — they were never applied, so the curator panel would
        otherwise show phantom actions with no matching store change."""
        user = _owner(request)
        return {"log": read_curation_log(user, limit=limit, include_dry_run=False)}

    @router.get("/context-doc")
    def get_context_doc(request: Request):
        """The caller's curator-maintained memory context document: core
        facts + tag registry, plus a markdown render for read-only display.
        Cold start (curator has never run for this owner) returns an empty
        document, not a 404 — there's nothing wrong, just nothing yet."""
        user = _owner(request)
        ctx = MemoryContext(user)
        doc = ctx.load()
        return {
            "markdown": ctx.render_markdown(),
            "core_facts": doc.get("core_facts") or [],
            "tag_registry": doc.get("tag_registry") or [],
            "updated_at": doc.get("updated_at") or 0,
        }

    @router.post("/curation-log/undo")
    def undo_curation_action(request: Request, memory_id: str = Form(...), action: str = Form("expire")):
        """Undo a curator changelog action by re-applying its snapshot.
        `action="expire"` (default, preserves existing callers) restores a
        deleted entry; `action="reword"` restores an entry's pre-reword
        text."""
        from src.auth_helpers import require_privilege
        require_privilege(request, "can_manage_memory")
        user = _owner(request)
        if action == "reword":
            ok = undo_reword(memory_manager, user, memory_id)
            not_found_detail = "No reworded memory found with that id"
        else:
            ok = undo_expire(memory_manager, memory_vector, user, memory_id)
            not_found_detail = "No expired memory found with that id"
        if not ok:
            raise HTTPException(404, not_found_detail)
        return {"ok": True}

    @router.get("/quarantine")
    def get_quarantined_memories(request: Request):
        """Entries the curator has stopped feeding to a given pass
        (tag_backfill/triage/dedupe/reword) after `memory_curator_quarantine_after`
        separate runs kept failing on that entry alone — see
        memory_curator.py's module docstring. Enriched with the entry's
        current text so the Curator tab
        can show something more useful than a bare id."""
        user = _owner(request)
        quarantined = list_quarantined(user)
        if not quarantined:
            return {"quarantined": []}
        by_id = {e["id"]: e for e in memory_manager.load(owner=user)}
        for q in quarantined:
            entry = by_id.get(q["id"])
            q["text"] = entry.get("text", "") if entry else None
        return {"quarantined": quarantined}

    @router.post("/quarantine/clear")
    def clear_quarantined_memory(request: Request, memory_id: str = Form(...), pass_name: str | None = Form(None)):
        """Manually clear a quarantine flag (Curator tab "retry" button) so
        the next curator run includes this entry in that pass again."""
        from src.auth_helpers import require_privilege
        require_privilege(request, "can_manage_memory")
        user = _owner(request)
        if not clear_quarantine(memory_manager, user, memory_id, pass_name):
            raise HTTPException(404, "No quarantine found for that entry/pass")
        return {"ok": True}

    @router.post("/import")
    async def import_memories_from_file(
        request: Request,
        session: str | None = Form(None),
        file: UploadFile = File(...)
    ):
        """Extract memory suggestions from an uploaded file (PDF, TXT, MD, etc.)."""
        from src.auth_helpers import require_privilege
        require_privilege(request, "can_manage_memory")

        endpoint_url = None
        model = None
        headers = {}

        user = _owner(request)

        if session:
            try:
                sess = session_manager.get_session(session)
                _assert_session_owner(sess, user)
            except KeyError:
                sess = None
            except HTTPException as exc:
                if exc.status_code != 404:
                    raise
                sess = None

            if sess is None:
                logger.warning("Session %s not found or inaccessible, falling back to utility endpoint", session)
                endpoint_url, model, headers = resolve_endpoint("utility", owner=user)
            else:
                endpoint_url, model, headers = resolve_task_endpoint(
                    sess.endpoint_url, sess.model, sess.headers, owner=user
                )
        else:
            endpoint_url, model, headers = resolve_task_endpoint(owner=user)
    
        if not endpoint_url or not model:
            raise HTTPException(400, "No LLM model configured. Set a default model in Settings.")

        content = await read_upload_limited(file, MEMORY_IMPORT_MAX_BYTES, "Memory import")
        filename = file.filename or "upload"
        _, ext = os.path.splitext(filename.lower())

        allowed = {".txt", ".md", ".pdf", ".csv", ".log", ".json", ".py", ".js", ".html"}
        if ext not in allowed:
            raise HTTPException(400, f"Unsupported file type: {ext}")

        # Extract text based on file type
        if ext == ".pdf":
            from src.document_processor import _process_pdf
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            try:
                text = _process_pdf(tmp_path, owner=_owner(request))
            finally:
                os.unlink(tmp_path)
        else:
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                from charset_normalizer import detect
                encoding = (detect(content) or {}).get("encoding") or "utf-8"
                text = content.decode(encoding, errors="replace")

        if not text.strip():
            return {"suggestions": [], "message": "No readable content found"}

        # Fast path: a .json upload that already looks like a memories export
        # (list of {text, category, ...} dicts, or list of strings) round-trips
        # directly without spending an LLM call to re-extract its own output.
        # Without this, re-importing a memories.json from another account
        # ran the file through the extractor, which often re-emitted the
        # entries as a numbered list (and the numbering leaked into the
        # `text` field).
        if ext == ".json":
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list) and parsed:
                direct = []
                for item in parsed:
                    if isinstance(item, dict) and item.get("text"):
                        # Post-upgrade exports carry `tags`; pre-upgrade ones
                        # carry `category`. Preserve both for the save step.
                        item_tags = normalize_tags(item.get("tags") or [])
                        direct.append({
                            "text": _strip_list_prefix(str(item["text"])),
                            "tags": item_tags,
                            "category": item.get("category") or (item_tags[0] if item_tags else "fact"),
                        })
                    elif isinstance(item, str) and item.strip():
                        direct.append({
                            "text": _strip_list_prefix(item.strip()),
                            "category": "fact",
                        })
                if direct:
                    return {"suggestions": direct, "filename": filename}

        # Truncate very long documents
        if len(text) > 15000:
            text = text[:15000] + "\n[Truncated]"

        # Send to LLM for memory extraction
        import_prompt = (
            "You are a memory extraction assistant. The user uploaded a document. "
            "Analyze the text below and extract specific, useful facts — things like "
            "names, preferences, jobs, locations, relationships, opinions, projects, "
            "goals, contacts, or any other personal details worth remembering.\n\n"
            "Rules:\n"
            "- Each fact should be a short, self-contained statement\n"
            "- Do NOT extract generic knowledge\n"
            "- Focus on personal, memorable information\n"
            "- If there are no useful facts, return an empty array\n\n"
            "Return a JSON array of objects with 'text' and 'category' fields.\n"
            "Categories: 'identity', 'preference', 'fact', 'contact', 'project', 'goal'\n\n"
            "Return ONLY valid JSON, no markdown fences."
        )

        try:
            raw = await llm_call_async(
                endpoint_url,
                model,
                [
                    {"role": "system", "content": import_prompt},
                    {"role": "user", "content": f"Document: {filename}\n\n{text}"},
                ],
                temperature=0.2,
                max_tokens=2000,
                headers=headers,
            )

            # Parse JSON
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            suggestions = json.loads(raw)
            if isinstance(suggestions, list):
                normalized = []
                for s in suggestions:
                    if not s:
                        continue
                    if isinstance(s, dict):
                        s = dict(s)
                        if s.get("text"):
                            s["text"] = _strip_list_prefix(str(s["text"]))
                        normalized.append(s)
                    else:
                        normalized.append({"text": _strip_list_prefix(str(s)), "category": "fact"})
                suggestions = normalized
            else:
                suggestions = []

            return {"suggestions": suggestions, "filename": filename}

        except json.JSONDecodeError:
            # Fallback: split by lines, stripping any "1.", "2)" markdown-list
            # numbering the model added so saved memories don't keep the prefix.
            lines = [_strip_list_prefix(l.strip()) for l in raw.splitlines() if l.strip() and len(l.strip()) > 5]
            return {"suggestions": [{"text": l, "category": "fact"} for l in lines[:20]], "filename": filename}
        except Exception as e:
            logger.error(f"Memory import extraction failed: {e}")
            raise HTTPException(502, f"LLM extraction failed: {str(e)}")

    @router.post("/{memory_id}/pin")
    def pin_memory(request: Request, memory_id: str, pinned: bool = Form(True)):
        """Pin or unpin a memory. Pinned memories are always included in context."""
        user = _owner(request)
        with memory_manager.lock:
            all_mem = _load_for_update(memory_manager)
            for i, memory in enumerate(all_mem):
                if memory["id"] == memory_id:
                    _verify_memory_owner(memory, user)
                    all_mem[i]["pinned"] = pinned
                    memory_manager.save(all_mem)
                    return {"ok": True, "pinned": pinned}
        raise HTTPException(404, f"Memory item {memory_id} not found")

    # Wildcard routes MUST come last — otherwise they swallow /import, /search, etc.
    @router.get("/{memory_id}")
    def get_memory_item(request: Request, memory_id: str):
        """Get a specific memory item by ID."""
        user = _owner(request)
        memories = memory_manager.load(owner=user)
        for memory in memories:
            if memory["id"] == memory_id:
                return {"memory": _with_compat(memory)}

        raise HTTPException(404, "Memory not found")

    @router.put("/{memory_id}")
    async def update_memory(request: Request, memory_id: str, text: str = Form(...),
                      tags: str = Form(None), category: str = Form(None)):
        """Update an existing memory item with new text and optional tags.

        `tags` (JSON array or comma-separated) replaces the tag list — the
        tagger is still called for a fresh generality/tier, but its own tag
        proposals are dropped in favor of the explicit list. `category` is
        the legacy alias: it swaps the first tag (which is the migrated
        category) and keeps the rest, then the tagger fills in the remainder.
        Neither given: the tagger re-tags from the new text and merges into
        the existing tags (nothing is dropped on a tagger failure).
        """
        user = _owner(request)
        explicit_tags: Optional[List[str]] = None
        with memory_manager.lock:
            all_mem = _load_for_update(memory_manager)
            target = next((m for m in all_mem if m["id"] == memory_id), None)
            if not target:
                raise HTTPException(404, f"Memory item {memory_id} not found")
            _verify_memory_owner(target, user)

            new_text = text.strip()
            target["text"] = new_text
            if tags is not None:
                try:
                    parsed = json.loads(tags)
                    tag_list = parsed if isinstance(parsed, list) else [tags]
                except json.JSONDecodeError:
                    tag_list = tags.split(",")
                explicit_tags = normalize_tags(tag_list)
                target["tags"] = explicit_tags
            elif category:
                rest = (target.get("tags") or [])[1:]
                target["tags"] = normalize_tags([category] + rest)
            target["timestamp"] = int(time.time())
            memory_manager.save(all_mem)

        # interactive=True: this route runs inline inside its own tracked HTTP
        # request, so waiting on interactive_gate's foreground-quiet check here
        # would deadlock the request against itself.
        from services.memory.memory_tagger import tag_memory, apply_tags
        tag_result = await tag_memory(new_text, owner=user, interactive=True)
        with memory_manager.lock:
            all_mem = _load_for_update(memory_manager)
            target = next((m for m in all_mem if m["id"] == memory_id), None)
            if target:
                apply_tags(target, tag_result, user_tags=explicit_tags)
                memory_manager.save(all_mem)

        # Sync vector index (remove old, add updated)
        if memory_vector and memory_vector.healthy:
            memory_vector.remove(memory_id)
            memory_vector.add(memory_id, new_text)
        return {"ok": True, "message": "Memory updated successfully"}

    @router.delete("/{memory_id}")
    def delete_memory(request: Request, memory_id: str):
        """Delete a memory item by its ID."""
        user = _owner(request)
        with memory_manager.lock:
            all_mem = _load_for_update(memory_manager)

            # Find and verify ownership before deleting
            target = next((m for m in all_mem if m["id"] == memory_id), None)
            if not target:
                raise HTTPException(404, f"Memory item {memory_id} not found")
            _verify_memory_owner(target, user)

            all_mem = [m for m in all_mem if m["id"] != memory_id]
            memory_manager.save(all_mem)
        # Sync vector index
        if memory_vector and memory_vector.healthy:
            memory_vector.remove(memory_id)
        return {"ok": True, "message": "Memory deleted successfully"}

    return router
