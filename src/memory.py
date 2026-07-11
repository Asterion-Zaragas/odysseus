
import json
import logging
import os
import threading
import time
import uuid
import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

# ── Tags ──
# Memories carry free-form facet tags (max MAX_TAGS, normalized kebab-case).
# Structured tags use one of the allowed prefixes, e.g. "person:sven".
# Tags proposed by the tagging agent but not yet in the curator's registry live
# in the entry's "provisional_tags" until a curator run promotes or merges them.

MAX_TAGS = 10
ALLOWED_TAG_PREFIXES = ("person", "place", "org", "project")

_TAG_CHARS_RE = re.compile(r"[^a-z0-9:-]+")
_HYPHENS_RE = re.compile(r"-{2,}")


def normalize_tag(tag: str) -> Optional[str]:
    """Normalize a single tag to kebab-case; None if nothing survives.

    "Dress Code" -> "dress-code"; "Person: Sven" -> "person:sven".
    A colon is only meaningful after an allowed prefix — otherwise it is
    treated as a word separator so free-form input can't invent prefixes.
    """
    if not tag or not isinstance(tag, str):
        return None
    t = tag.strip().lower().replace("_", "-").replace(" ", "-")
    t = _TAG_CHARS_RE.sub("", t)
    prefix, sep, rest = t.partition(":")
    if sep and prefix in ALLOWED_TAG_PREFIXES:
        rest = _HYPHENS_RE.sub("-", rest.replace(":", "-")).strip("-")
        t = f"{prefix}:{rest}" if rest else None
    else:
        t = _HYPHENS_RE.sub("-", t.replace(":", "-")).strip("-")
    if not t or len(t) > 48:
        return None
    return t


def normalize_tags(tags) -> List[str]:
    """Normalize a tag collection: dedupe (order-preserving), cap at MAX_TAGS."""
    if isinstance(tags, str):
        tags = [tags]
    out: List[str] = []
    for tag in tags or []:
        t = normalize_tag(tag)
        if t and t not in out:
            out.append(t)
        if len(out) >= MAX_TAGS:
            break
    return out


def compat_category(entry: Dict) -> str:
    """Legacy 'category' for API/tool output, until the UI speaks tags (Phase 7).

    After the category->tags migration the first tag IS the old category; once
    the tagger adds richer tags this is only a best-effort label.
    """
    tags = entry.get("tags") or []
    return tags[0] if tags else "fact"


def tokenize(text: str) -> List[str]:
    """Simple tokenizer that splits on whitespace and removes punctuation."""
    return [word.strip('.,!?";') for word in text.split()]

def get_text_similarity(text1: str, text2: str) -> float:
    """Calculate Jaccard similarity between two texts."""
    if not text1 or not text2:
        return 0.0
    
    tokens1 = set(tokenize(text1.lower()))
    tokens2 = set(tokenize(text2.lower()))
    
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
        
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    
    return len(intersection) / len(union)

class MemoryManager:
    def __init__(self, data_dir: str):
        self.memory_file = os.path.join(data_dir, "memory.json")
        # Guards load-modify-save sequences. Saves are atomic (os.replace) but
        # not transactional: the tagger, the curator, and increment_uses all
        # mutate the same file, and an unguarded concurrent RMW loses writes.
        # Everything runs in one process, so an RLock is enough; hold it around
        # any load→mutate→save block (external callers use `with mm.lock:`).
        self.lock = threading.RLock()
        self.ensure_file_exists()
        
    def extract_memory_from_chat(self, chat_history: List[Dict], session_id: str = None) -> List[Dict]:
        """
        Extract memory entries from chat history as a fallback when LLM fails.
        
        Args:
            chat_history: List of chat messages with 'role' and 'content' keys
            session_id: Optional session ID to associate with extracted memories
            
        Returns:
            List of memory entries with text, timestamp, and optional session_id
        """
        memories = []
        
        for msg in chat_history:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") == "assistant":
                content = str(msg.get("content", ""))
                lines = content.split('\n')
                
                for line in lines:
                    line = line.strip()
                    # Look for bullet points or numbered lists that might contain memories
                    if re.match(r'^[-*•]|\d+\.', line):
                        # Extract the text after the bullet/number. Group both
                        # markers so the capture applies to either — the previous
                        # `^[-*•]|\d+\.\s*(.*)` put the group on the numbered branch
                        # only, so a bullet line matched with group(1)=None and
                        # crashed on .strip().
                        text_match = re.match(r'^(?:[-*•]|\d+\.)\s*(.*)', line)
                        if text_match:
                            text = text_match.group(1).strip()
                            if text:
                                memories.append({
                                    "text": text,
                                    "timestamp": int(datetime.now().timestamp()),
                                    "session_id": session_id
                                })
                    # If we see a heading that suggests memories
                    elif re.search(r'memory|fact|note|remember', line, re.I):
                        pass
                    # If we see a clear separator or end
                    elif re.match(r'^={3,}|-{3,}|_{3,}', line):
                        pass
                        
        return memories
        
    def process_inline_memory_command(self, message: str) -> Tuple[bool, str]:
        """
        Check if a message is an inline memory command (e.g. "remember: X").
        
        Args:
            message: The user message to check
            
        Returns:
            Tuple of (is_command, extracted_text) where is_command is True if 
            the message matches the memory command pattern
        """
        # Pattern for memory commands: "remember: X", "memorize: X", "save: X", etc.
        pattern = r'^(?:remember|memorize|save|note|store)[:\-]?\s+(.+)$'
        match = re.match(pattern, message.strip(), re.IGNORECASE)
        
        if match:
            return True, match.group(1).strip()
        else:
            return False, ""
    
    def ensure_file_exists(self):
        """Create memory file if it doesn't exist."""
        if not os.path.exists(self.memory_file):
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=2)
    
    def load_all(self) -> List[Dict]:
        """Load all memory entries from JSON file (unfiltered)."""
        if not os.path.exists(self.memory_file):
            return []

        try:
            with open(self.memory_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return self._validate_entries(data)
        except (json.JSONDecodeError, PermissionError) as e:
            logger.error("Error loading memory.json: %s", e)
            return self._migrate_from_legacy()

        return []

    def load(self, owner: str = None) -> List[Dict]:
        """Load memory entries, optionally filtered by owner."""
        entries = self.load_all()
        if owner is None:
            return entries
        return [e for e in entries if e.get("owner") == owner]

    def claim_ownerless(self, owner: str):
        """Assign all ownerless memory entries to the given owner."""
        with self.lock:
            entries = self.load_all()
            changed = False
            claimed = 0
            for entry in entries:
                if not entry.get("owner"):
                    entry["owner"] = owner
                    changed = True
                    claimed += 1
            if changed:
                self.save(entries)
                logger.info("Claimed %d ownerless memories for %s", claimed, owner)
    
    def _validate_entries(self, entries: List[Dict]) -> List[Dict]:
        """Ensure all entries have required fields, migrating old-schema ones.

        Lazy in-place migration from the pre-tags schema: a legacy `category`
        value becomes the entry's first tag and the key is dropped. Persisted
        on the next save. `generality: None` marks an entry as not yet triaged
        by the curator; `tier` defaults to 2 (situational) until then.
        """
        validated = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if "id" not in entry:
                entry["id"] = str(uuid.uuid4())
            if "timestamp" not in entry:
                entry["timestamp"] = int(time.time())
            if "source" not in entry:
                entry["source"] = "unknown"
            if "uses" not in entry:
                entry["uses"] = 0
            legacy_category = entry.pop("category", None)
            tags = list(entry.get("tags") or [])
            if legacy_category:
                tags.append(legacy_category)
            entry["tags"] = normalize_tags(tags)
            entry.setdefault("provisional_tags", [])
            entry.setdefault("pinned", False)
            entry.setdefault("tier", 2)
            entry.setdefault("generality", None)
            entry.setdefault("last_used_at", 0)
            validated.append(entry)
        return validated
    
    def _migrate_from_legacy(self) -> List[Dict]:
        """Migrate from old text format to JSON if needed."""
        legacy_path = os.path.join(os.path.dirname(self.memory_file), "memory.txt")
        if not os.path.exists(legacy_path):
            return []
            
        logger.info("Converting legacy memory.txt to new JSON format")
        try:
            with open(legacy_path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
            
            entries = []
            for line in lines:
                entries.append({
                    "id": str(uuid.uuid4()),
                    "text": line,
                    "timestamp": int(time.time()),
                    "source": "user",
                })

            self.save(entries)
            return self._validate_entries(entries)
        except Exception as e:
            logger.error("Failed to convert legacy memory: %s", e)
            return []
    
    def save(self, entries: List[Dict]):
        """Save memory entries to JSON file."""
        entries = self._validate_entries(entries)

        with self.lock:
            # Use atomic write
            tmp_file = self.memory_file + ".tmp"
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(entries, f, ensure_ascii=False, indent=2)
            os.replace(tmp_file, self.memory_file)

    def add_entry(self, text: str, source: str = "user", tags: List[str] = None,
                  owner: str = None, category: str = None) -> Dict:
        """Build a new memory entry (not persisted — append + save yourself,
        holding `self.lock` around the load-append-save).

        `category` is a deprecated alias kept for old callers/imports; its
        value is folded into `tags`.
        """
        if not text.strip():
            raise ValueError("Memory text cannot be empty")

        all_tags = list(tags or [])
        if category:
            all_tags.append(category)

        entry = {
            "id": str(uuid.uuid4()),
            "text": text.strip(),
            "timestamp": int(time.time()),
            "source": source,
            "tags": normalize_tags(all_tags),
            "provisional_tags": [],
            "pinned": False,
            "tier": 2,
            "generality": None,
            "uses": 0,
            "last_used_at": 0,
        }
        if owner:
            entry["owner"] = owner
        return entry

    def increment_uses(self, ids: List[str]) -> None:
        """Bump the uses counter for each memory id. Called after a memory has
        actually been injected into a chat's context (not just retrieved)."""
        if not ids:
            return
        id_set = set(ids)
        now = int(time.time())
        with self.lock:
            entries = self.load_all()
            changed = False
            for e in entries:
                if e.get("id") in id_set:
                    e["uses"] = int(e.get("uses", 0) or 0) + 1
                    e["last_used_at"] = now
                    changed = True
            if changed:
                self.save(entries)
    
    def find_duplicates(self, text: str, entries: List[Dict] = None) -> List[Dict]:
        """Find duplicate memory entries based on text content."""
        if entries is None:
            entries = self.load()
            
        text_lower = text.strip().lower()
        return [entry for entry in entries if entry["text"].lower() == text_lower]
            
    def get_relevant_memories(self, query: str, memories: list, threshold: float = 0.05, max_items: int = 8):
        """Get memories that are relevant to the query based on text similarity and semantic keyword matching."""
        if not memories or not query.strip():
            return []
            
        # Define keyword categories for semantic matching
        identity_words = ["name", "who", "i", "am", "called", "identity", "myself", "me", "my"]
        contact_words = ["phone", "email", "address", "contact", "number", "where", "located", "reach"]
        preference_words = ["like", "prefer", "favorite", "want", "love", "hate", "dislike", "enjoy", "interested"]
        task_words = ["todo", "task", "remind", "meeting", "appointment", "schedule", "deadline"]
        fact_words = ["what", "when", "where", "how", "why", "explain", "describe", "information", "know"]
        
        query_lower = query.lower()
        
        # Determine query type based on keywords
        query_type = None
        if any(word in query_lower for word in identity_words):
            query_type = "identity"
        elif any(word in query_lower for word in contact_words):
            query_type = "contact"
        elif any(word in query_lower for word in preference_words):
            query_type = "preference"
        elif any(word in query_lower for word in task_words):
            query_type = "task"
        elif any(word in query_lower for word in fact_words):
            query_type = "fact"
        
        relevant = []
        identity_memories = []
        other_memories = []
        
        # Separate identity memories from others
        for memory in memories:
            memory_text = memory["text"].lower()
            # Check if this is an identity memory (contains name patterns or identity indicators)
            is_identity = any([
                re.search(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', memory["text"]),
                any(word in memory_text for word in ["name is", "i'm", "i am", "called", "my name", "named", "call me"])
            ])
            if is_identity:
                identity_memories.append(memory)
            else:
                other_memories.append(memory)
        
        # For identity queries, include all identity memories regardless of similarity
        if query_type == "identity" and identity_memories:
            # Give them high scores to ensure they're included first
            for memory in identity_memories:
                relevant.append((0.9, memory))  # High score for identity memories in identity queries
        
        # Process other memories with similarity scoring
        for memory in other_memories:
            memory_text = memory["text"].lower()
            memory_tokens = set(tokenize(memory_text))
            query_tokens = set(tokenize(query_lower))
            
            # Calculate base Jaccard similarity
            if not query_tokens or not memory_tokens:
                continue
                
            base_similarity = len(query_tokens & memory_tokens) / len(query_tokens | memory_tokens)
            final_score = base_similarity
            
            # Apply boosts based on semantic matching
            if query_type == "contact":
                # Boost memories with contact information
                has_contact_info = any(word in memory_text for word in ["@gmail.com", "@", ".com", 
                                                                     "phone", "number", "address", 
                                                                     "http", "www", "tel:"])
                if has_contact_info:
                    final_score *= 1.4  # 40% boost for contact-related memories
            
            elif query_type == "preference":
                # Boost memories with preference indicators
                has_preference = any(word in memory_text for word in ["like", "love", "hate", "dislike", 
                                                                   "prefer", "favorite", "enjoy", "interested"])
                if has_preference:
                    final_score *= 1.3  # 30% boost for preference-related memories
            
            elif query_type == "task":
                # Boost memories with task indicators
                has_task = any(word in memory_text for word in ["todo", "task", "remind", "meeting", 
                                                              "appointment", "schedule", "deadline", "need to"])
                if has_task:
                    final_score *= 1.3  # 30% boost for task-related memories
            
            # Always consider exact phrase matches as highly relevant
            if query.lower() in memory["text"].lower():
                final_score = max(final_score, 0.8)  # Ensure high relevance for exact matches
            
            # Include memory if it meets threshold after boosts
            if final_score >= threshold:
                relevant.append((final_score, memory))
        
        # Sort by final score (descending) and return top matches
        relevant.sort(key=lambda x: x[0], reverse=True)
        return [mem for _, mem in relevant[:max_items]]
