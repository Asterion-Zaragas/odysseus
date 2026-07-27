from types import SimpleNamespace

from src.chat_processor import ChatProcessor


class _Memory:
    def __init__(self, rows):
        self.rows = rows
        self.incremented = []

    def load(self, owner=None):
        return list(self.rows)

    def increment_uses(self, ids):
        self.incremented.extend(ids)


class _Docs:
    rag_manager = None


def _context_text(preface):
    return "\n".join(m.get("content", "") for m in preface)


def _processor(rows):
    return ChatProcessor(memory_manager=_Memory(rows), personal_docs_manager=_Docs())


async def test_pinned_memory_does_not_inject_every_unrelated_fact(monkeypatch):
    monkeypatch.setattr("src.settings.get_setting", lambda key, default=None: default)
    rows = [
        {
            "id": "identity",
            "text": "User's name is Felix.",
            "category": "identity",
            "pinned": True,
            "timestamp": 3,
        },
        {
            "id": "party",
            "text": "User is planning a birthday party with sack races.",
            "category": "fact",
            "pinned": True,
            "timestamp": 2,
        },
        {
            "id": "coffee",
            "text": "User likes dark roast coffee.",
            "category": "preference",
            "pinned": True,
            "timestamp": 1,
        },
    ]

    preface, _, _ = await _processor(rows).build_context_preface(
        message="Explain how Python decorators work",
        session=SimpleNamespace(),
        use_rag=False,
        use_memory=True,
        memory_effort="low",
    )

    text = _context_text(preface)
    assert "User's name is Felix." in text
    assert "birthday party with sack races" not in text
    assert "dark roast coffee" not in text


async def test_relevant_pinned_memory_is_still_injected(monkeypatch):
    monkeypatch.setattr("src.settings.get_setting", lambda key, default=None: default)
    rows = [
        {
            "id": "coffee",
            "text": "User likes dark roast coffee.",
            "category": "preference",
            "pinned": True,
            "tier": 1,
            "timestamp": 1,
        },
        {
            "id": "party",
            "text": "User is planning a birthday party with sack races.",
            "category": "fact",
            "pinned": True,
            "tier": 1,
            "timestamp": 2,
        },
    ]

    preface, _, _ = await _processor(rows).build_context_preface(
        message="likes coffee roast",
        session=SimpleNamespace(),
        use_rag=False,
        use_memory=True,
        memory_effort="low",
    )

    text = _context_text(preface)
    assert "User likes dark roast coffee." in text
    assert "birthday party with sack races" not in text


async def test_pinned_memory_injection_is_capped_at_five(monkeypatch):
    monkeypatch.setattr("src.settings.get_setting", lambda key, default=None: default)
    rows = [
        {
            "id": f"identity-{idx}",
            "text": f"User identity fact {idx} email marker.",
            "category": "identity",
            "pinned": True,
            "timestamp": idx,
        }
        for idx in range(10)
    ]

    processor = _processor(rows)
    await processor.build_context_preface(
        message="Who is the user?",
        session=SimpleNamespace(),
        use_rag=False,
        use_memory=True,
        memory_effort="low",
    )

    assert len(processor._last_used_memories) == 5


async def test_pinned_and_recalled_memories_are_capped_independently(monkeypatch):
    """Pinned/core and recalled ("rest") memories are two separate budgets
    (PINNED_MEMORY_LIMIT and the memory_recall_k setting), not one shared
    total — that split is what lets memory_recall_k be tuned independently
    of how many pinned/core facts a user happens to have. This replaces an
    older assumption of one combined cap of five across both groups.
    """
    monkeypatch.setattr("src.settings.get_setting", lambda key, default=None: default)
    rows = [
        {
            "id": f"identity-{idx}",
            "text": f"User identity fact {idx} email marker.",
            "category": "identity",
            "pinned": True,
            "timestamp": idx,
        }
        for idx in range(4)
    ]
    rows.extend([
        {
            "id": f"coffee-{idx}",
            "text": f"User likes coffee roast {idx}.",
            "category": "preference",
            "pinned": False,
            "timestamp": idx,
        }
        for idx in range(6)
    ])

    processor = _processor(rows)
    await processor.build_context_preface(
        message="likes coffee roast",
        session=SimpleNamespace(),
        use_rag=False,
        use_memory=True,
        memory_effort="low",
    )

    used = processor._last_used_memories
    pinned_count = sum(1 for m in used if m["type"] == "pinned")
    recalled_count = sum(1 for m in used if m["type"] == "recalled")
    assert pinned_count == 4  # all 4 match the identity marker, under PINNED_MEMORY_LIMIT (5)
    assert recalled_count <= 3  # default memory_recall_k
