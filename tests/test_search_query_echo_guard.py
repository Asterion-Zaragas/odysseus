"""Regression tests — reject query-extraction output that echoes the prompt.

The chat-mode web search flow (#4557) asks an LLM to distil a search query
from the user's message and previously accepted any non-empty reply. Weaker
instruct models paraphrase the task instead of performing it; the observed
failure sent this to SearXNG as the literal query:

    User message: "Hm, that's weird. I tweaked the settings, can you please
    try again?" * Goal: Extract a concise search query. * Constraint: Reply ONLY

which returned results unrelated to the conversation, and was long enough that
Mojeek answered HTTP 403. ``_is_plausible_search_query()`` gates acceptance so
these fall back to the first-line-of-message path instead.
"""
from src.chat_processor import _clean_search_query, _is_plausible_search_query


# ── Rejected: instruction echo ──

def test_rejects_observed_prompt_echo():
    """The exact echo captured from the SearXNG request log."""
    message = "Hm, that's weird. I tweaked the settings, can you please try again?"
    generated = (
        'User message: "Hm, that\'s weird. I tweaked the settings, can you '
        'please try again?" * Goal: Extract a concise search query. '
        "* Constraint: Reply ONLY"
    )

    assert _is_plausible_search_query(generated, message) is False


def test_rejects_reply_longer_than_source_message():
    """A distilled query is never longer than the message it came from."""
    message = "what's the weather in berlin"
    generated = "x" * 200

    assert _is_plausible_search_query(generated, message) is False


def test_rejects_prompt_vocabulary_even_when_short():
    """Echo markers catch restatements that slip under the length ceiling."""
    message = "I tweaked some settings, can you retry that lookup for me please?"

    assert _is_plausible_search_query("Reply ONLY with the query", message) is False
    assert _is_plausible_search_query("Goal: extract the query", message) is False


def test_rejects_empty_and_whitespace():
    assert _is_plausible_search_query("", "berlin news") is False
    assert _is_plausible_search_query("   \n  ", "berlin news") is False


# ── Accepted: genuine extractions ──

def test_accepts_normal_extraction():
    message = "Hey, I was wondering what the current news in Berlin is today?"

    assert _is_plausible_search_query("current news Berlin", message) is True


def test_accepts_expansion_of_a_very_short_message():
    """Short messages must not make the length rule reject a fair expansion."""
    message = "berlin?"

    assert _is_plausible_search_query("Berlin Germany current news", message) is True


def test_accepts_query_from_a_long_message():
    message = (
        "So I've been reading about transformer architectures for a while now "
        "and I keep running into the term 'grouped query attention' but I "
        "don't really understand how it differs from multi-head attention."
    )

    assert _is_plausible_search_query("grouped query attention vs multi-head", message) is True


# ── Wrapping-quote removal ──

def test_strips_quotes_the_model_wrapped_its_reply_in():
    """Observed in the SearXNG log as q="current weather" — an exact-phrase
    operator the user never asked for."""
    assert _clean_search_query('"current weather"') == "current weather"


def test_strips_typographic_quotes():
    assert _clean_search_query("“current weather”") == "current weather"


def test_preserves_a_deliberate_inner_phrase_search():
    assert _clean_search_query('berlin "hidden gems"') == 'berlin "hidden gems"'


def test_preserves_apostrophes():
    assert _clean_search_query("what's the weather") == "what's the weather"
