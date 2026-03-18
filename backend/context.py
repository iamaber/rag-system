import re
from dataclasses import dataclass

from session import Session


def _normalize_token(token: str) -> str:
    if token.endswith("দের"):
        return token

    for suffix in ("ির", "ের", "এর"):
        if token.endswith(suffix) and len(token) > len(suffix) + 1:
            return token[: -len(suffix)]
    return token


def _normalized_words(words: frozenset[str]) -> frozenset[str]:
    return frozenset(_normalize_token(word.casefold()) for word in words)


_TOKEN_RE = re.compile(r"[^\s!?.,:;()\"'`]+")
_SIZE_TOKEN_RE = re.compile(r"^[0-9০-৯]+(লিটার|কেজি|গ্রাম|গজ)$")
_MEASURE_PATTERNS = (
    (re.compile(r"([0-9০-৯]+)\s*লিটারের"), r"\1লিটার"),
    (re.compile(r"([0-9০-৯]+)\s*লিটার"), r"\1লিটার"),
    (re.compile(r"([0-9০-৯]+)\s*কেজির"), r"\1কেজি"),
    (re.compile(r"([0-9০-৯]+)\s*কেজি"), r"\1কেজি"),
    (re.compile(r"([0-9০-৯]+)\s*গ্রামের"), r"\1গ্রাম"),
    (re.compile(r"([0-9০-৯]+)\s*গ্রাম"), r"\1গ্রাম"),
)

# Price/availability words that alone don't identify a product
PRICE_WORDS = frozenset(
    ["দাম", "মূল্য", "কত", "টাকা", "দামটা", "মূল্যটা", "price", "cost", "how much"]
)
STOCK_WORDS = frozenset(
    ["আছে", "পাওয়া যায়", "স্টক", "available", "stock", "বিক্রি", "sell", "sells"]
)
DISCOUNT_WORDS = frozenset(
    ["discount", "discout", "discould", "offer", "sale", "ছাড়", "ছাড়", "ডিসকাউন্ট"]
)
# Pronouns that explicitly refer to something already said
PRONOUNS = frozenset(
    ["এটা", "এটি", "ওটা", "ওটি", "সেটা", "সেটি", "এর", "তার", "it", "that", "this"]
)

SIGNAL_WORDS = PRICE_WORDS | STOCK_WORDS | DISCOUNT_WORDS
GENERIC_WORDS = frozenset(
    [
        "আপনাদের",
        "আমাদের",
        "কোম্পানি",
        "কাছে",
        "কি",
        "বিক্রি",
        "করে",
        "করেন",
        "করো",
        "tell",
        "about",
        "me",
        "do",
        "you",
        "have",
        "is",
        "there",
        "any",
        "for",
    ]
)
NORMALIZED_SIGNAL_WORDS = _normalized_words(SIGNAL_WORDS)
NORMALIZED_GENERIC_WORDS = _normalized_words(GENERIC_WORDS)
NORMALIZED_PRONOUNS = _normalized_words(PRONOUNS)


@dataclass
class ResolvedQuery:
    original: str
    resolved: str  # what we actually search for
    entity: str | None  # product noun extracted (saved to session for next turn)
    was_referential: bool


def resolve(query: str, session: Session) -> ResolvedQuery:
    """Rule-based resolution, zero I/O. Rewrites the query if it's referential."""
    q = query.strip()
    entity = _extract_entity(q)

    if _is_referential(q) and session.last_entity:
        qualifiers = _content_tokens(q)
        resolved = session.last_entity
        if qualifiers:
            resolved = f"{resolved} {' '.join(qualifiers)}"
        return ResolvedQuery(
            original=q,
            resolved=resolved,
            entity=session.last_entity,
            was_referential=True,
        )

    if entity and any(word in q for word in SIGNAL_WORDS):
        return ResolvedQuery(
            original=q,
            resolved=entity,
            entity=entity,
            was_referential=False,
        )

    return ResolvedQuery(
        original=q, resolved=q, entity=entity, was_referential=False
    )


def _is_referential(query: str) -> bool:
    """True if query clearly refers to something already mentioned."""
    if any(p in query for p in PRONOUNS):
        return True

    if not any(word in query for word in SIGNAL_WORDS):
        return False

    tokens = _content_tokens(query)
    return not any(_is_entity_token(token) for token in tokens)


def _extract_entity(query: str) -> str | None:
    """Pick the most likely product noun from a non-referential query.
    Stored in session so the next referential turn can use it."""
    tokens = _content_tokens(query)
    return " ".join(tokens) if tokens else None


def _content_tokens(query: str) -> list[str]:
    return [token for token in _tokenize(query) if _is_content_token(token)]


def _tokenize(query: str) -> list[str]:
    normalized = query
    for pattern, replacement in _MEASURE_PATTERNS:
        normalized = pattern.sub(replacement, normalized)
    return [
        _normalize_token(token.casefold()) for token in _TOKEN_RE.findall(normalized)
    ]


def _is_content_token(token: str) -> bool:
    return (
        len(token) >= 3
        and token not in NORMALIZED_SIGNAL_WORDS
        and token not in NORMALIZED_GENERIC_WORDS
        and token not in NORMALIZED_PRONOUNS
    )


def _is_entity_token(token: str) -> bool:
    return not _SIZE_TOKEN_RE.match(token)
