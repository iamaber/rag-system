from context import resolve
from session import Session


def test_resolve_extracts_product_entity() -> None:
    resolved = resolve("লেমন বিস্কুটের দাম কত?", Session(session_id="s1"))

    assert resolved.entity == "লেমন বিস্কুট"
    assert resolved.resolved == "লেমন বিস্কুট"
    assert resolved.was_referential is False


def test_resolve_rewrites_referential_follow_up() -> None:
    session = Session(session_id="s2")
    first = resolve("লেমন বিস্কুটের দাম কত?", session)
    session.last_entity = first.entity

    follow_up = resolve("দাম কত?", session)

    assert follow_up.was_referential is True
    assert follow_up.resolved == "লেমন বিস্কুট"


def test_resolve_keeps_plain_query_without_history() -> None:
    resolved = resolve("শাওমি ইয়ারবাড", Session(session_id="s3"))

    assert resolved.resolved == "শাওমি ইয়ারবাড"
    assert resolved.entity == "শাওমি ইয়ারবাড"
    assert resolved.was_referential is False


def test_english_words_containing_pronoun_substrings_are_not_referential() -> None:
    session = Session(session_id="s4")
    session.last_entity = "লেমন বিস্কুট"

    digital = resolve("digital camera price", session)
    with_shipping = resolve("price with delivery", session)

    assert digital.was_referential is False
    assert digital.entity == "digital camera"
    assert with_shipping.was_referential is False
    assert with_shipping.entity == "with delivery"


def test_resolve_handles_inflected_bangla_pronouns() -> None:
    session = Session(session_id="s5")
    session.last_entity = "লেমন বিস্কুট"

    for follow_up in ("এটার দাম কত?", "ওটার দাম কত?", "এটির দাম কত?"):
        resolved = resolve(follow_up, session)

        assert resolved.was_referential is True
        assert resolved.resolved == "লেমন বিস্কুট"
