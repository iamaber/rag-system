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
