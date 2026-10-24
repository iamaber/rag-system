from session import SessionStore


def test_store_evicts_least_recently_used_session() -> None:
    store = SessionStore(max_sessions=2)

    store.get("a")
    store.get("b")
    store.get("a")  # refresh "a"
    store.get("c")

    assert set(store._sessions) == {"a", "c"}


def test_store_sweeps_expired_sessions_on_a_timer() -> None:
    store = SessionStore(ttl_seconds=1800, max_sessions=10)
    store.get("old")

    store._ttl = -1  # everything is already expired
    store._next_sweep = 0  # force the next call to sweep
    store.get("new")

    assert set(store._sessions) == {"new"}
