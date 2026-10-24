from security import RateLimiter, _token_matches


def test_rate_limiter_allows_requests_up_to_the_limit() -> None:
    limiter = RateLimiter(3, window_seconds=60)

    assert [limiter.allow("203.0.113.5") for _ in range(3)] == [True, True, True]
    assert limiter.allow("203.0.113.5") is False


def test_rate_limiter_counts_clients_separately() -> None:
    limiter = RateLimiter(1, window_seconds=60)

    assert limiter.allow("203.0.113.5") is True
    assert limiter.allow("203.0.113.6") is True


def test_rate_limiter_refills_after_the_window(monkeypatch) -> None:
    now = [1000.0]
    monkeypatch.setattr("security.time.monotonic", lambda: now[0])
    limiter = RateLimiter(1, window_seconds=60)

    assert limiter.allow("203.0.113.5") is True
    assert limiter.allow("203.0.113.5") is False

    now[0] += 61
    assert limiter.allow("203.0.113.5") is True


def test_token_matches_api_and_bearer_headers() -> None:
    assert _token_matches({"x-api-token": "s3cret"}, "s3cret") is True
    assert _token_matches({"authorization": "Bearer s3cret"}, "s3cret") is True
    assert _token_matches({"x-api-token": "wrong"}, "s3cret") is False
    assert _token_matches({}, "s3cret") is False
