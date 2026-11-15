from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from security import RateLimiter, RequestGuard, _token_matches


def _guarded_app(
    token: str | None = None, limit: int = 100, trust_proxy_headers: bool = False
) -> Starlette:
    async def chat(_request):
        return JSONResponse({"ok": True})

    app = Starlette(routes=[Route("/chat", chat, methods=["POST"])])
    app.add_middleware(
        RequestGuard,
        token=token,
        limiter=RateLimiter(limit, window_seconds=60),
        trust_proxy_headers=trust_proxy_headers,
    )
    return app


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


def test_guard_rejects_requests_without_the_configured_token() -> None:
    client = TestClient(_guarded_app(token="s3cret"))

    assert client.post("/chat").status_code == 401
    assert (
        client.post("/chat", headers={"X-API-Token": "s3cret"}).status_code == 200
    )


def test_guard_returns_429_when_the_limit_is_exhausted() -> None:
    client = TestClient(_guarded_app(limit=2))

    assert client.post("/chat").status_code == 200
    assert client.post("/chat").status_code == 200
    assert client.post("/chat").status_code == 429


def test_guard_ignores_forwarded_for_unless_proxy_headers_are_trusted() -> None:
    client = TestClient(_guarded_app(limit=1))

    first = client.post("/chat", headers={"X-Forwarded-For": "203.0.113.5"})
    second = client.post("/chat", headers={"X-Forwarded-For": "203.0.113.6"})

    assert first.status_code == 200
    assert second.status_code == 429


def test_guard_limits_per_forwarded_client_behind_a_trusted_proxy() -> None:
    client = TestClient(_guarded_app(limit=1, trust_proxy_headers=True))

    assert (
        client.post("/chat", headers={"X-Forwarded-For": "203.0.113.5"}).status_code
        == 200
    )
    assert (
        client.post("/chat", headers={"X-Forwarded-For": "203.0.113.6"}).status_code
        == 200
    )
    assert (
        client.post("/chat", headers={"X-Forwarded-For": "203.0.113.5"}).status_code
        == 429
    )


def test_guard_keys_on_the_hop_added_by_the_nearest_proxy() -> None:
    client = TestClient(_guarded_app(limit=1, trust_proxy_headers=True))

    # The leading entries are caller-supplied; rotating them must not help.
    first = client.post(
        "/chat", headers={"X-Forwarded-For": "198.51.100.1, 203.0.113.5"}
    )
    second = client.post(
        "/chat", headers={"X-Forwarded-For": "198.51.100.2, 203.0.113.5"}
    )

    assert first.status_code == 200
    assert second.status_code == 429


def test_guard_falls_back_to_the_peer_without_a_forwarded_header() -> None:
    client = TestClient(_guarded_app(limit=1, trust_proxy_headers=True))

    assert client.post("/chat").status_code == 200
    assert client.post("/chat").status_code == 429
