from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from security import RateLimiter, RequestGuard, _token_matches


def _guarded_app(token: str | None = None, limit: int = 100) -> Starlette:
    async def chat(_request):
        return JSONResponse({"ok": True})

    app = Starlette(routes=[Route("/chat", chat, methods=["POST"])])
    app.add_middleware(
        RequestGuard, token=token, limiter=RateLimiter(limit, window_seconds=60)
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
