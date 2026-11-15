"""Request protection for the endpoints that can spend LLM credits.

A sliding-window limiter bounds per-client request rate and an optional shared
token gates deployments that should not be publicly callable.

The client is the socket peer unless ``trust_proxy_headers`` is enabled, in
which case it is the address the nearest reverse proxy appended to
``X-Forwarded-For``. Behind a proxy with the setting off, every visitor shares
the proxy's address and therefore one bucket.
"""

import secrets
import time
from collections import OrderedDict, deque

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

PROTECTED_PATHS = frozenset({"/chat", "/chat/stream", "/session/reset"})

_WINDOW_SECONDS = 60.0
_MAX_TRACKED_CLIENTS = 10_000


class RateLimiter:
    """Sliding-window request counter keyed by client address."""

    def __init__(self, limit: int, window_seconds: float = _WINDOW_SECONDS) -> None:
        self._limit = max(1, limit)
        self._window = window_seconds
        self._hits: OrderedDict[str, deque[float]] = OrderedDict()

    def allow(self, client: str) -> bool:
        now = time.monotonic()
        hits = self._hits.get(client)
        if hits is None:
            if len(self._hits) >= _MAX_TRACKED_CLIENTS:
                self._hits.popitem(last=False)
            hits = deque()
            self._hits[client] = hits
        else:
            self._hits.move_to_end(client)

        cutoff = now - self._window
        while hits and hits[0] <= cutoff:
            hits.popleft()

        if len(hits) >= self._limit:
            return False
        hits.append(now)
        return True


class RequestGuard:
    """ASGI middleware enforcing the token and rate limit on paid endpoints."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        token: str | None,
        limiter: RateLimiter,
        trust_proxy_headers: bool = False,
    ) -> None:
        self.app = app
        self._token = token
        self._limiter = limiter
        self._trust_proxy_headers = trust_proxy_headers

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") not in PROTECTED_PATHS:
            await self.app(scope, receive, send)
            return

        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers", [])
        }

        if self._token and not _token_matches(headers, self._token):
            await _reject(
                scope, receive, send, 401, {"detail": "Invalid or missing API token."}
            )
            return

        client = _client_key(scope, headers, self._trust_proxy_headers)
        if not self._limiter.allow(client):
            await _reject(
                scope,
                receive,
                send,
                429,
                {"detail": "Too many requests; slow down."},
                headers={"retry-after": "60"},
            )
            return

        await self.app(scope, receive, send)


def _client_key(scope: Scope, headers: dict[str, str], trust_proxy_headers: bool) -> str:
    peer = (scope.get("client") or ("unknown", 0))[0]
    if not trust_proxy_headers:
        return peer
    # Only the last entry was written by the proxy we are connected to; earlier
    # entries are supplied by the caller and can be forged to dodge the limit.
    forwarded = headers.get("x-forwarded-for", "").rsplit(",", 1)[-1].strip()
    return forwarded or peer


def _token_matches(headers: dict[str, str], expected: str) -> bool:
    supplied = headers.get("x-api-token")
    if supplied is None:
        authorization = headers.get("authorization", "")
        if authorization.lower().startswith("bearer "):
            supplied = authorization[7:]
    return supplied is not None and secrets.compare_digest(supplied, expected)


async def _reject(
    scope: Scope,
    receive: Receive,
    send: Send,
    status_code: int,
    content: dict[str, str],
    headers: dict[str, str] | None = None,
) -> None:
    response = JSONResponse(content, status_code=status_code, headers=headers)
    await response(scope, receive, send)
