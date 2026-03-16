import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Maximum number of turns kept per session
MAX_HISTORY = 20


@dataclass
class Turn:
    role: str  # "user" or "assistant"
    content: str
    # The product entity resolved during this turn (if any)
    resolved_entity: str | None = None
    timestamp: float = field(default_factory=time.monotonic)


@dataclass
class Session:
    session_id: str
    history: list[Turn] = field(default_factory=list)
    last_entity: str | None = None  # last product entity mentioned
    created_at: float = field(default_factory=time.monotonic)
    last_active: float = field(default_factory=time.monotonic)


class SessionStore:
    def __init__(self, ttl_seconds: int = 1800) -> None:
        self._sessions: dict[str, Session] = {}
        self._ttl = ttl_seconds

    def get(self, session_id: str) -> Session:
        """Return session, creating one if it doesn't exist."""
        self._evict_expired_sync()
        if session_id not in self._sessions:
            self._sessions[session_id] = Session(session_id=session_id)
        session = self._sessions[session_id]
        session.last_active = time.monotonic()
        return session

    def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        resolved_entity: str | None = None,
    ) -> None:
        session = self.get(session_id)
        session.history.append(
            Turn(role=role, content=content, resolved_entity=resolved_entity)
        )
        # Keep history bounded
        if len(session.history) > MAX_HISTORY:
            session.history = session.history[-MAX_HISTORY:]
        if resolved_entity:
            session.last_entity = resolved_entity

    def update_last_entity(self, session_id: str, entity: str) -> None:
        session = self.get(session_id)
        session.last_entity = entity

    def reset(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def _evict_expired_sync(self) -> None:
        now = time.monotonic()
        expired = [
            sid for sid, s in self._sessions.items() if now - s.last_active > self._ttl
        ]
        for sid in expired:
            del self._sessions[sid]
            logger.debug("Evicted expired session: %s", sid)


# Module-level singleton — initialized in main.py startup
store: SessionStore | None = None


def get_store() -> SessionStore:
    if store is None:
        raise RuntimeError("SessionStore not initialized.")
    return store


def init_store(ttl_seconds: int = 1800) -> SessionStore:
    global store
    store = SessionStore(ttl_seconds=ttl_seconds)
    return store
