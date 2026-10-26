import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse

import answering
import context as ctx
import llm
import retrieval
import session as sess
from config import settings
from database import close_db, get_conn, init_db
from models import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    Product,
    RetrievalMethod,
    SessionResetRequest,
    StreamMetadata,
    Timing,
)
from security import RateLimiter, RequestGuard

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_start_time = time.monotonic()
_FRONTEND_INDEX = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
_MISSING_KEY_MESSAGE = (
    "This deployment has no LLM API key configured, so I can't answer open-ended "
    "questions. Price, discount, stock, and availability lookups still work. "
    "Set the GROQ_API_KEY secret (see .env.example) to enable full answers."
)
_UPSTREAM_ERROR_MESSAGE = (
    "The answer service is unavailable right now. Please try again."
)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    await init_db()
    sess.init_store(
        ttl_seconds=settings.session_ttl_seconds,
        max_sessions=settings.max_sessions,
    )
    try:
        yield
    finally:
        await close_db()


app = FastAPI(title="Bangla RAG System", lifespan=lifespan)
app.add_middleware(
    RequestGuard,
    token=settings.api_token,
    limiter=RateLimiter(settings.rate_limit_per_minute),
)

# Bound how many requests can hold a paid upstream generation at once.
_LLM_SLOTS = asyncio.Semaphore(settings.max_concurrent_llm_calls)


# shared logic used by both /chat and /chat/stream


@dataclass
class PreparedChat:
    resolved_query: str
    entity: str | None
    products: list[Product]
    retrieval_method: RetrievalMethod
    context_ms: float
    retrieval_ms: float


async def _prepare_chat(message: str, session: sess.Session) -> PreparedChat:
    t0 = time.monotonic()
    resolved = ctx.resolve(message, session)

    context_ms = (time.monotonic() - t0) * 1000

    t1 = time.monotonic()
    products, method = await retrieval.retrieve(resolved.resolved)
    retrieval_ms = (time.monotonic() - t1) * 1000

    return PreparedChat(
        resolved_query=resolved.resolved,
        entity=resolved.entity,
        products=products,
        retrieval_method=method,
        context_ms=context_ms,
        retrieval_ms=retrieval_ms,
    )


def _catalog_reply(prepared: PreparedChat, message: str) -> str | None:
    return answering.generate_catalog_reply(
        prepared.products, message, prepared.resolved_query
    )


async def _generate_response(prepared: PreparedChat, message: str) -> str:
    direct_response = _catalog_reply(prepared, message)
    if direct_response is not None:
        return direct_response

    if not llm.is_configured():
        return _MISSING_KEY_MESSAGE

    try:
        async with _LLM_SLOTS:
            return await llm.generate(
                prepared.products, message, prepared.resolved_query
            )
    except Exception as exc:
        logger.warning("LLM generation failed: %s", exc)
        raise HTTPException(
            status_code=502, detail=_UPSTREAM_ERROR_MESSAGE
        ) from exc


async def _response_deltas(
    prepared: PreparedChat, message: str
) -> AsyncGenerator[str, None]:
    direct_response = _catalog_reply(prepared, message)
    if direct_response is not None:
        yield direct_response
        return

    if not llm.is_configured():
        yield _MISSING_KEY_MESSAGE
        return

    async with _LLM_SLOTS:
        async for delta in llm.stream_generate(
            prepared.products, message, prepared.resolved_query
        ):
            yield delta


def _complete_chat(
    store: sess.SessionStore,
    session_id: str,
    prepared: PreparedChat,
    llm_ms: float,
    total_ms: float,
    stream: bool = False,
) -> None:
    entity = prepared.entity or _entity_from_products(prepared.products)
    if entity:
        store.update_last_entity(session_id, entity)

    prefix = "stream " if stream else ""
    logger.info(
        "%ssession=%s method=%s ctx=%.0fms ret=%.0fms llm=%.0fms total=%.0fms",
        prefix,
        session_id,
        prepared.retrieval_method.value,
        prepared.context_ms,
        prepared.retrieval_ms,
        llm_ms,
        total_ms,
    )


# endpoints


@app.get("/", include_in_schema=False)
async def frontend() -> FileResponse:
    return FileResponse(_FRONTEND_INDEX)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    t0 = time.monotonic()
    store = sess.get_store()
    session = store.get(req.session_id)

    prepared = await _prepare_chat(req.message, session)

    t_llm = time.monotonic()
    response_text = await _generate_response(prepared, req.message)
    llm_ms = (time.monotonic() - t_llm) * 1000
    total_ms = (time.monotonic() - t0) * 1000

    _complete_chat(
        store=store,
        session_id=req.session_id,
        prepared=prepared,
        llm_ms=llm_ms,
        total_ms=total_ms,
    )

    return ChatResponse(
        session_id=req.session_id,
        response=response_text,
        resolved_query=prepared.resolved_query,
        products=prepared.products,
        retrieval_method=prepared.retrieval_method,
        timing=Timing(
            context_ms=round(prepared.context_ms, 2),
            retrieval_ms=round(prepared.retrieval_ms, 2),
            llm_ms=round(llm_ms, 2),
            total_ms=round(total_ms, 2),
        ),
    )


@app.get("/chat/stream")
async def chat_stream(
    session_id: str = Query(..., min_length=1, max_length=128),
    message: str = Query(..., min_length=1, max_length=2000),
) -> EventSourceResponse:
    return EventSourceResponse(_stream_events(session_id, message))


async def _stream_events(session_id: str, message: str) -> AsyncGenerator[dict, None]:
    t0 = time.monotonic()
    store = sess.get_store()
    session = store.get(session_id)

    prepared = await _prepare_chat(message, session)

    # Send metadata before the first token so the client can show retrieved products immediately
    yield {
        "event": "metadata",
        "data": StreamMetadata(
            session_id=session_id,
            resolved_query=prepared.resolved_query,
            products=prepared.products,
            retrieval_method=prepared.retrieval_method,
            context_ms=round(prepared.context_ms, 2),
            retrieval_ms=round(prepared.retrieval_ms, 2),
        ).model_dump_json(),
    }

    t_llm = time.monotonic()

    try:
        async for delta in _response_deltas(prepared, message):
            yield {"event": "token", "data": json.dumps({"text": delta})}
    except Exception as exc:
        logger.warning("Streaming generation failed: %s", exc)
        yield {
            "event": "error",
            "data": json.dumps({"detail": _UPSTREAM_ERROR_MESSAGE}),
        }
        return

    llm_ms = (time.monotonic() - t_llm) * 1000
    total_ms = (time.monotonic() - t0) * 1000

    yield {
        "event": "done",
        "data": json.dumps(
            {"llm_ms": round(llm_ms, 2), "total_ms": round(total_ms, 2)}
        ),
    }

    _complete_chat(
        store=store,
        session_id=session_id,
        prepared=prepared,
        llm_ms=llm_ms,
        total_ms=total_ms,
        stream=True,
    )


@app.post("/session/reset")
async def reset_session(req: SessionResetRequest) -> dict:
    sess.get_store().reset(req.session_id)
    return {"status": "ok", "session_id": req.session_id}


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    uptime = round(time.monotonic() - _start_time, 1)
    try:
        async with get_conn() as conn:
            count = await conn.fetchval("SELECT COUNT(*) FROM products")
        return HealthResponse(
            status="ok",
            db_connected=True,
            llm_configured=llm.is_configured(),
            uptime_seconds=uptime,
            product_count=count,
        )
    except Exception as exc:
        logger.error("Health check failed: %s", exc)
        return HealthResponse(
            status="degraded",
            db_connected=False,
            llm_configured=llm.is_configured(),
            uptime_seconds=uptime,
        )


def _entity_from_products(products: list[Product]) -> str | None:
    """Fall back to base_item of first result when context resolution found nothing."""
    for p in products:
        if p.base_item:
            return p.base_item
    return None
