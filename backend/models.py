from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class RetrievalMethod(str, Enum):
    exact = "exact"
    fulltext = "fulltext"
    none = "none"


class Product(BaseModel):
    id: str
    name: str
    base_item: str | None
    subcategory: str | None
    size: str | None
    price_taka: int | None
    discounted_price_taka: int | None
    stock_qty: int | None
    rating: float | None


class Timing(BaseModel):
    context_ms: float = Field(
        description="Time spent on context/coreference resolution"
    )
    retrieval_ms: float = Field(description="Time spent on DB retrieval")
    llm_ms: float = Field(
        description="Time to complete LLM response (or first token for streaming)"
    )
    total_ms: float = Field(description="Total end-to-end time")


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    session_id: str
    response: str
    resolved_query: str
    products: list[Product]
    retrieval_method: RetrievalMethod
    timing: Timing


class StreamMetadata(BaseModel):
    """Sent as the first SSE event before token streaming begins."""

    session_id: str
    resolved_query: str
    products: list[Product]
    retrieval_method: RetrievalMethod
    context_ms: float
    retrieval_ms: float


class SessionResetRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    db_connected: bool
    uptime_seconds: float
    product_count: int | None = None
