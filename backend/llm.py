import json
from collections.abc import AsyncGenerator
from typing import Any

from groq import AsyncGroq

from config import settings
from models import Product

SYSTEM_PROMPT = """
You are the chat assistant of a Bangladeshi online store.

Rules:
- Answer in the same language the user asked in. If they ask in Bangla, reply in Bangla. If in English, reply in English.
- Only use the JSON product data provided to you — do not make up prices, sizes, stock, or products.
- Treat questions about "your company", "your shop", "আপনাদের কোম্পানি", or "আপনাদের কাছে" as questions about this store's catalog and inventory.
- If matching products are retrieved, that means this store has or sells those products.
- For availability questions, answer yes/no first. Do not jump to prices unless the user asked for prices.
- If multiple products match, list them briefly instead of picking just one.
- If no product is found, politely say that it is not available.
- When asked about price for a product family or category, list the matching products and their prices.
- When asked about price, mention both the original and discounted price if they differ.
- When asked about price, discount, or stock, answer only from the retrieved product fields.
- Do not compare prices with phrases like "smaller than", "lower than", or similar wording. State the prices directly.
- Do not claim any company metadata unless that metadata is explicitly present in the provided JSON.
- If the active product context is provided, treat it as the subject of short follow-up questions like "দাম কত?" or "Is there a discount?".
- Keep answers short (2-3 sentences max, unless listing products).
"""


def _format_context(products: list[Product], query: str, resolved_query: str) -> str:
    payload = {
        "question": query,
        "active_product_context": None if resolved_query == query else resolved_query,
        "products": [_product_payload(product) for product in products],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


# Lazy singletons — initialized on first use
_groq_client: AsyncGroq | None = None


def _require_groq_api_key() -> str:
    if settings.groq_api_key:
        return settings.groq_api_key
    raise RuntimeError("GROQ_API_KEY is not set. Set it in backend/.env.")


def get_groq_client() -> AsyncGroq:
    global _groq_client
    if _groq_client is None:
        _groq_client = AsyncGroq(api_key=_require_groq_api_key())
    return _groq_client


async def generate(products: list[Product], query: str, resolved_query: str) -> str:
    response = await get_groq_client().chat.completions.create(
        **_request_options(products, query, resolved_query)
    )
    return response.choices[0].message.content or ""


async def stream_generate(
    products: list[Product], query: str, resolved_query: str
) -> AsyncGenerator[str, None]:
    stream = await get_groq_client().chat.completions.create(
        stream=True,
        **_request_options(products, query, resolved_query),
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def _messages(
    products: list[Product], query: str, resolved_query: str
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _format_context(products, query, resolved_query)},
    ]


def _request_options(
    products: list[Product], query: str, resolved_query: str
) -> dict[str, Any]:
    return {
        "model": settings.groq_model,
        "messages": _messages(products, query, resolved_query),
        "temperature": 0.1,
        "max_completion_tokens": 350,
    }


def _product_payload(product: Product) -> dict[str, object]:
    return {
        "id": product.id,
        "name": product.name,
        "base_item": product.base_item,
        "subcategory": product.subcategory,
        "size": product.size,
        "price_taka": product.price_taka,
        "discounted_price_taka": product.discounted_price_taka,
        "stock_qty": product.stock_qty,
        "rating": product.rating,
    }
