import json
from collections.abc import AsyncGenerator

from groq import AsyncGroq

from config import settings
from models import Product

SYSTEM_PROMPT = """
You are a helpful e-commerce assistant for a Bangladeshi online store.

Rules:
- Answer in the same language the user asked in. If they ask in Bangla, reply in Bangla. If in English, reply in English.
- Only use the JSON product data provided to you — do not make up prices, sizes, stock, or products.
- If multiple products match, list them briefly instead of picking just one.
- If no product is found, politely say that it is not available.
- When asked about price for a product family or category, list the matching products and their prices.
- When asked about price, mention both the original and discounted price if they differ.
- When asked about price, discount, or stock, answer only from the retrieved product fields.
- Do not compare prices with phrases like "smaller than", "lower than", or similar wording. State the prices directly.
- Do not answer about generic entities like a company unless that is explicitly in the product data.
- If the active product context is provided, treat it as the subject of short follow-up questions like "দাম কত?" or "Is there a discount?".
- Keep answers short (2-3 sentences max, unless listing products).
"""

_COMPLETION_OPTIONS = {"temperature": 0.1, "max_completion_tokens": 350}


def _format_context(products: list[Product], query: str, resolved_query: str) -> str:
    return _payload_json(
        question=query,
        active_product_context=None if resolved_query == query else resolved_query,
        products=[_product_payload(product) for product in products],
    )


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
        *FEW_SHOT_EXAMPLES,
        {"role": "user", "content": _format_context(products, query, resolved_query)},
    ]


def _request_options(
    products: list[Product], query: str, resolved_query: str
) -> dict[str, object]:
    return {
        "model": settings.groq_model,
        "messages": _messages(products, query, resolved_query),
        **_COMPLETION_OPTIONS,
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


def _payload_json(
    question: str,
    active_product_context: str | None,
    products: list[dict[str, object]],
) -> str:
    return json.dumps(
        {
            "question": question,
            "active_product_context": active_product_context,
            "products": products,
        },
        ensure_ascii=False,
        indent=2,
    )


def _user_example(
    question: str,
    active_product_context: str | None,
    products: list[dict[str, object]],
) -> dict[str, str]:
    return {
        "role": "user",
        "content": _payload_json(question, active_product_context, products),
    }


def _assistant_example(content: str) -> dict[str, str]:
    return {"role": "assistant", "content": content}


FEW_SHOT_EXAMPLES: list[dict[str, str]] = [
    _user_example(
        question="দাম কত?",
        active_product_context="নুডুলস",
        products=[
            {
                "name": "নেসলে বিফ নুডুলস ১০০গ্রাম",
                "base_item": "নুডুলস",
                "size": "১০০গ্রাম",
                "price_taka": 103,
                "discounted_price_taka": 102,
                "stock_qty": 163,
            },
            {
                "name": "রয়্যাল চিংড়ি নুডুলস ১০০গ্রাম",
                "base_item": "নুডুলস",
                "size": "১০০গ্রাম",
                "price_taka": 118,
                "discounted_price_taka": 110,
                "stock_qty": 98,
            },
        ],
    ),
    _assistant_example(
        """নুডুলসের দামের কিছু উদাহরণ:
- নেসলে বিফ নুডুলস ১০০গ্রাম: মূল দাম 103 BDT, ছাড়ের দাম 102 BDT
- রয়্যাল চিংড়ি নুডুলস ১০০গ্রাম: মূল দাম 118 BDT, ছাড়ের দাম 110 BDT"""
    ),
    _user_example(
        question="৫ লিটারের দাম কত?",
        active_product_context="সয়াবিন তেল",
        products=[
            {
                "name": "সয়াবিন তেল ৫লিটার",
                "base_item": "সয়াবিন তেল",
                "size": "৫লিটার",
                "price_taka": 836,
                "discounted_price_taka": 786,
                "stock_qty": 47,
            },
            {
                "name": "সয়াবিন তেল ১লিটার",
                "base_item": "সয়াবিন তেল",
                "size": "১লিটার",
                "price_taka": 589,
                "discounted_price_taka": 539,
                "stock_qty": 30,
            },
        ],
    ),
    _assistant_example("সয়াবিন তেল ৫লিটারের মূল দাম 836 BDT, আর ছাড়ের দাম 786 BDT।"),
    _user_example(question="কফি মেশিন আছে?", active_product_context=None, products=[]),
    _assistant_example("দুঃখিত, এই নামে কোনো পণ্য পাইনি। চাইলে অন্য পণ্যের নাম বলুন।"),
]
