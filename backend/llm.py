from collections.abc import AsyncGenerator

from groq import AsyncGroq

from config import settings
from models import Product

SYSTEM_PROMPT = """
You are a helpful e-commerce assistant for a Bangladeshi online store.

Rules:
- Answer in the same language the user asked in. If they ask in Bangla, reply in Bangla. If in English, reply in English.
- Only use the product data provided to you — do not make up prices or products.
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

FEW_SHOT_EXAMPLES: list[dict[str, str]] = [
    {
        "role": "user",
        "content": """Question: নুডুলস আছে?

Relevant products:
- নেসলে বিফ নুডুলস ১০০গ্রাম: মূল দাম: 103 BDT, ছাড়ের দাম: 102 BDT, stock: 163
- রয়্যাল চিংড়ি নুডুলস ১০০গ্রাম: মূল দাম: 118 BDT, ছাড়ের দাম: 110 BDT, stock: 98""",
    },
    {
        "role": "assistant",
        "content": """হ্যাঁ, আমাদের নুডুলস আছে। যেমন:
- নেসলে বিফ নুডুলস ১০০গ্রাম: মূল দাম 103 BDT, ছাড়ের দাম 102 BDT
- রয়্যাল চিংড়ি নুডুলস ১০০গ্রাম: মূল দাম 118 BDT, ছাড়ের দাম 110 BDT""",
    },
    {
        "role": "user",
        "content": """Question: দাম কত?
Active product context: নুডুলস

Relevant products:
- নেসলে বিফ নুডুলস ১০০গ্রাম: মূল দাম: 103 BDT, ছাড়ের দাম: 102 BDT, stock: 163
- রয়্যাল চিংড়ি নুডুলস ১০০গ্রাম: মূল দাম: 118 BDT, ছাড়ের দাম: 110 BDT, stock: 98""",
    },
    {
        "role": "assistant",
        "content": """নুডুলসের দামের কিছু উদাহরণ:
- নেসলে বিফ নুডুলস ১০০গ্রাম: মূল দাম 103 BDT, ছাড়ের দাম 102 BDT
- রয়্যাল চিংড়ি নুডুলস ১০০গ্রাম: মূল দাম 118 BDT, ছাড়ের দাম 110 BDT""",
    },
    {
        "role": "user",
        "content": """Question: ৫ লিটারের দাম কত?
Active product context: সয়াবিন তেল

Relevant products:
- সয়াবিন তেল ৫লিটার: মূল দাম: 836 BDT, ছাড়ের দাম: 786 BDT, stock: 47
- সয়াবিন তেল ১লিটার: মূল দাম: 589 BDT, ছাড়ের দাম: 539 BDT, stock: 30""",
    },
    {
        "role": "assistant",
        "content": "সয়াবিন তেল ৫লিটারের মূল দাম 836 BDT, আর ছাড়ের দাম 786 BDT।",
    },
    {
        "role": "user",
        "content": """Question: এই পণ্যটাতে কি ছাড় আছে?
Active product context: প্রিমিয়াম সুতি শাড়ি

Relevant products:
- প্রিমিয়াম সুতি শাড়ি ৬গজ: মূল দাম: 6985 BDT, ছাড়ের দাম: 6504 BDT, stock: 235""",
    },
    {
        "role": "assistant",
        "content": "হ্যাঁ, এতে ছাড় আছে। মূল দাম 6985 BDT, আর ছাড়ের দাম 6504 BDT।",
    },
    {
        "role": "user",
        "content": "Question: কফি মেশিন আছে?\n\nNo matching products found.",
    },
    {
        "role": "assistant",
        "content": "দুঃখিত, এই নামে কোনো পণ্য পাইনি। চাইলে অন্য পণ্যের নাম বলুন।",
    },
]


def _format_context(products: list[Product], query: str, resolved_query: str) -> str:
    """Build the user message: query + matching products as a simple list."""
    context_lines = [f"Question: {query}"]
    if resolved_query != query:
        context_lines.append(f"Active product context: {resolved_query}")

    if not products:
        context_lines.append("")
        context_lines.append("No matching products found.")
        return "\n".join(context_lines)

    context_lines.extend(["", "Relevant products:"])
    for p in products:
        if (
            p.price_taka
            and p.discounted_price_taka
            and p.price_taka != p.discounted_price_taka
        ):
            price = (
                f"মূল দাম: {p.price_taka} BDT, "
                f"ছাড়ের দাম: {p.discounted_price_taka} BDT"
            )
        elif p.price_taka:
            price = f"দাম: {p.price_taka} BDT"
        else:
            price = "price unknown"

        stock = f", stock: {p.stock_qty}" if p.stock_qty is not None else ""
        context_lines.append(f"- {p.name}: {price}{stock}")

    return "\n".join(context_lines)


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
        model=settings.groq_model,
        messages=_messages(products, query, resolved_query),
        temperature=0.1,
        max_completion_tokens=350,
    )
    return response.choices[0].message.content or ""


async def stream_generate(
    products: list[Product], query: str, resolved_query: str
) -> AsyncGenerator[str, None]:
    stream = await get_groq_client().chat.completions.create(
        model=settings.groq_model,
        messages=_messages(products, query, resolved_query),
        stream=True,
        max_completion_tokens=350,
        temperature=0.1,
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
