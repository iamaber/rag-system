from collections.abc import Callable, Iterable
import re
from typing import Literal

from models import Product

_BANGLA_RE = re.compile(r"[\u0980-\u09FF]")
_PRODUCT_LIST_LIMIT = 5
_EXAMPLE_NAME_LIMIT = 3

_PRICE_TERMS = (
    "দাম",
    "মূল্য",
    "কত টাকা",
    "price",
    "cost",
)
_DISCOUNT_TERMS = (
    "ছাড়",
    "ছাড়",
    "ডিসকাউন্ট",
    "discount",
    "offer",
    "sale",
)
_STOCK_TERMS = (
    "স্টক",
    "মজুত",
    "কতটি আছে",
    "কতটা আছে",
    "কত পিস আছে",
    "stock",
)
_AVAILABILITY_TERMS = (
    "আছে",
    "পাওয়া যায়",
    "পাওয়া যায়",
    "বিক্রি",
    "রাখেন",
    "মিলবে",
    "available",
    "have",
    "sell",
)

Intent = Literal["availability", "price", "discount", "stock"]
ReplyBuilder = Callable[[list[Product], str | None, bool], str]


def generate_catalog_reply(
    products: list[Product], query: str, resolved_query: str
) -> str | None:
    intent = _classify_intent(query)
    if intent is None:
        return None

    subject = _subject_label(products, resolved_query)
    is_bangla = _is_bangla(query)
    reply_builder = _INTENT_HANDLERS.get(intent)
    if reply_builder is None:
        return None
    return reply_builder(products, subject, is_bangla)


def _classify_intent(query: str) -> Intent | None:
    q = query.casefold()
    if _contains_any(q, _DISCOUNT_TERMS):
        return "discount"
    if _contains_any(q, _STOCK_TERMS):
        return "stock"
    if _contains_any(q, _PRICE_TERMS) or ("কত" in q and "টাকা" in q):
        return "price"
    if _contains_any(q, _AVAILABILITY_TERMS):
        return "availability"
    return None


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    return any(term in text for term in terms)


def _is_bangla(text: str) -> bool:
    return bool(_BANGLA_RE.search(text))


def _subject_label(products: list[Product], resolved_query: str) -> str | None:
    candidate = resolved_query.strip()
    if candidate and not _looks_like_raw_question(candidate):
        return candidate

    base_items = {product.base_item.strip() for product in products if product.base_item}
    if len(base_items) == 1:
        return next(iter(base_items))

    subcategories = {
        product.subcategory.strip() for product in products if product.subcategory
    }
    if len(subcategories) == 1:
        return next(iter(subcategories))

    for product in products:
        if product.base_item:
            return product.base_item
        if product.subcategory:
            return product.subcategory
        if product.name:
            return product.name
    return candidate or None


def _looks_like_raw_question(text: str) -> bool:
    q = text.casefold()
    return "?" in q or _contains_any(
        q, _PRICE_TERMS + _DISCOUNT_TERMS + _STOCK_TERMS + _AVAILABILITY_TERMS
    )


def _availability_reply(
    products: list[Product], subject: str | None, is_bangla: bool
) -> str:
    label = _subject_text(subject, is_bangla)
    if products:
        names = _top_names(products, limit=_EXAMPLE_NAME_LIMIT)
        if is_bangla:
            reply = f"জি, আমাদের কাছে {label} আছে।"
            if names:
                reply += f" যেমন: {', '.join(names)}।"
            return reply
        reply = f"Yes, we have {label}."
        if names:
            reply += f" Examples: {', '.join(names)}."
        return reply

    if is_bangla:
        return f"দুঃখিত, আমাদের কাছে {label} পাইনি।"
    return f"Sorry, we could not find {label} in the catalog."


def _price_reply(products: list[Product], subject: str | None, is_bangla: bool) -> str:
    label = _subject_text(subject, is_bangla)
    if not products:
        if is_bangla:
            return f"দুঃখিত, {label} এর দামের তথ্য পাইনি।"
        return f"Sorry, I could not find pricing for {label}."

    lines = [_price_line(product, is_bangla) for product in products[:_PRODUCT_LIST_LIMIT]]
    if is_bangla:
        header = f"{label} এর দামের কিছু উদাহরণ:"
    else:
        header = f"Here are some prices for {label}:"
    return _listing_reply(lines, header)


def _discount_reply(
    products: list[Product], subject: str | None, is_bangla: bool
) -> str:
    discounted = [
        product
        for product in products
        if product.price_taka is not None
        and product.discounted_price_taka is not None
        and product.discounted_price_taka != product.price_taka
    ]
    label = _subject_text(subject, is_bangla)

    if not products:
        if is_bangla:
            return f"দুঃখিত, {label} এর ছাড়ের তথ্য পাইনি।"
        return f"Sorry, I could not find discount information for {label}."

    if not discounted:
        if is_bangla:
            return f"বর্তমানে {label} এ দৃশ্যমান কোনো ছাড় নেই।"
        return f"There is no visible discount right now for {label}."

    lines = [
        _discount_line(product, is_bangla)
        for product in discounted[:_PRODUCT_LIST_LIMIT]
    ]
    if is_bangla:
        header = f"{label} এর ছাড়ের কিছু উদাহরণ:"
    else:
        header = f"Here are some discounts for {label}:"
    return _listing_reply(lines, header)


def _stock_reply(products: list[Product], subject: str | None, is_bangla: bool) -> str:
    label = _subject_text(subject, is_bangla)
    if not products:
        if is_bangla:
            return f"দুঃখিত, {label} এর স্টকের তথ্য পাইনি।"
        return f"Sorry, I could not find stock information for {label}."

    lines = [_stock_line(product, is_bangla) for product in products[:_PRODUCT_LIST_LIMIT]]
    if is_bangla:
        header = f"{label} এর স্টকের কিছু উদাহরণ:"
    else:
        header = f"Here is some stock information for {label}:"
    return _listing_reply(lines, header)


def _price_line(product: Product, is_bangla: bool) -> str:
    if product.price_taka is None and product.discounted_price_taka is None:
        if is_bangla:
            return f"- {product.name}: দামের তথ্য নেই"
        return f"- {product.name}: pricing unavailable"

    effective_price = (
        product.discounted_price_taka
        if product.discounted_price_taka is not None
        else product.price_taka
    )
    if (
        product.price_taka is not None
        and product.discounted_price_taka is not None
        and product.discounted_price_taka != product.price_taka
    ):
        if is_bangla:
            return (
                f"- {product.name}: মূল দাম {product.price_taka} BDT, "
                f"ছাড়ের দাম {product.discounted_price_taka} BDT"
            )
        return (
            f"- {product.name}: original price {product.price_taka} BDT, "
            f"discounted price {product.discounted_price_taka} BDT"
        )

    if is_bangla:
        return f"- {product.name}: দাম {effective_price} BDT"
    return f"- {product.name}: price {effective_price} BDT"


def _discount_line(product: Product, is_bangla: bool) -> str:
    if is_bangla:
        return (
            f"- {product.name}: মূল দাম {product.price_taka} BDT, "
            f"ছাড়ের দাম {product.discounted_price_taka} BDT"
        )
    return (
        f"- {product.name}: original price {product.price_taka} BDT, "
        f"discounted price {product.discounted_price_taka} BDT"
    )


def _stock_line(product: Product, is_bangla: bool) -> str:
    if product.stock_qty is None:
        if is_bangla:
            return f"- {product.name}: স্টকের তথ্য নেই"
        return f"- {product.name}: stock unavailable"

    if is_bangla:
        return f"- {product.name}: স্টক {product.stock_qty}"
    return f"- {product.name}: stock {product.stock_qty}"


def _top_names(products: list[Product], limit: int) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for product in products:
        if not product.name or product.name in seen:
            continue
        names.append(product.name)
        seen.add(product.name)
        if len(names) >= limit:
            break
    return names


def _fallback_subject(is_bangla: bool) -> str:
    return "এই পণ্যটি" if is_bangla else "this product"


def _subject_text(subject: str | None, is_bangla: bool) -> str:
    return subject or _fallback_subject(is_bangla)


def _listing_reply(lines: list[str], header: str) -> str:
    if len(lines) == 1:
        return lines[0]
    return "\n".join([header, *lines])


_INTENT_HANDLERS: dict[Intent, ReplyBuilder] = {
    "availability": _availability_reply,
    "price": _price_reply,
    "discount": _discount_reply,
    "stock": _stock_reply,
}
