import logging
from collections.abc import Mapping
from collections import OrderedDict

import asyncpg

from config import settings
from database import get_conn, uses_sqlite
from models import Product, RetrievalMethod

logger = logging.getLogger(__name__)
_CACHE_SIZE = 256
_cache: OrderedDict[str, tuple[list[Product], RetrievalMethod]] = OrderedDict()

_SELECT = """
    id, name, base_item, subcategory, size,
    price_taka, discounted_price_taka, stock_qty, rating
"""


async def retrieve(query: str) -> tuple[list[Product], RetrievalMethod]:
    """Run the retrieval pipeline and return `(products, method_used)`."""
    cache_key = _normalize_query(query)
    cached = _cache.pop(cache_key, None)
    if cached is not None:
        _cache[cache_key] = cached
        return cached

    async with get_conn() as conn:
        for method, search in (
            (RetrievalMethod.exact, _exact_match),
            (RetrievalMethod.fulltext, _fulltext_search),
        ):
            products = await search(conn, query)
            if products:
                result = (products, method)
                _remember(cache_key, result)
                return result

    result = ([], RetrievalMethod.none)
    _remember(cache_key, result)
    return result


async def _exact_match(conn: asyncpg.Connection, query: str) -> list[Product]:
    if uses_sqlite():
        return await _fetch_products(
            conn,
            f"""
            SELECT {_SELECT}
            FROM products
            WHERE lower(base_item) = lower(?)
               OR lower(name) LIKE lower(?)
               OR lower(base_item) LIKE lower(?)
            ORDER BY
                CASE WHEN lower(base_item) = lower(?) THEN 0 ELSE 1 END,
                rating DESC
            LIMIT ?
            """,
            query,
            f"%{query}%",
            f"%{query}%",
            query,
            settings.max_retrieval_results,
        )

    return await _fetch_products(
        conn,
        f"""
        SELECT {_SELECT}
        FROM products
        WHERE base_item = $1
           OR name ILIKE $2
           OR base_item ILIKE $2
        ORDER BY
            CASE WHEN base_item = $1 THEN 0 ELSE 1 END,
            rating DESC NULLS LAST
        LIMIT $3
        """,
        query,
        f"%{query}%",
        settings.max_retrieval_results,
    )


async def _fulltext_search(conn: asyncpg.Connection, query: str) -> list[Product]:
    tokens = query.split()
    if not tokens:
        return []

    if uses_sqlite():
        return await _sqlite_fallback_search(conn, tokens)

    tsquery_str = " | ".join(tokens)

    try:
        return await _fetch_products(
            conn,
            f"""
            SELECT {_SELECT},
                   ts_rank(search_vector, to_tsquery('simple', $1)) AS rank
            FROM products
            WHERE search_vector @@ to_tsquery('simple', $1)
            ORDER BY rank DESC, rating DESC NULLS LAST
            LIMIT $2
            """,
            tsquery_str,
            settings.max_retrieval_results,
        )
    except Exception as exc:
        logger.debug("Full-text search failed for '%s': %s", query, exc)
        return []


def clear_cache() -> None:
    _cache.clear()


async def _fetch_products(
    conn: asyncpg.Connection, sql: str, *args: object
) -> list[Product]:
    rows = await conn.fetch(sql, *args)
    return [_row_to_product(row) for row in rows]


async def _sqlite_fallback_search(
    conn: asyncpg.Connection, tokens: list[str]
) -> list[Product]:
    where_clause = " AND ".join(
        [
            """lower(
                coalesce(name, '') || ' ' ||
                coalesce(base_item, '') || ' ' ||
                coalesce(subcategory, '') || ' ' ||
                coalesce(search_keywords, '')
            ) LIKE lower(?)"""
            for _ in tokens
        ]
    )
    sql = f"""
        SELECT {_SELECT}
        FROM products
        WHERE {where_clause}
        ORDER BY rating DESC
        LIMIT ?
    """
    args: list[object] = [f"%{token}%" for token in tokens]
    args.append(settings.max_retrieval_results)
    return await _fetch_products(conn, sql, *args)


def _row_to_product(row: asyncpg.Record | Mapping[str, object]) -> Product:
    return Product(
        id=row["id"],
        name=row["name"],
        base_item=row["base_item"],
        subcategory=row["subcategory"],
        size=row["size"],
        price_taka=row["price_taka"],
        discounted_price_taka=row["discounted_price_taka"],
        stock_qty=row["stock_qty"],
        rating=row["rating"],
    )


def _normalize_query(query: str) -> str:
    return " ".join(query.split())


def _remember(
    cache_key: str, result: tuple[list[Product], RetrievalMethod]
) -> None:
    _cache[cache_key] = result
    _cache.move_to_end(cache_key)
    if len(_cache) > _CACHE_SIZE:
        _cache.popitem(last=False)
