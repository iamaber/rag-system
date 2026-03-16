import csv
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import asyncpg

from config import settings

logger = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        raise RuntimeError("Database pool not initialized. Call init_db() first.")
    return _pool


@asynccontextmanager
async def get_conn() -> AsyncGenerator[asyncpg.Connection, None]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        yield conn


async def init_db() -> None:
    global _pool
    _pool = await asyncpg.create_pool(
        settings.database_url,
        min_size=2,
        max_size=10,
        command_timeout=30,
    )
    async with _pool.acquire() as conn:
        await _setup_schema(conn)
        await _seed_if_empty(conn)
    logger.info("Database initialized.")


async def close_db() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


async def _setup_schema(conn: asyncpg.Connection) -> None:
    # Create products table
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id              TEXT PRIMARY KEY,
            name            TEXT NOT NULL,
            subcategory     TEXT,
            base_item       TEXT,
            size            TEXT,
            price_taka      INTEGER,
            discounted_price_taka INTEGER,
            stock_qty       INTEGER,
            rating          FLOAT,
            reviews_count   INTEGER,
            sku             TEXT,
            description     TEXT,
            search_keywords TEXT
        )
    """)

    # Full-text search vector column (simple tokenizer works for Bangla tokens)
    await conn.execute("""
        ALTER TABLE products
            ADD COLUMN IF NOT EXISTS search_vector tsvector
    """)

    # Indexes — only create if not already present
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_products_base_item
            ON products (base_item)
    """)
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_products_search_vector
            ON products USING GIN(search_vector)
    """)

    logger.info("Schema ready.")


async def _seed_if_empty(conn: asyncpg.Connection) -> None:
    count = await conn.fetchval("SELECT COUNT(*) FROM products")
    if count and count > 0:
        logger.info("Products table already seeded (%d rows). Skipping.", count)
        return

    logger.info("Seeding products from CSV: %s", settings.csv_path)
    rows = _load_csv(settings.csv_path)

    # Batch insert for speed
    await conn.executemany(
        """
        INSERT INTO products
            (id, name, subcategory, base_item, size, price_taka,
             discounted_price_taka, stock_qty, rating, reviews_count,
             sku, description, search_keywords)
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
        ON CONFLICT (id) DO NOTHING
        """,
        rows,
    )

    # Update search_vector for all rows after bulk insert
    await conn.execute("""
        UPDATE products
        SET search_vector = to_tsvector(
            'simple',
            coalesce(name, '') || ' ' ||
            coalesce(base_item, '') || ' ' ||
            coalesce(subcategory, '') || ' ' ||
            coalesce(search_keywords, '')
        )
    """)

    logger.info("Seeded %d products.", len(rows))


def _load_csv(path: str) -> list[tuple]:
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                (
                    row["id"],
                    row["name"],
                    row.get("subcategory", ""),
                    row.get("base_item", ""),
                    row.get("size", ""),
                    _int(row.get("price_taka")),
                    _int(row.get("discounted_price_taka")),
                    _int(row.get("stock_qty")),
                    _float(row.get("rating")),
                    _int(row.get("reviews_count")),
                    row.get("sku", ""),
                    row.get("description", ""),
                    row.get("search_keywords", ""),
                )
            )
    return rows


def _int(val: str | None) -> int | None:
    try:
        return int(val) if val else None
    except (ValueError, TypeError):
        return None


def _float(val: str | None) -> float | None:
    try:
        return float(val) if val else None
    except (ValueError, TypeError):
        return None
