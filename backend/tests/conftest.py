import csv
from pathlib import Path

import pytest

FIELDS = [
    "id",
    "name",
    "subcategory",
    "base_item",
    "size",
    "price_taka",
    "discounted_price_taka",
    "currency",
    "stock_qty",
    "warehouse",
    "rating",
    "reviews_count",
    "sku",
    "description",
    "seller_notes",
    "search_keywords",
]

ROWS = [
    {
        "id": "PRD-1",
        "name": "ক্লাসিক লেমন বিস্কুট ১০০গ্রাম",
        "subcategory": "বিস্কুট",
        "base_item": "লেমন বিস্কুট",
        "size": "১০০গ্রাম",
        "price_taka": "200",
        "discounted_price_taka": "195",
        "currency": "টাকা",
        "stock_qty": "76",
        "warehouse": "রাজশাহী",
        "rating": "3.5",
        "reviews_count": "2319",
        "sku": "SKU-1",
        "description": "ক্লাসিক লেমন বিস্কুট",
        "seller_notes": "স্টক আছে",
        "search_keywords": "লেমন বিস্কুট দাম মূল্য",
    },
    {
        "id": "PRD-2",
        "name": "শাওমি ওয়্যারলেস ইয়ারবাড ওয়্যারলেস",
        "subcategory": "হেডফোন",
        "base_item": "শাওমি ওয়্যারলেস ইয়ারবাড",
        "size": "ওয়্যারলেস",
        "price_taka": "3907",
        "discounted_price_taka": "3782",
        "currency": "টাকা",
        "stock_qty": "372",
        "warehouse": "সিলেট",
        "rating": "3.8",
        "reviews_count": "1276",
        "sku": "SKU-2",
        "description": "শাওমি ওয়্যারলেস ইয়ারবাড",
        "seller_notes": "স্টক সীমিত",
        "search_keywords": "শাওমি ওয়্যারলেস হেডফোন অডিও",
    },
    {
        "id": "PRD-3",
        "name": "Wireless Earbuds Pro",
        "subcategory": "Headphones",
        "base_item": "Wireless Earbuds",
        "size": "Standard",
        "price_taka": "1500",
        "discounted_price_taka": "1400",
        "currency": "BDT",
        "stock_qty": "10",
        "warehouse": "Dhaka",
        "rating": "4.5",
        "reviews_count": "42",
        "sku": "SKU-3",
        "description": "Wireless earbuds with case",
        "seller_notes": "In stock",
        "search_keywords": "wireless earbuds audio",
    },
    {
        "id": "PRD-4",
        "name": "মিনিকেট চাল ৫ কেজি",
        "subcategory": "চাল",
        "base_item": "মিনিকেট চাল",
        "size": "৫ কেজি",
        "price_taka": "400",
        "discounted_price_taka": "",
        "currency": "টাকা",
        "stock_qty": "0",
        "warehouse": "ঢাকা",
        "rating": "4.0",
        "reviews_count": "8",
        "sku": "SKU-4",
        "description": "মিনিকেট চাল",
        "seller_notes": "স্টক শেষ",
        "search_keywords": "মিনিকেট চাল rice",
    },
]


@pytest.fixture
def catalog_csv(tmp_path: Path) -> Path:
    path = tmp_path / "catalog.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(ROWS)
    return path


@pytest.fixture
async def catalog_db(tmp_path: Path, catalog_csv: Path, monkeypatch: pytest.MonkeyPatch):
    import config
    import database
    import retrieval

    monkeypatch.setattr(
        config.settings, "database_url", f"sqlite:///{tmp_path / 'test.db'}"
    )
    monkeypatch.setattr(config.settings, "csv_path", str(catalog_csv))
    monkeypatch.setattr(config.settings, "max_retrieval_results", 5)

    retrieval.clear_cache()
    await database.close_db()
    await database.init_db()
    try:
        yield
    finally:
        await database.close_db()
        retrieval.clear_cache()
