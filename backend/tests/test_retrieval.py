import retrieval
from models import RetrievalMethod


def test_normalize_query_collapses_whitespace() -> None:
    assert retrieval._normalize_query("  লেমন \n\t বিস্কুট  ") == "লেমন বিস্কুট"


def test_query_variants_are_unique_and_keep_original() -> None:
    variants = retrieval._query_variants("  লেমন   বিস্কুট ")
    assert variants == ["লেমন বিস্কুট"]


def test_query_variants_include_bengali_spelling_variants() -> None:
    assert retrieval._query_variants("কয়") == ["কয়", "কয়"]
    assert retrieval._query_variants("কয়") == ["কয়", "কয়"]


async def test_exact_match_by_base_item(catalog_db) -> None:
    products, method = await retrieval.retrieve("লেমন বিস্কুট")

    assert method is RetrievalMethod.exact
    assert [product.id for product in products] == ["PRD-1"]
    assert products[0].price_taka == 200


async def test_exact_match_is_case_insensitive(catalog_db) -> None:
    products, method = await retrieval.retrieve("wireless earbuds")

    assert method is RetrievalMethod.exact
    assert {product.id for product in products} == {"PRD-3"}


async def test_fulltext_fallback_matches_all_tokens(catalog_db) -> None:
    products, method = await retrieval.retrieve("শাওমি ইয়ারবাড")

    assert method is RetrievalMethod.fulltext
    assert [product.id for product in products] == ["PRD-2"]


async def test_unknown_query_returns_no_products(catalog_db) -> None:
    products, method = await retrieval.retrieve("zzzz-not-in-catalog")

    assert products == []
    assert method is RetrievalMethod.none


async def test_repeated_query_is_served_from_cache(catalog_db) -> None:
    first_products, first_method = await retrieval.retrieve("মিনিকেট চাল")
    second_products, second_method = await retrieval.retrieve("মিনিকেট চাল")

    assert first_method is RetrievalMethod.exact
    assert second_method is first_method
    assert second_products is first_products

    retrieval.clear_cache()
    third_products, _ = await retrieval.retrieve("মিনিকেট চাল")
    assert third_products is not first_products
