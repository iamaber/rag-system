from answering import generate_catalog_reply
from models import Product


def _product(**overrides: object) -> Product:
    data: dict[str, object] = {
        "id": "P1",
        "name": "লেমন বিস্কুট",
        "base_item": "লেমন বিস্কুট",
        "subcategory": "বিস্কুট",
        "size": "১০০গ্রাম",
        "price_taka": 200,
        "discounted_price_taka": 195,
        "stock_qty": 76,
        "rating": 3.5,
    }
    data.update(overrides)
    return Product(**data)  # type: ignore[arg-type]


def test_price_reply_in_bangla() -> None:
    reply = generate_catalog_reply(
        [_product()], "লেমন বিস্কুটের দাম কত?", "লেমন বিস্কুট"
    )

    assert reply is not None
    assert "200" in reply
    assert "195" in reply


def test_price_reply_in_english() -> None:
    reply = generate_catalog_reply(
        [_product(name="Wireless Earbuds", price_taka=1500, discounted_price_taka=None)],
        "what is the price of wireless earbuds?",
        "wireless earbuds",
    )

    assert reply is not None
    assert "1500" in reply


def test_discount_reply_lists_discounted_price() -> None:
    reply = generate_catalog_reply(
        [_product()], "লেমন বিস্কুটে ছাড় আছে?", "লেমন বিস্কুট"
    )

    assert reply is not None
    assert "195" in reply


def test_discount_reply_when_nothing_is_discounted() -> None:
    reply = generate_catalog_reply(
        [_product(discounted_price_taka=None)],
        "লেমন বিস্কুটে ছাড় আছে?",
        "লেমন বিস্কুট",
    )

    assert reply is not None
    assert "ছাড় নেই" in reply


def test_stock_reply_reports_quantity() -> None:
    reply = generate_catalog_reply([_product()], "স্টক কত?", "লেমন বিস্কুট")

    assert reply is not None
    assert "76" in reply


def test_availability_reply_for_found_product() -> None:
    reply = generate_catalog_reply(
        [_product()], "লেমন বিস্কুট আছে?", "লেমন বিস্কুট"
    )

    assert reply is not None
    assert "আছে" in reply


def test_availability_reply_for_missing_product() -> None:
    reply = generate_catalog_reply([], "চকলেট আছে?", "চকলেট")

    assert reply is not None
    assert "পাইনি" in reply


def test_non_catalog_question_defers_to_llm() -> None:
    assert generate_catalog_reply([_product()], "hello there", "hello there") is None
