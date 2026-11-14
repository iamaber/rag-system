import config
import database


async def test_init_db_without_catalog_starts_with_empty_table(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        config.settings, "database_url", f"sqlite:///{tmp_path / 'empty.db'}"
    )
    monkeypatch.setattr(config.settings, "csv_path", str(tmp_path / "missing.csv"))

    await database.close_db()
    await database.init_db()
    try:
        async with database.get_conn() as conn:
            count = await conn.fetchval("SELECT COUNT(*) FROM products")
        assert count == 0
    finally:
        await database.close_db()
