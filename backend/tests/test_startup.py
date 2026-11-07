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


async def test_migrations_are_recorded_and_not_reapplied(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        config.settings, "database_url", f"sqlite:///{tmp_path / 'migrations.db'}"
    )
    monkeypatch.setattr(config.settings, "csv_path", str(tmp_path / "missing.csv"))

    await database.close_db()
    await database.init_db()
    try:
        async with database.get_conn() as conn:
            versions = await conn.fetch("SELECT version FROM schema_migrations")
        assert [row["version"] for row in versions] == [1]

        # A second startup must not re-run the baseline migration.
        await database.init_db()
        async with database.get_conn() as conn:
            versions = await conn.fetch("SELECT version FROM schema_migrations")
        assert [row["version"] for row in versions] == [1]
    finally:
        await database.close_db()
