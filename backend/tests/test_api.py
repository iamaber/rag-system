from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient

import config
import llm
import main as app_module
import retrieval


@pytest.fixture
def client(tmp_path, catalog_csv, monkeypatch):
    monkeypatch.setattr(
        config.settings, "database_url", f"sqlite:///{tmp_path / 'api.db'}"
    )
    monkeypatch.setattr(config.settings, "csv_path", str(catalog_csv))
    monkeypatch.setattr(config.settings, "max_retrieval_results", 5)
    monkeypatch.setattr(config.settings, "groq_api_key", None)

    retrieval.clear_cache()
    with TestClient(app_module.app) as test_client:
        yield test_client
    retrieval.clear_cache()


def test_health_reports_the_seeded_catalog(client) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["db_connected"] is True
    assert body["product_count"] == 4


def test_health_returns_503_when_the_database_is_unreachable(
    client, monkeypatch
) -> None:
    @asynccontextmanager
    async def broken_connection():
        raise RuntimeError("database unavailable")
        yield

    monkeypatch.setattr(app_module, "get_conn", broken_connection)

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["status"] == "degraded"


def test_chat_answers_a_price_question_from_the_catalog(client) -> None:
    response = client.post(
        "/chat",
        json={"session_id": "session-1", "message": "লেমন বিস্কুটের দাম কত?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["resolved_query"] == "লেমন বিস্কুট"
    assert [product["id"] for product in body["products"]] == ["PRD-1"]
    assert "200" in body["response"]


def test_chat_resolves_a_follow_up_to_the_previous_product(client) -> None:
    client.post(
        "/chat",
        json={"session_id": "session-2", "message": "লেমন বিস্কুটের দাম কত?"},
    )

    response = client.post(
        "/chat",
        json={"session_id": "session-2", "message": "এটার দাম কত?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["resolved_query"] == "লেমন বিস্কুট"
    assert [product["id"] for product in body["products"]] == ["PRD-1"]


def test_chat_stream_frames_metadata_tokens_and_done(client) -> None:
    with client.stream(
        "POST",
        "/chat/stream",
        json={"session_id": "session-3", "message": "লেমন বিস্কুটের দাম কত?"},
    ) as response:
        assert response.status_code == 200
        stream_text = "".join(response.iter_text())

    assert "event: metadata" in stream_text
    assert "event: token" in stream_text
    assert "event: done" in stream_text


def test_chat_returns_502_when_the_llm_fails(client, monkeypatch) -> None:
    monkeypatch.setattr(llm, "is_configured", lambda: True)

    async def failing_generate(*_args, **_kwargs):
        raise RuntimeError("upstream down")

    monkeypatch.setattr(llm, "generate", failing_generate)

    response = client.post(
        "/chat",
        json={"session_id": "session-4", "message": "tell me a joke"},
    )

    assert response.status_code == 502


def test_chat_stream_emits_an_error_event_when_the_llm_fails(
    client, monkeypatch
) -> None:
    monkeypatch.setattr(llm, "is_configured", lambda: True)

    async def failing_stream(*_args, **_kwargs):
        raise RuntimeError("upstream down")
        yield ""

    monkeypatch.setattr(llm, "stream_generate", failing_stream)

    with client.stream(
        "POST",
        "/chat/stream",
        json={"session_id": "session-5", "message": "tell me a joke"},
    ) as response:
        assert response.status_code == 200
        stream_text = "".join(response.iter_text())

    assert "event: error" in stream_text


def test_session_reset_clears_the_context(client) -> None:
    response = client.post("/session/reset", json={"session_id": "session-6"})

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "session_id": "session-6"}
