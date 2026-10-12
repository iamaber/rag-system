---
title: Bangla RAG System
emoji: 📚
colorFrom: yellow
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Bangla RAG System

A small Bangla product RAG system built with FastAPI, Groq, and a lightweight
retrieval pipeline over a product catalog.

The app:

- retrieves matching Bangla products from the dataset
- keeps short conversational context across turns
- answers price, discount, stock, and availability questions directly from the catalog
- uses Groq `llama-3.1-8b-instant` for open-ended answers when an API key is configured
- serves a simple frontend from the backend

## Stack

- FastAPI backend
- Groq `llama-3.1-8b-instant` (optional at runtime)
- SQL retrieval over SQLite or Postgres
- static frontend served by the backend

## Configuration

All settings are read from environment variables, `backend/.env`, or a `.env`
file at the repo root. Copy the example file first:

```bash
cp .env.example .env
```

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `GROQ_API_KEY` | No | unset | Enables LLM-generated answers. Catalog lookups work without it. |
| `GROQ_MODEL` | No | `llama-3.1-8b-instant` | Groq model name. |
| `DATABASE_URL` | No | `sqlite:////tmp/rag-system.db` | SQLAlchemy-style URL; `postgresql://` uses Postgres. |
| `CSV_PATH` | No | `data/bangla_products_5k.csv` | Product catalog used to seed an empty database. |
| `SESSION_TTL_SECONDS` | No | `1800` | Idle session expiry. |
| `MAX_RETRIEVAL_RESULTS` | No | `5` | Result cap per query. |

The app starts without `GROQ_API_KEY`. In that mode the UI loads and
catalog-backed questions are answered; open-ended questions return a clear
message asking for the key.

## Run with Docker Compose

From the repo root:

```bash
cp .env.example .env
docker compose up --build
```

Open <http://localhost:8000/>.

Compose starts Postgres (pgvector) and the backend. Both `.env` files are
optional, so the stack also starts without a key or env file.

## Run the backend directly

```bash
cd backend
uv sync
uv run uvicorn main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/>.

## Tests

```bash
cd backend
uv run pytest
```

The tests seed a temporary SQLite database from a small fixture catalog and
exercise the retrieval pipeline, query normalization, and context resolution.

## Deploy to Hugging Face Spaces

This repo is configured as a Docker Space:

- `sdk: docker`
- `app_port: 7860`
- process binds `0.0.0.0:7860` as a non-root user
- storage is SQLite at `/tmp`, so no external database is required

Set the optional API key as a Space secret:

1. Open the Space **Settings → Secrets and variables**.
2. Add `GROQ_API_KEY` with your key.

Do not commit a `.env` file. The Space works without the secret; only
LLM-generated answers are disabled.

## Data

`data/bangla_products_5k.csv` contains 5,000 rows with Bengali product names,
categories, sizes, prices in BDT, stock, ratings, and templated descriptions.

The repository does not document where this catalog came from or under what
terms it may be redistributed. The repeated, templated descriptions suggest it
is generated sample data. Treat it as demo data and confirm its provenance and
licensing before production or commercial use.

## License

No license is declared for this repository. Add a `LICENSE` file before
redistributing or reusing the code.
