# Bangla Product Q&A System

A small Bangla product Q&A system built with FastAPI, Groq, and keyword-based
SQL retrieval over a product catalog. There are no embeddings and no vector
search: retrieval matches catalog text with `LIKE` and Postgres full-text
`tsquery`.

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
| `MAX_SESSIONS` | No | `5000` | Session cap; least-recently-used sessions are evicted beyond it. |
| `MAX_RETRIEVAL_RESULTS` | No | `5` | Result cap per query. |
| `API_TOKEN` | No | unset | When set, `/chat`, `/chat/stream`, and `/session/reset` require an `X-API-Token` header (or a Bearer token). The bundled UI prompts for it on a 401. |
| `RATE_LIMIT_PER_MINUTE` | No | `30` | Per-client request cap on the endpoints above. |
| `MAX_CONCURRENT_LLM_CALLS` | No | `4` | Concurrent upstream generation cap. |

`docker compose` additionally reads `POSTGRES_DB`, `POSTGRES_USER`,
`POSTGRES_PASSWORD`, `APP_DB_USER`, and `APP_DB_PASSWORD` (see below).

The app starts without `GROQ_API_KEY`. In that mode the UI loads and
catalog-backed questions are answered; open-ended questions return a clear
message asking for the key.

## Run with Docker Compose

From the repo root:

```bash
cp .env.example .env
# set POSTGRES_PASSWORD and APP_DB_PASSWORD in .env first
docker compose up --build
```

Open <http://localhost:8000/>.

Compose runs Postgres 16 and the backend built from the root `Dockerfile`
(which contains the frontend and catalog, so it also runs without compose
mounts). The database container is not published on the host: only the backend
can reach it over the compose network, and it connects as the non-superuser
`APP_DB_USER` role created by `docker/postgres/init-app-role.sh`.

Both `.env` files are otherwise optional, so the stack also starts without a
Groq key. If you already have a `pgdata` volume from an older compose setup,
the app role does not exist in it; run `docker compose down -v` (destroys the
database volume) or create the role manually before starting.

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
cover the retrieval pipeline, query normalization, context resolution, the
catalog reply builders, and the HTTP endpoints (`/chat`, `/chat/stream`,
`/session/reset`, `/health`). GitHub Actions runs the same suite on every push
and pull request.

## Deploy to Hugging Face Spaces

The repo does not commit Space YAML metadata (it is kept out of the GitHub
README). When creating a Docker Space from this repo, use:

- SDK: `docker`
- app port: `7860`
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

The application code is released under the MIT License; see [LICENSE](LICENSE).
The product catalog provenance and redistribution terms remain undocumented;
see [Data](#data). This code licence does not establish rights in third-party data.
