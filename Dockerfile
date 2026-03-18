FROM python:3.14-slim

RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:/home/user/app/backend/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    DATABASE_URL=sqlite:////tmp/rag-system.db \
    CSV_PATH=/home/user/app/data/bangla_products_5k.csv

WORKDIR /home/user/app/backend

COPY --from=ghcr.io/astral-sh/uv:latest /uv /home/user/.local/bin/uv
COPY --chown=user backend/pyproject.toml backend/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY --chown=user backend /home/user/app/backend
COPY --chown=user frontend /home/user/app/frontend
COPY --chown=user data /home/user/app/data

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
