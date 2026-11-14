FROM python:3.14-slim

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:/home/user/app/backend/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    DATABASE_URL=sqlite:////tmp/rag-system.db \
    CSV_PATH=/home/user/app/data/bangla_products_5k.csv

RUN useradd -m -u 1000 user

WORKDIR /home/user/app

COPY --from=ghcr.io/astral-sh/uv:0.13.8 /uv /usr/local/bin/uv
COPY backend/pyproject.toml backend/uv.lock ./backend/

RUN chown -R user:user /home/user/app

USER user

WORKDIR /home/user/app/backend

RUN uv sync --frozen --no-dev --no-install-project

COPY --chown=user backend /home/user/app/backend
COPY --chown=user frontend /home/user/app/frontend
COPY --chown=user data /home/user/app/data

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
