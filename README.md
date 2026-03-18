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

A small Bangla product RAG system built with FastAPI, Groq, and a lightweight retrieval pipeline over a product catalog.

The app:
- retrieves matching Bangla products from the dataset
- keeps short conversational context across turns
- generates short grounded answers from the retrieved product data
- serves a simple frontend from the backend

## Stack

- FastAPI backend
- Groq `llama-3.1-8b-instant`
- SQL-based retrieval
- static frontend served by the backend

## Local Run

From the repo root:

```bash
docker compose up --build
```

Then open:

```text
http://localhost:8000/
```

Set `GROQ_API_KEY` in [`backend/.env`](/home/abir/Repos/rag-system/backend/.env) before starting.

## Hugging Face Spaces

This repo is also configured to run as a Docker Space.

Required secret:
- `GROQ_API_KEY`
