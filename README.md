---
title: Bangla RAG System
emoji: 📚
colorFrom: yellow
colorTo: blue
sdk: docker
app_port: 7860
---

# Bangla RAG System

FastAPI-based Bangla product RAG demo for Hugging Face Spaces.

## Required Space secret

- `GROQ_API_KEY`

## Optional runtime variables

- `GROQ_MODEL` defaults to `llama-3.1-8b-instant`
- `DATABASE_URL` defaults to `sqlite:////tmp/rag-system.db`
- `CSV_PATH` defaults to the bundled product CSV
