# Retrieval-Augmented Q&A for Financial Reports

Ask questions about SEC 10-K and 10-Q filings and get answers grounded in clickable,
passage-level citations. Filings are pulled from EDGAR, indexed in Elasticsearch for
hybrid BM25 + dense vector retrieval, re-ranked with a cross-encoder, and answered by
an LLM that cites the exact passages it used.

**Stack:** Python · Elasticsearch · Sentence Transformers · Streamlit · Claude

> Status: Phase 0 (scaffold). See [PLAN.md](PLAN.md) for the architecture and roadmap.

## Quickstart

Prerequisites: [uv](https://docs.astral.sh/uv/), Docker.

```bash
cp .env.example .env              # then set SEC_USER_AGENT
docker compose up -d              # Elasticsearch on localhost:9200
uv sync                           # creates .venv with Python 3.12
uv run python scripts/check_es.py # verifies the cluster is healthy
```

## Development

```bash
uv run ruff check .
uv run ruff format .
uv run pytest                     # unit tests
uv run pytest -m integration      # needs Elasticsearch running
```
