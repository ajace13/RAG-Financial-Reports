# Retrieval-Augmented Q&A for Financial Reports

Ask questions about SEC 10-K and 10-Q filings and get answers grounded in clickable,
passage-level citations. Filings are pulled from EDGAR, parsed into section-aware chunks,
indexed in Elasticsearch for hybrid BM25 + dense vector retrieval, fused with Reciprocal
Rank Fusion, and re-ranked with a cross-encoder.

**Stack:** Python · Elasticsearch 9 · Sentence Transformers · Streamlit (Phase 7) · Claude / OpenAI (Phase 6)

> **Status:** Phases 0–5 complete (ingestion → parsing → indexing → hybrid retrieval →
> re-ranking → retrieval evaluation). Generation with citations (Phase 6) and the Streamlit
> app (Phase 7) are next. See [PLAN.md](PLAN.md).

## How it works

```
EDGAR ─► fetch 10-K/10-Q ─► HTML → text/table blocks ─► Item sections ─► chunks + context header
                                                                              │
                                              ┌───────────────────────────────┤
                                              ▼                               ▼
                                      ES BM25 (english)            ES dense_vector (bge-base, HNSW)
                                              └────── Reciprocal Rank Fusion ─┘
                                                              ▼
                                           cross-encoder re-rank (bge-reranker-base)
                                                              ▼
                                               top passages with EDGAR source links
```

Design decisions worth knowing:

- **Fiscal labels, not calendar years.** Microsoft's FY2025 Q1 ends 2024-09-30. Labels are
  derived from each company's own 10-K year-ends (robust to fiscal-calendar changes), then
  overridden by the filing's inline-XBRL `dei` facts when present.
- **Tables are first-class.** Financial tables become pipe-delimited rows with `$` / `)`
  cells merged and units ("in millions") detected; large tables split by rows with header
  rows repeated, so a line item is never separated from its column years.
- **Context header.** Each chunk is indexed as `company | form | fiscal period | section |
  units` + body, so "Revenue increased 16%" is still findable by "Microsoft FY2024 revenue".
- **Client-side RRF.** Rank-based fusion avoids calibrating BM25 scores against cosine
  similarities, and makes per-retriever ranks available for debugging and ablations.

## Results (retrieval)

Evaluated on the [FinanceBench](https://github.com/patronus-ai/financebench) open sample:
every question that cites a 10-K/10-Q (127 of 150), with relevance judged by matching each
question's evidence text against chunks. See [eval/results](eval/results) and
`scripts/run_retrieval_eval.py`.

125 of 127 questions could be judged. Hit@k = at least one evidence chunk in the top k.
Latency is per query on a laptop RTX 5060 with a local single-node Elasticsearch.

**Document setting:** search restricted to the filing the question cites ("given this 10-K").

| Method | Hit@1 | Hit@5 | Hit@10 | MRR@10 | Latency |
|---|---|---|---|---|---|
| BM25 | 0.168 | 0.264 | 0.360 | 0.214 | 27 ms |
| Vector (bge-base) | **0.224** | **0.440** | 0.528 | **0.313** | 27 ms |
| Hybrid (RRF) | **0.224** | 0.416 | 0.536 | **0.313** | 32 ms |
| Hybrid + MiniLM cross-encoder | 0.144 | 0.376 | 0.480 | 0.243 | 145 ms |
| Hybrid + bge-reranker-base | 0.152 | 0.416 | **0.544** | 0.265 | 643 ms |

**Company setting:** search across all of that company's filings, which also tests whether
retrieval lands on the right fiscal year.

| Method | Hit@1 | Hit@5 | Hit@10 | MRR@10 | Latency |
|---|---|---|---|---|---|
| BM25 | 0.096 | 0.168 | 0.248 | 0.134 | 6 ms |
| Vector (bge-base) | **0.136** | 0.312 | 0.416 | **0.210** | 20 ms |
| Hybrid (RRF) | 0.112 | 0.312 | 0.376 | 0.194 | 31 ms |
| Hybrid + MiniLM cross-encoder | 0.088 | 0.256 | 0.336 | 0.154 | 120 ms |
| Hybrid + bge-reranker-base | 0.080 | **0.368** | **0.488** | 0.196 | 592 ms |

What the numbers say:

- **FinanceBench is table-heavy.** For 90 of 125 questions the evidence is only in tables.
  BM25 reaches Hit@10 of 0.26 on those versus 0.49 for dense retrieval, because the
  vocabulary doesn't match ("capital expenditure" vs "Purchases of property, plant and
  equipment").
- **RRF helps when evidence is prose** (Hit@10 0.71 vs 0.61 for vector alone), but on tables
  the weak BM25 list dilutes the fused ranking, so overall hybrid ≈ vector.
- **The off-the-shelf MS-MARCO cross-encoder hurts.** It was trained on web prose and does
  poorly on pipe-delimited tables. `bge-reranker-base` recovers Hit@5/10 and is the best
  method across a company's filings, but neither re-ranker improves Hit@1, and both cost
  4–20× the latency. `bge-reranker-base` is therefore the default re-ranker.

Next experiments: re-rank on the passage body instead of the header-prefixed text, strip
FinanceBench's answer-format instructions from queries, weighted RRF or BM25 boosting of
table titles, and an LLM re-ranker for table passages.

## Quickstart

Prerequisites: [uv](https://docs.astral.sh/uv/), Docker. A CUDA GPU is optional (CPU works,
embedding is just slower). On Windows, the lockfile pins CUDA 13.0 PyTorch wheels.

```bash
cp .env.example .env              # set SEC_USER_AGENT="Your Name you@example.com"
docker compose up -d              # Elasticsearch on localhost:9200
uv sync                           # Python 3.12 env with all dependencies
uv run python scripts/check_es.py
```

Build the corpus and index:

```bash
uv run python scripts/ingest.py                  # 5 tickers x 3 years of 10-K/10-Q
uv run python scripts/ingest_financebench.py     # filings cited by FinanceBench (eval)
uv run python scripts/build_chunks.py            # parse + chunk -> data/processed/
uv run python scripts/embed_chunks.py            # embeddings (cached by chunk id)
uv run python scripts/build_index.py --recreate  # load Elasticsearch
```

Search and evaluate:

```bash
uv run python scripts/search.py "What drove Azure growth?" --ticker MSFT --fiscal-year 2024
uv run python scripts/run_retrieval_eval.py
# variant run, e.g. a different re-ranker, written to *_minilm.* result files
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2 uv run python scripts/run_retrieval_eval.py \
    --methods hybrid+rerank --tag minilm
```

## Project layout

```
src/finrag/
  ingest/       EDGAR client, filing discovery, fiscal labels, raw cache + manifest
  parsing/      HTML → blocks, tables, Item sections, iXBRL dei facts, per-filing pipeline
  chunking.py   section-aware prose/table chunking with context headers
  embedding.py  Sentence-Transformers embedder + on-disk cache
  index/        Elasticsearch client, schema, bulk indexer
  retrieve/     BM25, kNN, RRF fusion, cross-encoder reranker, Retriever
  evaluation/   FinanceBench loading, evidence-based relevance, ranking metrics
scripts/        pipeline entry points (ingest → chunk → embed → index → search/eval)
tests/          unit tests (default) and live-ES integration tests (-m integration)
```

## Development

```bash
uv run ruff check . && uv run ruff format --check .
uv run pytest                     # unit tests
uv run pytest -m integration      # needs Elasticsearch running with the index built
```

## Known limitations

- **JPMorgan 10-Qs** place MD&A and financial statements outside the standard Item headings,
  so those chunks are labelled "Section not identified" (≈14% of all chunks).
- **Fiscal-year naming** for years ending late January/early February differs by company;
  derived labels follow the end-year convention and are corrected by `dei` facts on filings
  from ~2019 onward.
- Relevance labels are automatic (evidence-text matching), not human judgments; 2 of 127
  questions could not be matched and are excluded from metrics.
