# Build Plan — Retrieval-Augmented Q&A for Financial Reports

RAG over SEC 10-K / 10-Q filings from EDGAR. Hybrid BM25 + dense vector retrieval in
Elasticsearch, cross-encoder re-ranking, and LLM answers grounded in clickable,
passage-level citations. Served through Streamlit.

## Settled decisions

| Topic | Decision |
|---|---|
| Corpus | Demo set: MSFT, AMZN, JPM, PEP, MMM — last 3 years of 10-K + 10-Q. Eval set: every 10-K / 10-Q that the FinanceBench open sample cites. On-demand "enter a ticker → fetch + index" added in Phase 7. |
| LLM | Behind a `Generator` interface (Phase 6). **Default: local model via Ollama** (free; `qwen3:8b` suggested for 8 GB VRAM) using JSON-schema citations + verbatim quote verification, with `num_ctx` set per request. **Optional:** Claude via search-result citations (needs Console credits) and OpenAI (structured quotes). `auto` chains configured providers Claude → OpenAI → Ollama. Retrieval-only mode works with no LLM at all. |
| EDGAR access | Hand-rolled client over the free SEC endpoints (`data.sec.gov/submissions`, Archives). No `edgartools`. |
| Item segmentation | Own parser (Item 1A, 7, 7A, 8, …). Revisit only if it proves too brittle. |
| Search engine | Elasticsearch 9.x, single node in Docker, security disabled, bound to localhost. |
| Hybrid fusion | Client-side Reciprocal Rank Fusion over separate BM25 and kNN queries. |
| Embeddings | `BAAI/bge-base-en-v1.5` (768-d, cosine) to start; model name lives in config. |
| Re-ranker | `BAAI/bge-reranker-base` (chosen after the ablation: `cross-encoder/ms-marco-MiniLM-L-6-v2` lowered Hit@10 on table-heavy questions; bge raised it, at ~600 ms/query). |
| Tables | Each table → Markdown, kept as its own chunk with section title + units line attached. |
| Eval | FinanceBench open sample + ~20 hand-written questions on our tickers. |
| Tooling | uv, Python 3.12, ruff, pytest, GitHub Actions CI. |

## Architecture

```
EDGAR ─► fetch ─► parse HTML ─► sections + tables ─► chunks (+ metadata)
                                                        │
                                  ┌─────────────────────┤
                                  ▼                     ▼
                          ES text (BM25)        ES dense_vector (HNSW)
                                  │                     │
   question ─► metadata filter (ticker, form, fiscal period) ─► top-50 each
                                  └──────── RRF ────────┘
                                             ▼
                              cross-encoder rerank ─► top 6–8
                                             ▼
                              Claude + Citations API ─► answer + cited passages
                                             ▼
                              Streamlit UI (answer, citation chips, debug panel)
```

## Chunk schema (contract between ingestion, indexing and citations)

| Field | Type | Notes |
|---|---|---|
| `chunk_id` | keyword | Stable: `{accession}:{item}:{seq}` |
| `ticker`, `cik` | keyword | |
| `company_name` | keyword | |
| `form` | keyword | `10-K`, `10-Q` (amendments flagged separately) |
| `is_amendment` | boolean | `10-K/A`, `10-Q/A` |
| `fiscal_year`, `fiscal_period` | integer, keyword | Company fiscal labels, **not** calendar year (`FY`, `Q1`–`Q3`). Derived at ingestion from the company's `fiscalYearEnd` + `reportDate`; overridden by iXBRL `dei:DocumentFiscalYearFocus` / `dei:DocumentFiscalPeriodFocus` when present |
| `fiscal_year_end` | keyword | `MMDD` from EDGAR submissions |
| `period_of_report` | date | |
| `filing_date` | date | |
| `accession` | keyword | |
| `part`, `item` | keyword | e.g. part `II`, item `1A` (10-Q item numbers repeat across parts) |
| `section_title` | text + keyword | |
| `chunk_type` | keyword | `prose` or `table` |
| `units` | keyword | e.g. `in millions`, attached to table chunks |
| `seq` | integer | Order of the chunk within its filing |
| `text` | stored only | Passage body shown in citations |
| `search_text` | text (english analyzer) | Context header (company, form, period, section, units) + body. Used for BM25 **and** embedding |
| `embedding` | dense_vector (768, cosine) | Embedding of `search_text` |
| `source_url` | keyword | EDGAR document URL (text-fragment link built at render time) |

Character offsets were dropped from the original schema: text-fragment links locate passages by their text, so offsets were never needed.

## Phases

Each phase ends at a reviewable checkpoint.

| # | Phase | Done when | New deps | Status |
|---|---|---|---|---|
| 0 | Scaffold: uv, Py 3.12, ES in Docker, ruff, pytest, CI | `docker compose up -d` + `scripts/check_es.py` reports a healthy cluster | `elasticsearch`, `python-dotenv`; dev: `pytest`, `ruff` | ✅ |
| 1 | EDGAR ingestion | Raw filings + metadata for ~5 tickers cached in `data/raw/` | `requests` | ✅ 132 filings |
| 2 | Parse + chunk | Chunk JSONL inspected by eye, tables included and units attached | `beautifulsoup4`, `lxml` | ✅ ~47k chunks |
| 3 | ES index + BM25 baseline + CLI search | Sensible passages for hand-written queries | — | ✅ |
| 3.5 | Eval set + retrieval metrics (hit@k, MRR, nDCG) | Baseline numbers recorded | — | ✅ FinanceBench |
| 4 | Embeddings + kNN + RRF | Hybrid vs BM25 measured | `sentence-transformers`, `torch` (CUDA 13.0 build on Windows) | ✅ |
| 5 | Cross-encoder re-ranking | Rerank gain measured | — | ✅ |
| 6 | Generation with citations | Answers cite real passages; refuses when unsupported | `anthropic`, `openai` (Ollama via `requests`, no new dep) | ✅ live on Ollama `qwen3:8b` (5/6 spot-check questions correct; thinking off by default) |
| 7 | Streamlit app (+ on-demand ticker ingest) | End-to-end demo | `streamlit` |
| 8 | Polish | README with diagram, ablation table, demo GIF | — |

## Edge cases to handle

- **Amendments (10-K/A):** indexed with `is_amendment=true`; retrieval prefers originals unless asked.
- **Non-calendar fiscal years** (e.g. Apple, Sept FYE): filter on `fiscal_year` / `fiscal_period`.
- **Units** ("in millions") live in table headers and must travel with every table chunk.
- **Multi-year questions** need retrieval across multiple filings, not a single-filing filter.
- **Arithmetic** (growth rates, margins): LLMs are unreliable — show the computation; consider a calculator step later.
- **Older filings:** we parse HTML text, not XBRL tags, so pre-2019 HTML filings work (they just lack `dei` facts, so fiscal labels fall back to derivation). Only pre-2001 plain-text filings are out of scope.
- **Delisted / renamed companies** (e.g. Activision Blizzard, Block): absent from SEC's current ticker map, so FinanceBench companies map to CIKs explicitly.
- **Prolific filers** (e.g. JPM files thousands of prospectuses): the submissions feed only lists ~1000 recent filings inline; older history lives in paginated files that must be followed.
- **SEC fair-access:** descriptive `User-Agent` with contact info, ≤10 req/s, cache everything.
