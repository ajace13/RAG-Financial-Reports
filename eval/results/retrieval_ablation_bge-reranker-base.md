# Retrieval ablation: FinanceBench open sample

Judgeable questions: **125 / 127** (10-K/10-Q questions whose evidence text matched at least one chunk of the cited filing).

Embedding model: `BAAI/bge-base-en-v1.5` | Reranker: `BAAI/bge-reranker-base` | candidates per retriever: 50 | rerank pool: 30

## Setting: document

| Method | Hit@1 | Hit@5 | Hit@10 | MRR@10 | nDCG@10 | Latency (ms) |
|---|---|---|---|---|---|---|
| hybrid+rerank | 0.152 | 0.416 | 0.544 | 0.265 | 0.270 | 643 |

## Setting: company

| Method | Hit@1 | Hit@5 | Hit@10 | MRR@10 | nDCG@10 | Latency (ms) |
|---|---|---|---|---|---|---|
| hybrid+rerank | 0.080 | 0.368 | 0.488 | 0.196 | 0.206 | 592 |
