# Retrieval ablation: FinanceBench open sample

Judgeable questions: **125 / 127** (10-K/10-Q questions whose evidence text matched at least one chunk of the cited filing).

Embedding model: `BAAI/bge-base-en-v1.5` | Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` | candidates per retriever: 50 | rerank pool: 30

## Setting: document

| Method | Hit@1 | Hit@5 | Hit@10 | MRR@10 | nDCG@10 | Latency (ms) |
|---|---|---|---|---|---|---|
| bm25 | 0.168 | 0.264 | 0.360 | 0.214 | 0.187 | 27 |
| vector | 0.224 | 0.440 | 0.528 | 0.313 | 0.292 | 27 |
| hybrid | 0.224 | 0.416 | 0.536 | 0.313 | 0.289 | 32 |
| hybrid+rerank | 0.144 | 0.376 | 0.480 | 0.243 | 0.253 | 145 |

## Setting: company

| Method | Hit@1 | Hit@5 | Hit@10 | MRR@10 | nDCG@10 | Latency (ms) |
|---|---|---|---|---|---|---|
| bm25 | 0.096 | 0.168 | 0.248 | 0.134 | 0.115 | 6 |
| vector | 0.136 | 0.312 | 0.416 | 0.210 | 0.193 | 20 |
| hybrid | 0.112 | 0.312 | 0.376 | 0.194 | 0.174 | 31 |
| hybrid+rerank | 0.088 | 0.256 | 0.336 | 0.154 | 0.157 | 120 |
