# MindVault

[![CI](https://github.com/SA-Ark/mindvault/actions/workflows/ci.yml/badge.svg)](https://github.com/SA-Ark/mindvault/actions/workflows/ci.yml)
![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![PostgreSQL](https://img.shields.io/badge/postgres-pgvector-336791.svg)

**RAG + knowledge-graph memory engine on PostgreSQL/pgvector — hybrid recall that is measured, not vibed.**

MindVault is the packaged core of a production memory system that serves **68,000+ memories** with hybrid
semantic + keyword recall. It treats retrieval quality as an engineering discipline: the recall pipeline is
staged and explainable, every scoring constant earns its place, and the repo ships the **evaluation
harness** used to prove changes help before they ship.

## ▶ Try it live

**[mindvault.chakrakali.com](https://mindvault.chakrakali.com)** — an interactive playground (no signup).

Type a query and watch the six-stage hybrid pipeline run in real time. Every result is tagged with the
**signals that surfaced it** — `semantic`, `keyword`, `entity`, or `graph` — and an expandable panel shows
each retrieval leg's raw ranking. Ask *"how does rank fusion work?"* and you'll see the canonical hit come
back at #1 **and its caveat ride in at #2 on a graph edge** — the difference between "found the fact" and
"found the fact *and its caveat*". You can ingest your own memories and watch the knowledge graph grow.

The demo runs the *exact* recall code in this repo against a live PostgreSQL+pgvector, seeded with a small
hand-curated corpus. It is **fully self-hosted** — a single Rust binary (axum) serving both the API and the
UI, with a deterministic, dependency-free embedder so it needs nothing beyond Postgres (no embedding SaaS).

### Run the demo yourself

```bash
# 1. PostgreSQL with pgvector
docker run -d -e POSTGRES_PASSWORD=pw -p 5440:5432 pgvector/pgvector:pg17

# 2. the demo server (schema + demo corpus apply automatically on first boot)
DATABASE_URL=postgres://postgres:pw@localhost:5440/postgres \
  cargo run --release --bin mindvault-server
# → open http://localhost:3930
```

To use a real semantic embedder instead of the built-in deterministic one, point at any self-hosted
OpenAI-compatible `/embeddings` endpoint:

```bash
MINDVAULT_EMBED_ENDPOINT=http://localhost:8080/v1/embeddings \
MINDVAULT_EMBED_MODEL=bge-small-en MINDVAULT_EMBED_DIM=384 \
DATABASE_URL=postgres://... cargo run --release --bin mindvault-server
```


## The recall pipeline

```
  query ──────────────┬──────────────────────────────┐
                      ▼                              ▼
        ① semantic leg (pgvector)         ② keyword leg (Postgres FTS)
           cosine ANN, over-fetch 3×          ts_rank, over-fetch 3×
                      │                              │
                      └──────────┬───────────────────┘
                                 ▼
                  ③ Reciprocal Rank Fusion (k = 60)
                                 │
                  ④ graph-neighbor boost ◄─── memory_edges
                     (neighbors of top semantic hits          (similarity /
                      ride in at their anchor's rank)          co-access links)
                                 │
                  ⑤ entity-aware boost ◄───── kg_entities ∙ kg_entity_memories
                     (memories linked to query-matched         (typed knowledge
                      knowledge-graph entities)                 graph)
                                 │
                  ⑥ time decay × importance bonus
                     (per-type decay: lessons fade slowest,
                      operational notes fade fastest)
                                 ▼
                      deterministic top-k
```

Stages ③–⑥ are pure functions (`src/fuse.rs`, `src/model.rs`) — unit-tested without a database, so the
ranking behavior is verifiable in milliseconds and explainable line by line.

## Why these design choices

- **RRF instead of score blending.** BM25/ts_rank and cosine similarity live on incompatible scales;
  rank-based fusion sidesteps normalization entirely and is robust to either leg degrading.
- **A knowledge graph next to the vectors.** Pure similarity misses *related-but-differently-worded*
  memories. Graph edges (and entity links) let one strong hit pull in its cluster — stage ④ is routinely
  the difference between "found the fact" and "found the fact *and its caveat*".
- **Typed time decay.** A six-month-old `lesson` is gold; a six-month-old operational `note` is noise.
  Decay rates are per-type (`lesson` 0.5%/h ... `workflow` 5%/h on the score multiplier) — tuned in
  production against real recall quality.
- **Content-addressed storage.** Memories are keyed by SHA-256 of content: identical writes upsert
  instead of duplicating, and external systems can reference memories by stable hash.
- **Embeddings in a side table.** Swapping embedding models (or dimensionality) never rewrites the
  memories table.
- **Pluggable embedders.** Any OpenAI-compatible `/embeddings` endpoint via `HttpEmbedder`; a
  deterministic `HashEmbedder` keeps tests and CI hermetic.

## The eval harness

`mindvault-eval` runs a labeled JSONL query set against a live store and reports **recall@k, MRR, hit
rate, and latency percentiles**:

```bash
mindvault-eval --database-url postgres://... --cases eval/cases.jsonl --k 10
```

```json
{
  "cases": 48, "k": 10,
  "recall_at_k": 0.91, "mrr": 0.78, "hit_rate": 0.96,
  "latency_p50_ms": 19.4, "latency_p95_ms": 25.2
}
```

Retrieval changes (new boost, different decay, new embedding model) are judged by this harness, not by
eyeballing three queries.

## Benchmarks

Measured on this repo's `examples/bench.rs` (Intel i7-13700H, PostgreSQL 17 + pgvector, release build):

| Metric | Result |
|---|---|
| Hybrid recall p50 / p95 (10,000 memories, k=10, full 6-stage pipeline, exact scan) | **19.4 ms / 25.2 ms** |
| Pg integration suite (5 end-to-end scenarios) | 0.24 s |

The 10K benchmark runs **without** an HNSW index (exact cosine scan) — add
`CREATE INDEX ... USING hnsw (embedding vector_cosine_ops)` for larger corpora.

## Quickstart

```bash
# 1. a PostgreSQL with pgvector
docker run -d -e POSTGRES_PASSWORD=pw -p 5440:5432 pgvector/pgvector:pg17

# 2. schema applies automatically on connect
cargo test                                # pure unit tests (no DB needed)
MINDVAULT_TEST_DATABASE_URL=postgres://postgres:pw@localhost:5440/postgres \
  cargo test --test integration_pg       # full pipeline against real pgvector
```

```rust
use mindvault::{embed::{Embedder, HttpEmbedder}, store::MemoryStore};

let store = MemoryStore::connect(&database_url).await?;
let embedder = HttpEmbedder::new("http://localhost:8080/v1/embeddings", "bge-small", 384);

let emb = embedder.embed("the deploy pipeline gates on the test suite").await?;
store.store("the deploy pipeline gates on the test suite",
            Some("fact"), &["ci".into()], 1.0, &emb).await?;

let hits = store.hybrid_recall("what gates deploys?", &query_emb, 5).await?;
```

## Repo layout

| Path | What |
|---|---|
| `migrations/0001_init.sql` | Full schema: memories, embeddings, KG entities/relations, memory graph |
| `src/store.rs` | Persistence + the 6-stage `hybrid_recall` pipeline |
| `src/fuse.rs` | RRF fusion, boosts, final scoring — pure & unit-tested |
| `src/model.rs` | Memory model + typed time-decay policy |
| `src/embed.rs` | `Embedder` trait, HTTP + deterministic hash embedders |
| `src/eval.rs` + `src/bin/eval.rs` | The retrieval eval harness |
| `src/server.rs` + `src/bin/server.rs` | HTTP API (axum) + the live playground demo |
| `src/seed.rs` | Hand-curated demo corpus, graph edges, and knowledge graph |
| `static/index.html` | The playground UI (zero third-party JS; served by the binary) |
| `tests/integration_pg.rs` | End-to-end scenarios against real pgvector (CI service container) |

## License

MIT — see [LICENSE](LICENSE).
