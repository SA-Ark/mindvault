# mindvault

mindvault is a memory engine for AI applications, combining hybrid RAG retrieval with a knowledge graph so agents remember facts and how they connect.

It gives AI systems durable, searchable, connected memory: hybrid semantic + keyword recall over a knowledge graph, with retrieval quality treated as an engineering discipline. Today it serves **180,000+ memories** live — 100% embedding coverage, 60,000+ knowledge-graph edges, backing a production agent fleet around the clock.

## Features

- **Hybrid retrieval** — BM25 keyword precision and vector semantic similarity, fused with reciprocal-rank fusion; each leg covers the other's blind spots.
- **Knowledge graph** — memories connect to memories, so recall can follow relationships, not just similarity.
- **Built-in eval harness** — retrieval changes are proven with recall@k, MRR, and latency percentiles *before* they ship. No eval, no quality claim.
- **Never stale by policy** — a contradicted fact gets corrected the moment it's detected.

A note on scale, kept honest: the architecture is *designed* for billions of memory nodes — partitioned staged recall so no query ever scans the full corpus, ANN indexing, bounded candidate sets at every stage. That's a property of the design. The 180,000+ memories serving today are the *measured* number. The two never get conflated, which is the same standard the engine holds its own retrieval changes to.

## How it works

Recall runs in stages. A query fans out to the keyword and vector legs in parallel, each returning a bounded candidate set; reciprocal-rank fusion merges them into one ranked list. The knowledge graph then lets recall walk from a matched memory to the ones it's connected to, so an answer can follow relationships instead of stopping at similarity. Every stage is measured by the built-in eval harness before a change to it ships.

Built on **[scour](https://github.com/SA-Ark/scour)** — the zero-dependency retrieval primitives underneath: BM25, HNSW, RRF, chunking, full source, live demo.
