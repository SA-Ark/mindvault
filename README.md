# MindVault

**Memory for AI systems — architected for billion-node scale, measured every step of the way.**

MindVault is a production memory engine. It gives AI systems durable, searchable, connected memory: hybrid semantic + keyword recall over a knowledge graph, with retrieval quality treated as an engineering discipline instead of a vibe. This repo is an architecture showcase; the engine is a commercial product and its source and internals stay private.

## The scale claim, honestly

| Rung | Claim | Basis |
|---|---|---|
| **Architecture** | Designed for **billions of memory nodes** | Partitioned staged recall — no query ever scans the full corpus; ANN indexing; bounded candidate sets at every stage. Capacity is a property of the design, not a promise about hardware. |
| **Production** | **180,000+ memories** live today | 100% embedding coverage · 60,000+ knowledge-graph edges · serving a production agent fleet around the clock |

Architecture is a design claim, production is a measured claim, and the two never get conflated — which is the same standard the engine holds its own retrieval changes to.

## What this buys you

Most AI systems start every conversation from zero — whatever your business taught them yesterday is gone by morning. MindVault is the other path: an AI that remembers the customers, the decisions, and the hard-won facts, and surfaces the right one when it matters. Institutional memory that compounds instead of evaporating.

## What it does

- **Hybrid recall** — semantic similarity and keyword precision, fused; each covers the other's blind spots
- **Knowledge graph** — memories connect to memories, so recall can follow relationships, not just similarity
- **Built-in evaluation** — retrieval changes are proven with recall@k, MRR, and latency percentiles *before* they ship. No eval, no quality claim.
- **Never stale by policy** — a contradicted fact gets corrected the moment it's detected

## Who uses it

MindVault is the memory layer behind a production AI agent fleet running 120+ services. Every architectural decision above was forced by real operational pain, not sketched on a whiteboard.

## Why the source is private

The recall pipeline's staging, scoring, and tuning *are* the product. What's shareable is open: [**scour**](https://github.com/SA-Ark/scour) is the zero-dependency retrieval primitives layer underneath it — BM25, HNSW, RRF, chunking — full source, live demo.

---

**Product & contact:** [ark.chakrakali.com](https://ark.chakrakali.com)
