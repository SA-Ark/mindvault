# MindVault

**Memory for AI systems — architected for billion-node scale, measured every step of the way.**

MindVault is a production memory engine: it gives AI systems durable, searchable, connected
memory — hybrid semantic + keyword recall over a knowledge graph, with retrieval quality treated
as an engineering discipline rather than a vibe. This repo is an architecture showcase; the
engine is a commercial product and its source and internals stay private.

## The scale claim, honestly

| Rung | Claim | Basis |
|---|---|---|
| **Architecture** | Designed for **billions of memory nodes** | Partitioned staged recall — no query ever scans the full corpus; ANN indexing; bounded candidate sets at every stage. Capacity is a property of the design, not a promise about hardware. |
| **Production** | **180,000+ memories** live today | 100% embedding coverage · 60,000+ knowledge-graph edges · serving a production agent fleet around the clock |

That's the same standard the engine applies internally: architecture is a design claim,
production is a measured claim, and the two are never conflated.

## What this means if you're building with AI

Most AI systems wake up every morning with amnesia. Every conversation starts from zero; every
lesson your business taught it yesterday is gone. MindVault is the other path: **an AI that
remembers every customer, every decision, every hard-won fact — permanently — and surfaces the
right one at the moment it matters.** Institutional memory that compounds instead of
evaporating. That's not a feature; it's the difference between a chatbot and an asset.

## What it does

- **Hybrid recall** — semantic similarity and keyword precision, fused; each compensates for the
  other's blind spots
- **Knowledge graph** — memories connect to memories; recall follows relationships, not just similarity
- **Built-in evaluation** — retrieval changes are proven with recall@k, MRR, and latency
  percentiles *before* they ship. If there's no eval, there's no quality claim.
- **Never stale by policy** — contradicted facts are corrected at the moment of detection

## Who uses it

MindVault is the memory layer behind a production AI agent fleet operating 120+ services —
every architectural decision above was forced by real operational pain, not designed on a whiteboard.

## Why the source is private

The recall pipeline's staging, scoring, and tuning **are** the product. What's shareable is open:
[**scour**](https://github.com/SA-Ark/scour) is the open-source, zero-dependency retrieval
primitives layer (BM25, HNSW, RRF, chunking) — full source, live demo.

---

**Product & contact:** [ark.chakrakali.com](https://ark.chakrakali.com)
