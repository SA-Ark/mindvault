//! # MindVault
//!
//! RAG + knowledge-graph memory engine on PostgreSQL/pgvector.
//!
//! - [`store::MemoryStore`] — content-addressed memory persistence with a
//!   six-stage hybrid recall pipeline (pgvector cosine + FTS + RRF fusion
//!   + graph-neighbor boost + entity boost + time decay)
//! - [`embed::Embedder`] — pluggable embedding providers (any
//!   OpenAI-compatible endpoint, or the deterministic test embedder)
//! - [`eval`] — a retrieval evaluation harness (recall@k, MRR, latency
//!   percentiles) so recall quality is measured, not assumed
//! - [`model`] — the memory model and the per-type time-decay policy
//!
//! ## Quick start
//!
//! ```no_run
//! use mindvault::{embed::{Embedder, HashEmbedder}, store::MemoryStore};
//!
//! # async fn demo() -> anyhow::Result<()> {
//! let store = MemoryStore::connect("postgres://localhost/mindvault").await?;
//! let embedder = HashEmbedder::new(384); // swap for HttpEmbedder in prod
//!
//! let embedding = embedder.embed("the deploy pipeline gates on tests").await?;
//! store.store(
//!     "the deploy pipeline gates on tests",
//!     Some("fact"),
//!     &["ci".into()],
//!     1.0,
//!     &embedding,
//! ).await?;
//!
//! let hits = store.hybrid_recall(
//!     "what gates the deploy?",
//!     &embedder.embed("what gates the deploy?").await?,
//!     5,
//! ).await?;
//! # Ok(()) }
//! ```

pub mod embed;
pub mod eval;
pub mod fuse;
pub mod model;
pub mod store;

pub use model::{Memory, ScoredMemory};
pub use store::MemoryStore;
