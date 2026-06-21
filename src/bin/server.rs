//! `mindvault-server` — the demo API + playground UI.
//!
//! Env:
//!   DATABASE_URL              PostgreSQL+pgvector connection string (required)
//!   PORT                      listen port (default 3930)
//!   MINDVAULT_EMBED_DIM       embedding dimensions (default 384)
//!   MINDVAULT_SEED            "1" (default) seeds the demo corpus if the store is empty
//!
//! Embedder selection (self-hosted by default):
//!   MINDVAULT_EMBED_ENDPOINT + MINDVAULT_EMBED_MODEL [+ MINDVAULT_EMBED_API_KEY]
//!     → OpenAI-compatible HTTP embedder (e.g. a local TEI / llama.cpp server)
//!   otherwise → deterministic HashEmbedder (no external service required)

use anyhow::{Context, Result};
use mindvault::embed::{Embedder, HashEmbedder, HttpEmbedder};
use mindvault::seed;
use mindvault::server::{router, AppState};
use mindvault::store::MemoryStore;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    let database_url = std::env::var("DATABASE_URL")
        .context("DATABASE_URL (a PostgreSQL+pgvector URL) is required")?;
    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(3930);
    let dim: usize = std::env::var("MINDVAULT_EMBED_DIM")
        .ok()
        .and_then(|d| d.parse().ok())
        .unwrap_or(384);

    let store = MemoryStore::connect(&database_url).await?;

    let (embedder, embedder_name): (Box<dyn Embedder>, String) = match (
        std::env::var("MINDVAULT_EMBED_ENDPOINT"),
        std::env::var("MINDVAULT_EMBED_MODEL"),
    ) {
        (Ok(endpoint), Ok(model)) => {
            let mut e = HttpEmbedder::new(endpoint, &model, dim);
            if let Ok(key) = std::env::var("MINDVAULT_EMBED_API_KEY") {
                e = e.with_api_key(key);
            }
            (Box::new(e), format!("http:{model}"))
        }
        _ => (
            Box::new(HashEmbedder::new(dim)),
            "hash (deterministic, self-hosted)".to_string(),
        ),
    };

    // Seed the demo corpus on first boot so the playground is instantly
    // testable. Idempotent (content-addressed upsert), but we only auto-seed
    // an empty store to avoid surprising an operator with a real corpus.
    let seed_on = std::env::var("MINDVAULT_SEED")
        .map(|v| v != "0")
        .unwrap_or(true);
    if seed_on && store.count().await? == 0 {
        eprintln!("seeding demo corpus ...");
        let n = seed::seed(&store, embedder.as_ref()).await?;
        eprintln!("seeded — {n} memories");
    }

    let state = Arc::new(AppState {
        store,
        embedder,
        embedder_name,
        embed_dim: dim,
    });

    let app = router(state);
    let addr = format!("0.0.0.0:{port}");
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("failed to bind {addr}"))?;
    eprintln!("mindvault-server listening on http://{addr}");
    axum::serve(listener, app).await?;
    Ok(())
}
