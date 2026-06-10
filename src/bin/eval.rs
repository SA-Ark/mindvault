//! `mindvault-eval` — run the retrieval eval harness against a live store.
//!
//! ```text
//! mindvault-eval --database-url postgres://... --cases eval/cases.jsonl --k 10
//! ```
//!
//! Embedding provider is selected by env:
//! - `MINDVAULT_EMBED_ENDPOINT` + `MINDVAULT_EMBED_MODEL` + `MINDVAULT_EMBED_DIM`
//!   → OpenAI-compatible HTTP embedder (`MINDVAULT_EMBED_API_KEY` optional)
//! - otherwise → deterministic hash embedder (plumbing/smoke runs only)

use anyhow::{Context, Result};
use mindvault::embed::{Embedder, HashEmbedder, HttpEmbedder};
use mindvault::eval;
use mindvault::store::MemoryStore;

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let database_url = arg_value(&args, "--database-url")
        .or_else(|| std::env::var("DATABASE_URL").ok())
        .context("--database-url or DATABASE_URL required")?;
    let cases_path = arg_value(&args, "--cases").context("--cases <file.jsonl> required")?;
    let k: usize = arg_value(&args, "--k")
        .unwrap_or_else(|| "10".into())
        .parse()
        .context("--k must be an integer")?;

    let store = MemoryStore::connect(&database_url).await?;
    let cases = eval::parse_cases(&std::fs::read_to_string(&cases_path)?)?;

    let embedder: Box<dyn Embedder> = match (
        std::env::var("MINDVAULT_EMBED_ENDPOINT"),
        std::env::var("MINDVAULT_EMBED_MODEL"),
        std::env::var("MINDVAULT_EMBED_DIM"),
    ) {
        (Ok(endpoint), Ok(model), Ok(dim)) => {
            let mut e = HttpEmbedder::new(endpoint, model, dim.parse()?);
            if let Ok(key) = std::env::var("MINDVAULT_EMBED_API_KEY") {
                e = e.with_api_key(key);
            }
            Box::new(e)
        }
        _ => {
            eprintln!("note: no MINDVAULT_EMBED_* env set — using the hash embedder (smoke mode)");
            Box::new(HashEmbedder::new(384))
        }
    };

    let report = eval::run(&store, embedder.as_ref(), &cases, k).await?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
