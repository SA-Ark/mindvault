//! Seed a synthetic corpus and measure hybrid recall latency.
//!
//! ```text
//! MINDVAULT_BENCH_DATABASE_URL=postgres://postgres:pw@localhost:5440/postgres \
//!   cargo run --release --example bench
//! ```

use mindvault::embed::hash_embed;
use mindvault::eval::percentile;
use mindvault::store::MemoryStore;
use std::time::Instant;

const DIM: usize = 384;
const N: usize = 10_000;
const QUERIES: usize = 200;

const TOPICS: [&str; 8] = [
    "deployment pipelines and release gating",
    "vector search and embedding indexes",
    "postgres tuning and connection pooling",
    "incident response and on-call runbooks",
    "authentication tokens and session storage",
    "gpu inference batching and latency",
    "frontend bundle size and hydration cost",
    "message queues and retry semantics",
];

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let url = std::env::var("MINDVAULT_BENCH_DATABASE_URL")
        .expect("set MINDVAULT_BENCH_DATABASE_URL to a scratch pgvector database");
    let store = MemoryStore::connect(&url).await?;

    let existing = store.count().await?;
    if (existing as usize) < N {
        println!("seeding {N} memories ...");
        let started = Instant::now();
        for i in existing as usize..N {
            let topic = TOPICS[i % TOPICS.len()];
            let content = format!("memory {i}: notes about {topic}, variant {}", i / 8);
            store
                .store(&content, Some("fact"), &[], 0.5, &hash_embed(&content, DIM))
                .await?;
            if i % 1000 == 0 {
                println!("  {i}/{N}");
            }
        }
        println!("seeded in {:.1?}", started.elapsed());
    } else {
        println!("corpus already seeded ({existing} memories)");
    }

    let mut latencies = Vec::with_capacity(QUERIES);
    for q in 0..QUERIES {
        let query = format!("notes about {}", TOPICS[q % TOPICS.len()]);
        let embedding = hash_embed(&query, DIM);
        let started = Instant::now();
        let results = store.hybrid_recall(&query, &embedding, 10).await?;
        latencies.push(started.elapsed().as_secs_f64() * 1000.0);
        assert!(!results.is_empty());
    }
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

    println!(
        "\nhybrid recall over {N} memories, {QUERIES} queries, k=10:\n  p50 {:.1} ms · p95 {:.1} ms · max {:.1} ms",
        percentile(&latencies, 0.50),
        percentile(&latencies, 0.95),
        latencies.last().unwrap()
    );
    Ok(())
}
