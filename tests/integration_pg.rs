//! Integration tests against a real PostgreSQL + pgvector.
//!
//! Skipped unless `MINDVAULT_TEST_DATABASE_URL` is set (CI provides a
//! pgvector service container; locally: `docker run -e POSTGRES_PASSWORD=pw
//! -p 5440:5432 pgvector/pgvector:pg17`).

use mindvault::embed::hash_embed;
use mindvault::store::{content_hash, MemoryStore};

const DIM: usize = 64;

async fn test_store() -> Option<MemoryStore> {
    let url = std::env::var("MINDVAULT_TEST_DATABASE_URL").ok()?;
    let store = MemoryStore::connect(&url)
        .await
        .expect("failed to connect to test database");
    // Clean slate per run.
    sqlx::raw_sql(
        "TRUNCATE memory_edges, kg_entity_memories, kg_relations, kg_entities, \
         memory_embeddings, memories RESTART IDENTITY CASCADE",
    )
    .execute(store.pool())
    .await
    .expect("truncate failed");
    Some(store)
}

macro_rules! require_db {
    () => {
        match test_store().await {
            Some(store) => store,
            None => {
                eprintln!("skipping: MINDVAULT_TEST_DATABASE_URL not set");
                return;
            }
        }
    };
}

#[tokio::test]
async fn store_and_hybrid_recall_round_trip() {
    let store = require_db!();

    let corpus = [
        (
            "the rust borrow checker prevents data races at compile time",
            "fact",
        ),
        ("simmer the ragu for three hours on low heat", "note"),
        (
            "postgres connection pooling caps at eight connections",
            "architecture",
        ),
        (
            "the deploy pipeline gates on the full test suite",
            "workflow",
        ),
    ];
    for (content, mtype) in corpus {
        store
            .store(content, Some(mtype), &[], 0.5, &hash_embed(content, DIM))
            .await
            .unwrap();
    }
    assert_eq!(store.count().await.unwrap(), 4);

    let query = "rust borrow checker data races";
    let results = store
        .hybrid_recall(query, &hash_embed(query, DIM), 2)
        .await
        .unwrap();

    assert!(!results.is_empty());
    assert!(results[0].memory.content.contains("borrow checker"));
}

#[tokio::test]
async fn dedup_by_content_hash() {
    let store = require_db!();

    let content = "exactly the same content stored twice";
    let id1 = store
        .store(content, Some("fact"), &[], 0.1, &hash_embed(content, DIM))
        .await
        .unwrap();
    let id2 = store
        .store(content, Some("fact"), &[], 0.9, &hash_embed(content, DIM))
        .await
        .unwrap();

    assert_eq!(id1, id2);
    assert_eq!(store.count().await.unwrap(), 1);
    // Upsert refreshed importance.
    let m = store
        .get_by_hash(&content_hash(content))
        .await
        .unwrap()
        .unwrap();
    assert!((m.importance - 0.9).abs() < 1e-9);
}

#[tokio::test]
async fn soft_delete_excludes_from_recall() {
    let store = require_db!();

    let content = "a memory that will be deleted";
    store
        .store(content, Some("note"), &[], 0.5, &hash_embed(content, DIM))
        .await
        .unwrap();
    assert!(store.delete(&content_hash(content)).await.unwrap());
    assert!(!store.delete(&content_hash(content)).await.unwrap()); // idempotent

    let results = store
        .hybrid_recall("deleted memory", &hash_embed("deleted memory", DIM), 5)
        .await
        .unwrap();
    assert!(results
        .iter()
        .all(|r| r.memory.content_hash != content_hash(content)));
}

#[tokio::test]
async fn graph_neighbor_boost_surfaces_linked_memory() {
    let store = require_db!();

    let anchor = "kubernetes ingress routes traffic to services";
    let neighbor = "unrelated text about watercolor painting techniques";
    let anchor_id = store
        .store(anchor, Some("fact"), &[], 0.5, &hash_embed(anchor, DIM))
        .await
        .unwrap();
    let neighbor_id = store
        .store(neighbor, Some("fact"), &[], 0.5, &hash_embed(neighbor, DIM))
        .await
        .unwrap();
    store.link(anchor_id, neighbor_id, 1.0).await.unwrap();

    let query = "kubernetes ingress traffic";
    let results = store
        .hybrid_recall(query, &hash_embed(query, DIM), 5)
        .await
        .unwrap();

    // The watercolor memory shares nothing with the query but rides in on
    // the graph edge from the anchor.
    assert!(results.iter().any(|r| r.memory.id == neighbor_id));
    assert_eq!(results[0].memory.id, anchor_id);
}

#[tokio::test]
async fn eval_harness_runs_end_to_end() {
    let store = require_db!();

    let docs = [
        "alpha document about distributed consensus and raft",
        "beta document about sourdough starter hydration",
        "gamma document about gpu memory bandwidth limits",
    ];
    for d in docs {
        store
            .store(d, Some("fact"), &[], 0.5, &hash_embed(d, DIM))
            .await
            .unwrap();
    }

    let cases = vec![
        mindvault::eval::EvalCase {
            query: "distributed consensus raft".into(),
            relevant: vec![content_hash(docs[0])],
        },
        mindvault::eval::EvalCase {
            query: "gpu memory bandwidth".into(),
            relevant: vec![content_hash(docs[2])],
        },
    ];

    let embedder = mindvault::embed::HashEmbedder::new(DIM);
    let report = mindvault::eval::run(&store, &embedder, &cases, 3)
        .await
        .unwrap();

    assert_eq!(report.cases, 2);
    assert!(report.recall_at_k > 0.99, "recall {}", report.recall_at_k);
    assert!(report.mrr > 0.99, "mrr {}", report.mrr);
    assert!(report.latency_p95_ms > 0.0);
}
