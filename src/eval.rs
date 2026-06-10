//! Retrieval evaluation harness: measure recall@k, MRR, and latency
//! percentiles against a labeled query set. "RAG that isn't measured is
//! RAG that's vibing."
//!
//! Labeled set format (JSONL), one case per line:
//! ```json
//! {"query": "how does decay work", "relevant": ["<content_hash>", "..."]}
//! ```

use crate::embed::Embedder;
use crate::store::MemoryStore;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Debug, Clone, Deserialize)]
pub struct EvalCase {
    pub query: String,
    /// Content hashes of the memories that should be retrieved.
    pub relevant: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EvalReport {
    pub cases: usize,
    pub k: usize,
    pub recall_at_k: f64,
    pub mrr: f64,
    pub hit_rate: f64,
    pub latency_p50_ms: f64,
    pub latency_p95_ms: f64,
}

/// Parse a JSONL labeled set.
pub fn parse_cases(jsonl: &str) -> Result<Vec<EvalCase>> {
    jsonl
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| serde_json::from_str(l).with_context(|| format!("bad eval case: {l}")))
        .collect()
}

/// Run the labeled set through `store.hybrid_recall` and score it.
pub async fn run(
    store: &MemoryStore,
    embedder: &dyn Embedder,
    cases: &[EvalCase],
    k: usize,
) -> Result<EvalReport> {
    anyhow::ensure!(!cases.is_empty(), "eval set is empty");
    let mut recall_sum = 0.0;
    let mut rr_sum = 0.0;
    let mut hits = 0usize;
    let mut latencies = Vec::with_capacity(cases.len());

    for case in cases {
        let embedding = embedder.embed(&case.query).await?;
        let started = Instant::now();
        let results = store.hybrid_recall(&case.query, &embedding, k).await?;
        latencies.push(started.elapsed().as_secs_f64() * 1000.0);

        let retrieved: Vec<&str> = results
            .iter()
            .map(|r| r.memory.content_hash.as_str())
            .collect();

        let relevant_found = case
            .relevant
            .iter()
            .filter(|hash| retrieved.contains(&hash.as_str()))
            .count();
        recall_sum += relevant_found as f64 / case.relevant.len().max(1) as f64;
        if relevant_found > 0 {
            hits += 1;
        }

        let first_rank = retrieved
            .iter()
            .position(|hash| case.relevant.iter().any(|r| r == hash));
        rr_sum += first_rank.map(|p| 1.0 / (p as f64 + 1.0)).unwrap_or(0.0);
    }

    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    Ok(EvalReport {
        cases: cases.len(),
        k,
        recall_at_k: recall_sum / cases.len() as f64,
        mrr: rr_sum / cases.len() as f64,
        hit_rate: hits as f64 / cases.len() as f64,
        latency_p50_ms: percentile(&latencies, 0.50),
        latency_p95_ms: percentile(&latencies, 0.95),
    })
}

/// Nearest-rank percentile over an ascending-sorted slice.
pub fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = (p.clamp(0.0, 1.0) * sorted.len() as f64).ceil() as usize;
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_jsonl_cases_and_skips_comments() {
        let jsonl = r#"
# labeled retrieval set
{"query": "alpha", "relevant": ["h1", "h2"]}
{"query": "beta", "relevant": ["h3"]}
"#;
        let cases = parse_cases(jsonl).unwrap();
        assert_eq!(cases.len(), 2);
        assert_eq!(cases[0].relevant.len(), 2);
    }

    #[test]
    fn malformed_line_is_an_error() {
        assert!(parse_cases("{not json}").is_err());
    }

    #[test]
    fn percentile_nearest_rank() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        assert_eq!(percentile(&v, 0.50), 5.0);
        assert_eq!(percentile(&v, 0.95), 10.0);
        assert_eq!(percentile(&v, 1.0), 10.0);
        assert_eq!(percentile(&[], 0.5), 0.0);
    }
}
