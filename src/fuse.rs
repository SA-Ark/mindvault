//! Reciprocal Rank Fusion over scored memory lists, plus the final
//! re-scoring (decay × fused score + importance bonus) used by hybrid
//! recall. Pure functions — fully unit-testable without a database.

use crate::model::{decay_multiplier, Memory, ScoredMemory};
use std::collections::HashMap;

pub const RRF_K: f64 = 60.0;
/// Additive bonus per unit of importance in final scoring.
pub const IMPORTANCE_WEIGHT: f64 = 0.05;
/// Flat additive boost for memories linked to a query-matched entity.
pub const ENTITY_BOOST: f64 = 0.05;

/// Accumulate RRF contributions (`1 / (k + rank + 1)`) from one ranked
/// list (best-first) into `fused`, keyed by memory id.
pub fn fuse_ranked(fused: &mut HashMap<i64, (Memory, f64)>, ranked: &[(Memory, f64)], k: f64) {
    for (rank, (memory, _native_score)) in ranked.iter().enumerate() {
        let entry = fused
            .entry(memory.id)
            .or_insert_with(|| (memory.clone(), 0.0));
        entry.1 += 1.0 / (k + rank as f64 + 1.0);
    }
}

/// Add a graph-neighbor boost: a neighbor of the rank-`rank` semantic hit
/// receives the same reciprocal contribution that hit would at that rank.
pub fn boost_neighbor(fused: &mut HashMap<i64, (Memory, f64)>, neighbor: &Memory, rank: usize) {
    let entry = fused
        .entry(neighbor.id)
        .or_insert_with(|| (neighbor.clone(), 0.0));
    entry.1 += 1.0 / (RRF_K + rank as f64 + 1.0);
}

/// Add the flat entity boost for a memory linked to a matched KG entity.
pub fn boost_entity(fused: &mut HashMap<i64, (Memory, f64)>, memory: &Memory) {
    let entry = fused
        .entry(memory.id)
        .or_insert_with(|| (memory.clone(), 0.0));
    entry.1 += ENTITY_BOOST;
}

/// Final stage: apply time decay and importance bonus, sort best-first
/// (deterministic: ties break by memory id), truncate to `limit`.
pub fn finalize(
    fused: HashMap<i64, (Memory, f64)>,
    now_unix: i64,
    limit: usize,
) -> Vec<ScoredMemory> {
    let mut results: Vec<ScoredMemory> = fused
        .into_values()
        .map(|(memory, score)| {
            let decay = decay_multiplier(&memory, now_unix);
            let final_score = (score * decay) + (memory.importance * IMPORTANCE_WEIGHT);
            ScoredMemory {
                score: final_score,
                memory,
            }
        })
        .collect();
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.memory.id.cmp(&b.memory.id))
    });
    results.truncate(limit);
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    fn memory(id: i64, importance: f64, last_accessed: i64) -> Memory {
        Memory {
            id,
            content_hash: format!("h{id}"),
            content: format!("content {id}"),
            memory_type: Some("fact".into()),
            tags: vec![],
            importance,
            access_count: 0,
            last_accessed: Some(last_accessed),
            created_at: last_accessed,
            updated_at: last_accessed,
        }
    }

    const NOW: i64 = 1_700_000_000;

    #[test]
    fn memory_in_both_legs_wins() {
        let semantic = vec![(memory(1, 0.0, NOW), 0.9), (memory(2, 0.0, NOW), 0.8)];
        let keyword = vec![(memory(3, 0.0, NOW), 12.0), (memory(2, 0.0, NOW), 8.0)];

        let mut fused = HashMap::new();
        fuse_ranked(&mut fused, &semantic, RRF_K);
        fuse_ranked(&mut fused, &keyword, RRF_K);
        let results = finalize(fused, NOW, 3);

        assert_eq!(results[0].memory.id, 2);
    }

    #[test]
    fn importance_breaks_near_ties() {
        let semantic = vec![(memory(1, 0.0, NOW), 0.9), (memory(2, 2.0, NOW), 0.89)];
        let mut fused = HashMap::new();
        fuse_ranked(&mut fused, &semantic, RRF_K);
        let results = finalize(fused, NOW, 2);

        // id=2 ranks second by RRF but its importance bonus dominates the
        // tiny rank-1-vs-rank-2 reciprocal gap.
        assert_eq!(results[0].memory.id, 2);
    }

    #[test]
    fn stale_operational_memory_decays_below_fresh_one() {
        let week_ago = NOW - 7 * 24 * 3600;
        let mut stale = memory(1, 0.0, week_ago);
        stale.memory_type = Some("note".into());
        let fresh = memory(2, 0.0, NOW);

        let semantic = vec![(stale, 0.95), (fresh, 0.94)];
        let mut fused = HashMap::new();
        fuse_ranked(&mut fused, &semantic, RRF_K);
        let results = finalize(fused, NOW, 2);

        assert_eq!(results[0].memory.id, 2);
    }

    #[test]
    fn neighbor_boost_pulls_in_unranked_memory() {
        let semantic = vec![(memory(1, 0.0, NOW), 0.9)];
        let mut fused = HashMap::new();
        fuse_ranked(&mut fused, &semantic, RRF_K);
        boost_neighbor(&mut fused, &memory(99, 0.0, NOW), 0);

        let results = finalize(fused, NOW, 5);
        assert!(results.iter().any(|r| r.memory.id == 99));
    }

    #[test]
    fn finalize_is_deterministic_on_exact_ties() {
        let mut fused = HashMap::new();
        fuse_ranked(&mut fused, &[(memory(7, 0.0, NOW), 0.5)], RRF_K);
        fuse_ranked(&mut fused, &[(memory(3, 0.0, NOW), 0.5)], RRF_K);
        let results = finalize(fused, NOW, 2);
        assert_eq!(results[0].memory.id, 3); // tie -> lower id first
    }
}
